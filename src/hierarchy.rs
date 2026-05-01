//! Hierarchical OCR result types and walker.
//!
//! Tesseract restituisce un albero gerarchico delle entità riconosciute:
//! `Page → Block → Paragraph → TextLine → Word → Symbol`. La maggior parte
//! dei consumer usa solo line-level o aggrega prematuramente, perdendo
//! l'informazione word/bbox utile per highlight, layout analysis e
//! per consumer come Edge che vogliono renderizzare bounding box per
//! parola sopra le immagini.
//!
//! Questo modulo espone l'intera gerarchia come tipi Rust serializzabili
//! e fornisce `TesseractAPI::get_hierarchy()` che cammina il
//! `ResultIterator` interno e produce la struttura completa **senza
//! aggregare nulla**. I consumer possono poi:
//! - usare il livello di granularità che gli serve (line/word/symbol)
//! - serializzare in JSON/YAML per audit / sidecar
//! - aggregare a posteriori per backwards compat (es. Edge mantiene
//!   `recognize_with_bbox` come adapter sopra l'hierarchy).
//!
//! ## Filosofia: il crate ESPONE, il consumer SCEGLIE
//!
//! Il modello ha generato i dati word-level: il crate non ha il diritto
//! di buttarli via prima che il consumer li veda. Vedi memory
//! `project_ocr_granularity_design`.

use serde::{Deserialize, Serialize};

use crate::api::TesseractAPI;
use crate::enums::TessPageIteratorLevel;
use crate::error::Result;

/// Bounding box axis-aligned in coordinate pixel dell'immagine
/// originale (top-left origin). Float in serde per future estensioni
/// a coordinate normalizzate, ma valori sempre integer in Tesseract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub left:   i32,
    pub top:    i32,
    pub right:  i32,
    pub bottom: i32,
}

impl BoundingBox {
    pub fn from_lrtb(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self { left, top, right, bottom }
    }
    pub fn width(&self)  -> i32 { (self.right  - self.left).max(0) }
    pub fn height(&self) -> i32 { (self.bottom - self.top).max(0) }
    pub fn is_empty(&self) -> bool { self.width() == 0 && self.height() == 0 }
}

/// Singola parola riconosciuta. Confidence 0.0..=100.0 (Tesseract scale).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TesseractWord {
    pub text:       String,
    pub bbox:       BoundingBox,
    pub confidence: f32,
}

/// Linea di testo. La confidence è la media delle word.
/// `text` è la concatenazione delle word con spazio singolo (preserva la
/// reading order ma non le distanze inter-word originali).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TesseractTextLine {
    pub text:       String,
    pub bbox:       BoundingBox,
    pub confidence: f32,
    pub words:      Vec<TesseractWord>,
}

/// Paragrafo: 1+ linee con bbox unificato. In documenti senza struttura
/// (immagini scansionate) Tesseract spesso emette 1 paragrafo per blocco.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TesseractParagraph {
    pub bbox:  BoundingBox,
    pub lines: Vec<TesseractTextLine>,
}

/// Block: unità di layout di alto livello (colonna di testo, tabella,
/// figure, ecc.). La maggior parte dei documenti single-column ha 1 block
/// per pagina con 1+ paragrafi.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TesseractBlock {
    pub bbox:       BoundingBox,
    pub paragraphs: Vec<TesseractParagraph>,
}

/// Risultato gerarchico completo dell'OCR su una singola immagine.
/// L'API è single-page (Tesseract supporta multi-page solo via PDF /
/// renderer, non via image direct); per multi-page i consumer possono
/// concatenare più `TesseractHierarchy` ognuno annotato con `page_number`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TesseractHierarchy {
    pub blocks: Vec<TesseractBlock>,
}

impl TesseractHierarchy {
    /// Numero totale di word riconosciute (utile per metrics/log).
    pub fn word_count(&self) -> usize {
        self.blocks.iter()
            .flat_map(|b| &b.paragraphs)
            .flat_map(|p| &p.lines)
            .map(|l| l.words.len())
            .sum()
    }

    /// Iteratore flat su tutte le linee (path comodo per consumer
    /// line-level che vogliono il vec piatto invece dell'hierarchy).
    pub fn iter_lines(&self) -> impl Iterator<Item = &TesseractTextLine> {
        self.blocks.iter()
            .flat_map(|b| b.paragraphs.iter())
            .flat_map(|p| p.lines.iter())
    }

    /// Iteratore flat su tutte le word.
    pub fn iter_words(&self) -> impl Iterator<Item = &TesseractWord> {
        self.iter_lines().flat_map(|l| l.words.iter())
    }
}

impl TesseractAPI {
    /// Esegue OCR e ritorna l'**intera gerarchia** del riconoscimento
    /// senza aggregare prematuramente: ogni livello (block / paragraph
    /// / line / word) è esposto con il proprio bbox e confidence.
    ///
    /// Pre-requisiti:
    /// - immagine già caricata via `set_image`
    /// - lingue configurate via `init`
    ///
    /// Cammina il `ResultIterator` word-by-word usando
    /// `is_at_beginning_of(BLOCK/PARA/TEXTLINE)` per detectare i confini
    /// e popolare la struttura nidificata. La complessità è O(N) con
    /// N = numero di word; nessun pass extra rispetto all'iteratore
    /// nativo.
    ///
    /// Filosofia: questa API è l'inverso di `get_utf8_text()` (testo
    /// piatto) e `get_tsv_text()` (TSV serializzato): ritorna una
    /// struttura Rust nativa che i consumer possono navigare senza
    /// re-parsare.
    pub fn get_hierarchy(&self) -> Result<TesseractHierarchy> {
        // Recognize first (idempotente se già fatto).
        self.recognize()?;
        let it = self.get_iterator()?;

        let mut hierarchy = TesseractHierarchy::default();
        // Working accumulators: l'API Tesseract emette word in ordine
        // reading-order, e i flag is_at_beginning_of(BLOCK/PARA/LINE)
        // ci dicono quando aprire un nuovo "scope" nidificato. Quando
        // il flag è true, prima committiamo il scope precedente
        // (push nel parent), poi apriamo il nuovo.
        let mut cur_block: Option<TesseractBlock>     = None;
        let mut cur_para:  Option<TesseractParagraph> = None;
        let mut cur_line:  Option<TesseractTextLine>  = None;

        loop {
            // Per ogni word: leggi text/bbox/conf e gestisci boundary
            let word_text = match it.get_utf8_text(TessPageIteratorLevel::RIL_WORD) {
                Ok(t)  => t,
                Err(_) => String::new(), // word vuota = noise, skip
            };
            let word_text_trim = word_text.trim();

            // Se la word ha contenuto, processa; altrimenti skip ma
            // continua l'iterazione (avanti pure su word vuote).
            if !word_text_trim.is_empty() {
                let bbox_w = it.get_bounding_box(TessPageIteratorLevel::RIL_WORD)
                    .map(|(l, t, r, b)| BoundingBox::from_lrtb(l, t, r, b))
                    .unwrap_or_default();
                let conf_w = it.confidence(TessPageIteratorLevel::RIL_WORD)
                    .unwrap_or(-1.0);

                // BLOCK boundary
                if it.is_at_beginning_of(TessPageIteratorLevel::RIL_BLOCK)? {
                    commit_para(&mut cur_para, &mut cur_block);
                    commit_block(&mut cur_block, &mut hierarchy);
                    let bbox_b = it.get_bounding_box(TessPageIteratorLevel::RIL_BLOCK)
                        .map(|(l, t, r, b)| BoundingBox::from_lrtb(l, t, r, b))
                        .unwrap_or_default();
                    cur_block = Some(TesseractBlock { bbox: bbox_b, paragraphs: Vec::new() });
                }
                if cur_block.is_none() {
                    // Defensive: se TextLine inizia senza BLOCK boundary signal
                    // (raro ma possibile su immagini degenerate), creiamo un
                    // block "synthetic" con bbox vuoto.
                    cur_block = Some(TesseractBlock::default());
                }

                // PARAGRAPH boundary
                if it.is_at_beginning_of(TessPageIteratorLevel::RIL_PARA)? {
                    commit_para(&mut cur_para, &mut cur_block);
                    let bbox_p = it.get_bounding_box(TessPageIteratorLevel::RIL_PARA)
                        .map(|(l, t, r, b)| BoundingBox::from_lrtb(l, t, r, b))
                        .unwrap_or_default();
                    cur_para = Some(TesseractParagraph { bbox: bbox_p, lines: Vec::new() });
                }
                if cur_para.is_none() {
                    cur_para = Some(TesseractParagraph::default());
                }

                // TEXTLINE boundary
                if it.is_at_beginning_of(TessPageIteratorLevel::RIL_TEXTLINE)? {
                    commit_line(&mut cur_line, &mut cur_para);
                    let bbox_l = it.get_bounding_box(TessPageIteratorLevel::RIL_TEXTLINE)
                        .map(|(l, t, r, b)| BoundingBox::from_lrtb(l, t, r, b))
                        .unwrap_or_default();
                    cur_line = Some(TesseractTextLine {
                        bbox: bbox_l,
                        ..Default::default()
                    });
                }
                if cur_line.is_none() {
                    cur_line = Some(TesseractTextLine::default());
                }

                // Append word to current line
                let line = cur_line.as_mut().unwrap();
                line.words.push(TesseractWord {
                    text:       word_text_trim.to_string(),
                    bbox:       bbox_w,
                    confidence: conf_w.max(0.0),
                });
            }

            // Advance to next word
            if !it.next(TessPageIteratorLevel::RIL_WORD)? {
                break;
            }
        }

        // Commit eventuali scope ancora aperti alla fine dell'iterazione
        commit_line(&mut cur_line, &mut cur_para);
        commit_para(&mut cur_para, &mut cur_block);
        commit_block(&mut cur_block, &mut hierarchy);

        Ok(hierarchy)
    }
}

// ── Helpers privati di commit ────────────────────────────────────────

fn commit_line(cur: &mut Option<TesseractTextLine>, parent: &mut Option<TesseractParagraph>) {
    if let Some(mut line) = cur.take() {
        // Aggrega `text` dalle words con un singolo spazio. Tesseract
        // non ci dà la sequenza esatta inter-word (avremmo dovuto leggere
        // a livello SYMBOL per i caratteri spazio); per il caso d'uso
        // line-level questo è sufficiente.
        line.text = line.words.iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        // Confidence linea = media delle word
        if !line.words.is_empty() {
            line.confidence = line.words.iter().map(|w| w.confidence).sum::<f32>()
                / line.words.len() as f32;
        }
        if let Some(p) = parent {
            p.lines.push(line);
        }
    }
}

fn commit_para(cur: &mut Option<TesseractParagraph>, parent: &mut Option<TesseractBlock>) {
    if let Some(para) = cur.take() {
        if let Some(b) = parent {
            // Skip empty paragraphs (no lines committed)
            if !para.lines.is_empty() {
                b.paragraphs.push(para);
            }
        }
    }
}

fn commit_block(cur: &mut Option<TesseractBlock>, parent: &mut TesseractHierarchy) {
    if let Some(block) = cur.take() {
        if !block.paragraphs.is_empty() {
            parent.blocks.push(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_box_constructor_and_dims() {
        let b = BoundingBox::from_lrtb(10, 20, 110, 60);
        assert_eq!(b.width(), 100);
        assert_eq!(b.height(), 40);
        assert!(!b.is_empty());

        let zero = BoundingBox::default();
        assert!(zero.is_empty());
    }

    #[test]
    fn hierarchy_aggregations_empty() {
        let h = TesseractHierarchy::default();
        assert_eq!(h.word_count(), 0);
        assert_eq!(h.iter_lines().count(), 0);
        assert_eq!(h.iter_words().count(), 0);
    }

    #[test]
    fn hierarchy_aggregations_synthetic() {
        let line = TesseractTextLine {
            text: "Mario Rossi".into(),
            bbox: BoundingBox::from_lrtb(0, 0, 200, 30),
            confidence: 95.0,
            words: vec![
                TesseractWord {
                    text: "Mario".into(),
                    bbox: BoundingBox::from_lrtb(0, 0, 90, 30),
                    confidence: 96.0,
                },
                TesseractWord {
                    text: "Rossi".into(),
                    bbox: BoundingBox::from_lrtb(100, 0, 200, 30),
                    confidence: 94.0,
                },
            ],
        };
        let para = TesseractParagraph {
            bbox: line.bbox,
            lines: vec![line],
        };
        let block = TesseractBlock {
            bbox: para.bbox,
            paragraphs: vec![para],
        };
        let h = TesseractHierarchy { blocks: vec![block] };

        assert_eq!(h.word_count(), 2);
        assert_eq!(h.iter_lines().count(), 1);
        assert_eq!(h.iter_words().count(), 2);
        assert_eq!(h.iter_words().nth(1).unwrap().text, "Rossi");
    }

    #[test]
    fn hierarchy_serializes_to_json() {
        let line = TesseractTextLine {
            text: "ciao".into(),
            bbox: BoundingBox::from_lrtb(1, 2, 3, 4),
            confidence: 90.0,
            words: vec![],
        };
        let json = serde_json::to_string(&line).unwrap();
        assert!(json.contains("\"text\":\"ciao\""));
        assert!(json.contains("\"left\":1"));
        // round-trip
        let parsed: TesseractTextLine = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, line);
    }
}
