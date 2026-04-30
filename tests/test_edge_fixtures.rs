//! Test E2E su un fixture reale del workspace Edge: il TIFF "Manleva e
//! Verbale di Consegna" che vive in `<workspace>/tests/ROGITO SINTETICO TIFF/`.
//!
//! Verifica che la build dynamic (DLL Leptonica + Tesseract) carichi
//! correttamente, accetti un buffer RGB raw, e produca testo OCR non
//! banale (lingua: `eng` di default — abbiamo solo eng+tur nei tessdata
//! pre-scaricati). Per un OCR italiano accurato bisognerebbe aggiungere
//! `ita.traineddata`; questo test verifica il pipeline FFI base, non la
//! qualità OCR.
//!
//! Skip silenzioso se il fixture non esiste (CI minimal o checkout
//! parziale).

mod common;
use common::*;

use std::path::PathBuf;
use tesseract_rs::TesseractAPI;

/// Risale dal `CARGO_MANIFEST_DIR` di tesseract-rs (=
/// `Semplifica.Tesseract/tesseract-rs/`) alla root del workspace
/// (`/c/Progetti/anonimator/`) e poi a `tests/`.
fn workspace_tests_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // ../../tests   (Semplifica.Tesseract/tesseract-rs -> ../.. = workspace root)
    let candidate = manifest.parent()?.parent()?.join("tests");
    if candidate.is_dir() { Some(candidate) } else { None }
}

#[test]
fn ocr_first_frame_of_workspace_tiff() {
    let Some(tests_dir) = workspace_tests_dir() else {
        eprintln!("[edge-fixture] SKIP: workspace tests/ root non trovata");
        return;
    };
    let tiff_path = tests_dir
        .join("ROGITO SINTETICO TIFF")
        .join("Manleva e Verbale di Consegna.tiff");
    if !tiff_path.is_file() {
        eprintln!("[edge-fixture] SKIP: fixture TIFF mancante: {}", tiff_path.display());
        return;
    }
    eprintln!("[edge-fixture] TIFF input: {}", tiff_path.display());

    // Decodifica TIFF → RGB8 con `image` crate (gestisce multi-page
    // selezionando la prima frame in `image::open`).
    let img = image::open(&tiff_path).expect("image::open TIFF");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let buf: Vec<u8> = rgb.into_raw();
    eprintln!("[edge-fixture] decoded: {}x{} px, {} bytes RGB8", width, height, buf.len());

    // Inizializza Tesseract con tessdata path (l'env var TESSDATA_PREFIX
    // potrebbe non essere settata nel runner test, fallback al path noto).
    let tessdata = get_tessdata_dir();
    eprintln!("[edge-fixture] tessdata dir: {}", tessdata.display());
    let api = TesseractAPI::new();
    api.init(tessdata.to_str().expect("tessdata path utf8"), "eng")
        .expect("Tesseract init eng");

    // RGB 24-bit packed: 3 bytes/pixel, bytes_per_line = width*3
    api.set_image(&buf, width as i32, height as i32, 3, (width * 3) as i32)
        .expect("set_image");

    let confidence_avg = api.recognize().ok();
    eprintln!("[edge-fixture] recognize result: {:?}", confidence_avg);

    let text = api.get_utf8_text().expect("get_utf8_text");
    let preview: String = text.chars().take(400).collect();
    eprintln!("[edge-fixture] OCR preview (first 400 chars):\n{}", preview);
    eprintln!("[edge-fixture] total chars: {}", text.len());

    // Smoke assert: dovremmo aver estratto QUALCOSA. La qualità con
    // tessdata `eng` su un documento italiano sarà bassa, ma deve
    // comunque rilevare > 50 char di testo "alfabetico".
    let alpha_count = text.chars().filter(|c| c.is_alphabetic()).count();
    assert!(
        alpha_count > 50,
        "OCR ha estratto solo {} char alfabetici — pipeline Tesseract+Leptonica probabilmente rotta. Output: {:?}",
        alpha_count, preview
    );
}
