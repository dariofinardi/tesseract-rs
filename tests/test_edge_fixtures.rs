//! E2E test on a real TIFF fixture via the `TESSERACT_TEST_TIFF` environment variable.
//!
//! Verifies that the dynamic build (Leptonica + Tesseract DLL) loads correctly,
//! accepts an RGB raw buffer, and produces non-trivial OCR output.
//!
//! Silently skipped if the env var is not set (CI minimal or partial checkout).

mod common;
use common::*;

use std::path::PathBuf;
use tesseract_rs::TesseractAPI;

fn tiff_fixture() -> Option<PathBuf> {
    let p = std::env::var("TESSERACT_TEST_TIFF").ok().map(PathBuf::from)?;
    if p.is_file() { Some(p) } else { None }
}

#[test]
fn ocr_tiff_fixture() {
    let Some(tiff_path) = tiff_fixture() else {
        eprintln!("[tiff-fixture] SKIP: set TESSERACT_TEST_TIFF=path/to/file.tiff to run");
        return;
    };
    eprintln!("[tiff-fixture] TIFF input: {}", tiff_path.display());

    let img = image::open(&tiff_path).expect("image::open TIFF");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let buf: Vec<u8> = rgb.into_raw();
    eprintln!("[tiff-fixture] decoded: {}x{} px, {} bytes RGB8", width, height, buf.len());

    let tessdata = get_tessdata_dir();
    eprintln!("[tiff-fixture] tessdata dir: {}", tessdata.display());
    let api = TesseractAPI::new();
    api.init(tessdata.to_str().expect("tessdata path utf8"), "eng")
        .expect("Tesseract init eng");

    api.set_image(&buf, width as i32, height as i32, 3, (width * 3) as i32)
        .expect("set_image");

    let confidence_avg = api.recognize().ok();
    eprintln!("[tiff-fixture] recognize result: {:?}", confidence_avg);

    let text = api.get_utf8_text().expect("get_utf8_text");
    let preview: String = text.chars().take(400).collect();
    eprintln!("[tiff-fixture] OCR preview (first 400 chars):\n{}", preview);
    eprintln!("[tiff-fixture] total chars: {}", text.len());

    let alpha_count = text.chars().filter(|c| c.is_alphabetic()).count();
    assert!(
        alpha_count > 50,
        "OCR extracted only {} alphabetic chars — Tesseract+Leptonica pipeline likely broken. Output: {:?}",
        alpha_count, preview
    );
}
