//! Demo — show code-specific single-token hits.

use anyhow::{Context, Result};
use v11_core::Tokenizer;

const VOCAB: &str = "v11/artifacts/v11.vocab.bin";

fn main() -> Result<()> {
    // See basic.rs — this one is also a documented release-process step, so a
    // wrong-directory run should say which file it wanted, not just "No such
    // file or directory".
    let tok = Tokenizer::from_file(VOCAB)
        .with_context(|| format!("couldn't load {VOCAB} — run this from the repo root"))?;

    let checks = [
        (
            "Python",
            &[
                "def", "class", "import", "return", "yield", "async", "await", "lambda",
            ][..],
        ),
        (
            "Rust",
            &[
                "fn", "let", "mut", "struct", "enum", "trait", "impl", "pub", "Vec", "String",
                "i32", "u64", "f64", "Option", "Result",
            ][..],
        ),
        (
            "JS/TS",
            &[
                "function",
                "const",
                "interface",
                "type",
                "extends",
                "implements",
            ][..],
        ),
        (
            "C",
            &[
                "int", "char", "void", "struct", "typedef", "malloc", "printf", "size_t",
                "uint32_t",
            ][..],
        ),
        ("Go", &["func", "package", "chan", "go", "defer"][..]),
        ("Acronyms", &["JSON", "HTTP", "SQL", "API", "URL"][..]),
        ("Greek", &["α", "β", "γ", "π", "σ"][..]),
        (
            "Science",
            &[
                "photosynthesis",
                "chromosome",
                "electromagnetic",
                "hydrogen",
                "thermodynamics",
                "algorithm",
            ][..],
        ),
    ];

    let mut total_hits = 0usize;
    let mut total_checks = 0usize;
    for (label, words) in checks {
        let mut hits = 0usize;
        for w in words {
            let ids = tok.encode(w);
            if ids.len() == 1 {
                hits += 1;
            }
        }
        total_hits += hits;
        total_checks += words.len();
        println!("{label:12}  {hits}/{}", words.len());
    }
    println!("{:12}  {total_hits}/{total_checks}", "TOTAL");
    Ok(())
}
