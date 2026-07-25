//! Demo — encode + decode a few sample sentences.

use anyhow::{Context, Result};
use v11_core::Tokenizer;

const VOCAB: &str = "v11/artifacts/v11.vocab.bin";

fn main() -> Result<()> {
    // The path is repo-relative and the demos take no arguments, so the only
    // way this fails in practice is being run from the wrong directory. Say so
    // -- bare `?` here reports "No such file or directory" naming no file at
    // all, which is a poor thing to hand someone following the release steps.
    let tok = Tokenizer::from_file(VOCAB)
        .with_context(|| format!("couldn't load {VOCAB} — run this from the repo root"))?;

    let samples = [
        "The capital of France is Paris.",
        "def fibonacci(n): return fibonacci(n - 1) + fibonacci(n - 2)",
        "fn main() { println!(\"Hello, world!\"); }",
        "Photosynthesis converts sunlight into chemical energy.",
        "α + β = γ",
    ];

    for s in samples {
        let ids = tok.encode(s);
        let pieces = tok.decode_pieces(&ids);
        let back = tok.decode(&ids);
        println!("\ntext:   {s}");
        println!("pieces: {pieces:?}");
        println!("ids:    {ids:?}");
        println!("back:   {back}");
    }

    Ok(())
}
