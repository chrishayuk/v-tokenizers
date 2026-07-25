//! v11 — knowledge-first tokenizer CLI.
//!
//! Subcommands:
//!   encode   tokenize text → ids
//!   decode   ids → text
//!   pieces   tokenize text → piece strings (for inspection)
//!   info     print vocab size, specials, sample pieces
//!   lookup   look up specific token text or id
//!   vocab    walk the vocabulary by id, or summarize its block structure

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use v11_core::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "v11", author, version, about = "v11 knowledge-first tokenizer")]
struct Cli {
    /// Path to v11.vocab.bin. If omitted, searches a few conventional
    /// locations.
    #[arg(long, global = true)]
    model: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode text to a list of token IDs
    Encode {
        /// Text to encode (use `-` or omit to read stdin)
        #[arg(long)]
        text: Option<String>,

        /// Print pieces alongside IDs
        #[arg(long)]
        show_pieces: bool,

        /// Output as JSON {"ids": [...], "pieces": [...]}
        #[arg(long)]
        json: bool,
    },

    /// Decode a comma-separated list of token IDs back to text
    Decode {
        /// Comma-separated token IDs (e.g. "2,1024,5,8")
        #[arg(long)]
        ids: String,
    },

    /// Show the piece-string form of each token for inspection
    Pieces {
        #[arg(long)]
        text: Option<String>,
    },

    /// Print vocabulary size, special tokens, and sample entries
    Info,

    /// Look up a token by text or by id
    Lookup {
        /// Token text (wraps in optional ▁ prefix if word-like)
        #[arg(long, conflicts_with = "id")]
        text: Option<String>,

        /// Token ID
        #[arg(long, conflicts_with = "text")]
        id: Option<u32>,
    },

    /// Walk the vocabulary by id, or summarize its block structure.
    ///
    /// A knowledge-first vocabulary is assembled rather than discovered, so
    /// where a piece sits is meaningful: `--blocks` reports the contiguous
    /// runs of same-kind pieces that assembly produces. Kinds are derived from
    /// the pieces themselves, not from a hardcoded map, so this stays honest
    /// if the vocabulary is ever rebuilt — and works on any v11-format vocab.
    Vocab {
        /// First id to show
        #[arg(long, default_value_t = 0)]
        from: u32,

        /// How many ids to show
        #[arg(long, default_value_t = 16)]
        count: u32,

        /// Summarize contiguous same-kind runs across the whole vocabulary
        /// instead of listing individual pieces
        #[arg(long)]
        blocks: bool,

        /// Runs shorter than this are folded into the previous block, so a
        /// stray piece doesn't shatter the summary into noise
        #[arg(long, default_value_t = 4)]
        min_run: usize,

        #[arg(long)]
        json: bool,
    },
}

/// What a piece *is*, judged from its own text. Deliberately coarse: the point
/// is to make assembly visible, not to build a taxonomy.
fn piece_kind(text: &str) -> &'static str {
    if text.starts_with("<0x") && text.ends_with('>') {
        return "byte fallback";
    }
    if text.starts_with('<') && text.ends_with('>') {
        return "special";
    }
    let bare = text.strip_prefix('\u{2581}').unwrap_or(text);
    let mut chars = bare.chars();
    let (first, rest) = (chars.next(), chars.next());
    if let (Some(c), None) = (first, rest) {
        // Single character: the hand-placed prelude of an assembled vocab.
        if c.is_ascii_digit() {
            return "digit";
        }
        if c.is_ascii_alphabetic() {
            return "latin letter";
        }
        if ('\u{0370}'..='\u{03FF}').contains(&c) {
            return "greek letter";
        }
        if c.is_ascii_punctuation() {
            return "punctuation";
        }
        if !c.is_ascii() {
            return "symbol";
        }
        return "other";
    }
    if !bare.is_empty() && bare.chars().all(|c| c.is_ascii_punctuation()) {
        return "operator";
    }
    if !bare.is_empty() && !bare.is_ascii() {
        return "symbol";
    }
    "word piece"
}

/// Contiguous runs of one kind: (start_id, end_id_inclusive, kind, count).
fn blocks_of(pieces: &[(u32, String)], min_run: usize) -> Vec<(u32, u32, &'static str, usize)> {
    let mut out: Vec<(u32, u32, &'static str, usize)> = Vec::new();
    for (id, text) in pieces {
        let kind = piece_kind(text);
        match out.last_mut() {
            Some(last) if last.2 == kind => {
                last.1 = *id;
                last.3 += 1;
            }
            _ => out.push((*id, *id, kind, 1)),
        }
    }
    // Fold short runs into the preceding block so one outlier doesn't split
    // it -- and then merge neighbours that ended up the same kind, otherwise
    // absorbing a stray leaves two adjacent blocks with identical labels.
    let mut folded: Vec<(u32, u32, &'static str, usize)> = Vec::new();
    for run in out {
        match folded.last_mut() {
            Some(prev) if run.3 < min_run || prev.2 == run.2 => {
                prev.1 = run.1;
                prev.3 += run.3;
            }
            _ => folded.push(run),
        }
    }
    folded
}

fn resolve_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("model not found at {}", p.display());
        }
        return Ok(p);
    }
    let candidates = [
        "v11/artifacts/v11.vocab.bin",
        "../v11/artifacts/v11.vocab.bin",
        "../../v11/artifacts/v11.vocab.bin",
        "artifacts/v11.vocab.bin",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }
    bail!(
        "couldn't find v11.vocab.bin — pass --model <path>.\n\
         searched: {}",
        candidates.join(", ")
    );
}

fn read_text_arg(text_arg: Option<String>) -> Result<String> {
    match text_arg.as_deref() {
        Some("-") | None => {
            let mut s = String::new();
            io::stdin().read_to_string(&mut s)?;
            Ok(s.trim_end().to_string())
        }
        Some(t) => Ok(t.to_string()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model_path = resolve_model(cli.model)?;
    let tok = Tokenizer::from_file(&model_path)
        .with_context(|| format!("load {}", model_path.display()))?;

    match cli.command {
        Command::Encode {
            text,
            show_pieces,
            json,
        } => {
            let text = read_text_arg(text)?;
            let ids = tok.encode(&text);
            if json {
                let pieces = tok.decode_pieces(&ids);
                let body = serde_json::json!({
                    "text": text,
                    "ids": ids,
                    "pieces": pieces,
                    "n_tokens": ids.len(),
                });
                println!("{}", serde_json::to_string_pretty(&body)?);
            } else if show_pieces {
                let pieces = tok.decode_pieces(&ids);
                for (id, p) in ids.iter().zip(&pieces) {
                    println!("{id:6}  {p}");
                }
            } else {
                let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                println!("{}", s.join(","));
            }
        }
        Command::Decode { ids } => {
            let ids: Vec<u32> = ids
                .split(',')
                .map(|s| s.trim().parse::<u32>())
                .collect::<std::result::Result<_, _>>()
                .context("failed to parse --ids (expected comma-separated u32s)")?;
            println!("{}", tok.decode(&ids));
        }
        Command::Pieces { text } => {
            let text = read_text_arg(text)?;
            let ids = tok.encode(&text);
            let pieces = tok.decode_pieces(&ids);
            for p in pieces {
                println!("{p}");
            }
        }
        Command::Info => {
            let v = tok.vocab();
            println!("model: {}", model_path.display());
            println!("vocab_size: {}", v.len());
            println!(
                "special: pad={} unk={} bos={} eos={}",
                v.special.pad_id, v.special.unk_id, v.special.bos_id, v.special.eos_id,
            );
            println!("sample pieces:");
            for p in v.pieces.iter().take(20) {
                println!("  {:5}  {}", p.id, p.text);
            }
        }
        Command::Lookup { text, id } => {
            let v = tok.vocab();
            if let Some(text) = text {
                match v.get_id(&text) {
                    Some(id) => println!("{id}\t{text}"),
                    None => {
                        // Try ▁-prefix
                        let prefixed = format!("\u{2581}{text}");
                        match v.get_id(&prefixed) {
                            Some(id) => println!("{id}\t{prefixed}  (word-boundary form)"),
                            None => println!("not in vocab: {text}"),
                        }
                    }
                }
            } else if let Some(id) = id {
                match v.get_text(id) {
                    Some(text) => println!("{id}\t{text}"),
                    None => println!("id out of range: {id}"),
                }
            } else {
                bail!("lookup requires --text or --id");
            }
        }
        Command::Vocab {
            from,
            count,
            blocks,
            min_run,
            json,
        } => {
            let v = tok.vocab();
            let all: Vec<(u32, String)> = v.pieces.iter().map(|p| (p.id, p.text.clone())).collect();

            if blocks {
                let runs = blocks_of(&all, min_run);
                if json {
                    let body: Vec<_> = runs
                        .iter()
                        .map(|(lo, hi, kind, n)| {
                            serde_json::json!({"from": lo, "to": hi, "kind": kind, "count": n})
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&body)?);
                } else {
                    println!("{} pieces in {} blocks", all.len(), runs.len());
                    for (lo, hi, kind, n) in runs {
                        // Sample from inside the block only -- taking a fixed 6
                        // spills into the next block on short runs and makes
                        // the boundary look wrong.
                        let sample: Vec<&str> = all
                            .iter()
                            .filter(|(id, _)| *id >= lo && *id <= hi)
                            .take(6)
                            .map(|(_, t)| t.as_str())
                            .collect();
                        println!("{lo:>6}-{hi:<6} {n:>6}  {kind:<14} {}", sample.join(" "));
                    }
                }
                return Ok(());
            }

            let lo = from as usize;
            let hi = (lo + count as usize).min(all.len());
            if lo >= all.len() {
                bail!(
                    "--from {} is past the end of the vocabulary ({})",
                    from,
                    all.len()
                );
            }
            let window = &all[lo..hi];
            if json {
                let body: Vec<_> = window
                    .iter()
                    .map(|(id, text)| {
                        serde_json::json!({"id": id, "piece": text, "kind": piece_kind(text)})
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&body)?);
            } else {
                for (id, text) in window {
                    println!("{id:>6}  {text}");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(pairs: &[(u32, &str)]) -> Vec<(u32, String)> {
        pairs.iter().map(|(i, t)| (*i, (*t).to_string())).collect()
    }

    #[test]
    fn piece_kind_separates_the_assembled_prelude() {
        assert_eq!(piece_kind("<pad>"), "special");
        assert_eq!(piece_kind("<0x1F>"), "byte fallback");
        assert_eq!(piece_kind("7"), "digit");
        assert_eq!(piece_kind("a"), "latin letter");
        assert_eq!(piece_kind("\u{2581}A"), "latin letter");
        assert_eq!(piece_kind("\u{03B1}"), "greek letter");
        assert_eq!(piece_kind("\u{2581}\u{03A9}"), "greek letter");
        assert_eq!(piece_kind("!"), "punctuation");
        assert_eq!(piece_kind("\u{00D7}"), "symbol");
        assert_eq!(piece_kind("=="), "operator");
        assert_eq!(piece_kind("\u{2581}the"), "word piece");
    }

    #[test]
    fn byte_fallback_is_not_mistaken_for_a_special() {
        // Both are <...>-shaped; ordering in piece_kind decides, and getting it
        // wrong would collapse the 256-piece byte block into "special".
        assert_eq!(piece_kind("<0x00>"), "byte fallback");
        assert_eq!(piece_kind("<unk>"), "special");
    }

    #[test]
    fn blocks_merge_runs_of_the_same_kind() {
        let pieces = v(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "1"),
            (5, "2"),
            (6, "3"),
            (7, "4"),
        ]);
        let runs = blocks_of(&pieces, 4);
        assert_eq!(runs.len(), 2);
        assert_eq!(
            (runs[0].0, runs[0].1, runs[0].2, runs[0].3),
            (0, 3, "latin letter", 4)
        );
        assert_eq!(
            (runs[1].0, runs[1].1, runs[1].2, runs[1].3),
            (4, 7, "digit", 4)
        );
    }

    #[test]
    fn a_stray_piece_does_not_shatter_a_block() {
        // One digit in the middle of letters must not split them into three
        // blocks -- and the two letter runs must come back out as ONE block,
        // not two adjacent ones with the same label.
        let pieces = v(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "7"),
            (5, "e"),
            (6, "f"),
            (7, "g"),
            (8, "h"),
        ]);
        let runs = blocks_of(&pieces, 4);
        assert_eq!(runs.len(), 1, "expected one merged block, got {runs:?}");
        assert_eq!((runs[0].0, runs[0].1, runs[0].3), (0, 8, 9));
    }

    #[test]
    fn min_run_of_one_keeps_every_boundary() {
        let pieces = v(&[(0, "a"), (1, "7"), (2, "b")]);
        let runs = blocks_of(&pieces, 1);
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn empty_vocabulary_yields_no_blocks() {
        assert!(blocks_of(&[], 4).is_empty());
    }
}
