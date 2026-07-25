use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use v11_ws_exact::WhitespaceExactTokenizer;

#[derive(Parser)]
#[command(name = "v11-ws-exact")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Encode {
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Decode {
        #[arg(long)]
        ids: String,
    },
    /// Read a JSON array of strings from stdin and encode/decode them in one process.
    Batch,
    Info,
}

fn read_exact_text(text: Option<String>) -> Result<String> {
    if let Some(text) = text {
        return Ok(text);
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let tokenizer = WhitespaceExactTokenizer::from_file(&cli.model)
        .with_context(|| format!("load {}", cli.model.display()))?;
    match cli.command {
        Command::Encode { text, json } => {
            let text = read_exact_text(text)?;
            let ids = tokenizer.encode(&text);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "adapter": "v11-ws-exact",
                        "text": text,
                        "ids": ids,
                        "n_tokens": ids.len(),
                    }))?
                );
            } else {
                println!(
                    "{}",
                    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
                );
            }
        }
        Command::Decode { ids } => {
            let ids = if ids.is_empty() {
                Vec::new()
            } else {
                ids.split(',')
                    .map(|value| value.trim().parse::<u32>())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("failed to parse --ids")?
            };
            println!("{}", tokenizer.decode(&ids));
        }
        Command::Batch => {
            let mut input = String::new();
            io::stdin().read_to_string(&mut input)?;
            let texts: Vec<String> =
                serde_json::from_str(&input).context("batch input must be a JSON string array")?;
            let rows = texts
                .into_iter()
                .map(|text| {
                    let ids = tokenizer.encode(&text);
                    let decoded = tokenizer.decode(&ids);
                    serde_json::json!({"text": text, "ids": ids, "decoded": decoded})
                })
                .collect::<Vec<_>>();
            println!("{}", serde_json::to_string(&rows)?);
        }
        Command::Info => {
            println!("adapter: v11-ws-exact");
            println!("vocab_size: {}", tokenizer.vocab_size());
            println!("space_byte_id: {}", tokenizer.space_byte_id());
            println!("rows_added: 0");
        }
    }
    Ok(())
}
