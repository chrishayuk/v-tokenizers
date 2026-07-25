//! Whitespace-exact experiment adapter over the immutable v11 tokenizer.
//!
//! Published v11 uses `Metaspace(prepend_scheme=always)`, which consumes one
//! literal leading ASCII space as though it were the synthetic word-start
//! prefix. This adapter prefixes v11's existing `<0x20>` byte-fallback row
//! when the complete input begins with a literal space, then delegates the
//! complete input unchanged. The adapter decoder restores that one byte.
//! It adds no vocabulary rows and does not alter `v11_core::Tokenizer::encode`.

use std::path::Path;

use v11_core::{Result, Tokenizer};

pub struct WhitespaceExactTokenizer {
    inner: Tokenizer,
    space_byte_id: u32,
}

impl WhitespaceExactTokenizer {
    pub fn from_tokenizer(inner: Tokenizer) -> Result<Self> {
        let space_byte_id = inner.vocab().byte_fallback_id(b' ').ok_or_else(|| {
            v11_core::Error::InvalidVocab(
                "v11-ws-exact requires the existing <0x20> byte-fallback row".to_string(),
            )
        })?;
        Ok(Self {
            inner,
            space_byte_id,
        })
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_tokenizer(Tokenizer::from_file(path)?)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.starts_with(' ') {
            let mut ids = Vec::with_capacity(text.len() / 3 + 2);
            ids.push(self.space_byte_id);
            ids.extend(self.inner.encode(text));
            ids
        } else {
            self.inner.encode(text)
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if ids.first() == Some(&self.space_byte_id) {
            let mut decoded = String::from(" ");
            decoded.push_str(&self.inner.decode(&ids[1..]));
            decoded
        } else {
            self.inner.decode(ids)
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn space_byte_id(&self) -> u32 {
        self.space_byte_id
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../v11/artifacts/v11.vocab.bin")
            .canonicalize()
            .expect("v11 vocab artifact")
    }

    #[test]
    fn preserves_vocab_and_non_leading_ids() {
        let exact = WhitespaceExactTokenizer::from_file(model_path()).unwrap();
        assert_eq!(exact.vocab_size(), 71_260);
        for text in ["", "x", "\tx", "\n x", "a  b", "café", "👩‍💻"] {
            assert_eq!(exact.encode(text), exact.inner().encode(text), "{text:?}");
        }
    }

    #[test]
    fn leading_spaces_use_existing_byte_row() {
        let exact = WhitespaceExactTokenizer::from_file(model_path()).unwrap();
        let mut expected = vec![exact.space_byte_id()];
        expected.extend(exact.inner().encode("  x"));
        assert_eq!(exact.encode("  x"), expected);
        assert_eq!(exact.decode(&expected), "  x");
    }

    #[test]
    fn preserves_canonical_suffix_for_leading_space_inputs() {
        let exact = WhitespaceExactTokenizer::from_file(model_path()).unwrap();
        for text in [" x", "  x", "   "] {
            let encoded = exact.encode(text);
            assert_eq!(encoded[0], exact.space_byte_id());
            assert_eq!(&encoded[1..], exact.inner().encode(text).as_slice());
            assert_eq!(exact.decode(&encoded), text);
        }
    }

    #[test]
    fn whole_input_contract_minimal_battery() {
        let exact = WhitespaceExactTokenizer::from_file(model_path()).unwrap();
        for text in [
            "x",
            " x",
            "  x",
            "\tx",
            "\n x",
            "",
            " ",
            "   ",
            " trailing ",
        ] {
            assert_eq!(exact.decode(&exact.encode(text)), text, "{text:?}");
        }
    }
}
