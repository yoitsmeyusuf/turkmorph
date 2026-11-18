//! Python binding: `turkmorph._fast.FastTokenizer` (built with maturin, feature "python").
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::tokenizer::Tokenizer;

#[pyclass(frozen)]
struct FastTokenizer {
    tok: Tokenizer,
}

#[pymethods]
impl FastTokenizer {
    #[new]
    fn new(py: Python<'_>, data_dir: &str) -> Self {
        FastTokenizer { tok: py.allow_threads(|| Tokenizer::load(data_dir)) }
    }

    /// text -> token ids
    fn encode_ids(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.allow_threads(|| self.tok.encode(text))
    }

    /// text -> token strings
    fn encode(&self, py: Python<'_>, text: &str) -> Vec<String> {
        let ids = py.allow_threads(|| self.tok.encode(text));
        self.tok.token_strs(&ids).into_iter().map(String::from).collect()
    }

    /// one word -> LEX/byte pieces only, no morphological analysis (lossless:
    /// decode gives the word back). For vocabulary pruning: a word whose
    /// analysis uses a pruned root falls back to this.
    fn encode_lex(&self, word: &str) -> Vec<u32> {
        let w: Vec<char> = word.chars().collect();
        self.tok.bpe_toks(&w, true)
    }

    /// many texts -> token ids, on `threads` threads (the GIL is released)
    #[pyo3(signature = (texts, threads = 6))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, threads: usize) -> Vec<Vec<u32>> {
        py.allow_threads(|| self.tok.encode_batch(&texts, threads))
    }

    fn decode_ids(&self, ids: Vec<u32>) -> PyResult<String> {
        let n = self.tok.vocab.toks.len() as u32;
        if let Some(&bad) = ids.iter().find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!("token id out of range: {bad}")));
        }
        self.tok.decode(&ids).map_err(PyValueError::new_err)
    }

    fn decode(&self, toks: Vec<String>) -> PyResult<String> {
        let mut ids = Vec::with_capacity(toks.len());
        for t in &toks {
            match self.tok.vocab.index.get(t.as_str()) {
                Some(&i) => ids.push(i),
                None => return Err(PyValueError::new_err(format!("not a token: {t:?}"))),
            }
        }
        self.tok.decode(&ids).map_err(PyValueError::new_err)
    }

    fn token_to_id(&self, t: &str) -> PyResult<u32> {
        self.tok.vocab.index.get(t).copied().ok_or_else(|| PyKeyError::new_err(t.to_string()))
    }

    fn id_to_token(&self, i: u32) -> PyResult<String> {
        self.tok.vocab.toks.get(i as usize).cloned().ok_or_else(|| PyKeyError::new_err(i))
    }

    fn vocab(&self) -> Vec<String> {
        self.tok.vocab.toks.clone()
    }

    fn __len__(&self) -> usize {
        self.tok.vocab.toks.len()
    }
}

#[pymodule]
fn _fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastTokenizer>()?;
    Ok(())
}
