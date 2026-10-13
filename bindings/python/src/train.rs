//! `train_bpe`: BPE training from a text file with toki's parallel word counting and PR #2501's merge loop.
use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Train a BPE vocab on the text file at `path`, one document or paragraph per line.
///
/// Same vocab and merges as HF's `BpeTrainer` fed the file line by line. `pre_tokenizer`: bytelevel (GPT-2),
/// llama3, gpt4o, whitespace, whitespace_split, metaspace, bert, punctuation, digits, unicode_scripts,
/// char_delimiter. `normalizers`: applied in order, from nfc, nfd, nfkc, nfkd, lowercase, strip_accents.
/// `strip_newlines`: drop each line's newline before pre-tokenizing (as feeding `f.read().splitlines()`).
/// Returns `(vocab, merges)`: `{token: id}` and `[(left, right), ...]` in merge order.
#[pyfunction]
#[pyo3(signature = (path, vocab_size, pre_tokenizer="bytelevel", normalizers=Vec::new(), min_frequency=0, threads=None, strip_newlines=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn train_bpe(
    py: Python<'_>,
    path: PathBuf,
    vocab_size: usize,
    pre_tokenizer: &str,
    normalizers: Vec<String>,
    min_frequency: u64,
    threads: Option<usize>,
    strip_newlines: bool,
) -> PyResult<(HashMap<String, u32>, Vec<(String, String)>)> {
    let d = toki::train::Options::default();
    let o = toki::train::Options {
        vocab_size,
        min_frequency,
        threads: threads.unwrap_or(d.threads),
        pre_tokenizer: pre_tokenizer.to_string(),
        normalizer: normalizers.join("\x1e"),
        strip_newlines,
        block_kb: d.block_kb,
    };
    let path = path.to_string_lossy().into_owned();
    let (vocab, merges, _) = py.detach(|| toki::train::train_file(&path, &o, || {})).map_err(PyValueError::new_err)?;
    Ok((vocab, merges))
}
