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

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to 256 + n.
fn byte_chars() -> [char; 256] {
    let mut bs: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut t = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        t[*b as usize] = char::from_u32(c).unwrap();
    }
    t
}

thread_local! {
    static SCRATCH: std::cell::RefCell<(tk_encode::pipeline::PreTokenizerScratch, Vec<tk_encode::pipeline::Span>)> =
        std::cell::RefCell::new(Default::default());
}

/// Train a BPE vocab on the text file at `path` with PR #2501's own feed: each line (with its `\n`) is normalized
/// and pre-tokenized by the normalizers and pre-tokenizer of `tokenizer_json` (HF's own parts), byte-level words
/// are mapped to the GPT-2 byte alphabet, and the words are fed to its `BpeTrainer`, as `Tokenizer::train` did in
/// 0.22. The baseline `train_bpe` is compared against; threads: `RAYON_NUM_THREADS`.
/// Returns `(vocab, merges)`.
#[pyfunction]
#[pyo3(signature = (path, tokenizer_json, vocab_size, byte_level=true))]
#[allow(clippy::type_complexity)]
pub fn train_bpe_pr2501(
    py: Python<'_>,
    path: PathBuf,
    tokenizer_json: PathBuf,
    vocab_size: usize,
    byte_level: bool,
) -> PyResult<(HashMap<String, u32>, Vec<(String, String)>)> {
    use std::io::BufRead;
    use tk_encode::pipeline::{normalize_all, PreTokenizer};
    use tk_train::Trainer;
    let e = |x: &dyn std::fmt::Display| PyValueError::new_err(x.to_string());
    let text = std::fs::read_to_string(&tokenizer_json).map_err(|x| e(&x))?;
    let mut j: serde_json::Value = serde_json::from_str(&text).map_err(|x| e(&x))?;
    if byte_level {
        // HF main refuses a byte-level vocab that lacks one of the 256 bytes: start from the bare alphabet
        let alpha: serde_json::Map<String, serde_json::Value> =
            byte_chars().iter().enumerate().map(|(i, c)| (c.to_string(), i.into())).collect();
        j["model"]["vocab"] = alpha.into();
        j["model"]["merges"] = serde_json::Value::Array(vec![]);
    }
    let canonical = tk_convert::canonicalize_str(&j.to_string()).map_err(|x| e(&x))?;
    let tk = tk_serialize::from_json(&canonical).map_err(|x| e(&x))?;
    let file = std::fs::File::open(&path).map_err(|x| e(&x))?;
    py.detach(|| {
        let (norms, pre, map) = (tk.get_normalizers(), tk.get_pre_tokenizer(), byte_chars());
        let mut trainer = tk_train::trainers::BpeTrainer::builder().vocab_size(vocab_size).show_progress(false).build();
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
        let lines = std::iter::from_fn(move || {
            let mut l = Vec::new();
            (reader.read_until(b'\n', &mut l).ok()? > 0).then(|| String::from_utf8_lossy(&l).into_owned())
        });
        trainer
            .feed(lines, |s: &str| {
                let text = normalize_all(norms, s, 0)?;
                SCRATCH.with(|c| {
                    let (scratch, spans) = &mut *c.borrow_mut();
                    spans.clear();
                    pre.pre_tokenize(&text, scratch, spans)?;
                    Ok(spans
                        .iter()
                        .map(|sp| {
                            let w = &text[sp.start as usize..sp.end as usize];
                            if byte_level { w.bytes().map(|x| map[x as usize]).collect() } else { w.to_string() }
                        })
                        .collect())
                })
            })
            .map_err(|x| e(&x))?;
        let (vocab, merges, _) = trainer.train_vocab().map_err(|x| e(&x))?;
        Ok((vocab.into_iter().collect(), merges))
    })
}
