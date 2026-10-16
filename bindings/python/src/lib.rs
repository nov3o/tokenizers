//! Python bindings for Hugging Face's Tokenizers rust library
//!
//! Encode text to token ids and decode token ids back to text
mod encoding;
mod error;
mod options;
mod repr;
mod stream;
mod tokenizer;
mod train;
mod type_hints;

#[pyo3::pymodule]
mod tokenizers {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::options::Padding;
    #[pymodule_export]
    use crate::options::Truncation;
    #[pymodule_export]
    use crate::stream::EncodeStream;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;

    #[pymodule_export]
    use crate::train::train_bpe;
    #[pymodule_export]
    use crate::train::train_bpe_pr2501;
}
