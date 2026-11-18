//! TurkMorph runtime in Rust: a port of the Python reference implementation
//! (turkmorph/). Data files are built by the Python pipeline (scripts/).

pub mod bpe;
pub mod langid;
pub mod morph;
pub mod normalize;
pub mod phon;
pub mod text;
pub mod tokenizer;

#[cfg(feature = "python")]
mod py;
