//! Print the vocabulary, one token per line, in id order (compare with Python's Vocab.toks).
use std::io::Write;
use turkmorph_rs::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = Tokenizer::load(a.get(1).map(String::as_str).unwrap_or("../data"));
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for t in &tok.vocab.toks {
        writeln!(out, "{t}").unwrap();
    }
}
