//! Conformance tool: stdin = one JSON string per line; stdout = tokens joined by \x1f,
//! then \t, then "ok"/"BAD" for decode(encode(x)) == x. (scripts/conformance_encode.py)

use std::io::{self, BufRead, Write};
use turkmorph_rs::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = Tokenizer::load(a.get(1).map(String::as_str).unwrap_or("../data"));
    let mut out = io::BufWriter::new(io::stdout());
    for line in io::stdin().lock().lines() {
        let text: String = serde_json::from_str(&line.unwrap()).unwrap();
        let ids = tok.encode(&text);
        let ok = tok.decode(&ids).map_or(false, |d| d == text);
        writeln!(out, "{}\t{}", tok.token_strs(&ids).join("\x1f"), if ok { "ok" } else { "BAD" }).unwrap();
    }
}
