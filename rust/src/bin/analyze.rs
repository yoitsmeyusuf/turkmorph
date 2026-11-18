//! Conformance tool: analyses of each stdin word, one line per word:
//!   word \t key chain cost | key chain cost | ...
//! (compare with scripts/conformance_analyze.py)

use std::io::{self, BufRead, Write};
use turkmorph_rs::morph::{analyze_known, Costs, Lexicon, Tables};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a.get(1).map(String::as_str).unwrap_or("../data");
    let t = Tables::load(&format!("{data}/morph_tables.json"));
    let lex = Lexicon::from_tsv(&format!("{data}/lexicon.tsv"), &t);
    let costs = Costs::load(&format!("{data}/costs.json"), &t, &lex);
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let w = line.unwrap();
        let wc: Vec<char> = w.chars().collect();
        let an = analyze_known(&t, &lex, &costs, &wc);
        let parts: Vec<String> = an
            .iter()
            .map(|x| {
                let ch: Vec<String> = x.chain.iter().map(|s| s.name(&t)).collect();
                format!("{} {} {:.12}", lex.roots[x.root as usize].key, ch.join("+"), x.cost)
            })
            .collect();
        writeln!(out, "{}\t{}", w, parts.join(" | ")).unwrap();
    }
}
