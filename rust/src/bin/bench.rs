//! Throughput on a text file (one JSON string per line): first pass (cold cache) and second pass.
//!     bench DATA_DIR FILE [THREADS]
use std::time::Instant;
use turkmorph_rs::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a.get(3).map(|x| x.parse().unwrap()).unwrap_or(1);
    let t0 = Instant::now();
    let tok = Tokenizer::load(&a[1]);
    let load = t0.elapsed().as_secs_f64();
    let texts: Vec<String> = std::fs::read_to_string(&a[2]).unwrap().lines()
        .map(|l| serde_json::from_str(l).unwrap()).collect();
    let words: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
    for pass in ["cold", "warm"] {
        let t = Instant::now();
        let n: usize = tok.encode_batch(&texts, threads).iter().map(|v| v.len()).sum();
        let dt = t.elapsed().as_secs_f64();
        println!("{pass} ({threads} threads): {words} words, {n} tokens in {dt:.3}s -> {:.0} words/s", words as f64 / dt);
    }
    println!("load {load:.2}s");
}
