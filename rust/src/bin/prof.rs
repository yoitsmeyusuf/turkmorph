//! Where does a cold word cost go?  prof DATA_DIR WORDS.txt (one type per line)
//! analyze_known: best of 3 passes (timing noise); encode_word: one cold pass.
use std::time::Instant;
use turkmorph_rs::morph::analyze_known;
use turkmorph_rs::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = Tokenizer::load(&a[1]);
    let words: Vec<Vec<char>> = std::fs::read_to_string(&a[2]).unwrap().lines().map(|l| l.chars().collect()).collect();
    let n = words.len() as f64;
    let lows: Vec<Vec<char>> = words.iter().map(|w| tok.text.tr_lower(w)).collect();
    let mut best = f64::MAX;
    let mut k = 0usize;
    for _ in 0..3 {
        let t = Instant::now();
        for w in &lows {
            k += analyze_known(&tok.t, &tok.lex, &tok.costs, w).len();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("analyze_known: {:.1} us/type ({} analyses)", best / n * 1e6, k / 3);
    let t = Instant::now();
    for w in &words {
        k += tok.encode_word(w).len();
    }
    println!("encode_word (cold): {:.1} us/type", t.elapsed().as_secs_f64() / n * 1e6);
}
