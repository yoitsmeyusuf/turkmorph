//! LEX BPE encoder (port of turkmorph/bpe.py LexBPE.encode); trained in Python.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde_json::Value;

pub const START: char = '▁';

pub struct Bpe {
    ranks: HashMap<(String, String), u32>,
    pub vocab: HashSet<String>,
}

pub enum Piece {
    Lex(String),
    Byte(u8),
}

impl Bpe {
    pub fn load(path: &str) -> Bpe {
        let d: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("lex_bpe.json")).unwrap();
        let base = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîûñáéíóú".to_string()
            + "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXÂÎÛÑÁÉÍÓÚ" + "▁";
        let mut vocab: HashSet<String> = base.chars().map(String::from).collect();
        for a in d["alphabet"].as_array().unwrap() {
            vocab.insert(a.as_str().unwrap().to_string());
        }
        let mut ranks = HashMap::default();
        for (i, m) in d["merges"].as_array().unwrap().iter().enumerate() {
            let (a, b) = (m[0].as_str().unwrap().to_string(), m[1].as_str().unwrap().to_string());
            vocab.insert(format!("{a}{b}"));
            ranks.insert((a, b), i as u32);
        }
        Bpe { ranks, vocab }
    }

    pub fn encode(&self, word: &[char], start: bool) -> Vec<Piece> {
        let mut sym: Vec<String> = Vec::with_capacity(word.len() + 1);
        if start {
            sym.push(START.to_string());
        }
        sym.extend(word.iter().map(|c| c.to_string()));
        while sym.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..sym.len() - 1 {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    let m = format!("{}{}", sym[i], sym[i + 1]);
                    sym.splice(i..i + 2, [m]);
                }
            }
        }
        let mut out = Vec::new();
        for p in sym {
            if self.vocab.contains(&p) {
                out.push(Piece::Lex(p));
            } else {
                out.extend(p.as_bytes().iter().map(|&b| Piece::Byte(b)));
            }
        }
        out
    }
}
