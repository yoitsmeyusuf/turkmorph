//! Typo / ASCII-spelling normalization (port of turkmorph/normalize.py).

use crate::langid::LangId;
use crate::text::asciify;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::OnceLock;

pub const EDIT_CHARS: &str = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû";
pub const MAX_POS: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub kind: &'static str, // del ins sub swap
    pub p: usize,
    pub ch: Option<char>,
}

impl Op {
    pub fn token(&self) -> String {
        match self.ch {
            Some(c) => format!("E:{}:{}:{}", self.kind, self.p, c),
            None => format!("E:{}:{}", self.kind, self.p),
        }
    }
    pub fn parse(t: &str) -> Option<Op> {
        let parts: Vec<&str> = t.split(':').collect();
        if !(parts.len() == 3 || parts.len() == 4) || parts[0] != "E" {
            return None;
        }
        let kind = match parts[1] { "del" => "del", "ins" => "ins", "sub" => "sub", "swap" => "swap", _ => return None };
        if parts[2].is_empty() || !parts[2].chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let p = parts[2].parse().ok()?;
        let ch = if parts.len() == 4 { parts[3].chars().next() } else { None };
        Some(Op { kind, p, ch })
    }
}

pub fn apply_edit(c: &[char], op: &Op) -> Vec<char> {
    let p = op.p.min(c.len());
    let mut out = c[..p].to_vec();
    match op.kind {
        "del" => out.extend_from_slice(c.get(p + 1..).unwrap_or(&[])),
        "ins" => {
            out.extend(op.ch);
            out.extend_from_slice(&c[p..]);
        }
        "sub" => {
            out.extend(op.ch);
            out.extend_from_slice(c.get(p + 1..).unwrap_or(&[]));
        }
        _ => {
            out.extend_from_slice(c.get(p + 1..p + 2).unwrap_or(&[]));
            out.extend_from_slice(c.get(p..p + 1).unwrap_or(&[]));
            out.extend_from_slice(c.get(p + 2..).unwrap_or(&[]));
        }
    }
    out
}

/// The single op turning `correct` into `written` (Damerau distance 1), else None.
pub fn edit_op(correct: &[char], written: &[char]) -> Option<Op> {
    let (lc, lw) = (correct.len(), written.len());
    let mut i = 0;
    while i < lc.min(lw) && correct[i] == written[i] {
        i += 1;
    }
    if lw + 1 == lc && correct[..i] == written[..i] && correct[i + 1..] == written[i..] {
        return Some(Op { kind: "del", p: i, ch: None });
    }
    if lw == lc + 1 && correct[i..] == written[i + 1..] {
        return Some(Op { kind: "ins", p: i, ch: Some(written[i]) });
    }
    if lw == lc && i < lc {
        if correct[i + 1..] == written[i + 1..] {
            return Some(Op { kind: "sub", p: i, ch: Some(written[i]) });
        }
        if i + 1 < lc && correct[i + 1] == written[i] && correct[i] == written[i + 1] && correct[i + 2..] == written[i + 2..] {
            return Some(Op { kind: "swap", p: i, ch: None });
        }
    }
    None
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û')
}

fn key_neighbours() -> HashMap<char, Vec<char>> {
    let rows: Vec<Vec<char>> = ["qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç"].iter().map(|r| r.chars().collect()).collect();
    let mut out = HashMap::default();
    for (r, row) in rows.iter().enumerate() {
        for (c, &ch) in row.iter().enumerate() {
            let mut near = Vec::new();
            for j in [c as i64 - 1, c as i64 + 1] {
                if j >= 0 && (j as usize) < row.len() {
                    near.push(row[j as usize]);
                }
            }
            for rr in [r as i64 - 1, r as i64 + 1] {
                if rr >= 0 && (rr as usize) < rows.len() {
                    let other = &rows[rr as usize];
                    for j in [c as i64 - 1, c as i64, c as i64 + 1] {
                        if j >= 0 && (j as usize) < other.len() {
                            near.push(other[j as usize]);
                        }
                    }
                }
            }
            out.insert(ch, near);
        }
    }
    out
}

fn tr_pair(a: char, b: char) -> bool {
    let p = |x: char, y: char| (a == x && b == y) || (a == y && b == x);
    p('c', 'ç') || p('g', 'ğ') || p('ı', 'i') || p('o', 'ö') || p('s', 'ş') || p('u', 'ü')
}

pub struct Normalizer {
    pub counts: HashMap<String, i64>,
    pub known: HashSet<String>,
    pub proper: HashMap<String, i64>,
    pub ascii_map: HashMap<String, String>,
    keys: HashMap<char, Vec<char>>,
    edit_chars: Vec<char>,
    del_index: OnceLock<HashMap<String, Vec<String>>>,
    pub min_len: usize, pub ratio: i64, pub short_ratio: i64, pub ascii_ratio: i64, pub dominant: i64,
    pub ed2_min_len: usize, pub ed2_ratio: i64, pub ed2_index_min: i64, pub foreign_override: i64,
}

impl Normalizer {
    pub fn load(wordfreq: &str, proper: &str, cfg: &serde_json::Value) -> Normalizer {
        let mut counts = HashMap::default();
        let mut known = HashSet::default();
        for line in std::fs::read_to_string(wordfreq).expect("wordfreq.tsv").lines() {
            let mut it = line.split('\t');
            let (w, c, k) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            counts.insert(w.to_string(), c.parse().unwrap());
            if k == "1" {
                known.insert(w.to_string());
            }
        }
        let mut pr = HashMap::default();
        if let Ok(t) = std::fs::read_to_string(proper) {
            for line in t.lines() {
                let (w, c) = line.split_once('\t').unwrap_or((line, ""));
                pr.insert(w.to_string(), if c.is_empty() { 1 << 30 } else { c.parse().unwrap() });
            }
        }
        // most frequent Turkish spelling per ASCII form; ties: smallest word
        let mut ascii_map: HashMap<String, String> = HashMap::default();
        for w in &known {
            let wc: Vec<char> = w.chars().collect();
            let a: String = asciify(&wc).into_iter().collect();
            if a != *w {
                let cw = *counts.get(w).unwrap_or(&0);
                match ascii_map.get(&a) {
                    Some(cur) => {
                        let cc = *counts.get(cur).unwrap_or(&0);
                        if (-cw, w.as_str()) < (-cc, cur.as_str()) {
                            ascii_map.insert(a, w.clone());
                        }
                    }
                    None => {
                        ascii_map.insert(a, w.clone());
                    }
                }
            }
        }
        let g = |k: &str| cfg[k].as_i64().unwrap();
        Normalizer {
            counts, known, proper: pr, ascii_map, keys: key_neighbours(), edit_chars: EDIT_CHARS.chars().collect(),
            del_index: OnceLock::new(),
            min_len: g("min_len") as usize, ratio: g("ratio"), short_ratio: g("short_ratio"), ascii_ratio: g("ascii_ratio"),
            dominant: g("dominant"), ed2_min_len: g("ed2_min_len") as usize, ed2_ratio: g("ed2_ratio"),
            ed2_index_min: g("ed2_index_min_count"), foreign_override: g("foreign_override"),
        }
    }

    fn count(&self, w: &str) -> i64 {
        *self.counts.get(w).unwrap_or(&0)
    }

    fn is_name(&self, w: &str) -> bool {
        self.proper.get(w).copied().unwrap_or(0) * 2 >= self.count(w).max(1)
    }

    pub fn ascii_target(&self, w: &[char], ratio: Option<i64>) -> Option<String> {
        let ws: String = w.iter().collect();
        let t = self.ascii_map.get(&ws)?;
        if asciify(w) != w || self.is_name(&ws) {
            return None;
        }
        let need = ratio.unwrap_or(self.ascii_ratio) * self.count(&ws).max(1);
        if self.count(t) >= need { Some(t.clone()) } else { None }
    }

    pub fn dominant_ascii(&self, w: &[char]) -> Option<String> {
        self.ascii_target(w, Some(self.dominant))
    }

    fn prior(&self, op: &Op, correct: &[char], written: &[char]) -> f64 {
        match op.kind {
            "del" => if is_vowel(correct[op.p]) { 4.0 } else { 2.0 },
            "ins" => {
                let ch = op.ch.unwrap();
                let mut near = Vec::new();
                if op.p >= 1 {
                    if let Some(&c) = written.get(op.p - 1) {
                        near.push(c);
                    }
                }
                if let Some(&c) = written.get(op.p + 1) {
                    near.push(c);
                }
                if near.contains(&ch) {
                    3.0
                } else if near.iter().any(|n| self.keys.get(n).map_or(false, |k| k.contains(&ch))) {
                    1.5
                } else {
                    0.3
                }
            }
            "sub" => {
                let ch = op.ch.unwrap();
                if tr_pair(correct[op.p], ch) {
                    4.0
                } else if self.keys.get(&correct[op.p]).map_or(false, |k| k.contains(&ch)) {
                    3.0
                } else {
                    1.0
                }
            }
            _ => 2.5,
        }
    }

    fn neighbours(&self, w: &[char]) -> Vec<Vec<char>> {
        let n = w.len();
        let mut out = Vec::new();
        for i in 0..n {
            let mut d = w[..i].to_vec();
            d.extend_from_slice(&w[i + 1..]);
            out.push(d);
            if i + 1 < n {
                let mut s = w.to_vec();
                s.swap(i, i + 1);
                out.push(s);
            }
            for &c in &self.edit_chars {
                if c != w[i] {
                    let mut s = w.to_vec();
                    s[i] = c;
                    out.push(s);
                }
            }
        }
        for i in 0..=n {
            for &c in &self.edit_chars {
                let mut s = w[..i].to_vec();
                s.push(c);
                s.extend_from_slice(&w[i..]);
                out.push(s);
            }
        }
        out
    }

    /// (intended word, ops) for a one- or two-letter typo.
    pub fn edit_target(&self, w: &[char], langid: Option<&LangId>) -> Option<(String, Vec<Op>)> {
        let ws: String = w.iter().collect();
        if w.len() < self.min_len || self.is_name(&ws) || w.iter().any(|c| !self.edit_chars.contains(c)) {
            return None;
        }
        let own = self.count(&ws).max(1);
        let mut need = (if w.len() == self.min_len { self.short_ratio } else { self.ratio }) * own;
        if langid.map_or(false, |l| l.is_foreign(w)) {
            need = need.max(self.foreign_override * own);
        }
        let mut best: Option<((f64, String), String, Vec<Op>)> = None;
        let mut seen: HashSet<Vec<char>> = HashSet::default();
        for cand in self.neighbours(w) {
            if !seen.insert(cand.clone()) {
                continue;
            }
            let cs: String = cand.iter().collect();
            let c = self.count(&cs);
            if c >= need && self.known.contains(&cs) {
                if let Some(op) = edit_op(&cand, w) {
                    if op.p < MAX_POS {
                        let score = (c as f64 * self.prior(&op, &cand, w), cs.clone());
                        if best.as_ref().map_or(true, |b| gt(&score, &b.0)) {
                            best = Some((score, cs, vec![op]));
                        }
                    }
                }
            }
        }
        if best.is_none() && w.len() >= self.ed2_min_len {
            best = self.edit2(w, &ws, self.ed2_ratio * own);
        }
        best.map(|b| (b.1, b.2))
    }

    fn edit2(&self, w: &[char], ws: &str, need: i64) -> Option<((f64, String), String, Vec<Op>)> {
        let idx = self.del_index.get_or_init(|| {
            let mut idx: HashMap<String, Vec<String>> = HashMap::default();
            for v in &self.known {
                let vc: Vec<char> = v.chars().collect();
                if vc.len() >= self.ed2_min_len && self.count(v) >= self.ed2_index_min {
                    let mut ks: HashSet<String> = HashSet::default();
                    ks.insert(v.clone());
                    for i in 0..vc.len() {
                        let mut d = vc[..i].to_vec();
                        d.extend_from_slice(&vc[i + 1..]);
                        ks.insert(d.into_iter().collect());
                    }
                    for k in ks {
                        idx.entry(k).or_default().push(v.clone());
                    }
                }
            }
            idx
        });
        let n = w.len();
        let mut keys: HashSet<String> = HashSet::default();
        keys.insert(ws.to_string());
        for i in 0..n {
            let mut d = w[..i].to_vec();
            d.extend_from_slice(&w[i + 1..]);
            keys.insert(d.into_iter().collect());
            for &c in &self.edit_chars {
                let mut s = w.to_vec();
                s[i] = c;
                keys.insert(s.into_iter().collect());
            }
            if i + 1 < n {
                let mut s = w.to_vec();
                s.swap(i, i + 1);
                keys.insert(s.into_iter().collect());
            }
        }
        for i in 0..=n {
            for &c in &self.edit_chars {
                let mut s = w[..i].to_vec();
                s.push(c);
                s.extend_from_slice(&w[i..]);
                keys.insert(s.into_iter().collect());
            }
        }
        let mut cands: HashSet<&String> = HashSet::default();
        for k in &keys {
            if let Some(v) = idx.get(k) {
                cands.extend(v.iter());
            }
        }
        let mut best: Option<((f64, String), String, Vec<Op>)> = None;
        for cand in cands {
            let c = self.count(cand);
            if c < need || cand == ws {
                continue;
            }
            let cc: Vec<char> = cand.chars().collect();
            for x in self.neighbours(&cc) {
                let Some(op2) = edit_op(&x, w) else { continue };
                if let Some(op1) = edit_op(&cc, &x) {
                    if op1.p < MAX_POS && op2.p < MAX_POS {
                        let score = (c as f64 * self.prior(&op1, &cc, &x) * self.prior(&op2, &x, w), cand.clone());
                        if best.as_ref().map_or(true, |b| gt(&score, &b.0)) {
                            best = Some((score, cand.clone(), vec![op1, op2]));
                        }
                    }
                }
            }
        }
        best
    }
}

/// Python tuple comparison (score, word) > (score, word).
fn gt(a: &(f64, String), b: &(f64, String)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 > b.1)
}
