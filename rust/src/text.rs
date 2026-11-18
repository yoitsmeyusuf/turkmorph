//! Unicode behaviour that must match Python exactly: regex classes ([^\W\d_], \s),
//! str.lower/upper (from data/tokenizer_tables.json), Turkish casing, the chunker.

use rustc_hash::FxHashMap as HashMap;
use serde_json::Value;

pub struct Text {
    letter: Vec<(u32, u32)>,
    space: Vec<(u32, u32)>,
    cased: Vec<(u32, u32)>,
    ignorable: Vec<(u32, u32)>,
    lower: HashMap<char, String>,
    upper: HashMap<char, String>,
}

fn ranges(v: &Value) -> Vec<(u32, u32)> {
    v.as_array().unwrap().iter().map(|r| (r[0].as_u64().unwrap() as u32, r[1].as_u64().unwrap() as u32)).collect()
}
fn in_ranges(r: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    match r.binary_search_by(|&(lo, hi)| if hi < c { std::cmp::Ordering::Less } else if lo > c { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }) {
        Ok(_) => true,
        Err(_) => false,
    }
}
fn cmap(v: &Value) -> HashMap<char, String> {
    v.as_object().unwrap().iter().map(|(k, v)| (k.chars().next().unwrap(), v.as_str().unwrap().to_string())).collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkKind { Word, Num, Space, Other }

impl Text {
    pub fn load(d: &Value) -> Text {
        Text { letter: ranges(&d["letter_ranges"]), space: ranges(&d["space_ranges"]),
               cased: ranges(&d["cased_ranges"]), ignorable: ranges(&d["ignorable_ranges"]),
               lower: cmap(&d["lower"]), upper: cmap(&d["upper"]) }
    }
    pub fn is_letter(&self, c: char) -> bool { in_ranges(&self.letter, c) }
    pub fn is_space(&self, c: char) -> bool { in_ranges(&self.space, c) }

    /// Python str.lower: per-char table + CPython's Final_Sigma rule (Σ -> ς when a cased
    /// letter precedes and none follows, skipping case-ignorable chars both ways).
    pub fn lower(&self, s: &[char]) -> Vec<char> {
        let mut out = Vec::with_capacity(s.len());
        let cased_near = |it: &mut dyn Iterator<Item = &char>| -> bool {
            for &x in it {
                if in_ranges(&self.ignorable, x) {
                    continue;
                }
                return in_ranges(&self.cased, x);
            }
            false
        };
        for (i, &c) in s.iter().enumerate() {
            if c == 'Σ' {
                let prev = cased_near(&mut s[..i].iter().rev());
                let next = cased_near(&mut s[i + 1..].iter());
                out.push(if prev && !next { 'ς' } else { 'σ' });
                continue;
            }
            match self.lower.get(&c) {
                Some(m) => out.extend(m.chars()),
                None => out.push(c),
            }
        }
        out
    }
    pub fn upper(&self, s: &[char]) -> Vec<char> {
        let mut out = Vec::with_capacity(s.len());
        for &c in s {
            match self.upper.get(&c) {
                Some(m) => out.extend(m.chars()),
                None => out.push(c),
            }
        }
        out
    }
    pub fn tr_lower(&self, s: &[char]) -> Vec<char> {
        let r: Vec<char> = s.iter().map(|&c| match c { 'I' => 'ı', 'İ' => 'i', x => x }).collect();
        self.lower(&r)
    }
    pub fn tr_upper(&self, s: &[char]) -> Vec<char> {
        let r: Vec<char> = s.iter().map(|&c| match c { 'i' => 'İ', 'ı' => 'I', x => x }).collect();
        self.upper(&r)
    }
    pub fn tr_cap(&self, s: &[char]) -> Vec<char> {
        if s.is_empty() {
            return Vec::new();
        }
        let mut out = self.tr_upper(&s[..1]);
        out.extend_from_slice(&s[1..]);
        out
    }

    /// The CHUNK regex: word ([^\W\d_]+ ('’ [^\W\d_]+)?), number ([0-9]+ (apos word)?),
    /// whitespace run, or any single char. Returns (kind, start, end) in char offsets.
    pub fn chunks(&self, s: &[char]) -> Vec<(ChunkKind, usize, usize)> {
        let mut out = Vec::new();
        let n = s.len();
        let mut i = 0;
        let apos_tail = |j: usize| -> usize {
            if j + 1 < n && (s[j] == '\'' || s[j] == '’') && self.is_letter(s[j + 1]) {
                let mut k = j + 1;
                while k < n && self.is_letter(s[k]) {
                    k += 1;
                }
                k
            } else {
                j
            }
        };
        while i < n {
            let c = s[i];
            let start = i;
            if self.is_letter(c) {
                while i < n && self.is_letter(s[i]) {
                    i += 1;
                }
                i = apos_tail(i);
                out.push((ChunkKind::Word, start, i));
            } else if c.is_ascii_digit() {
                while i < n && s[i].is_ascii_digit() {
                    i += 1;
                }
                i = apos_tail(i);
                out.push((ChunkKind::Num, start, i));
            } else if self.is_space(c) {
                while i < n && self.is_space(s[i]) {
                    i += 1;
                }
                out.push((ChunkKind::Space, start, i));
            } else {
                i += 1;
                out.push((ChunkKind::Other, start, i));
            }
        }
        out
    }
}

pub fn asciify(s: &[char]) -> Vec<char> {
    s.iter()
        .map(|&c| match c {
            'ç' => 'c', 'ğ' => 'g', 'ı' => 'i', 'ö' => 'o', 'ş' => 's', 'ü' => 'u', 'â' => 'a', 'î' => 'i', 'û' => 'u',
            'Ç' => 'C', 'Ğ' => 'G', 'İ' => 'I', 'Ö' => 'O', 'Ş' => 'S', 'Ü' => 'U', 'Â' => 'A', 'Î' => 'I', 'Û' => 'U',
            x => x,
        })
        .collect()
}

pub fn s(v: &[char]) -> String {
    v.iter().collect()
}
pub fn c(v: &str) -> Vec<char> {
    v.chars().collect()
}
