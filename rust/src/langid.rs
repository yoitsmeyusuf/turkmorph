//! Turkish vs foreign word id (port of turkmorph/langid.py CharNgramLangId.is_foreign).

use rustc_hash::FxHashMap as HashMap;
use serde_json::Value;

pub struct LangId {
    tr: HashMap<String, f64>,
    fr: HashMap<String, f64>,
    unk_tr: f64,
    unk_fr: f64,
    threshold: f64,
}

impl LangId {
    pub fn load(path: &str) -> LangId {
        let d: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("langid.json")).unwrap();
        let m = |k: &str| d[k].as_object().unwrap().iter().map(|(g, v)| (g.clone(), v.as_f64().unwrap())).collect();
        LangId { tr: m("lp_tr"), fr: m("lp_fr"), unk_tr: d["unk_tr"].as_f64().unwrap(),
                 unk_fr: d["unk_fr"].as_f64().unwrap(), threshold: d["threshold"].as_f64().unwrap() }
    }

    pub fn score(&self, word: &[char]) -> f64 {
        let mut w = vec!['<'];
        w.extend_from_slice(word);
        w.push('>');
        let mut s = 0.0;
        let mut n = 0usize;
        let mut g = String::new();
        for k in 1..=4usize {
            if w.len() < k {
                continue;
            }
            for i in 0..=w.len() - k {
                g.clear();
                g.extend(&w[i..i + k]);
                s += self.fr.get(&g).copied().unwrap_or(self.unk_fr) - self.tr.get(&g).copied().unwrap_or(self.unk_tr);
                n += 1;
            }
        }
        s / n as f64
    }

    pub fn is_foreign(&self, word: &[char]) -> bool {
        word.len() >= 3 && self.score(word) > self.threshold
    }
}
