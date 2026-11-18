//! Morphotactics, generator, analysis-by-synthesis (port of turkmorph/morphology.py).
//!
//! Tables (suffix inventory, FSA states, suppletion, ...) are read from
//! data/morph_tables.json, exported by `python3 -m turkmorph.export`; the rules
//! (aorist, causative, narrowing, softening, ...) are implemented here and must
//! stay identical to gen_start / gen_step in Python.

use crate::phon::{drop_last_vowel, is_vowel, n_vowels, realize_c, soften};
use rustc_hash::FxHashMap as HashMap;
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc as Rc;

pub type Sid = u16;

// ------------------------------------------------------------------ flags --
/// Bit i of a root's flag mask = FLAG_NAMES[i] (the F_* constants below follow this order).
pub const FLAG_NAMES: [&str; 19] = [
    "soften", "drop", "drop_pass", "geminate", "front", "aor_hr", "aor_ar", "cp3", "temporal",
    "num", "caus_Hr", "caus_Ar", "ki_harm", "pron_n", "y_poss", "narrow_y", "case_n", "no_buffer",
    "coll",
];
fn flag(name: &str) -> u64 {
    match FLAG_NAMES.iter().position(|&f| f == name) {
        Some(i) => 1 << i,
        None => panic!("unknown root flag {name}"),
    }
}
pub fn parse_flags(s: &str) -> u64 {
    s.split(',').filter(|x| !x.is_empty()).map(flag).fold(0, |a, b| a | b)
}

// ----------------------------------------------------------------- tables --
pub struct Suffix {
    pub name: String,
    pub tmpl: String,
    pub nxt: u16,
    pub n_tmpl: Option<String>,
    pub tmpl_c: Vec<char>,
    pub n_tmpl_c: Option<Vec<char>>,
    pub soften: bool,
    pub cost: f64,
}

/// Ids of the suffixes the rules refer to by name.
pub struct Special {
    pub loc_c: Sid, pub abl_c: Sid, pub prog: Sid, pub prog_c: Sid, pub aor: Sid,
    pub z1sg: Sid, pub z1pl: Sid, pub caus: Sid, pub pass: Sid, pub rel: Sid, pub p3sg: Sid,
    pub gen: Sid, pub pl: Sid, pub neg: Sid, pub impot: Sid,
}

pub struct Tables {
    pub suffixes: Vec<Suffix>,
    pub sfx_index: HashMap<String, Sid>,
    pub state_names: Vec<String>,
    pub state_index: HashMap<String, u16>,
    pub states: Vec<Vec<Sid>>,
    pub final_: Vec<bool>,
    pub pos_start: HashMap<String, u16>,
    pub pron_n_state: Vec<bool>,
    pub root_gated: HashMap<(u16, Sid), u64>,
    /// states with each transition's root gate (0 = ungated), for the DFS
    pub trans: Vec<Vec<(Sid, u64)>>,
    pub suppletion: HashMap<(String, Sid), Vec<char>>,
    pub suppletion_by_stem: HashMap<String, Vec<String>>,
    /// suffixes that occur in some suppletion entry (gate for the lookup)
    pub supp_sid: Vec<bool>,
    pub markable: Vec<Sid>,
    pub marked: HashMap<Sid, Vec<String>>,
    pub first_chars: Vec<Vec<char>>,
    pub no_filter: Vec<bool>,
    pub cp3_swap: Vec<bool>,
    pub de_keep: Vec<bool>,
    pub predicative: Vec<bool>,
    pub marked_cost: f64,
    pub union_margin: f64,
    pub predicative_penalty: f64,
    pub letter_names: HashMap<char, String>,
    pub sp: Special,
    /// rank of each suffix name in sorted order: compare chains like Python compares names
    pub name_rank: Vec<u32>,
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

impl Tables {
    pub fn load(path: &str) -> Tables {
        let d: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("morph_tables.json"))
            .expect("bad morph_tables.json");
        let st_obj = d["states"].as_object().unwrap();
        let state_names: Vec<String> = st_obj.keys().cloned().collect();
        let state_index: HashMap<String, u16> =
            state_names.iter().enumerate().map(|(i, s)| (s.clone(), i as u16)).collect();
        let sl = d["suffixes"].as_array().unwrap();
        let sfx_index: HashMap<String, Sid> = sl
            .iter()
            .enumerate()
            .map(|(i, s)| (s["id"].as_str().unwrap().to_string(), i as Sid))
            .collect();
        let suffixes: Vec<Suffix> = sl
            .iter()
            .map(|s| Suffix {
                name: s["id"].as_str().unwrap().into(),
                tmpl: s["tmpl"].as_str().unwrap().into(),
                nxt: state_index[s["nxt"].as_str().unwrap()],
                n_tmpl: s["n_tmpl"].as_str().map(String::from),
                tmpl_c: s["tmpl"].as_str().unwrap().chars().collect(),
                n_tmpl_c: s["n_tmpl"].as_str().map(|x| x.chars().collect()),
                soften: s["soften"].as_bool().unwrap(),
                cost: s["cost"].as_f64().unwrap(),
            })
            .collect();
        let id = |n: &str| sfx_index[n];
        let n = suffixes.len();
        let set_of = |key: &str| -> Vec<bool> {
            let mut v = vec![false; n];
            for x in strs(&d[key]) {
                v[id(&x) as usize] = true;
            }
            v
        };
        let states: Vec<Vec<Sid>> = state_names
            .iter()
            .map(|s| strs(&st_obj[s]).iter().map(|x| id(x)).collect())
            .collect();
        let mut final_ = vec![false; state_names.len()];
        for s in strs(&d["final"]) {
            final_[state_index[&s] as usize] = true;
        }
        let mut pron_n_state = vec![false; state_names.len()];
        for s in strs(&d["pron_n_states"]) {
            pron_n_state[state_index[&s] as usize] = true;
        }
        let pos_start = d["pos_start"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), state_index[v.as_str().unwrap()]))
            .collect();
        let root_gated: HashMap<(u16, Sid), u64> = d["root_gated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let g = strs(g);
                ((state_index[&g[0]], id(&g[1])), flag(&g[2]))
            })
            .collect();
        let mut suppletion = HashMap::default();
        let mut suppletion_by_stem: HashMap<String, Vec<String>> = HashMap::default();
        for e in d["suppletion"].as_array().unwrap() {
            let e = strs(e);
            suppletion.insert((e[0].clone(), id(&e[1])), e[2].chars().collect());
            suppletion_by_stem.entry(e[0].clone()).or_default().push(e[2].clone());
        }
        let trans: Vec<Vec<(Sid, u64)>> = states.iter().enumerate()
            .map(|(st, v)| v.iter().map(|&sid| (sid, root_gated.get(&(st as u16, sid)).copied().unwrap_or(0))).collect())
            .collect();
        let mut supp_sid = vec![false; n];
        for (_, sid) in suppletion.keys() {
            supp_sid[*sid as usize] = true;
        }
        let markable: Vec<Sid> = strs(&d["markable"]).iter().map(|x| id(x)).collect();
        let marked = d["marked"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (id(k), strs(v)))
            .collect();
        let mut first_chars = vec![Vec::new(); n];
        for (k, v) in d["first_chars"].as_object().unwrap() {
            first_chars[id(k) as usize] = v.as_str().unwrap().chars().collect();
        }
        let letter_names = d["letter_names"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.chars().next().unwrap(), v.as_str().unwrap().to_string()))
            .collect();
        let sp = Special {
            loc_c: id("LOC_C"), abl_c: id("ABL_C"), prog: id("PROG"), prog_c: id("PROG_C"),
            aor: id("AOR"), z1sg: id("Z1SG"), z1pl: id("Z1PL"), caus: id("CAUS"), pass: id("PASS"),
            rel: id("REL"), p3sg: id("P3SG"), gen: id("GEN"), pl: id("PL"), neg: id("NEG"),
            impot: id("IMPOT"),
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| suffixes[a].name.cmp(&suffixes[b].name));
        let mut name_rank = vec![0u32; n];
        for (r, &i) in order.iter().enumerate() {
            name_rank[i] = r as u32;
        }
        Tables {
            name_rank,
            no_filter: set_of("no_filter"),
            cp3_swap: set_of("cp3_swap"),
            de_keep: set_of("de_keep"),
            predicative: set_of("predicative_start"),
            marked_cost: d["marked_cost"].as_f64().unwrap(),
            union_margin: d["union_margin"].as_f64().unwrap(),
            predicative_penalty: d["predicative_penalty"].as_f64().unwrap(),
            suffixes, sfx_index, state_names, state_index, states, final_, pos_start, pron_n_state,
            root_gated, trans, suppletion, suppletion_by_stem, supp_sid, markable, marked, first_chars, letter_names, sp,
        }
    }
}

// ------------------------------------------------------------- suffix use --
/// One suffix in a chain: a plain id, or a marked allomorph (S:DAT!ya: id + literal surface).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sfx {
    pub id: Sid,
    pub marked: Option<Rc<[char]>>,
}
impl Sfx {
    pub fn plain(id: Sid) -> Sfx {
        Sfx { id, marked: None }
    }
    pub fn name(&self, t: &Tables) -> String {
        let base = &t.suffixes[self.id as usize].name;
        match &self.marked {
            Some(m) => format!("{base}!{}", m.iter().collect::<String>()),
            None => base.clone(),
        }
    }
}

// ------------------------------------------------------------------ roots --
pub struct Root {
    pub form: Vec<char>,
    pub form_s: String,
    pub pos: String,
    pub flags: u64,
    pub freq: i64,
    pub key: String,
    pub reading: String,
    pub inner: Option<(Vec<char>, u64)>,
    pub inner_s: String,
}

pub struct Lexicon {
    pub roots: Vec<Root>,
    pub by_key: HashMap<String, u32>,
    pub by_form: HashMap<String, Vec<u32>>,
    pub index: HashMap<String, Vec<u32>>,
}

const F_SOFTEN: u64 = 1 << 0;
const F_DROP: u64 = 1 << 1;
const F_DROP_PASS: u64 = 1 << 2;
const F_GEMINATE: u64 = 1 << 3;
const F_FRONT: u64 = 1 << 4;
const F_AOR_HR: u64 = 1 << 5;
const F_AOR_AR: u64 = 1 << 6;
const F_CP3: u64 = 1 << 7;
const F_CAUS_HR: u64 = 1 << 10;
const F_CAUS_AR: u64 = 1 << 11;
const F_KI_HARM: u64 = 1 << 12;
const F_PRON_N: u64 = 1 << 13;
const F_Y_POSS: u64 = 1 << 14;
const F_NARROW_Y: u64 = 1 << 15;
const F_CASE_N: u64 = 1 << 16;
const F_NO_BUFFER: u64 = 1 << 17;
const F_COLL: u64 = 1 << 18;
pub const FRONT_FLAG: u64 = F_FRONT;
pub const CP3_FLAG: u64 = F_CP3;

fn joined(form: &[char], flags: u64) -> Vec<char> {
    let mut f = form.to_vec();
    if flags & (F_DROP | F_DROP_PASS) != 0 {
        drop_last_vowel(&mut f);
    }
    if flags & F_SOFTEN != 0 {
        soften(&mut f);
    }
    if flags & F_GEMINATE != 0 {
        if let Some(&c) = f.last() {
            f.push(c);
        }
    }
    f
}

fn stem_variants(r: &Root, t: &Tables) -> Vec<String> {
    let mut out: Vec<Vec<char>> = Vec::new();
    let mut stems: Vec<(Vec<char>, u64)> = vec![(r.form.clone(), r.flags)];
    if let Some((st, fl)) = &r.inner {
        stems.push((st.clone(), *fl));
    }
    for (form, flags) in &stems {
        out.push(form.clone());
        out.push(joined(form, *flags));
        if r.pos == "V" && matches!(form.last(), Some('a') | Some('e')) {
            out.push(form[..form.len() - 1].to_vec());
        }
        if flags & F_NARROW_Y != 0 && !form.is_empty() {
            let mut f = form[..form.len() - 1].to_vec();
            f.push('i');
            out.push(f);
        }
        if flags & F_COLL != 0 && !form.is_empty() {
            out.push(form[..form.len() - 1].to_vec());
        }
    }
    if let Some(v) = t.suppletion_by_stem.get(&r.form_s) {
        out.extend(v.iter().map(|x| x.chars().collect()));
    }
    let mut s: Vec<String> = out.into_iter().filter(|v| !v.is_empty()).map(|v| v.into_iter().collect()).collect();
    s.sort();
    s.dedup();
    s
}

impl Lexicon {
    pub fn from_tsv(path: &str, t: &Tables) -> Lexicon {
        let text = std::fs::read_to_string(path).expect("lexicon.tsv");
        let mut roots = Vec::new();
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            let (form, pos, tag, flags, freq, reading) = (c[0], c[1], c[2], c[3], c[4], c[5]);
            let inner_s = c.get(6).copied().unwrap_or("").to_string();
            let inner = if inner_s.is_empty() {
                None
            } else {
                let (st, fl) = inner_s.split_once('|').unwrap();
                Some((st.chars().collect(), parse_flags(fl)))
            };
            roots.push(Root {
                form: form.chars().collect(),
                form_s: form.into(),
                pos: pos.into(),
                flags: parse_flags(flags),
                freq: freq.parse().unwrap(),
                key: format!("{form}#{pos}{tag}"),
                reading: reading.into(),
                inner,
                inner_s,
            });
        }
        let mut lex = Lexicon { roots, by_key: HashMap::default(), by_form: HashMap::default(), index: HashMap::default() };
        for (i, r) in lex.roots.iter().enumerate() {
            let i = i as u32;
            lex.by_key.insert(r.key.clone(), i);
            lex.by_form.entry(r.form_s.clone()).or_default().push(i);
            for v in stem_variants(r, t) {
                lex.index.entry(v).or_default().push(i);
            }
        }
        lex
    }

    /// Roots one of whose stem variants is a prefix of the word (insertion order, deduplicated).
    pub fn candidates(&self, word: &[char]) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        let mut prefix = String::with_capacity(word.len() * 2);
        for &c in word {
            prefix.push(c);
            if let Some(rs) = self.index.get(&prefix) {
                for &r in rs {
                    if !out.contains(&r) {
                        out.push(r);
                    }
                }
            }
        }
        out
    }
}

// -------------------------------------------------------------- generator --
/// generated surface so far
pub type Chars = Vec<char>;

#[derive(Clone)]
pub struct GenState {
    pub s: Chars,
    pub flags: u64,
    prev_root: bool,
    prev_soften: bool,
    prev_id: Option<Sid>,
    pron_n: bool,
    vowel_from_root: bool,
    neg_aor: bool,
    cp3_pl: bool,
    pub i: usize,
}

pub fn gen_start(t: &Tables, stem: &[char], flags: u64, first: Option<&Sfx>, inner: Option<&(Vec<char>, u64)>) -> (GenState, bool) {
    let (mut stem, mut flags) = (stem, flags);
    let mut cp3_pl = false;
    if let (Some((ist, ifl)), Some(f)) = (inner, first) {
        if t.cp3_swap[f.id as usize] {
            stem = ist;
            flags = *ifl;
            cp3_pl = f.id == t.sp.pl;
        }
    }
    let mut st = GenState {
        s: stem.to_vec(),
        flags,
        prev_root: true,
        prev_soften: false,
        prev_id: None,
        pron_n: flags & (F_PRON_N | F_CP3 | F_CASE_N) != 0,
        vowel_from_root: true,
        neg_aor: false,
        cp3_pl,
        i: 0,
    };
    if let Some(f) = first {
        if f.marked.is_none() && t.supp_sid[f.id as usize] {
            let key: String = stem.iter().collect();
            if let Some(sup) = t.suppletion.get(&(key, f.id)) {
                st.s = sup.clone();
                st.prev_root = false;
                st.vowel_from_root = false;
                st.prev_id = Some(f.id);
                st.i = 1;
                return (st, true);
            }
        }
    }
    (st, false)
}

fn aorist(t: &Tables, s: &[char], prev_root: bool, prev_id: Option<Sid>, flags: u64) -> &'static str {
    if prev_id == Some(t.sp.neg) || prev_id == Some(t.sp.impot) {
        return "z";
    }
    if s.last().map_or(false, |&c| is_vowel(c)) {
        return "r";
    }
    if prev_root {
        if flags & F_AOR_AR != 0 {
            return "Ar";
        }
        return if flags & F_AOR_HR != 0 || n_vowels(s) > 1 { "Hr" } else { "Ar" };
    }
    "Hr"
}

fn causative(s: &[char], prev_root: bool, flags: u64) -> &'static str {
    if prev_root && flags & F_CAUS_HR != 0 {
        return "Hr";
    }
    if prev_root && flags & F_CAUS_AR != 0 {
        return "Ar";
    }
    let last = *s.last().unwrap_or(&' ');
    if n_vowels(s) > 1 && (is_vowel(last) || last == 'l' || last == 'r') {
        return "t";
    }
    "DHr"
}

fn passive(s: &[char]) -> &'static str {
    match s.last() {
        Some(&c) if is_vowel(c) => "n",
        Some('l') => "Hn",
        _ => "Hl",
    }
}

/// Append one suffix; `nxt` (the following suffix) matters only for the aorist.
pub fn gen_step(t: &Tables, st: &GenState, sfx: &Sfx, nxt: Option<Sid>) -> GenState {
    let sp = &t.sp;
    let sid = sfx.id;
    let sf = &t.suffixes[sid as usize];
    let flags = st.flags;
    let front = st.vowel_from_root && flags & F_FRONT != 0;
    let mut s = st.s.clone();
    let mut neg_aor = st.neg_aor;
    let piece: crate::phon::Piece;
    if let Some(m) = &sfx.marked {
        piece = crate::phon::Piece::from_slice(m);
    } else if sid == sp.loc_c || sid == sp.abl_c {
        s.pop();
        piece = realize_c(&sf.tmpl_c, &s, false);
    } else if (sid == sp.prog || sid == sp.prog_c) && matches!(s.last(), Some('a') | Some('e')) {
        let dropped = s.pop().unwrap();
        let mut p = if s.iter().any(|&c| is_vowel(c)) { realize_c(&['H'], &s, false) } else { realize_c(&['H'], &[dropped], false) };
        p.extend(if sid == sp.prog { "yor" } else { "yo" }.chars());
        piece = p;
    } else {
        use std::borrow::Cow;
        let lit = |x: &str| -> Cow<[char]> { Cow::Owned(x.chars().collect()) };
        let mut tmpl: Cow<[char]> = match &sf.n_tmpl_c {
            Some(n) if st.pron_n && (sid != sp.pl || flags & F_PRON_N != 0) => Cow::Borrowed(n),
            _ => Cow::Borrowed(&sf.tmpl_c),
        };
        if (st.prev_root && flags & F_NO_BUFFER != 0) || st.prev_id == Some(sp.prog_c) {
            if tmpl.windows(3).any(|w| w == ['(', 'y', ')']) {
                let mut v = Vec::with_capacity(tmpl.len());
                let mut i = 0;
                while i < tmpl.len() {
                    if tmpl[i..].starts_with(&['(', 'y', ')']) {
                        i += 3;
                    } else {
                        v.push(tmpl[i]);
                        i += 1;
                    }
                }
                tmpl = Cow::Owned(v);
            }
        }
        if sid == sp.aor {
            let a = aorist(t, &s, st.prev_root, st.prev_id, flags);
            tmpl = lit(a);
            if a == "z" && (nxt == Some(sp.z1sg) || nxt == Some(sp.z1pl)) {
                tmpl = Cow::Owned(Vec::new());
                neg_aor = true;
            }
        } else if sid == sp.z1sg && neg_aor {
            tmpl = lit("m");
        } else if sid == sp.caus {
            tmpl = lit(causative(&s, st.prev_root, flags));
        } else if sid == sp.pass {
            tmpl = lit(passive(&s));
        } else if st.cp3_pl && st.i == 0 {
            tmpl = lit("lArH");
        } else if sid == sp.rel && st.prev_root && flags & F_KI_HARM != 0 {
            tmpl = lit("kü");
        }
        if (sid == sp.p3sg || sid == sp.gen) && st.prev_root && flags & F_Y_POSS != 0 {
            tmpl = lit(if sid == sp.p3sg { "(y)H" } else { "(y)Hn" });
        }
        let p = realize_c(&tmpl, &s, front);
        if st.prev_root && flags & F_NARROW_Y != 0 && p.first() == Some(&'y') && s.last() == Some(&'e')
            && !(s[..] == ['d', 'e'] && t.de_keep[sid as usize])
        {
            let n = s.len();
            s[n - 1] = 'i';
        }
        if p.first().map_or(false, |&c| is_vowel(c)) {
            if st.prev_root {
                if flags & F_DROP != 0 || (flags & F_DROP_PASS != 0 && sid == sp.pass) {
                    if let Some(i) = s.iter().rposition(|&c| is_vowel(c)) {
                        s.remove(i);
                    }
                }
                if flags & F_SOFTEN != 0 {
                    soften(&mut s);
                }
                if flags & F_GEMINATE != 0 {
                    if let Some(&c) = s.last() {
                        s.push(c);
                    }
                }
            } else if st.prev_soften {
                soften(&mut s);
            }
        }
        piece = p;
    }
    let mut vowel_from_root = st.vowel_from_root;
    if vowel_from_root && piece.iter().any(|&c| is_vowel(c)) {
        vowel_from_root = false;
    }
    s.extend_from_slice(&piece);
    GenState {
        s,
        flags,
        prev_root: false,
        prev_soften: sf.soften,
        prev_id: Some(sid),
        pron_n: t.pron_n_state[sf.nxt as usize] || (st.cp3_pl && st.i == 0),
        vowel_from_root,
        neg_aor,
        cp3_pl: st.cp3_pl,
        i: st.i + 1,
    }
}

pub fn generate(t: &Tables, stem: &[char], flags: u64, chain: &[Sfx], inner: Option<&(Vec<char>, u64)>) -> Vec<char> {
    let (mut st, _) = gen_start(t, stem, flags, chain.first(), inner);
    for k in st.i..chain.len() {
        let nxt = chain.get(k + 1).filter(|x| x.marked.is_none()).map(|x| x.id);
        st = gen_step(t, &st, &chain[k], nxt);
    }
    st.s
}

// ---------------------------------------------------------------- search --
struct Search<'a> {
    t: &'a Tables,
    word: &'a [char],
    stem: &'a [char],
    flags: u64,
    inner: Option<&'a (Vec<char>, u64)>,
    max_depth: usize,
    out: Vec<Vec<Sfx>>,
    base: Option<GenState>,
}

impl<'a> Search<'a> {
    fn dfs(&mut self, state: u16, sfx: &mut Vec<Sfx>, pre: Option<&GenState>, full: Option<&GenState>, surf: &[char]) {
        let t = self.t;
        if surf == self.word && t.final_[state as usize] {
            self.out.push(sfx.clone());
        }
        if sfx.len() >= self.max_depth {
            return;
        }
        let keep = if sfx.is_empty() { 2 } else { 1 };
        let safe = surf.len().saturating_sub(keep);
        if safe > self.word.len() || self.word[..safe] != surf[..safe] {
            return;
        }
        let last = sfx.last().cloned();
        for &(sid, gate) in &t.trans[state as usize] {
            if gate != 0 && (!sfx.is_empty() || self.flags & gate == 0) {
                continue;
            }
            let cand = Sfx::plain(sid);
            let (c_pre, c_full) = match &last {
                None => {
                    // gen_start depends on the suffix only through the compound swap and suppletion
                    if t.supp_sid[sid as usize] || (self.inner.is_some() && t.cp3_swap[sid as usize]) {
                        let (st0, consumed) = gen_start(t, self.stem, self.flags, Some(&cand), self.inner);
                        if consumed {
                            (None, st0)
                        } else {
                            let f = gen_step(t, &st0, &cand, None);
                            (if sid == t.sp.aor { Some(st0) } else { None }, f)
                        }
                    } else {
                        let st0 = self.base.as_ref().unwrap();
                        let f = gen_step(t, st0, &cand, None);
                        (if sid == t.sp.aor { Some(st0.clone()) } else { None }, f)
                    }
                }
                Some(l) => {
                    // the parent state, re-stepped only after an aorist (gelmez / gelmem)
                    let restep = if l.id == t.sp.aor && l.marked.is_none() && pre.is_some() {
                        Some(gen_step(t, pre.unwrap(), l, Some(sid)))
                    } else {
                        None
                    };
                    let p: &GenState = restep.as_ref().unwrap_or_else(|| full.unwrap());
                    if !t.no_filter[sid as usize] {
                        let k = p.s.len();
                        if k >= self.word.len() || !t.first_chars[sid as usize].contains(&self.word[k]) {
                            continue;
                        }
                    }
                    let f = gen_step(t, p, &cand, None);
                    // a child keeps its parent state only if it is an aorist itself
                    let keep = if sid == t.sp.aor { Some(p.clone()) } else { None };
                    (keep, f)
                }
            };
            sfx.push(cand);
            self.dfs(t.suffixes[sid as usize].nxt, sfx, c_pre.as_ref(), Some(&c_full), &c_full.s);
            sfx.pop();
        }
    }
}

/// All suffix chains from `state` realizing `word` from `stem` (port of _search).
pub fn search(t: &Tables, word: &[char], stem: &[char], flags: u64, state: u16, start: &[Sfx],
              max_depth: usize, inner: Option<&(Vec<char>, u64)>) -> Vec<Vec<Sfx>> {
    let base = if start.is_empty() { Some(gen_start(t, stem, flags, None, inner).0) } else { None };
    let mut s = Search { t, word, stem, flags, inner, max_depth, out: Vec::new(), base };
    let mut sfx: Vec<Sfx> = start.to_vec();
    if start.is_empty() {
        s.dfs(state, &mut sfx, None, None, stem);
    } else {
        let (mut full, consumed) = gen_start(t, stem, flags, start.first(), inner);
        let mut pre: Option<GenState> = None;
        for k in full.i..start.len() {
            pre = Some(full.clone());
            full = gen_step(t, &full, &start[k], None);
        }
        if consumed && start.len() == 1 {
            pre = None;
        }
        let surf = full.s.clone();
        s.dfs(state, &mut sfx, pre.as_ref(), Some(&full), &surf);
    }
    s.out
}

// ----------------------------------------------------------------- costs --
fn get(m: &serde_json::Map<String, Value>, k: &str) -> f64 {
    m.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}
fn sum_vals(m: &serde_json::Map<String, Value>) -> f64 {
    m.values().fold(0.0, |a, v| a + v.as_f64().unwrap())
}

/// Port of LearnedCosts: -log P(root) - sum log P(s_i | s_{i-1}) (+ backoff to FSA state, unigram).
pub struct Costs {
    root_cost: Vec<f64>,          // per lexicon root
    m: Vec<Vec<f64>>,             // [prev code][s code]; prev: 0..n suffixes, n.. POS; s: 0..n, n = END
    pos_code: HashMap<String, usize>,
    prev_state_is_n: Vec<bool>,   // _state_after(prev).startswith("N")
    penalty: f64,
}

impl Costs {
    pub fn load(path: &str, t: &Tables, lex: &Lexicon) -> Costs {
        let d: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("costs.json")).unwrap();
        let alpha = d["alpha"].as_f64().unwrap();
        let k = d["k"].as_f64().unwrap();
        let rc = d["root_counts"].as_object().unwrap();
        let big = d["bigram_counts"].as_object().unwrap();
        let uni = d["unigram_counts"].as_object().unwrap();
        let total = sum_vals(rc);
        let n_roots = lex.roots.len().max(rc.len()) as f64;
        let root_den = total + alpha * n_roots;
        let n_sfx = (t.suffixes.len() + 1) as f64;
        let uni_den = sum_vals(uni) + n_sfx;
        let pos_names: Vec<String> = t.pos_start.keys().cloned().collect();
        let state_after = |prev: &str| -> String {
            if let Some(p) = prev.strip_prefix('^') {
                match t.pos_start.get(p) {
                    Some(&s) => t.state_names[s as usize].clone(),
                    None => prev.to_string(),
                }
            } else {
                match t.sfx_index.get(prev) {
                    Some(&i) => t.state_names[t.suffixes[i as usize].nxt as usize].clone(),
                    None => prev.to_string(),
                }
            }
        };
        // state_counts in Python's insertion order
        let mut sc_order: Vec<String> = Vec::new();
        let mut state_counts: HashMap<String, serde_json::Map<String, Value>> = HashMap::default();
        for (prev, dv) in big {
            let st = state_after(prev);
            if !state_counts.contains_key(&st) {
                sc_order.push(st.clone());
                state_counts.insert(st.clone(), serde_json::Map::new());
            }
            let sc = state_counts.get_mut(&st).unwrap();
            for (x, c) in dv.as_object().unwrap() {
                let v = get(sc, x) + c.as_f64().unwrap();
                sc.insert(x.clone(), Value::from(v));
            }
        }
        let state_tot: HashMap<String, f64> = state_counts.iter().map(|(k2, m)| (k2.clone(), sum_vals(m))).collect();
        let n = t.suffixes.len();
        let mut prevs: Vec<String> = t.suffixes.iter().map(|s| s.name.clone()).collect();
        let mut pos_code = HashMap::default();
        for p in &pos_names {
            pos_code.insert(p.clone(), prevs.len());
            prevs.push(format!("^{p}"));
        }
        let mut targets: Vec<String> = t.suffixes.iter().map(|s| s.name.clone()).collect();
        targets.push("$".into());
        let empty = serde_json::Map::new();
        let mut m = vec![vec![0.0; n + 1]; prevs.len()];
        let mut prev_state_is_n = vec![false; prevs.len()];
        for (pi, prev) in prevs.iter().enumerate() {
            let st = state_after(prev);
            prev_state_is_n[pi] = st.starts_with('N');
            let sc = state_counts.get(&st).unwrap_or(&empty);
            let stot = *state_tot.get(&st).unwrap_or(&0.0);
            let bg = big.get(prev).and_then(|v| v.as_object()).unwrap_or(&empty);
            let ptot = if big.contains_key(prev) { sum_vals(bg) } else { 0.0 };
            for (si, s) in targets.iter().enumerate() {
                let p_uni = (get(uni, s) + 1.0) / uni_den;
                let p_state = (get(sc, s) + k * p_uni) / (stot + k);
                let p = (get(bg, s) + k * p_state) / (ptot + k);
                m[pi][si] = -p.ln();
            }
        }
        let root_cost = lex
            .roots
            .iter()
            .map(|r| -((get(rc, &r.key) + alpha) / root_den).ln())
            .collect();
        let _ = sc_order;
        Costs { root_cost, m, pos_code, prev_state_is_n, penalty: t.predicative_penalty }
    }

    pub fn analysis_cost(&self, t: &Tables, root: u32, pos: &str, chain: &[Sfx]) -> f64 {
        let mut c = self.root_cost[root as usize];
        let mut prev = self.pos_code[pos];
        for s in chain {
            let si = s.id as usize;
            c += self.m[prev][si];
            if t.predicative[si] && self.prev_state_is_n[prev] {
                c += self.penalty;
            }
            prev = si;
        }
        c + self.m[prev][t.suffixes.len()]
    }
}

/// Hand costs (suffix_cost) used by analyze_tail.
pub fn suffix_cost(t: &Tables, chain: &[Sfx]) -> f64 {
    let mut extra = if chain.first().map_or(false, |s| t.predicative[s.id as usize]) { 0.3 } else { 0.0 };
    let mut m = 0.0;
    for s in chain {
        if s.marked.is_some() {
            m += t.marked_cost;
        }
    }
    extra += m;
    let mut total = 0.0;
    for s in chain {
        total += t.suffixes[s.id as usize].cost;
    }
    extra + total
}

// -------------------------------------------------------------- analyses --
#[derive(Clone)]
pub struct Analysis {
    pub cost: f64,
    pub root: u32,
    pub chain: Vec<Sfx>,
}

fn cmp_chain(t: &Tables, a: &[Sfx], b: &[Sfx]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let o = if x.marked.is_none() && y.marked.is_none() {
            t.name_rank[x.id as usize].cmp(&t.name_rank[y.id as usize])
        } else {
            x.name(t).cmp(&y.name(t))
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

/// All analyses with a known root, sorted like Python: (cost, -freq, key, chain).
pub fn analyze_known(t: &Tables, lex: &Lexicon, costs: &Costs, word: &[char]) -> Vec<Analysis> {
    let mut res = Vec::new();
    for r in lex.candidates(word) {
        let root = &lex.roots[r as usize];
        let state = t.pos_start[&root.pos];
        for ch in search(t, word, &root.form, root.flags, state, &[], 10, root.inner.as_ref()) {
            let cost = costs.analysis_cost(t, r, &root.pos, &ch);
            res.push(Analysis { cost, root: r, chain: ch });
        }
    }
    res.sort_by(|a, b| {
        let (ra, rb) = (&lex.roots[a.root as usize], &lex.roots[b.root as usize]);
        a.cost
            .partial_cmp(&b.cost)
            .unwrap()
            .then((-ra.freq).cmp(&-rb.freq))
            .then(ra.key.cmp(&rb.key))
            .then_with(|| cmp_chain(t, &a.chain, &b.chain))
    });
    res
}

/// Unknown-root guesser (port of analyze_guess): (cost, stem, chain), sorted.
pub fn analyze_guess(t: &Tables, word: &[char], attested: &rustc_hash::FxHashSet<String>) -> Vec<(f64, Vec<char>, Vec<Sfx>)> {
    let mut res = Vec::new();
    let n_state = t.state_index["N"];
    if word.len() < 3 {
        return res;
    }
    for i in (2..word.len()).rev() {
        let stem = &word[..i];
        if n_vowels(stem) < 2 {
            continue;
        }
        let stem_s: String = stem.iter().collect();
        for ch in search(t, word, stem, 0, n_state, &[], 6, None) {
            if ch.is_empty() || (ch.len() < 2 && !attested.contains(&stem_s)) {
                continue;
            }
            let cost = 5.0 - ch.len() as f64 + 0.01 * stem.len() as f64;
            res.push((cost, stem.to_vec(), ch));
        }
    }
    res.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then_with(|| cmp_chain(t, &a.2, &b.2))
    });
    res
}

/// Suffix chain after an apostrophe (port of analyze_tail), marked allomorph fallback.
pub fn analyze_tail(t: &Tables, reading: &[char], tail: &[char], flags: u64) -> Option<Vec<Sfx>> {
    let mut word = reading.to_vec();
    word.extend_from_slice(tail);
    let n_state = t.state_index["N"];
    let mut chains: Vec<Vec<Sfx>> = search(t, &word, reading, flags, n_state, &[], 6, None)
        .into_iter()
        .filter(|c| !c.is_empty())
        .collect();
    if chains.is_empty() {
        for &sid in &t.states[n_state as usize] {
            if let Some(ms) = t.marked.get(&sid) {
                for a in ms {
                    let ac: Vec<char> = a.chars().collect();
                    if tail.starts_with(&ac) {
                        let start = [Sfx { id: sid, marked: Some(ac.into()) }];
                        let nxt = t.suffixes[sid as usize].nxt;
                        chains.extend(search(t, &word, reading, flags, nxt, &start, 6, None));
                    }
                }
            }
        }
    }
    chains.into_iter().min_by(|a, b| {
        suffix_cost(t, a).partial_cmp(&suffix_cost(t, b)).unwrap().then_with(|| cmp_chain(t, a, b))
    })
}

// ------------------------------------------------------------ unions (A) --
fn boundaries(t: &Tables, root: &Root, chain: &[Sfx]) -> Vec<usize> {
    (0..=chain.len()).map(|i| generate(t, &root.form, root.flags, &chain[..i], root.inner.as_ref()).len()).collect()
}

fn union_id(t: &Tables, members: &mut Vec<String>) -> String {
    let key = |m: &String| -> (usize, Vec<Sid>) {
        let ids: Vec<Sid> = m.split('+').map(|x| t.sfx_index[x]).collect();
        (ids.len(), ids)
    };
    members.sort_by(|a, b| key(a).cmp(&key(b)));
    members.dedup();
    members.join("|")
}

/// Port of underspecify: best's chain (as token names) with spans that near-equal
/// alternatives segment or label differently merged into union ids (P3SG|ACC).
pub fn underspecify(t: &Tables, lex: &Lexicon, best: &Analysis, alts: &[Analysis], margin: f64) -> Vec<String> {
    let names = |c: &[Sfx]| -> Vec<String> { c.iter().map(|s| s.name(t)).collect() };
    let near: Vec<&Analysis> = alts
        .iter()
        .filter(|a| lex.roots[a.root as usize].key == lex.roots[best.root as usize].key && a.chain != best.chain
            && a.cost - best.cost <= margin && a.chain.iter().all(|s| s.marked.is_none()))
        .collect();
    if near.is_empty() || best.chain.iter().any(|s| s.marked.is_some()) {
        return names(&best.chain);
    }
    let root = &lex.roots[best.root as usize];
    let chains: Vec<&[Sfx]> = std::iter::once(best.chain.as_slice()).chain(near.iter().map(|a| a.chain.as_slice())).collect();
    let bnds: Vec<Vec<usize>> = chains.iter().map(|c| boundaries(t, root, c)).collect();
    let common: Vec<usize> = bnds[0].iter().copied().filter(|x| bnds[1..].iter().all(|b| b.contains(x))).collect();
    let mut segs: Vec<Vec<Vec<String>>> = Vec::new();
    for (c, b) in chains.iter().zip(&bnds) {
        let (mut cur, mut out): (Vec<String>, Vec<Vec<String>>) = (Vec::new(), Vec::new());
        for (i, s) in c.iter().enumerate() {
            cur.push(s.name(t));
            if common.contains(&b[i + 1]) {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        segs.push(out);
    }
    if segs.iter().any(|x| x.len() != segs[0].len()) {
        return names(&best.chain);
    }
    let mut out = Vec::new();
    for k in 0..segs[0].len() {
        let mut members: Vec<String> = segs.iter().map(|x| x[k].join("+")).collect();
        members.sort();
        members.dedup();
        if members.len() == 1 {
            out.extend(segs[0][k].iter().cloned());
        } else {
            out.push(union_id(t, &mut members));
        }
    }
    out
}
