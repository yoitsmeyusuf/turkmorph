//! Encode / decode (port of turkmorph/tokenizer.py, deterministic mode).
//! Tokens are vocabulary ids; the vocabulary is built in exactly Python's order.

use crate::bpe::{Bpe, Piece};
use crate::langid::LangId;
use crate::morph::{analyze_guess, analyze_known, analyze_tail, generate, suffix_cost, underspecify, Analysis, Costs,
                   Lexicon, Sfx, Sid, Tables, CP3_FLAG, FRONT_FLAG};
use crate::normalize::{apply_edit, Normalizer, Op, EDIT_CHARS, MAX_POS};
use crate::phon::{n_vowels, number_reading};
use crate::text::{asciify, c, s, ChunkKind, Text};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::RwLock;

const SHARDS: usize = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sp { Pad, Bos, Eos, Cap, Caps, Capsb, Glue, Sp, Nl, Tab, Cr, Apos, Apos2, Lit, Raw, Ascii, Bare, Fuse }

#[derive(Clone, Debug)]
pub enum Kind {
    Special(Sp),
    Byte(u8),
    Sfx(Vec<Sfx>),           // what the generator realizes (a union's first member)
    Root(u32),
    Lex(String, bool),       // piece (without '▁' if start), is_start
    Edit(Op),
    Digit(char),
    Punct(String),
}

pub struct Vocab {
    pub toks: Vec<String>,
    pub index: HashMap<String, u32>,
    pub kinds: Vec<Kind>,
}

impl Vocab {
    pub fn id(&self, t: &str) -> u32 {
        match self.index.get(t) {
            Some(&i) => i,
            None => panic!("token not in vocabulary: {t}"),
        }
    }
}

fn special(name: &str) -> Sp {
    match name {
        "<pad>" => Sp::Pad, "<bos>" => Sp::Bos, "<eos>" => Sp::Eos, "<cap>" => Sp::Cap, "<caps>" => Sp::Caps,
        "<capsb>" => Sp::Capsb, "<glue>" => Sp::Glue, "<sp>" => Sp::Sp, "<nl>" => Sp::Nl, "<tab>" => Sp::Tab,
        "<cr>" => Sp::Cr, "<apos>" => Sp::Apos, "<apos2>" => Sp::Apos2, "<lit>" => Sp::Lit, "<raw>" => Sp::Raw,
        "<ascii>" => Sp::Ascii, "<bare>" => Sp::Bare, "<fuse>" => Sp::Fuse,
        _ => panic!("unknown special {name}"),
    }
}

fn parse_sfx(t: &Tables, name: &str) -> Vec<Sfx> {
    let first = name.split('|').next().unwrap();
    if let Some((base, surf)) = first.split_once('!') {
        return vec![Sfx { id: t.sfx_index[base], marked: Some(surf.chars().collect::<Vec<_>>().into()) }];
    }
    first.split('+').map(|x| Sfx::plain(t.sfx_index[x])).collect()
}

pub struct Tokenizer {
    pub t: Tables,
    pub lex: Lexicon,
    pub costs: Costs,
    pub text: Text,
    pub bpe: Bpe,
    pub langid: Option<LangId>,
    pub norm: Option<Normalizer>,
    pub overrides: HashMap<String, (String, Vec<String>)>,
    pub unions: HashSet<String>,
    pub attested: HashSet<String>,
    pub vocab: Vocab,
    punct: HashSet<char>,
    glued_clitics: Vec<Vec<char>>,
    glued_conj: HashSet<String>,
    bare_first: HashSet<Sid>,
    /// word cache, sharded so that threads rarely wait on each other
    cache: Vec<RwLock<HashMap<String, Vec<u32>>>>,
    pub cache_size: usize,
    sp_ids: HashMap<&'static str, u32>,
    byte0: u32,
    root_tok: Vec<u32>,
    sfx_tok: Vec<u32>,
}

const SPECIAL_TOKS: [&str; 18] = ["<pad>", "<bos>", "<eos>", "<cap>", "<caps>", "<capsb>", "<glue>", "<sp>", "<nl>",
    "<tab>", "<cr>", "<apos>", "<apos2>", "<lit>", "<raw>", "<ascii>", "<bare>", "<fuse>"];

impl Tokenizer {
    pub fn load(data: &str) -> Tokenizer {
        let p = |f: &str| format!("{data}/{f}");
        let exists = |f: &str| std::path::Path::new(&p(f)).exists();
        let t = Tables::load(&p("morph_tables.json"));
        let tt: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p("tokenizer_tables.json")).unwrap()).unwrap();
        let text = Text::load(&tt);
        let lex = Lexicon::from_tsv(&p("lexicon.tsv"), &t);
        let costs = Costs::load(&p("costs.json"), &t, &lex);
        let bpe = Bpe::load(&p("lex_bpe.json"));
        let langid = if exists("langid.json") { Some(LangId::load(&p("langid.json"))) } else { None };
        let norm = if exists("wordfreq.tsv") {
            Some(Normalizer::load(&p("wordfreq.tsv"), &p("proper_words.txt"), &tt["normalizer"]))
        } else {
            None
        };
        let read_lines = |f: &str| -> Vec<String> {
            if exists(f) { std::fs::read_to_string(p(f)).unwrap().lines().map(String::from).collect() } else { vec![] }
        };
        let attested: HashSet<String> = read_lines("attested_stems.txt").into_iter().filter(|x| !x.is_empty()).collect();
        let mut overrides = HashMap::default();
        for line in read_lines("overrides.tsv") {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let chain: Vec<String> = f[2].split_whitespace().map(String::from).collect();
            overrides.insert(f[0].to_string(), (f[1].to_string(), chain));
        }
        let union_list: Vec<String> = read_lines("unions.txt").into_iter().filter(|x| !x.is_empty()).collect();

        // ---- vocabulary, in Python's order
        let mut toks: Vec<String> = Vec::new();
        let mut kinds: Vec<Kind> = Vec::new();
        for sname in tt["specials"].as_array().unwrap() {
            let sn = sname.as_str().unwrap();
            toks.push(sn.into());
            kinds.push(Kind::Special(special(sn)));
        }
        for b in 0..256u32 {
            toks.push(format!("B:{b:02x}"));
            kinds.push(Kind::Byte(b as u8));
        }
        for sf in &t.suffixes {
            toks.push(format!("S:{}", sf.name));
            kinds.push(Kind::Sfx(parse_sfx(&t, &sf.name)));
        }
        for &sid in &t.markable {
            for a in &t.marked[&sid] {
                let name = format!("{}!{a}", t.suffixes[sid as usize].name);
                kinds.push(Kind::Sfx(parse_sfx(&t, &name)));
                toks.push(format!("S:{name}"));
            }
        }
        for (i, r) in lex.roots.iter().enumerate() {
            toks.push(format!("R:{}", r.key));
            kinds.push(Kind::Root(i as u32));
        }
        let mut lexp: Vec<String> = bpe.vocab.iter().map(|p| format!("L:{p}")).collect();
        lexp.sort();
        for lp in lexp {
            let piece = &lp[2..];
            kinds.push(match piece.strip_prefix('▁') {
                Some(rest) => Kind::Lex(rest.to_string(), true),
                None => Kind::Lex(piece.to_string(), false),
            });
            toks.push(lp);
        }
        let ec: Vec<char> = EDIT_CHARS.chars().collect();
        for k in ["del", "swap"] {
            for pos in 0..MAX_POS {
                let tok = format!("E:{k}:{pos}");
                kinds.push(Kind::Edit(Op::parse(&tok).unwrap()));
                toks.push(tok);
            }
        }
        for k in ["ins", "sub"] {
            for pos in 0..MAX_POS {
                for &ch in &ec {
                    let tok = format!("E:{k}:{pos}:{ch}");
                    kinds.push(Kind::Edit(Op::parse(&tok).unwrap()));
                    toks.push(tok);
                }
            }
        }
        for d in '0'..='9' {
            toks.push(format!("D:{d}"));
            kinds.push(Kind::Digit(d));
        }
        let punct_list: Vec<String> = tt["punct"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let mut pt: Vec<String> = punct_list.iter().map(|x| format!("P:{x}")).collect();
        pt.sort();
        for x in pt {
            kinds.push(Kind::Punct(x[2..].to_string()));
            toks.push(x);
        }
        for u in &union_list {
            toks.push(format!("S:{u}"));
            kinds.push(Kind::Sfx(parse_sfx(&t, u)));
        }
        let index = toks.iter().enumerate().map(|(i, x)| (x.clone(), i as u32)).collect();
        let vocab = Vocab { toks, index, kinds };
        let sp_ids = SPECIAL_TOKS.iter().map(|&n| (n, vocab.id(n))).collect();
        let byte0 = vocab.id("B:00");
        let root_tok = lex.roots.iter().map(|r| vocab.id(&format!("R:{}", r.key))).collect();
        let sfx_tok = t.suffixes.iter().map(|s| vocab.id(&format!("S:{}", s.name))).collect();
        let glued_clitics = tt["glued_clitics"].as_array().unwrap().iter().map(|x| c(x.as_str().unwrap())).collect();
        let glued_conj = tt["glued_conj"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let bare_first = tt["bare_first"].as_array().unwrap().iter().map(|x| t.sfx_index[x.as_str().unwrap()]).collect();
        let punct = punct_list.iter().map(|x| x.chars().next().unwrap()).collect();
        Tokenizer {
            t, lex, costs, text, bpe, langid, norm, overrides, unions: union_list.into_iter().collect(), attested, vocab,
            punct, glued_clitics, glued_conj, bare_first, cache: (0..SHARDS).map(|_| RwLock::new(HashMap::default())).collect(), cache_size: 4_000_000, sp_ids, byte0, root_tok, sfx_tok,
        }
    }

    fn sp(&self, n: &str) -> u32 {
        self.sp_ids[n]
    }
    fn tid(&self, tok: &str) -> u32 {
        self.vocab.id(tok)
    }

    // ------------------------------------------------------------ encode --
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let chunks = self.text.chunks(&chars);
        let mut toks: Vec<u32> = Vec::new();
        let mut implicit_space = false;
        for (i, &(kind, a, b)) in chunks.iter().enumerate() {
            let piece = &chars[a..b];
            match kind {
                ChunkKind::Space => {
                    let nxt = chunks.get(i + 1);
                    let last_digit = toks.last().map_or(false, |&x| matches!(self.vocab.kinds[x as usize], Kind::Digit(_)));
                    let fuses = nxt.map_or(false, |n| n.0 == ChunkKind::Num) && last_digit;
                    if piece == [' '] && !toks.is_empty()
                        && nxt.map_or(false, |n| n.0 == ChunkKind::Word || n.0 == ChunkKind::Num) && !fuses
                    {
                        implicit_space = true;
                        continue;
                    }
                    for &ch in piece {
                        match ch {
                            ' ' => toks.push(self.sp("<sp>")),
                            '\n' => toks.push(self.sp("<nl>")),
                            '\t' => toks.push(self.sp("<tab>")),
                            '\r' => toks.push(self.sp("<cr>")),
                            _ => {
                                toks.push(self.sp("<raw>"));
                                self.push_bytes(&mut toks, ch);
                            }
                        }
                    }
                }
                ChunkKind::Word | ChunkKind::Num => {
                    if !toks.is_empty() && !implicit_space {
                        toks.push(self.sp("<glue>"));
                    }
                    implicit_space = false;
                    if kind == ChunkKind::Word {
                        toks.extend(self.encode_word(piece));
                    } else {
                        toks.extend(self.encode_number(piece));
                    }
                }
                ChunkKind::Other => {
                    implicit_space = false;
                    let ch = piece[0];
                    if self.punct.contains(&ch) {
                        toks.push(self.tid(&format!("P:{ch}")));
                    } else {
                        toks.push(self.sp("<raw>"));
                        self.push_bytes(&mut toks, ch);
                    }
                }
            }
        }
        toks
    }

    fn push_bytes(&self, toks: &mut Vec<u32>, ch: char) {
        let mut buf = [0u8; 4];
        for &b in ch.encode_utf8(&mut buf).as_bytes() {
            toks.push(self.byte0 + b as u32);
        }
    }

    pub fn bpe_toks(&self, w: &[char], start: bool) -> Vec<u32> {
        self.bpe.encode(w, start).into_iter().map(|p| match p {
            Piece::Lex(x) => self.tid(&format!("L:{x}")),
            Piece::Byte(b) => self.byte0 + b as u32,
        }).collect()
    }

    fn apos_tok(&self, a: char) -> u32 {
        if a == '\'' { self.sp("<apos>") } else { self.sp("<apos2>") }
    }

    fn encode_number(&self, chunk: &[char]) -> Vec<u32> {
        let nd = chunk.iter().take_while(|c| c.is_ascii_digit()).count();
        let digits = &chunk[..nd];
        let mut toks: Vec<u32> = digits.iter().map(|d| self.tid(&format!("D:{d}"))).collect();
        if nd < chunk.len() {
            toks.push(self.apos_tok(chunk[nd]));
            let reading = c(&number_reading(&s(digits)));
            toks.extend(self.encode_tail(&reading, &chunk[nd + 1..], 0));
        }
        toks
    }

    fn encode_tail(&self, reading: &[char], tail: &[char], flags: u64) -> Vec<u32> {
        if let Some(ch) = analyze_tail(&self.t, reading, tail, flags) {
            let g = generate(&self.t, reading, flags, &ch, None);
            if g.len() >= reading.len() && g[reading.len()..] == *tail {
                return self.sfx_toks(&ch).collect();
            }
        }
        let mut out = vec![self.sp("<lit>")];
        out.extend(self.bpe_toks(tail, false));
        out
    }

    pub fn encode_word(&self, chunk: &[char]) -> Vec<u32> {
        let key = s(chunk);
        let mut h = rustc_hash::FxBuildHasher.build_hasher();
        key.hash(&mut h);
        let shard = &self.cache[h.finish() as usize % SHARDS];
        if let Some(v) = shard.read().unwrap().get(&key) {
            return v.clone();
        }
        let r = self.encode_word_uncached(chunk);
        let mut cache = shard.write().unwrap();
        if cache.len() < self.cache_size / SHARDS {
            cache.insert(key, r.clone());
        }
        r
    }

    /// Encode many texts on `threads` threads (the cache is shared); output order = input order.
    pub fn encode_batch<S: AsRef<str> + Sync>(&self, texts: &[S], threads: usize) -> Vec<Vec<u32>> {
        let threads = threads.max(1).min(texts.len().max(1));
        if threads == 1 {
            return texts.iter().map(|t| self.encode(t.as_ref())).collect();
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); texts.len()];
        let slots: Vec<std::sync::Mutex<&mut Vec<u32>>> = out.iter_mut().map(std::sync::Mutex::new).collect();
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| loop {
                    // small chunks keep the threads balanced when text lengths vary
                    let i0 = next.fetch_add(16, std::sync::atomic::Ordering::Relaxed);
                    if i0 >= texts.len() {
                        break;
                    }
                    for i in i0..(i0 + 16).min(texts.len()) {
                        let v = self.encode(texts[i].as_ref());
                        **slots[i].lock().unwrap() = v;
                    }
                });
            }
        });
        drop(slots);
        out
    }

    fn case_of(&self, chunk: &[char], low: &[char]) -> Option<Vec<u32>> {
        if chunk == low {
            Some(vec![])
        } else if chunk == self.text.tr_cap(low) {
            Some(vec![self.sp("<cap>")])
        } else if chunk.len() > 1 && chunk == self.text.tr_upper(low) {
            Some(vec![self.sp("<caps>")])
        } else {
            None
        }
    }

    fn verify(&self, toks: &[u32], chunk: &[char]) -> bool {
        match self.decode(toks) {
            Ok(d) => d.chars().eq(chunk.iter().copied()),
            Err(_) => false,
        }
    }

    fn encode_word_uncached(&self, chunk: &[char]) -> Vec<u32> {
        if let Some(ap) = chunk.iter().position(|&x| x == '\'' || x == '’') {
            if ap > 0 && ap + 1 < chunk.len() {
                return self.encode_apos(chunk, &chunk[..ap], chunk[ap], &chunk[ap + 1..]);
            }
        }
        let low = self.text.tr_lower(chunk);
        if let Some(case) = self.case_of(chunk, &low) {
            if let Some(t) = self.candidates(&low, !case.is_empty(), &case, chunk) {
                return t;
            }
        }
        self.bpe_toks(chunk, true)
    }

    fn encode_apos(&self, chunk: &[char], base: &[char], apos: char, tail: &[char]) -> Vec<u32> {
        let tx = &self.text;
        let low = tx.tr_lower(chunk);
        let low_base = tx.tr_lower(base);
        let tail_low = tx.tr_lower(tail);
        let mut head: Option<Vec<u32>> = None;
        let case: Option<Vec<u32>> = if chunk == low.as_slice() {
            Some(vec![])
        } else if chunk == tx.tr_cap(&low).as_slice() {
            Some(vec![self.sp("<cap>")])
        } else if chunk.len() > 1 && chunk == tx.tr_upper(&low).as_slice() {
            Some(vec![self.sp("<caps>")])
        } else if tail == tail_low.as_slice() && base.len() > 1 && base == tx.tr_upper(&low_base).as_slice() {
            Some(vec![self.sp("<capsb>")])
        } else if tail == tail_low.as_slice() {
            head = Some(self.bpe_toks(base, true));
            Some(vec![])
        } else {
            None
        };
        if let Some(case) = case {
            let mut reading = self.spelled_reading(&low_base);
            let mut flags = 0u64;
            let head = match head {
                Some(h) => h,
                None => {
                    let mut known: Vec<u32> = self.lex.by_form.get(&s(&low_base)).cloned().unwrap_or_default()
                        .into_iter().filter(|&r| matches!(self.lex.roots[r as usize].pos.as_str(), "PROPER" | "N" | "PRON")).collect();
                    known.sort_by_key(|&r| self.lex.roots[r as usize].pos != "PROPER");
                    if let Some(&r) = known.first() {
                        let root = &self.lex.roots[r as usize];
                        reading = if root.reading.is_empty() { root.form.clone() } else { c(&root.reading) };
                        flags = root.flags & (FRONT_FLAG | CP3_FLAG);
                        vec![self.root_tok[r as usize]]
                    } else {
                        self.bpe_toks(&low_base, true)
                    }
                }
            };
            let mut toks = case;
            toks.extend(head);
            toks.push(self.apos_tok(apos));
            toks.extend(self.encode_tail(&reading, &tail_low, flags));
            if self.verify(&toks, chunk) {
                return toks;
            }
        }
        self.bpe_toks(chunk, true)
    }

    fn spelled_reading(&self, base: &[char]) -> Vec<char> {
        if n_vowels(base) > 0 || base.is_empty() || base.iter().any(|x| !self.t.letter_names.contains_key(x)) {
            return base.to_vec();
        }
        base.iter().flat_map(|x| self.t.letter_names[x].chars()).collect()
    }

    fn sfx_toks<'a>(&'a self, chain: &'a [Sfx]) -> impl Iterator<Item = u32> + 'a {
        chain.iter().map(move |x| match &x.marked {
            None => self.sfx_tok[x.id as usize],
            Some(_) => self.tid(&format!("S:{}", x.name(&self.t))),
        })
    }

    fn analysis_toks(&self, a: &Analysis) -> Vec<u32> {
        let mut v = vec![self.root_tok[a.root as usize]];
        v.extend(self.sfx_toks(&a.chain));
        v
    }

    /// Port of _known: up to 8 analyses as token lists (an underspecified one first).
    fn known(&self, low: &[char], cased: bool) -> Vec<Vec<u32>> {
        let mut an = analyze_known(&self.t, &self.lex, &self.costs, low);
        if !cased {
            an.sort_by_key(|a| self.lex.roots[a.root as usize].pos == "PROPER");
        }
        if let Some((k, ch)) = self.overrides.get(&s(low)) {
            an.sort_by_key(|a| {
                let r = &self.lex.roots[a.root as usize];
                !(r.key == *k && a.chain.len() == ch.len() && a.chain.iter().zip(ch).all(|(x, y)| x.name(&self.t) == *y))
            });
        }
        let mut out: Vec<Vec<u32>> = an.iter().take(8).map(|a| self.analysis_toks(a)).collect();
        if !self.unions.is_empty() && an.len() > 1 {
            let u = underspecify(&self.t, &self.lex, &an[0], &an[1..], self.t.union_margin);
            if u.iter().any(|x| x.contains('|')) && u.iter().all(|x| !x.contains('|') || self.unions.contains(x)) {
                let mut v = vec![self.root_tok[an[0].root as usize]];
                v.extend(u.iter().map(|x| self.tid(&format!("S:{x}"))));
                out.insert(0, v);
            }
        }
        out
    }

    fn root_key_of(&self, toks: &[u32]) -> &str {
        match self.vocab.kinds[toks[0] as usize] {
            Kind::Root(r) => &self.lex.roots[r as usize].key,
            _ => "",
        }
    }

    fn ops_toks(&self, ops: &[Op]) -> Vec<u32> {
        ops.iter().map(|o| self.tid(&o.token())).collect()
    }

    /// The cascade of _candidates; returns the first candidate that regenerates the word.
    fn candidates(&self, low: &[char], cased: bool, case: &[u32], chunk: &[char]) -> Option<Vec<u32>> {
        let attempt = |cand: Vec<u32>| -> Option<Vec<u32>> {
            let mut t = case.to_vec();
            t.extend(cand);
            if self.verify(&t, chunk) { Some(t) } else { None }
        };
        let known = self.known(low, cased);
        let lows = s(low);
        // yada = ya da: a conjunction + clitic beats a rare reading
        for cl in &self.glued_clitics {
            if low.ends_with(cl) && self.glued_conj.contains(&s(&low[..low.len() - cl.len()])) {
                let left = self.known(&low[..low.len() - cl.len()], cased);
                if let Some(l0) = left.first() {
                    if self.root_key_of(l0).ends_with("#X") && l0.len() == 1 {
                        if let Some(r0) = self.known(cl, false).into_iter().next() {
                            let mut v = l0.clone();
                            v.push(self.sp("<glue>"));
                            v.extend(r0);
                            if let Some(t) = attempt(v) { return Some(t); }
                        }
                    }
                }
            }
        }
        let norm = self.norm.as_ref();
        let weak = !known.is_empty() && !cased && known.iter().all(|k| self.root_key_of(k).ends_with("#PROPER"));
        if let Some(n) = norm {
            if !known.is_empty() && (weak || n.dominant_ascii(low).is_some()) {
                if let Some(t) = n.ascii_target(low, None) {
                    if let Some(k0) = self.known(&c(&t), cased).into_iter().next() {
                        let mut v = vec![self.sp("<ascii>")];
                        v.extend(k0);
                        if let Some(r) = attempt(v) { return Some(r); }
                    }
                }
                if weak {
                    if let Some((tw, ops)) = n.edit_target(low, self.langid.as_ref()) {
                        if let Some(k0) = self.known(&c(&tw), cased).into_iter().next() {
                            if !self.root_key_of(&k0).ends_with("#PROPER") {
                                let mut v = k0;
                                v.extend(self.ops_toks(&ops));
                                if let Some(r) = attempt(v) { return Some(r); }
                            }
                        }
                    }
                }
            }
        }
        for k in &known {
            if let Some(r) = attempt(k.clone()) { return Some(r); }
        }
        if !known.is_empty() {
            return None;
        }
        if let Some(n) = norm {
            if let Some(t) = n.ascii_target(low, None) {
                if let Some(k0) = self.known(&c(&t), cased).into_iter().next() {
                    let mut v = vec![self.sp("<ascii>")];
                    v.extend(k0);
                    if let Some(r) = attempt(v) { return Some(r); }
                }
            }
        }
        for cl in &self.glued_clitics {
            if low.ends_with(cl) && low.len() >= cl.len() + 2 {
                let left = self.known(&low[..low.len() - cl.len()], cased);
                let right = self.known(cl, false);
                if let (Some(l0), Some(r0)) = (left.first(), right.first()) {
                    let mut v = l0.clone();
                    v.push(self.sp("<glue>"));
                    v.extend(r0.iter().copied());
                    if let Some(r) = attempt(v) { return Some(r); }
                }
            }
        }
        if low.len() >= 4 && low[0] == 'n' && "aeıioöuü".contains(low[1]) {
            let rest = &low[1..];
            for pre in [None, Some('y'), Some('h')] {
                let mut rw: Vec<char> = pre.into_iter().collect();
                rw.extend_from_slice(rest);
                let right = self.known(&rw, false);
                if let Some(r0) = right.first() {
                    if self.root_key_of(r0).ends_with("#V") || s(&rw) == "haber" {
                        let mut v = vec![self.tid("R:ne#PRON"), self.sp("<fuse>")];
                        v.extend(r0.iter().copied());
                        if let Some(r) = attempt(v) { return Some(r); }
                        break;
                    }
                }
            }
        }
        if (2..=7).contains(&low.len()) {
            if let Some(ch) = analyze_tail(&self.t, &[], low, 0) {
                if self.bare_first.contains(&ch[0].id) {
                    let mut v = vec![self.sp("<bare>")];
                    v.extend(ch.iter().map(|x| self.tid(&format!("S:{}", x.name(&self.t)))));
                    if let Some(r) = attempt(v) { return Some(r); }
                }
            }
        }
        if let Some(n) = norm {
            if let Some((tw, ops)) = n.edit_target(low, self.langid.as_ref()) {
                if let Some(k0) = self.known(&c(&tw), cased).into_iter().next() {
                    if !self.root_key_of(&k0).ends_with("#PROPER") {
                        let mut v = k0;
                        v.extend(self.ops_toks(&ops));
                        if let Some(r) = attempt(v) { return Some(r); }
                    }
                }
            }
        }
        let foreign = match &self.langid {
            Some(l) => l.is_foreign(low),
            None => foreign_hint(&lows),
        };
        if !foreign {
            for (_, stem, ch) in analyze_guess(&self.t, low, &self.attested).into_iter().take(4) {
                let mut v = self.bpe_toks(&stem, true);
                v.extend(ch.iter().map(|x| self.tid(&format!("S:{}", x.name(&self.t)))));
                if let Some(r) = attempt(v) { return Some(r); }
            }
        }
        attempt(self.bpe_toks(low, true))
    }

    // ------------------------------------------------------------ decode --
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut out: Vec<String> = Vec::new();
        let mut w: Option<Word> = None;
        let mut raw: Option<Vec<u8>> = None;
        let mut prev: Option<Sp> = None;
        let mut fuse_pending = false;
        macro_rules! flush {
            () => {
                if let Some(word) = w.take() {
                    let mut surf = self.surface(&word);
                    if fuse_pending && !out.is_empty() {
                        surf = self.fuse(&out.pop().unwrap(), &surf);
                        fuse_pending = false;
                    }
                    out.push(surf);
                }
            };
        }
        macro_rules! start_word {
            () => {
                flush!();
                let space = !out.is_empty() && !matches!(prev, Some(Sp::Glue) | Some(Sp::Fuse));
                if space {
                    out.push(" ".into());
                }
                w = Some(Word::default());
            };
        }
        macro_rules! cur {
            () => {{
                if w.is_none() {
                    start_word!();
                }
                w.as_mut().unwrap()
            }};
        }
        for &id in ids {
            let kind = self.vocab.kinds.get(id as usize).ok_or_else(|| format!("bad token id {id}"))?;
            if raw.is_some() && !matches!(kind, Kind::Byte(_)) {
                out.push(String::from_utf8_lossy(&raw.take().unwrap()).into_owned());
            }
            let mut this_sp: Option<Sp> = None;
            match kind {
                Kind::Special(sp) => {
                    this_sp = Some(*sp);
                    match sp {
                        Sp::Cap | Sp::Caps | Sp::Capsb => {
                            start_word!();
                            w.as_mut().unwrap().case = Some(*sp);
                        }
                        Sp::Ascii | Sp::Bare => {
                            if w.as_ref().map_or(true, |x| x.has_stem() || !x.sfx.is_empty()) {
                                start_word!();
                            }
                            let x = w.as_mut().unwrap();
                            if *sp == Sp::Ascii { x.ascii = true } else { x.bare = true }
                        }
                        Sp::Apos => cur!().apos = Some('\''),
                        Sp::Apos2 => cur!().apos = Some('’'),
                        Sp::Lit => cur!().lit = Some(Vec::new()),
                        Sp::Raw => {
                            flush!();
                            raw = Some(Vec::new());
                        }
                        Sp::Glue => flush!(),
                        Sp::Fuse => {
                            flush!();
                            fuse_pending = true;
                        }
                        Sp::Sp | Sp::Nl | Sp::Tab | Sp::Cr => {
                            flush!();
                            out.push(match sp { Sp::Sp => " ", Sp::Nl => "\n", Sp::Tab => "\t", _ => "\r" }.into());
                        }
                        Sp::Bos | Sp::Eos | Sp::Pad => flush!(),
                    }
                }
                Kind::Edit(op) => cur!().edit.push(op.clone()),
                Kind::Root(r) => {
                    if w.as_ref().map_or(true, |x| x.has_stem() || !x.sfx.is_empty() || x.bare) {
                        start_word!();
                    }
                    w.as_mut().unwrap().root = Some(*r);
                }
                Kind::Lex(p, true) => {
                    if w.as_ref().map_or(true, |x| x.has_stem() || !x.sfx.is_empty() || x.bare) {
                        start_word!();
                    }
                    w.as_mut().unwrap().stem.push(Part::S(p.clone()));
                }
                Kind::Digit(d) => {
                    if w.as_ref().map_or(true, |x| x.digits.is_empty() || x.apos.is_some() || !x.sfx.is_empty()) {
                        start_word!();
                    }
                    w.as_mut().unwrap().digits.push(*d);
                }
                Kind::Lex(p, false) => {
                    let x = cur!();
                    let piece = Part::S(p.clone());
                    match &mut x.lit { Some(l) => l.push(piece), None => x.stem.push(piece) }
                }
                Kind::Byte(b) => match &mut raw {
                    Some(r) => r.push(*b),
                    None => {
                        let x = cur!();
                        match &mut x.lit { Some(l) => l.push(Part::B(*b)), None => x.stem.push(Part::B(*b)) }
                    }
                },
                Kind::Sfx(_) => cur!().sfx.push(id),
                Kind::Punct(p) => {
                    flush!();
                    out.push(p.clone());
                }
            }
            prev = this_sp;
        }
        if let Some(r) = raw.take() {
            out.push(String::from_utf8_lossy(&r).into_owned());
        }
        flush!();
        Ok(out.concat())
    }

    fn sfx_chain(&self, ids: &[u32]) -> Vec<Sfx> {
        ids.iter().flat_map(|&i| match &self.vocab.kinds[i as usize] { Kind::Sfx(v) => v.clone(), _ => vec![] }).collect()
    }

    fn surface(&self, w: &Word) -> String {
        let tx = &self.text;
        let chain = self.sfx_chain(&w.sfx);
        let mut tflags = 0u64;
        let (base, reading, flags): (Vec<char>, Vec<char>, u64);
        if !w.digits.is_empty() {
            base = c(&w.digits);
            reading = c(&number_reading(&w.digits));
            flags = 0;
        } else if w.bare {
            base = vec![];
            reading = vec![];
            flags = 0;
        } else if let Some(r) = w.root {
            let root = &self.lex.roots[r as usize];
            base = root.form.clone();
            flags = root.flags;
            reading = if root.reading.is_empty() { root.form.clone() } else { c(&root.reading) };
            tflags = root.flags & (FRONT_FLAG | CP3_FLAG);
        } else {
            let j = join(&w.stem);
            let j = j.strip_prefix('▁').unwrap_or(&j).to_string();
            base = c(&j);
            reading = self.spelled_reading(&tx.tr_lower(&base));
            flags = 0;
        }
        let mut out: Vec<char>;
        if let Some(ap) = w.apos {
            let tail: Vec<char> = match &w.lit {
                Some(l) => c(&join(l)),
                None => {
                    let g = generate(&self.t, &reading, tflags, &chain, None);
                    if g.len() >= reading.len() { g[reading.len()..].to_vec() } else { vec![] }
                }
            };
            out = if w.case == Some(Sp::Capsb) { tx.tr_upper(&base) } else { base };
            out.push(ap);
            out.extend(tail);
        } else if !w.digits.is_empty() {
            out = base;
        } else {
            let inner = w.root.and_then(|r| self.lex.roots[r as usize].inner.as_ref());
            out = generate(&self.t, &base, flags, &chain, inner);
            for op in &w.edit {
                out = apply_edit(&out, op);
            }
            if w.ascii {
                out = asciify(&out);
            }
        }
        match w.case {
            Some(Sp::Cap) => out = tx.tr_cap(&out),
            Some(Sp::Caps) => out = tx.tr_upper(&out),
            _ => {}
        }
        s(&out)
    }

    fn lower1_in(&self, ch: char, set: &str) -> bool {
        let l = self.text.lower(&[ch]);
        l.len() == 1 && set.contains(l[0])
    }

    fn fuse(&self, left: &str, right: &str) -> String {
        let mut l: Vec<char> = left.chars().collect();
        let mut r: Vec<char> = right.chars().collect();
        if l.last().map_or(false, |&x| self.lower1_in(x, "aeıioöuü")) {
            l.pop();
        }
        if r.len() > 1 && self.lower1_in(r[0], "yh") && self.lower1_in(r[1], "aeıioöuü") {
            r.remove(0);
        }
        l.extend(r);
        s(&l)
    }

    pub fn token_strs(&self, ids: &[u32]) -> Vec<&str> {
        ids.iter().map(|&i| self.vocab.toks[i as usize].as_str()).collect()
    }
}

#[derive(Clone)]
enum Part { S(String), B(u8) }

fn join(parts: &[Part]) -> String {
    let mut out = String::new();
    let mut buf: Vec<u8> = Vec::new();
    for p in parts {
        match p {
            Part::B(b) => buf.push(*b),
            Part::S(x) => {
                if !buf.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&buf));
                    buf.clear();
                }
                out.push_str(x);
            }
        }
    }
    if !buf.is_empty() {
        out.push_str(&String::from_utf8_lossy(&buf));
    }
    out
}

#[derive(Default)]
struct Word {
    case: Option<Sp>,
    root: Option<u32>,
    stem: Vec<Part>,
    digits: String,
    apos: Option<char>,
    sfx: Vec<u32>,
    lit: Option<Vec<Part>>,
    ascii: bool,
    bare: bool,
    edit: Vec<Op>,
}

impl Word {
    fn has_stem(&self) -> bool {
        self.root.is_some() || !self.stem.is_empty() || !self.digits.is_empty()
    }
}

fn foreign_hint(w: &str) -> bool {
    w.contains(['q', 'w', 'x', 'ñ', 'á', 'é', 'í', 'ó', 'ú'])
        || ["th", "sh", "ck", "ee", "oo", "ph"].iter().any(|x| w.contains(x))
}

#[allow(dead_code)]
fn _unused(t: &Tables, ch: &[Sfx]) -> f64 {
    suffix_cost(t, ch)
}
