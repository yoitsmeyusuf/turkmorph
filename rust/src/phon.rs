//! Turkish morphophonology: archiphoneme realization (port of turkmorph/phonology.py).

pub fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û')
}
fn is_back(c: char) -> bool {
    matches!(c, 'a' | 'ı' | 'o' | 'u' | 'â' | 'û')
}
fn is_round(c: char) -> bool {
    matches!(c, 'o' | 'ö' | 'u' | 'ü' | 'û')
}
pub fn is_hard(c: char) -> bool {
    matches!(c, 'ç' | 'f' | 'h' | 'k' | 'p' | 's' | 'ş' | 't')
}

pub fn last_vowel(s: &[char]) -> Option<char> {
    s.iter().rev().copied().find(|&c| is_vowel(c))
}

pub fn n_vowels(s: &[char]) -> usize {
    s.iter().filter(|&&c| is_vowel(c)).count()
}

/// kitap -> kitab, renk -> reng, köpek -> köpeğ, psikolog -> psikoloğ
pub fn soften(s: &mut [char]) {
    let n = s.len();
    if n == 0 {
        return;
    }
    let c = s[n - 1];
    if c == 'k' && n >= 2 && s[n - 2] == 'n' {
        s[n - 1] = 'g';
        return;
    }
    let r = match c {
        'p' => 'b',
        'ç' => 'c',
        't' => 'd',
        'k' => 'ğ',
        'g' => 'ğ',
        _ => return,
    };
    s[n - 1] = r;
}

/// ağız -> ağz
pub fn drop_last_vowel(s: &mut Vec<char>) {
    if let Some(i) = s.iter().rposition(|&c| is_vowel(c)) {
        s.remove(i);
    }
}

fn harm_h(v: Option<char>, fo: bool) -> char {
    let back = v.map_or(false, is_back) && !fo;
    let rnd = v.map_or(false, is_round);
    match (back, rnd) {
        (true, true) => 'u',
        (true, false) => 'ı',
        (false, true) => 'ü',
        (false, false) => 'i',
    }
}

fn harm_a(v: Option<char>, fo: bool) -> char {
    if v.map_or(false, is_back) && !fo {
        'a'
    } else {
        'e'
    }
}

fn trigger(s: &[char], out: &[char], fo: bool) -> (Option<char>, bool) {
    match last_vowel(out) {
        Some(v) => (Some(v), false),
        None => (last_vowel(s), fo),
    }
}

/// Realize an archiphonemic template as the piece appended to `s`.
pub fn realize(tmpl: &str, s: &[char], front_override: bool) -> Vec<char> {
    let t: Vec<char> = tmpl.chars().collect();
    realize_c(&t, s, front_override).to_vec()
}

pub type Piece = smallvec::SmallVec<[char; 8]>;

/// realize() on a pre-split template, without heap allocation for short suffixes
pub fn realize_c(t: &[char], s: &[char], front_override: bool) -> Piece {
    let mut out = Piece::new();
    let mut i = 0;
    while i < t.len() {
        let last = out.last().copied().or_else(|| s.last().copied());
        let ends_v = last.map_or(false, is_vowel);
        let c = t[i];
        if c == '(' {
            let j = i + t[i..].iter().position(|&x| x == ')').expect("unclosed template group");
            let inner = &t[i + 1..j];
            if inner == ['H'] {
                if !ends_v {
                    let (v, fo) = trigger(s, &out, front_override);
                    out.push(harm_h(v, fo));
                }
            } else if ends_v {
                out.extend_from_slice(inner);
            }
            i = j + 1;
            continue;
        }
        match c {
            'A' => {
                let (v, fo) = trigger(s, &out, front_override);
                out.push(harm_a(v, fo));
            }
            'H' => {
                let (v, fo) = trigger(s, &out, front_override);
                out.push(harm_h(v, fo));
            }
            'D' => out.push(if last.map_or(false, is_hard) { 't' } else { 'd' }),
            'C' => out.push(if last.map_or(false, is_hard) { 'ç' } else { 'c' }),
            'G' => out.push(if last.map_or(false, is_hard) { 'k' } else { 'g' }),
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Harmony reading of a numeral: that of its last number word (2023 -> üç).
pub fn number_reading(digits: &str) -> String {
    const UNITS: [&str; 10] = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
    const TENS: [&str; 10] = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
    let d: Vec<u32> = digits.chars().filter_map(|c| c.to_digit(10)).collect();
    let at = |k: usize| -> u32 { if k < d.len() { d[d.len() - 1 - k] } else { 0 } };
    let group = |lo: usize| -> u32 { at(lo) + 10 * at(lo + 1) + 100 * at(lo + 2) };
    if d.iter().all(|&x| x == 0) {
        return "sıfır".into();
    }
    if at(0) != 0 {
        return UNITS[at(0) as usize].into();
    }
    if at(1) != 0 {
        return TENS[at(1) as usize].into();
    }
    if at(2) != 0 {
        return "yüz".into();
    }
    if group(3) != 0 {
        return "bin".into();
    }
    if group(6) != 0 {
        return "milyon".into();
    }
    "milyar".into()
}
