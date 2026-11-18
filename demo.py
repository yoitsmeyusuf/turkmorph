"""Build a demo tokenizer, print segmentations, and run the lossless invariant."""
import sys

from turkmorph.presets import demo_tokenizer, full_tokenizer

# --full: Zemberek lexicon + EM costs + LEX BPE + langid (needs scripts/build_all.sh)
tok = full_tokenizer() if "--full" in sys.argv else demo_tokenizer()

EXAMPLES = [
    "evlerimizdekiler", "kitabımızdan", "kalktığımızda", "gidiyorum", "ağzından",
    "hakkında", "saatlerde", "kalbimiz", "rengi", "topu", "suyu", "bana", "onda",
    "diyor", "söylüyor", "arıyor", "geleceğim", "gelmiyor", "gelmez", "alır",
    "yapar", "okur", "yapabilir", "reddi", "şehrimizde", "oğlu", "kalıplaşmış",
    "geçmişten", "günümüze", "bakımından", "mecazlı", "sözlerdir",
    "Danimarka'dan", "Danimarka", "komünistlerden", "Avrupa", "2023'te",
    "6'dan", "10'a", "meeting'e", "Bordeaux'ya", "ANKARA'DA", "iPhone",
    # added in v0.2: voice, compound verbs, converbs, -ken, -ki harmony, pronouns
    "yazıldı", "okundu", "bilinen", "yaptırdım", "okuttu", "batırdı", "çıkardık",
    "getirdi", "gidiverdi", "yapamadım", "gelmeli", "gelip", "gelince", "gelirken",
    "çocukken", "evdeyken", "dünkü", "yarınki", "gelseydik", "geldiyse", "gelsinler",
    "suyun", "onlar", "bunlara", "benim", "bizimki", "onunla", "görüştük",
]

for w in EXAMPLES:
    t = tok.encode(w)
    assert tok.decode(t) == w, (w, t, tok.decode(t))
    print(f"{w:22s} -> {' '.join(t)}")

SENTENCES = [
    "Kalktığımızda hep birlikte yürüdük.",
    "Evlerimizdekiler, Danimarka'dan 2023'te geldi!",
    "The meeting was postponed; el gobierno  dijo  que no.\n\tYeni satır (parantez) \"tırnak\" ☃ emoji 🙂",
    "  baştaki boşluk ve sondaki   ",
    "İSTANBUL'DA Iğdır ısı İzmir",
    "1 2 3 ve 12 34 5'te",
]
print()
for s in SENTENCES:
    t = tok.encode(s)
    ok = tok.decode(t) == s
    ids = tok.encode_ids(s)
    assert tok.decode_ids(ids) == s
    print(f"[{'OK' if ok else 'FAIL'}] {len(t):3d} tokens | {s!r}")
    assert ok, (s, t, tok.decode(t))

# the 'dan' in Danimarka is a LEX piece; ablative -DAn is S:ABL -> disjoint ids
v = tok.vocab
print("\nvocab size:", len(v), {k: b - a for k, (a, b) in v.ranges.items()})
print("ABL id", v.id("S:ABL"), v.namespace(v.id("S:ABL")))
for t in tok.encode("komünistlerden"):
    print(f"  {t:14s} id={v.id(t):5d} ns={v.namespace(v.id(t))}")
print("encode paths:", tok.stats)
