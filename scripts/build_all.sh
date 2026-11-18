#!/usr/bin/env bash
# Full pipeline: download raw data (if missing) -> lexicon -> EM -> LEX/langid -> eval.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p $RAW/ud $RAW/ud_foreign
Z=https://raw.githubusercontent.com/ahmetaa/zemberek-nlp/master/morphology/src/main/resources/tr
for f in master-dictionary non-tdk proper locations-tr person-names abbreviations proper-from-corpus; do
  [ -s $RAW/zemberek-$f.dict ] || curl -sSfLo $RAW/zemberek-$f.dict $Z/$f.dict
done
[ -s $RAW/zemberek-LICENSE ] || curl -sSfLo $RAW/zemberek-LICENSE https://raw.githubusercontent.com/ahmetaa/zemberek-nlp/master/LICENSE
U=https://raw.githubusercontent.com/UniversalDependencies
for tb in boun:BOUN imst:IMST kenet:Kenet penn:Penn tourism:Tourism framenet:FrameNet; do
  code=${tb%%:*}; name=${tb##*:}
  for sp in train dev test; do
    f=$RAW/ud/tr_$code-ud-$sp.conllu
    [ -s $f ] || curl -sSfLo $f $U/UD_Turkish-$name/master/tr_$code-ud-$sp.conllu
  done
done
for tb in en_ewt:English-EWT es_gsd:Spanish-GSD de_gsd:German-GSD fr_gsd:French-GSD it_isdt:Italian-ISDT; do
  code=${tb%%:*}; name=${tb##*:}
  for sp in train test; do
    f=$RAW/ud_foreign/$code-ud-$sp.conllu
    [ -s $f ] || curl -sSfLo $f $U/UD_$name/master/$code-ud-$sp.conllu
  done
done
python3 scripts/build_lexicon.py
python3 scripts/train_em.py
cp data/costs_em_lemma.json data/costs.json
python3 scripts/build_lex.py
python3 scripts/evaluate.py
