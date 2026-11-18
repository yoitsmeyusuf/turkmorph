#!/usr/bin/env bash
# Rebuild everything from already-downloaded data (see build_all.sh for downloads).
# Prints a timestamp per stage; long stages print progress with elapsed / ETA.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export TURKMORPH_PROCS=${TURKMORPH_PROCS:-6}            # 6 workers: keeps a laptop CPU cooler
T0=$(date +%s)
stage() { echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $*"; }
[ -s data/cache/bt_counts.pkl ] || { [ -d data/raw/bellaturca ] && stage "prepare BellaTurca (~2 min)" && python3 scripts/prepare_bellaturca.py; } || true
[ "${SKIP_LEXICON:-0}" = 1 ] || { stage "lexicon (~1.5 min)"; python3 scripts/build_lexicon.py; }
stage "EM (~1 min UD, ~2 min BellaTurca analysis, ~4 min EM)"; python3 scripts/train_em.py
if [ -s data/costs_bt.json ]; then cp data/costs_bt.json data/costs.json; else cp data/costs_em_lemma.json data/costs.json; fi
stage "langid + LEX BPE (~4 min)";     python3 scripts/build_lex.py
stage "frequency audit + underspecified ids (~3 min)"; python3 scripts/build_overrides.py && python3 scripts/build_unions.py
# (B) context model: tied to the vocabulary ids, so retrain after a rebuild (GPU, ~12 min)
[ "${CONTEXT:-0}" = 1 ] && { stage "context model (~12 min)"; python3 scripts/train_context.py; }
stage "evaluate (~4 min)";             python3 scripts/evaluate.py "$@"
stage "done"
