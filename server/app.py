"""Local test server: how TurkMorph splits text and puts it back together.

    python server/app.py [--port 8765]
    then open http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turkmorph.morphology import SUFFIXES, suffix_base             # noqa: E402
from turkmorph.presets import full_tokenizer                       # noqa: E402
from turkmorph.tokenizer import group_flags                        # noqa: E402

TOK = full_tokenizer()
PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
try:                                                               # (B) context mode, optional
    from turkmorph.context import ContextTokenizer
    CTX = ContextTokenizer(TOK, ROOT / "data" / "context.pt")
except Exception:
    CTX = None

SFX_NAMES = {
    "PL": "çoğul", "P1SG": "1. tekil iyelik", "P2SG": "2. tekil iyelik", "P3SG": "3. tekil iyelik",
    "P1PL": "1. çoğul iyelik", "P2PL": "2. çoğul iyelik", "P3PL": "3. çoğul iyelik",
    "ACC": "belirtme", "DAT": "yönelme", "LOC": "bulunma", "ABL": "ayrılma", "GEN": "ilgi",
    "INS": "vasıta", "EQU": "eşitlik", "REL": "-ki", "COP3": "ek-fiil -DIr", "PCOP": "hikâye -(y)DI",
    "ECOP": "rivayet -(y)mIş", "CCOP": "şart -(y)sA", "KEN": "-(y)ken", "NEG": "olumsuz",
    "PROG": "şimdiki zaman", "PROG2": "-mAktA", "PAST": "görülen geçmiş", "FUT": "gelecek",
    "EVID": "öğrenilen geçmiş", "AOR": "geniş zaman", "COND": "şart", "NECES": "gereklilik",
    "OPT": "istek", "CAUS": "ettirgen", "PASS": "edilgen", "RECP": "işteş", "ABIL": "yeterlik",
    "IMPOT": "yetersizlik", "PART_DHK": "sıfat-fiil -DIK", "PART_AN": "sıfat-fiil -(y)An",
    "INF": "mastar -mAk", "VN": "isim-fiil -mA", "VN_HS": "isim-fiil -(y)Iş", "PROG_C": "şimdiki (konuşma)",
    "FUT_C": "gelecek (konuşma)", "FUT_C2": "gelecek (konuşma)", "LOC_C": "bulunma (konuşma)",
    "ABL_C": "ayrılma (konuşma)", "DIM": "küçültme", "ORD": "sıra sayı", "DIST": "üleştirme",
    "LH": "-lI", "LHK": "-lIk", "CH": "-CI", "SHZ": "-sIz", "AGT": "-(y)IcI", "SAL": "-sAl",
}
NS_HELP = {"R": "kök (sözlük + ses bayrakları)", "S": "ek (arkifonemik ID)", "L": "LEX parçası (BPE)",
           "B": "bayt", "D": "rakam", "P": "noktalama", "E": "yazım hatası düzeltmesi", "<": "özel"}
MAX_BODY_BYTES = 256_000


def explain_token(t: str) -> dict:
    ns = t[0] if t[0] == "<" else t.split(":", 1)[0]
    info = {"tok": t, "id": TOK.vocab.index.get(t), "ns": ns, "help": NS_HELP.get(ns[:1], "")}
    if t.startswith("S:"):
        sid = suffix_base(t[2:])
        sf = SUFFIXES.get(sid)
        marked = t[2:].partition("!")[2]
        info["help"] = (SFX_NAMES.get(sid, sid) + (f"  şablon {sf.tmpl}" if sf else "")
                        + (f"  (işaretli yüzey '{marked}')" if marked else ""))
    elif t.startswith("R:"):
        r = TOK.lex.by_key.get(t[2:])
        if r:
            info["help"] = f"kök '{r.form}' ({r.pos})" + (f"  bayraklar: {', '.join(sorted(r.flags))}" if r.flags else "")
    elif t.startswith("E:"):
        info["help"] = "yazım farkı: " + t[2:].replace(":", " ")
    return info


def turkmorph(text: str, context: bool = False) -> dict:
    t0 = time.perf_counter()
    toks = CTX.encode(text) if (context and CTX) else TOK.encode(text)
    enc_ms = (time.perf_counter() - t0) * 1000
    decoded = TOK.decode(toks)
    words, cur = [], None
    for t, start in zip(toks, group_flags(toks)):
        if start:
            cur = {"tokens": []}
            words.append(cur)
        cur["tokens"].append(t)
    for w in words:                                    # how decode rebuilds each word, token by token
        steps, shown = [], ""
        for i in range(1, len(w["tokens"]) + 1):
            s = TOK.decode(w["tokens"][:i])
            if s != shown:
                steps.append({"after": w["tokens"][i - 1], "surface": s})
                shown = s
        w["surface"] = TOK.decode(w["tokens"])
        w["steps"] = steps
        w["tokens"] = [explain_token(t) for t in w["tokens"]]
    return {"n_tokens": len(toks), "decoded": decoded, "lossless": decoded == text,
            "words": words, "encode_ms": round(enc_ms, 2)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/tokenize":
            return self._send(404, "not found", "text/plain")
        def bad_request(code, message):
            self.close_connection = True
            return self._send(code, json.dumps({"error": message}), "application/json; charset=utf-8")

        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return bad_request(400, "invalid Content-Length")
        if n < 0:
            return bad_request(400, "invalid Content-Length")
        if n > MAX_BODY_BYTES:
            return bad_request(413, "request body too large")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return bad_request(400, "invalid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("text", ""), str):
            return bad_request(400, "text must be a string in a JSON object")
        text = body.get("text", "")[:20000]
        ctx = bool(body.get("context"))
        res = {"turkmorph": turkmorph(text, ctx),
               "vocab": {"turkmorph": len(TOK.vocab)}, "context_available": CTX is not None,
               "context": ctx and CTX is not None}
        self._send(200, json.dumps(res, ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"TurkMorph test server on http://localhost:{args.port}  "
          f"(context mode: {'on' if CTX else 'off'})",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
