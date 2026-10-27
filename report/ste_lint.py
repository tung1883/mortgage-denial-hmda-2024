"""Small STE lint for report prose (approximate; not the official ASD check)."""
import re, sys
from html.parser import HTMLParser

SKIP = {"script", "style", "svg", "table", "code", "pre", "title"}
BLOCK = {"p", "li", "figcaption", "h1", "h2", "h3", "span", "footer"}
APPROVED_ING = {"mating", "missing", "remaining", "lighting", "opening", "routing", "servicing", "during"}
BAD_MODAL = re.compile(r"\b(should|would|may|might|shall)\b", re.I)
HAVE_PP = re.compile(r"\b(has|have|had)\s+(been|\w+ed|\w+en)\b", re.I)
CONTR = re.compile(r"\b\w+('|\u2019)(t|s|re|ve|ll|d)\b", re.I)
PASSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+(\w+ed|made|done|given|shown|known|seen|taken|chosen)\b(?!\s+(as|to)\b)", re.I)
# technical names that end in -ing and are allowed (rule 3.5)
TECH_ING = {"boosting", "bootstrap", "ceiling", "string", "building", "thing", "something", "nothing", "everything", "during", "training", "testing"}

class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.skip = 0; self.stack = []; self.cur = []; self.paras = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP or "eq" in (a.get("class") or "").split() or tag.startswith("mjx"): self.skip += 1; self.stack.append(True)
        else: self.stack.append(False); self.flush() if tag in BLOCK|{"div","figure"} else None
    def handle_endtag(self, tag):
        if self.stack and self.stack.pop(): self.skip -= 1
        if tag in BLOCK|{"div","figure"}: self.flush()
    def handle_data(self, d):
        if not self.skip: self.cur.append(d)
    def flush(self):
        t = re.sub(r"\s+", " ", "".join(self.cur)).strip()
        if t: self.paras.append(t)
        self.cur = []

html = open(sys.argv[1], encoding="utf-8").read()
p = P(); p.feed(html); p.flush()
issues = 0
seen = set()
for para in p.paras:
    if para in seen: continue
    seen.add(para)
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", para) if s]
    if len(sents) > 6 and len(para) > 200:
        print(f"[para {len(sents)} sentences] {para[:70]}..."); issues += 1
    for s in sents:
        w = len(s.split())
        if w > 25: print(f"[{w} words] {s}"); issues += 1
        if ";" in s: print(f"[semicolon] {s}"); issues += 1
        for rx, name in ((BAD_MODAL, "modal"), (HAVE_PP, "have+pp"), (CONTR, "contraction")):
            m = rx.search(s)
            if m: print(f"[{name}: {m.group(0)}] {s[:110]}"); issues += 1
        for m in re.finditer(r"\b([A-Za-z]+ing)\b", s):
            if m.group(1).lower() not in APPROVED_ING|TECH_ING: print(f"[-ing: {m.group(1)}] {s[:110]}"); issues += 1
        m = PASSIVE.search(s)
        if m: print(f"[passive?: {m.group(0)}] {s[:110]}"); issues += 1
print("issues:", issues, "paragraphs:", len(seen))
