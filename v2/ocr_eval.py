"""OCR-only evaluation at several strictness levels (no parser involved).

For every confirmed ground-truth value we ask: is it present in the OCR output of the invoice?
Text fields are scored at 3 levels so that "OCR cannot read it" can be separated from
"the matching rule is too strict":
  L1  contiguous phrase after NFKC+casefold+whitespace   (what the original benchmark does)
  L2  contiguous phrase after removing diacritics/punctuation
  L3  token coverage >= 0.9 (bag of folded tokens; ignores reading order / line breaks)
Numeric / identifier / date fields use one tolerant rule each.
Usage: HOADON_DATA=40HD.json python ocr_eval.py <thư mục có ocr_regions.jsonl> [--split dev|test|all]
"""
import json, re, sys, unicodedata, collections
from pathlib import Path
import numpy as np

import os
# Đường dẫn tới 40HD.json (giải nén từ 40HD_colab_bundle.zip). Đổi bằng biến môi trường HOADON_DATA
DATA = Path(os.environ.get("HOADON_DATA", "40HD.json"))
CONFIRMED = "Đã kiểm tra PDF"
NUMERIC = {"quantity", "unit_price", "tax_rate", "line_total", "total_amount"}
TEXT = {"seller_name", "item_description", "unit"}


def nfkc(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).casefold()).strip()


def fold(s):
    s = unicodedata.normalize("NFD", nfkc(s).replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def digits(s): return re.sub(r"\D", "", str(s))


def num_tokens(text):
    out = set()
    for t in re.findall(r"\d[\d.,]*", text):
        out.add(digits(t))
        m = re.match(r"^(.*?)[.,]0+$", t)
        if m: out.add(digits(m.group(1)))
        m = re.match(r"^(\d+)[.,]\d+$", t)       # decimal quantity such as 1,000
        if m and len(t.split(",")[-1].split(".")[-1]) != 3: out.add(digits(m.group(1)))
    return out


DATE_RES = [r"(\d{1,2})\D{0,25}thang\D{0,25}(\d{1,2})\D{0,25}nam\D{0,25}(\d{4})",
            r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})"]


def date_present(value, folded_text):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not m: return False
    y, mo, d = m.groups()
    for rx in DATE_RES:
        for g in re.findall(rx, folded_text):
            if (int(g[0]), int(g[1]), g[2]) == (int(d), int(mo), y): return True
    return False


def score_value(field, value, page_text):
    """returns dict level -> bool"""
    f_text = fold(page_text); n_text = nfkc(page_text)
    if field in NUMERIC:
        v = digits(value)
        ok = bool(v) and v in num_tokens(page_text)
        return {"tolerant": ok}
    if field == "seller_tax_code":
        return {"tolerant": digits(value) in digits(page_text)}
    if field == "invoice_number":
        return {"tolerant": nfkc(value) in n_text or fold(value) in f_text.split()}
    if field == "invoice_date":
        return {"tolerant": date_present(value, f_text)}
    toks = fold(value).split()
    page_c = collections.Counter(f_text.split())
    need = collections.Counter(toks)
    cov = sum(min(c, page_c[t]) for t, c in need.items()) / max(1, len(toks))
    return {"L1_strict": nfkc(value) in n_text,
            "L2_no_diacritics": fold(value) in f_text,
            "L3_token_cov90": cov >= 0.9}


def page_texts(regions_path, order="row"):
    """-> {engine: {invoice_id: text}}  (file có thể chứa nhiều engine)"""
    pages = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(regions_path, encoding="utf-8"):
        p = json.loads(line)
        regs = sorted(p["regions"], key=lambda r: (round(r["bbox"][1] / 12), r["bbox"][0])) if order == "row" else p["regions"]
        pages[p.get("engine", "ocr")][p["invoice_id"]].append(" ".join(r["text"] for r in regs))
    return {e: {k: "\n".join(v) for k, v in d.items()} for e, d in pages.items()}


def slots(ds, ids):
    for inv in ds["invoices"]:
        if ids and inv["id"] not in ids: continue
        for k, v in inv["header"].items():
            if v["status"] == CONFIRMED: yield inv["id"], k, (v.get("normalized") or v.get("raw") or "")
        for it in inv["items"]:
            for k, v in it["fields"].items():
                if v["status"] == CONFIRMED: yield inv["id"], k, (v.get("normalized") or v.get("raw") or "")


def evaluate(ocr_dir, split="all", splits_file=None):
    ds = json.load(open(DATA))
    ids = None
    if split != "all":
        ids = set(json.load(open(splits_file or Path(__file__).parent / "splits.json", encoding="utf-8"))[split])
    all_texts = page_texts(Path(ocr_dir) / "ocr_regions.jsonl")
    out = {}
    for engine, texts in all_texts.items():
        rows = []
        for iid, field, value in slots(ds, ids):
            if not str(value).strip(): continue
            res = score_value(field, str(value), texts.get(iid, ""))
            rows.append((iid, field, res))
        out[engine] = rows
    return out


def boot_ci(per_inv, B=2000, seed=0):
    rng = np.random.default_rng(seed); keys = list(per_inv); a = np.array([per_inv[k] for k in keys], float)
    vals = [a[rng.integers(0, len(a), len(a))].sum(0) for _ in range(B)]
    r = np.array([v[0] / max(1, v[1]) for v in vals]); return np.percentile(r, [2.5, 97.5])


def report(rows, title):
    print(f"\n=== {title} ===")
    levels = ["tolerant", "L1_strict", "L2_no_diacritics", "L3_token_cov90"]
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    inv_level = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for iid, field, res in rows:
        for lv, ok in res.items():
            by[field][lv][0] += ok; by[field][lv][1] += 1
            inv_level[lv][iid][0] += ok; inv_level[lv][iid][1] += 1
    print(f"{'trường':17s} {'n':>4s} | " + " | ".join(f"{l[:16]:>16s}" for l in levels))
    for field in sorted(by):
        n = max(v[1] for v in by[field].values())
        cells = [f"{by[field][l][0]/by[field][l][1]:6.1%}" if l in by[field] else "      -" for l in levels]
        print(f"{field:17s} {n:4d} | " + " | ".join(f"{c:>16s}" for c in cells))
    # overall: tolerant for non-text, and each level for text fields
    for lv in ["L1_strict", "L2_no_diacritics", "L3_token_cov90"]:
        tp = {k: [0, 0] for k in inv_level[lv]}
        for iid, field, res in rows:
            if field in TEXT: tp[iid][0] += res[lv]; tp[iid][1] += 1
            else: tp[iid][0] += res["tolerant"]; tp[iid][1] += 1
        a = sum(v[0] for v in tp.values()); n = sum(v[1] for v in tp.values())
        lo, hi = boot_ci(tp)
        print(f"TỔNG (số/ID/ngày: tolerant; chữ: {lv:16s}) = {a}/{n} = {a/n:.1%}  CI95 theo hóa đơn [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    d = sys.argv[1]; sp = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "all"
    for eng, rows in evaluate(d, sp).items():
        report(rows, f"engine={eng} | {d} | split={sp}")
