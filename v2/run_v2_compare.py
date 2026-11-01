"""So parser CŨ (repo) với parser v2 trên cùng đầu ra OCR của thư mục kết quả.
Dùng:  HOADON_REPO=<repo> python run_v2_compare.py <thư_mục_kết_quả> --data 40HD.json [--split dev|test|all]
Thư mục kết quả cần có predictions.csv và ocr_regions.jsonl (có thể chứa nhiều engine)."""
import sys, os, json, csv, argparse, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import parser_v2 as pv
sm = pv.sm

def run(res_dir, data, out, parser, ids):
    res_dir, out = Path(res_dir), Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(res_dir / "predictions.csv", encoding="utf-8-sig")))
    if ids: rows = [r for r in rows if r["invoice_id"] in ids]
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    orig = sm.extract_fields
    if parser == "v2": sm.extract_fields = pv.extract_fields_v2
    try: sm.evaluate_run(out / "predictions.csv", Path(data), out, res_dir / "ocr_regions.jsonl")
    finally: sm.extract_fields = orig
    return list(csv.DictReader(open(out / "structured_field_scores.csv", encoding="utf-8-sig")))

def f1(a):
    tp, fp, fn = np.array(a).sum(0); return tp, fp, fn, 2 * tp / max(1, 2 * tp + fp + fn)

def boot(per_inv, B=2000, seed=0):
    rng = np.random.default_rng(seed); a = np.array(list(per_inv.values())); n = len(a)
    v = [f1(a[rng.integers(0, n, n)])[3] for _ in range(B)]; return np.percentile(v, [2.5, 97.5])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="all"); ap.add_argument("--splits-file", default=str(Path(__file__).resolve().parent / "splits.json"))
    ap.add_argument("--out", default="so_sanh_v2"); a = ap.parse_args()
    ids = set(json.load(open(a.splits_file, encoding="utf-8"))[a.split]) if a.split != "all" else None
    scores = {p: run(a.results, a.data, Path(a.out) / p, p, ids) for p in ("old", "v2")}
    engines = sorted({r["engine"] for r in scores["old"]})
    print(f"\n=== Parser cũ vs v2 | split={a.split} | {a.results} ===")
    print(f"{'engine':12s} {'parser':6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}   CI95 F1 (theo hóa đơn)")
    for e in engines:
        for p in ("old", "v2"):
            per = collections.defaultdict(lambda: [0, 0, 0])
            for r in scores[p]:
                if r["engine"] != e: continue
                x = per[r["invoice_id"]]; x[0] += int(r["tp"]); x[1] += int(r["fp"]); x[2] += int(r["fn"])
            tp, fp, fn, F = f1(list(per.values())); lo, hi = boot(per)
            P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
            print(f"{e:12s} {p:6s} {tp:4d} {fp:4d} {fn:4d} {P:6.3f} {R:6.3f} {F:6.3f}   [{lo:.3f}, {hi:.3f}]")
    print("\n=== Số ô đúng (TP) theo trường: cũ -> v2 ===")
    fields = sorted({r["field"] for r in scores["v2"]})
    print(f"{'trường':17s} " + " ".join(f"{e[:10]:>16s}" for e in engines))
    for fld in fields:
        cells = []
        for e in engines:
            c = {p: sum(int(r["tp"]) for r in scores[p] if r["engine"] == e and r["field"] == fld) for p in ("old", "v2")}
            n = sum(1 for r in scores["v2"] if r["engine"] == e and r["field"] == fld and r["reference_status"] == "Đã kiểm tra PDF")
            cells.append(f"{c['old']:3d}->{c['v2']:3d} /{n:3d}")
        print(f"{fld:17s} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"\nChi tiết từng ô: {a.out}/old và {a.out}/v2 (structured_field_scores.csv)")
