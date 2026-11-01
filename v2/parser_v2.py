"""Parser v2: geometry-based extraction on OCR boxes (P0-P4). Labels are never read."""
import re, sys, unicodedata
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
# structured_metrics_40hd.py nằm trong repo gốc. Đặt biến môi trường HOADON_REPO trỏ tới thư mục repo,
# hoặc để parser_v2.py cùng thư mục với structured_metrics_40hd.py.
for _p in (os.environ.get("HOADON_REPO"), str(Path(__file__).resolve().parent), os.getcwd()):
    if _p and _p not in sys.path: sys.path.insert(0, _p)
import structured_metrics_40hd as sm


def fold(s):
    s = unicodedata.normalize("NFD", str(s).replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.casefold()


class W:
    __slots__ = ("text", "f", "x0", "y0", "x1", "y1", "conf", "page")

    def __init__(self, text, x0, y0, x1, y1, conf=1.0, page=1):
        self.text, self.f = text, fold(text)
        self.x0, self.y0, self.x1, self.y1, self.conf, self.page = x0, y0, x1, y1, conf, page

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def h(self): return max(1.0, self.y1 - self.y0)


# ---------------------------------------------------------------- P0
def to_words(page):
    """Regions (word- or line-level) -> word boxes. Multi-token regions are split
    proportionally to character counts."""
    out = []
    for r in page.get("regions", []):
        t = str(r.get("text", "")).strip()
        if not t: continue
        x0, y0, x1, y1 = r["bbox"]
        toks = t.split()
        if len(toks) == 1:
            out.append(W(t, x0, y0, x1, y1, r.get("confidence", 1.0), page.get("page", 1)))
            continue
        total = sum(len(k) for k in toks) + (len(toks) - 1)
        pos = 0
        for k in toks:
            a = x0 + (x1 - x0) * pos / total
            b = x0 + (x1 - x0) * (pos + len(k)) / total
            out.append(W(k, a, y0, b, y1, r.get("confidence", 1.0), page.get("page", 1)))
            pos += len(k) + 1
    return out


def cluster_rows(words, ratio=0.55):
    rows = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        if rows:
            cy = sum(v.cy for v in rows[-1]) / len(rows[-1])
            hh = max(w.h, sum(v.h for v in rows[-1]) / len(rows[-1]))
            if abs(w.cy - cy) <= ratio * hh:
                rows[-1].append(w); continue
        rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def merge_spaced_digits(words):
    """'0 1 0 8 0 7' -> '010807' (some invoices print digits with spacing)."""
    out, rows = [], cluster_rows(words)
    for row in rows:
        i = 0
        while i < len(row):
            w = row[i]
            if re.fullmatch(r"\d", w.text):
                j, buf = i, [w]
                while j + 1 < len(row) and re.fullmatch(r"\d", row[j + 1].text) and row[j + 1].x0 - row[j].x1 < 1.2 * w.h:
                    j += 1; buf.append(row[j])
                if len(buf) >= 4:
                    out.append(W("".join(b.text for b in buf), buf[0].x0, min(b.y0 for b in buf),
                                 buf[-1].x1, max(b.y1 for b in buf), min(b.conf for b in buf), w.page))
                    i = j + 1; continue
            out.append(w); i += 1
    return out


def num(s):
    """Parse Vietnamese-style number to Decimal or None."""
    t = re.sub(r"[^\d.,]", "", s)
    if not t or not re.search(r"\d", t): return None
    if "." in t and "," in t:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        t = t.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        parts = t.split(sep)
        if len(parts) > 2 or (len(parts[-1]) == 3 and len(parts[0]) <= 3 and parts[0] != "0"):
            t = "".join(parts)
        else:
            t = ".".join(parts)
    try: return Decimal(t)
    except InvalidOperation: return None


AMOUNT_RE = re.compile(r"^-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$")


def is_amount(w): return bool(AMOUNT_RE.match(w.text.strip("()")))


# ---------------------------------------------------------------- table (P2/P3)
STOP_ROW = re.compile(r"(cong tien hang|tong cong|tong tien|cong tien|so tien (?:viet )?bang chu|tong so tien|"
                      r"tong gia tri|tien thanh toan|tong thanh toan|cong thanh tien)")
HDR_VOC = set("""stt tt ten hang hoa dich vu don vi tinh dvt so luong gia thue suat thanh tien gtgt chua gom truoc sau chiet khau
vat rate amount price unit quantity description total not including before after tax name goods and services of no discount
ma sp ghi chu quy cach nhan hieu xuat xu""".split())


def _fw(w): return re.sub(r"[^\w()%]", "", w.f)


def find_table(words):
    rows = cluster_rows(words)
    stts = [w for w in words if _fw(w) in ("stt", "tt")]
    for s in sorted(stts, key=lambda w: w.y0):
        zone = [w for w in words if s.y0 - 1.5 * s.h <= w.cy <= s.y1 + 9 * s.h and w.x0 >= 0]
        zr = cluster_rows(zone)
        anchors = {}
        flat = sorted(zone, key=lambda w: (w.cy, w.x0))
        F2 = lambda w: re.sub(r"[():.,%\-]", "", w.f)
        for r in zr:
            for i, w in enumerate(r):
                prev = r[i - 1] if i else None
                nxt = r[i + 1] if i + 1 < len(r) else None
                f = F2(w)
                if f == "ten" and nxt and F2(nxt) in ("hang", "mat"):
                    chain = [w]
                    for v in r[i + 1:]:
                        if v.x0 - chain[-1].x1 < 2.5 * w.h: chain.append(v)
                        else: break
                    anchors.setdefault("item_description", []).append(tuple(chain))
                if f in ("dvt", "vi", "tinh"): anchors.setdefault("unit", []).append((w,))
                if f in ("luong", "sl") or (f.startswith("luon") and len(f) >= 4): anchors.setdefault("quantity", []).append((w,))
                if f == "gia": anchors.setdefault("unit_price", []).append((w,))
                if f == "suat" or (f == "rate" and prev and F2(prev) == "vat"): anchors.setdefault("tax_rate", []).append((w,))
                if f == "thanh": anchors.setdefault("line_total", []).append((w,))
        if len([k for k in ("item_description", "quantity", "unit_price", "line_total") if k in anchors]) >= 3:
            return s, anchors, zone
    return None


def _ext(ws):
    return min(w.x0 for w in ws), max(w.x1 for w in ws), min(w.y0 for w in ws), max(w.y1 for w in ws)


def _extend(ws, zone):
    x0, x1, y0, y1 = _ext(ws)
    wd = max(x1 - x0, 1)
    out = list(ws)
    for v in zone:
        if v in out: continue
        f = re.sub(r"[():.,%\-]", "", v.f)
        if not (f in HDR_VOC or ((v.text.startswith("(") or v.text.endswith(")")) and not re.search(r"\d", v.text))): continue
        if x0 - 0.3 * wd <= v.cx <= x1 + 0.3 * wd and abs(v.cy - (y0 + y1) / 2) < 5 * v.h:
            out.append(v)
    return out


def build_columns(stt, anchors, ltotal_pick="right", zone=()):
    cols = {"__stt__": (stt.x0, stt.x1)}
    for k, lst in anchors.items():
        if k == "line_total":
            pick = sorted(lst, key=lambda t: t[0].x0)
            t = pick[-1] if ltotal_pick == "right" else pick[0]
        else:
            t = sorted(lst, key=lambda t: t[0].y0)[0]
        ws = _extend([x for x in t if isinstance(x, W)], zone) if k != "item_description" else [x for x in t if isinstance(x, W)]
        x0, x1, _, _ = _ext(ws)
        cols[k] = (x0, x1)
    order = sorted(cols.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / 2)
    bounds = {}
    for i, (k, (x0, x1)) in enumerate(order):
        lo = -1e9 if i == 0 else None
        hi = 1e9 if i == len(order) - 1 else None
        if lo is None:
            pk, (px0, px1) = order[i - 1]
            lo = (px1 + x0) / 2 if px1 < x0 else ((px0 + px1) / 2 + (x0 + x1) / 2) / 2
        if hi is None:
            nk, (nx0, nx1) = order[i + 1]
            hi = (x1 + nx0) / 2 if x1 < nx0 else ((x0 + x1) / 2 + (nx0 + nx1) / 2) / 2
        bounds[k] = (lo, hi)
    return bounds, cols


def refine_bounds(bounds, cols, body_words):
    """Data-driven column cuts: cluster x-projection of body words, assign each cluster to the
    nearest header, cut at the middle of the gap between neighbouring columns' clusters."""
    iv = sorted((w.x0 - 2, w.x1 + 2) for w in body_words)
    clusters = []
    for a_, b_ in iv:
        if clusters and a_ <= clusters[-1][1] + 6: clusters[-1][1] = max(clusters[-1][1], b_)
        else: clusters.append([a_, b_])
    keys = sorted(cols, key=lambda k: (cols[k][0] + cols[k][1]) / 2)
    owned = {k: [] for k in keys}
    for c0, c1 in clusters:
        best, bd = None, 1e18
        for k in keys:
            h0, h1 = cols[k]
            d = 0 if (c0 < h1 and c1 > h0) else min(abs(c0 - h1), abs(c1 - h0))
            ov = max(0, min(c1, h1) - max(c0, h0))
            sc = d - ov * 0.01
            if sc < bd: best, bd = k, sc
        owned[best].append((c0, c1))
    cuts = []
    for i in range(len(keys) - 1):
        A, B = keys[i], keys[i + 1]
        if owned[A] and owned[B]:
            cuts.append((max(c[1] for c in owned[A]) + min(c[0] for c in owned[B])) / 2)
        else:
            cuts.append(bounds[B][0])
    new = {}
    for i, k in enumerate(keys):
        new[k] = (-1e9 if i == 0 else cuts[i - 1], 1e9 if i == len(keys) - 1 else cuts[i])
    return new


def col_of(w, bounds):
    for k, (lo, hi) in bounds.items():
        if lo <= w.cx < hi: return k
    return None


def parse_table(pages_words, ltotal_pick="right"):
    """Returns (items, info). items: {line: {field: (text, [words])}}"""
    items, info = {}, {"ok": False}
    bounds = None; last_no = 0; last_ce = {}
    for pi, words in enumerate(pages_words):
        ft = find_table(words)
        if ft:
            stt, anchors, zone = ft
            bounds, cols_ext = build_columns(stt, anchors, ltotal_pick, zone)
            top = stt.y0 - 2 * stt.h
        elif bounds is None:
            continue
        else:
            top = -1
        stt_lo, stt_hi = bounds["__stt__"]
        cand = sorted([w for w in words if re.fullmatch(r"\d{1,3}", w.text) and stt_lo <= w.cx < stt_hi and w.y0 > top + (stt.h * 3 if ft else 0)], key=lambda w: w.y0)
        # runs of consecutive numbers
        runs, cur = [], []
        for w in cand:
            n = int(w.text)
            if cur and n == int(cur[-1].text) + 1: cur.append(w)
            else:
                if cur: runs.append(cur)
                cur = [w]
        if cur: runs.append(cur)
        if pi > 0 and last_no:
            runs = [r for r in runs if int(r[0].text) == last_no + 1] or runs
        if not runs: continue
        best = max(runs, key=lambda r: (len(r), r[0].y0))
        if pi == 0 and len(best) == 1 and len(runs) > 1: best = runs[-1]
        # table end
        after = [w for w in words if w.y0 > best[-1].y1]
        end_y = 1e9
        for r in cluster_rows(after):
            rt = " ".join(fold(w.text) for w in r)
            if STOP_ROW.search(rt) and r[0].y0 > best[-1].y0:
                end_y = r[0].y0 - 2; break
        # header bottom (row-level classification of rows above the first STT)
        hdr_bottom = (top if ft else -1)
        if ft:
            for r in cluster_rows([w for w in words if top <= w.cy < best[0].cy - 0.4 * best[0].h]):
                toks = [_fw(w) for w in r]
                voc = sum(1 for t in toks if t in HDR_VOC or ((t.startswith("(") or t.endswith(")")) and not re.search(r"\d", t)))
                numbering = len(toks) >= 3 and all(re.fullmatch(r"[\d=x+*/().,\-]+", t) for t in toks)
                if voc >= 0.5 * len(toks) or numbering:
                    hdr_bottom = max(hdr_bottom, max(w.y1 for w in r))
        sset = set(id(w) for w in best)
        body = [w for w in words if w.y0 >= hdr_bottom - 1 and w.y0 < end_y]
        if ft:
            ce = dict(cols_ext)
            ce["__stt__"] = (min(w.x0 for w in best), max(w.x1 for w in best))
            bounds = refine_bounds(bounds, ce, [w for w in body if id(w) not in sset])
        lines = cluster_rows(body)
        line_of = {}
        for li, r in enumerate(lines):
            for w in r: line_of[id(w)] = li
        ly0 = [min(w.y0 for w in r) for r in lines]; ly1 = [max(w.y1 for w in r) for r in lines]
        sl = [line_of[id(w)] for w in best]
        splits = []
        for i in range(len(best) - 1):
            a_, b_ = sl[i], sl[i + 1]
            if b_ <= a_: splits.append(a_); continue
            gaps = [(ly0[j + 1] - ly1[j], j) for j in range(a_, b_)]
            splits.append(max(gaps)[1])
        for i, srow in enumerate(best):
            start = 0 if i == 0 else splits[i - 1] + 1
            end = len(lines) if i == len(best) - 1 else splits[i] + 1
            cells = {}
            for li in range(start, end):
                for w in lines[li]:
                    if id(w) in sset: continue
                    k = col_of(w, bounds)
                    if k and k != "__stt__": cells.setdefault(k, []).append(w)
            no = int(srow.text)
            if ft: last_ce = ce
            for k in ("quantity", "unit_price", "line_total", "tax_rate"):
                if k in cells and k in last_ce:
                    toks = [w for w in cells[k] if re.search(r"\d", w.text)]
                    h0, h1 = last_ce[k]; hc = (h0 + h1) / 2
                    def dist(w): return 0 if (w.x0 < h1 and w.x1 > h0) else min(abs(w.x0 - h1), abs(w.x1 - h0))
                    toks.sort(key=lambda w: (dist(w), abs(w.cx - hc)))
                    if len(toks) > 1:
                        d0 = dist(toks[0])
                        alts = [t for t in toks[1:] if dist(t) <= d0 * 1.3 + 12 and t.text != toks[0].text]
                        cells[k] = [toks[0]] + alts
                    else:
                        cells[k] = toks
            items[no] = cells
            last_no = no
        info["ok"] = True
    return items, info


# ---------------------------------------------------------------- header (P1)
BUYER = re.compile(r"(nguoi mua|don vi mua|khach hang|buyer|ten don vi|nguoi nhan|customer)")


def row_text(r): return " ".join(w.text for w in r)


def header_fields(words, table_top):
    out = {}
    zone = [w for w in words if w.page == 1 and w.cy < table_top]
    rows = cluster_rows(zone)
    # buyer boundary
    buyer_y = 1e9
    for r in rows:
        if BUYER.search(fold(row_text(r))) and not re.search(r"ban hang|seller", fold(row_text(r))):
            buyer_y = r[0].y0; break
    seller_rows = [r for r in rows if r[0].y0 < buyer_y]

    # --- invoice number
    labels = []
    for w in zone:
        f = _fw(w)
        if not (f.startswith("so") or f in ("no.:", "no:", "no.", "no", "(no.):", "(no.)", "(no)")):
            continue
        same = sorted([v for v in zone if v is not w and abs(v.cy - w.cy) < 0.8 * w.h and v.x0 >= w.x0], key=lambda v: v.x0)
        nf = _fw(same[0]) if same else ""
        n2 = _fw(same[1]) if len(same) > 1 else ""
        if f in ("so", "so:") and (":" in w.text or nf.startswith("(no") or nf in ("no", "no.", "hoa", "hd", "(invoice", "invoice")):
            labels.append(w)
        elif f.startswith("so:") or f.startswith("so(no"):
            labels.append(w)
        elif f in ("no.:", "no:", "(no.):") and ":" in w.text:
            labels.append(w)
    best = None
    for lb in labels:
        for v in zone:
            if v is lb or not re.fullmatch(r"(?:[A-Za-z]{0,3})?\d{1,10}", v.text.strip(":")): continue
            if v.cx < lb.x0: continue
            dy = abs(v.cy - lb.cy)
            if dy > 1.6 * lb.h: continue
            dx = v.x0 - lb.x1
            if dx < -0.3 * lb.h: continue
            if dx > 14 * lb.h and dy > 0.6 * lb.h: continue
            sc = dx + 3 * dy
            if best is None or sc < best[0]: best = (sc, v)
    if best: out["invoice_number"] = (best[1].text.strip(":"), [best[1]])

    # --- date
    for r in rows:
        t = " ".join(w.text for w in r)
        ft = fold(t)
        m = sm.DATE_WORDS.search(ft)
        iso = None
        if m: iso = (m.group(3), m.group(2), m.group(1))
        else:
            m = sm.DATE_DMY.search(ft)
            if m: iso = (m.group(3), m.group(2), m.group(1))
            else:
                m = sm.DATE_YMD.search(ft)
                if m: iso = (m.group(1), m.group(2), m.group(3))
        if iso:
            y, mo, d = iso
            try:
                if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31 and 1990 <= int(y) <= 2100:
                    out["invoice_date"] = (f"{int(d):02d}/{int(mo):02d}/{y}", r); break
            except ValueError: pass

    # --- seller tax code
    def mst_with_suffix(r, w):
        txt = w.text
        i = r.index(w)
        if re.fullmatch(r"\d{10}", re.sub(r"\D", "", txt)) and len(re.sub(r"\D", "", txt)) == 10 and i + 1 < len(r):
            nx = r[i + 1]
            if nx.x0 - w.x1 < 0.9 * w.h and re.fullmatch(r"-?\d{3}", nx.text):
                txt = txt + ("" if nx.text.startswith("-") else "-") + nx.text
        return txt
    for r in seller_rows:
        ft = fold(row_text(r))
        if re.search(r"(thue|mst|tax)", ft):
            for w in r:
                d = re.sub(r"\D", "", w.text)
                if len(d) in (10, 13) and re.fullmatch(r"[\d\-\s.]+", w.text):
                    out["seller_tax_code"] = (mst_with_suffix(r, w), [w]); break
            if "seller_tax_code" in out: break
    if "seller_tax_code" not in out:
        for r in seller_rows:
            for w in r:
                if re.fullmatch(r"\d{10}(-?\d{3})?", w.text):
                    out["seller_tax_code"] = (mst_with_suffix(r, w), [w]); break
            if "seller_tax_code" in out: break

    # --- seller name
    STOPC = re.compile(r"(dia chi|ma so thue|dien thoai|so tai khoan|mst|address|tel\b|fax|website|email|hinh thuc|\(tax)")
    LABW = {"ten", "don", "vi", "ban", "hang", "nguoi", "seller", "name", "of", "company", "(seller)", "(name", "(company"}

    def continuation(idx, x0, h):
        extra = []
        for r2 in seller_rows[idx + 1: idx + 3]:
            t2 = fold(row_text(r2))
            if STOPC.search(t2) or ":" in t2 or abs(r2[0].x0 - x0) > 1.2 * h: break
            extra.extend(r2)
        return extra

    name = None
    for idx, r in enumerate(seller_rows):
        ft = fold(row_text(r))
        if re.search(r"(don vi ban hang|ten don vi ban|nguoi ban hang|nguoi ban|\bseller\b)", ft):
            i = 0
            while i < len(r) and (re.sub(r"[():.\-]", "", fold(r[i].text)).strip() in LABW or r[i].text.strip() in (":", "-", ".")):
                i += 1
            ws = r[i:]
            if ws and len(" ".join(w.text for w in ws)) > 5:
                ws = ws + continuation(idx, ws[0].x0, ws[0].h)
                name = (" ".join(w.text for w in ws).strip(" :-."), ws); break
    if not name:
        kw = re.compile(r"(cong ty|chi nhanh|doanh nghiep|cua hang|hop tac xa|tnhh|co phan|trung tam|ngan hang|vien|truong)")
        mst_y = out["seller_tax_code"][1][0].y0 if "seller_tax_code" in out else 1e9
        for idx, r in enumerate(seller_rows):
            if r[0].y0 >= mst_y: break
            ft = fold(row_text(r))
            if kw.search(ft) and not re.search(r"(hoa don|invoice|ky hieu|ma cua|so \()", ft):
                ws = r + continuation(idx, r[0].x0, r[0].h)
                name = (" ".join(w.text for w in ws).strip(" :-"), ws); break
    if name: out["seller_name"] = name
    return out


def total_candidates(words, ystart):
    cands = []
    for r in cluster_rows([w for w in words if w.cy > ystart]):
        ft = fold(row_text(r))
        ft = re.sub(r"hinh thuc thanh toan", "", ft)
        amts = [w for w in r if is_amount(w) and num(w.text) is not None and num(w.text) >= 1000]
        if not amts or "bang chu" in ft: continue
        if re.search(r"(tien thanh toan|tong thanh toan|total payment|tong cong thanh toan|tong gia tri thanh toan)", ft):
            cands.append((3, amts[-1], r))
        elif re.search(r"cong tien hang", ft) and len(amts) >= 3:
            cands.append((2, amts[-1], r))
        elif re.search(r"tong cong|tong tien|total", ft):
            cands.append((1, amts[-1], r))
    return cands


# ---------------------------------------------------------------- P4 constraints
def row_consistent(q, p, t, tax, tol=Decimal("2")):
    if q is None or p is None or t is None: return None
    if abs(q * p - t) <= max(tol, t * Decimal("0.0005")): return True
    if tax is not None and abs(q * p * (1 + tax / 100) - t) <= max(tol, t * Decimal("0.002")): return True
    return False


# ---------------------------------------------------------------- main
CFG = {"ltotal_pick": "right", "use_p4": True}


def extract_fields_v2(ocr_text, ocr_pages=None):
    if not ocr_pages: return []
    pages_words = [merge_spaced_digits(to_words(p)) for p in ocr_pages]
    pages_words = [pw for pw in pages_words if pw]
    if not pages_words: return []
    preds = []

    def add(field, line, value, src, words, status="ok", reason=""):
        if value in (None, ""): return
        w0 = words[0] if words else None
        preds.append({"field": field, "line": line, "predicted_value": str(value), "source_ocr_line": src,
                      "assignment_status": status, "assignment_reason": reason, "source_page": getattr(w0, "page", ""),
                      "bbox": [w0.x0, w0.y0, w0.x1, w0.y1] if w0 else []})

    items, info = parse_table(pages_words, CFG["ltotal_pick"])
    first = pages_words[0]
    ft = find_table(first)
    table_top = ft[0].y0 - 1 if ft else 0.62 * max(w.y1 for w in first)
    hdr = header_fields(first, table_top)
    for k, (txt, ws) in hdr.items():
        add(k, "", txt, txt, ws if isinstance(ws, list) else ws)
    # total (chosen after items are parsed, using arithmetic when possible)
    allw = [w for pw in pages_words for w in pw]
    # items
    parsed_rows = {}
    for no, cells in items.items():
        row = {}
        for k, ws in cells.items():
            ws = sorted(ws, key=lambda w: (round(w.cy / (w.h * 0.6)), w.x0))
            if k in ("quantity", "unit_price", "line_total", "tax_rate"):
                cand = [w for w in ws if re.search(r"\d", w.text)]
                row[k] = cand
            else:
                row[k] = ws
        parsed_rows[no] = row
    for no, row in parsed_rows.items():
        q = p = t = tax = None
        pick = {}
        for k in ("quantity", "unit_price", "line_total", "tax_rate"):
            c = row.get(k, [])
            if len(c) == 1: pick[k] = c[0]
            elif len(c) > 1: pick[k] = c  # ambiguous list
        # P4: resolve multi-number cells by arithmetic if possible
        def val(k):
            x = pick.get(k)
            return num(x.text) if isinstance(x, W) else None
        q, p, t = val("quantity"), val("unit_price"), val("line_total")
        tx = pick.get("tax_rate")
        if CFG["use_p4"]:
            lists = {k: v for k, v in pick.items() if isinstance(v, list)}
            if lists:
                import itertools
                choices = [lists.get(k) or ([pick[k]] if k in pick else [None]) for k in ("quantity", "unit_price", "line_total")]
                sols = []
                for qq, pp, tt in itertools.product(*choices):
                    if qq is None or pp is None or tt is None: continue
                    if row_consistent(num(qq.text), num(pp.text), num(tt.text), None) is True: sols.append((qq, pp, tt))
                if len(sols) == 1:
                    for k, v in zip(("quantity", "unit_price", "line_total"), sols[0]): pick[k] = v
        cons = row_consistent(val("quantity"), val("unit_price"), val("line_total"),
                              num(tx.text.replace("%", "")) if isinstance(tx, W) else None)
        for k in ("quantity", "unit_price", "line_total", "tax_rate"):
            x = pick.get(k)
            if isinstance(x, W):
                add(k, str(no), x.text, x.text, [x], reason=f"consistent={cons}")
            elif isinstance(x, list):
                add(k, str(no), x[0].text, x[0].text, x, status="ambiguous", reason="multiple_numbers")
        for k in ("item_description", "unit"):
            ws = row.get(k, [])
            if ws:
                txt = " ".join(w.text for w in ws)
                add(k, str(no), txt, txt, ws)
    ls = sum((num(w.text) for row in parsed_rows.values() for w in row.get("line_total", [])[:1] if num(w.text) is not None), Decimal(0))
    tot = _pick_total(pages_words, table_top, ls)
    if tot: add("total_amount", "", tot.text, tot.text, [tot])
    return preds


def _pick_total(pages_words, table_top, line_sum):
    cands = []
    for pi, pw in enumerate(pages_words):
        cands += total_candidates(pw, table_top if pi == 0 else 0)
    if not cands: return None
    if line_sum:
        for pr, w, r in sorted(cands, key=lambda c: -c[0]):
            v = num(w.text)
            if v is not None and abs(v - line_sum) <= max(Decimal(2), line_sum * Decimal("0.0005")): return w
    top = max(c[0] for c in cands)
    return [c for c in cands if c[0] == top][-1][1]
