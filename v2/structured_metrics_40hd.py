"""Independently extract field/row assignments from three OCR outputs.

The rule-based parser reads OCR text only. Ground truth is supplied afterward
to score each predicted slot; it is never used to choose a predicted value.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

from run_40hd_benchmark import (
    CONFIRMED_STATUS, FIELD_LABELS, HEADER_FIELDS, ITEM_FIELDS,
    NUMERIC_FIELDS, canonical_date, normalize_number, normalize_phrase,
    value_found, value_found_in_regions,
)

ABSENT_STATUS = "Không có trên hóa đơn"
PREDICTION_COLUMNS = ("engine", "invoice_id", "pdf", "field", "field_label", "line",
                      "predicted_value", "source_ocr_line", "assignment_status",
                      "assignment_reason", "source_page", "bbox")
SCORE_COLUMNS = ("engine", "invoice_id", "pdf", "field", "field_label", "line",
                 "reference_status", "expected_value", "predicted_value",
                 "source_ocr_line", "assignment_status", "assignment_reason",
                 "source_page", "bbox", "diagnostic_status", "raw_value_found",
                 "linked_prediction_field", "linked_prediction_line",
                 "outcome", "tp", "fp", "fn")
SUMMARY_COLUMNS = ("engine", "invoices", "tp", "fp", "fn", "precision", "recall",
                   "f1", "avg_ocr_seconds_per_invoice")
FIELD_SUMMARY_COLUMNS = ("engine", "field", "field_label", "tp", "fp", "fn",
                         "precision", "recall", "f1")
INVOICE_SUMMARY_COLUMNS = ("engine", "invoice_id", "pdf", "tp", "fp", "fn",
                           "precision", "recall", "f1", "ocr_seconds")


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").casefold()


LABELS = {
    "invoice_number": re.compile(r"\b(?:so\s+hoa\s+don|so\s*\(\s*no\.?\s*\)|invoice\s*(?:no\.?|number))", re.I),
    "invoice_date": re.compile(r"\bngay\b(?:\s*\(\s*date\s*\))?|\binvoice\s+date\b", re.I),
    "seller_name": re.compile(r"\b(?:ten\s+don\s+vi\s+ban\s+hang|don\s+vi\s+ban\s+hang|nguoi\s+ban(?:\s+hang)?|seller(?:\s+name)?)\b", re.I),
    "seller_tax_code": re.compile(r"\b(?:ma\s+so\s+thue|mst|tax\s+code)\b", re.I),
    "total_amount": re.compile(r"\b(?:tong\s+(?:cong\s+)?(?:so\s+)?tien\s+thanh\s+toan|tong\s+cong\s+thanh\s+toan|total\s+payment)\b", re.I),
}
BUYER_MARKER = re.compile(r"\b(?:nguoi\s+mua|don\s+vi\s+mua|buyer)\b", re.I)
TAX_CODE = re.compile(r"(?<!\d)\d{10}(?:-?\d{3})?(?!\d)")
INVOICE_NUMBER = re.compile(r"(?<![\w])\d{1,10}(?![\w])")
AMOUNT = re.compile(r"(?<!\w)-?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d{1,4})?(?!\w)")
DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})(?!\d)")
DATE_YMD = re.compile(r"(?<!\d)(\d{4})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{1,2})(?!\d)")
DATE_WORDS = re.compile(r"(\d{1,2})\D{0,25}thang\D{0,25}(\d{1,2})\D{0,25}nam\D{0,25}(\d{4})", re.I)

ITEM_HEADERS = {
    "__stt__": ("stt", "so thu tu"),
    "item_description": ("ten hang hoa dich vu", "ten hang hoa", "ten hang", "hang hoa dich vu"),
    "unit": ("don vi tinh", "dvt"),
    "quantity": ("so luong",),
    "unit_price": ("don gia truoc thue", "don gia"),
    "tax_rate": ("thue suat gtgt", "thue suat"),
    "line_total": ("thanh tien sau thue gtgt", "thanh tien"),
}
ITEM_VALUE_FIELDS = tuple(ITEM_HEADERS.keys())
TABLE_END_MARKERS = ("tong cong", "cong tien hang", "tong tien thanh toan", "tong thanh toan")


def _box(region: dict) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(value) for value in region.get("bbox", []))
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 and values[2] > values[0] and values[3] > values[1] else None


def _cluster_regions_by_y(regions: list[dict]) -> list[dict]:
    valid = [(region, _box(region)) for region in regions]
    valid = [(region, box) for region, box in valid if box is not None and str(region.get("text", "")).strip()]
    lines: list[dict] = []
    for region, box in sorted(valid, key=lambda pair: ((pair[1][1] + pair[1][3]) / 2, pair[1][0])):
        center_y = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        possible = []
        for line in lines:
            tolerance = max(4.0, min(height, line["height"]) * 0.7)
            distance = abs(center_y - line["center_y"])
            if distance <= tolerance:
                possible.append((distance, line))
        if possible:
            _, line = min(possible, key=lambda item: item[0])
            line["regions"].append(region)
            centers = []
            heights = []
            for item in line["regions"]:
                item_box = _box(item)
                centers.append((item_box[1] + item_box[3]) / 2)
                heights.append(item_box[3] - item_box[1])
            line["center_y"] = statistics.mean(centers)
            line["height"] = statistics.median(heights)
        else:
            lines.append({"center_y": center_y, "height": height, "regions": [region]})
    return sorted(lines, key=lambda line: line["center_y"])


def _header_centers(regions: list[dict]) -> dict[str, float]:
    tokens = []
    for region in sorted(regions, key=lambda item: _box(item)[0]):
        box = _box(region)
        for token in re.findall(r"[a-z0-9]+", _fold(str(region.get("text", "")))):
            tokens.append((token, box))
    centers = {}
    for field, aliases in ITEM_HEADERS.items():
        matches = []
        for alias in aliases:
            target = alias.split()
            for start in range(len(tokens) - len(target) + 1):
                if [token for token, _ in tokens[start:start + len(target)]] != target:
                    continue
                boxes = [box for _, box in tokens[start:start + len(target)]]
                x0 = min(box[0] for box in boxes)
                x1 = max(box[2] for box in boxes)
                matches.append((len(target), (x0 + x1) / 2))
        if matches:
            longest = max(length for length, _ in matches)
            candidates = sorted({round(center, 1) for length, center in matches if length == longest})
            if len(candidates) == 1:
                centers[field] = candidates[0]
    return centers


def _table_header(lines: list[dict]) -> tuple[int, dict[str, float]] | None:
    candidates = []
    for index, line in enumerate(lines):
        centers = _header_centers(line["regions"])
        data_columns = sum(field in centers for field in ITEM_FIELDS)
        if "item_description" in centers and data_columns >= 3:
            candidates.append((len(centers), -line["center_y"], index, centers))
    if not candidates:
        return None
    _, _, index, centers = max(candidates)
    return index, centers


def _field_for_x(x: float, centers: dict[str, float]) -> str | None:
    ordered = sorted((center, field) for field, center in centers.items())
    if len(ordered) < 2:
        return None
    gaps = [right[0] - left[0] for left, right in zip(ordered, ordered[1:]) if right[0] > left[0]]
    if not gaps:
        return None
    field_order = ("__stt__", *ITEM_FIELDS)
    field_positions = {field: index for index, field in enumerate(field_order)}
    typical_gap = statistics.median(gaps)
    # A missing header creates a wide gap between known columns. Do not let a
    # token near the interpolated missing-column position fall into its nearest
    # neighbor (e.g. unit-price digits becoming part of quantity).
    for (left_x, left_field), (right_x, right_field) in zip(ordered, ordered[1:]):
        left_position = field_positions.get(left_field)
        right_position = field_positions.get(right_field)
        if left_position is None or right_position is None:
            continue
        skipped = abs(right_position - left_position) - 1
        if skipped <= 0:
            continue
        steps = abs(right_position - left_position)
        step_width = (right_x - left_x) / steps
        for offset in range(1, steps):
            missing_center = left_x + (right_x - left_x) * offset / steps
            if abs(x - missing_center) <= max(12.0, step_width * 0.55):
                return None
    tolerance = max(16.0, typical_gap * 0.72)
    distance, field = min((abs(x - center), field) for center, field in ordered)
    return field if distance <= tolerance else None


def _line_cells(regions: list[dict], centers: dict[str, float]) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = defaultdict(list)
    for region in regions:
        box = _box(region)
        if box is None:
            continue
        field = _field_for_x((box[0] + box[2]) / 2, centers)
        if field:
            cells[field].append(region)
    return cells


def _cell_text(regions: list[dict]) -> str:
    ordered = sorted(regions, key=lambda region: (_box(region)[1], _box(region)[0]))
    return " ".join(str(region.get("text", "")).strip() for region in ordered if str(region.get("text", "")).strip())


def _cell_bbox(regions: list[dict]) -> list[float]:
    boxes = [_box(region) for region in regions]
    boxes = [box for box in boxes if box is not None]
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _extract_layout_items(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    active_centers: dict[str, float] | None = None
    active_width = 0
    for page in sorted(pages, key=lambda item: int(item.get("page", 1))):
        lines = _cluster_regions_by_y(page.get("regions", []))
        header = _table_header(lines)
        page_width = int(page.get("width", 0) or 0)
        if header:
            header_index, centers = header
            active_centers = centers
            active_width = page_width
        elif active_centers:
            # Continuation pages commonly omit the table header. Carry forward
            # the last known column layout, scaling it when page raster widths
            # differ (for example, mixed page sizes in one invoice).
            scale = page_width / active_width if page_width and active_width else 1.0
            header_index = -1
            centers = {field: center * scale for field, center in active_centers.items()}
        else:
            continue
        median_height = statistics.median(
            max(1.0, _box(region)[3] - _box(region)[1])
            for line in lines for region in line["regions"] if _box(region)
        )
        candidate_lines = []
        table_lines = []
        table_ended = False
        for line in lines[header_index + 1:]:
            source = _cell_text(line["regions"])
            folded = _fold(source)
            if any(marker in folded for marker in TABLE_END_MARKERS):
                table_ended = True
                break
            table_lines.append(line)
            cells = _line_cells(line["regions"], centers)
            stt_value = ""
            if "__stt__" in cells:
                match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", _cell_text(cells["__stt__"]))
                stt_value = match.group(1) if match else ""
            populated = sum(bool(cells.get(field)) for field in ITEM_FIELDS)
            if stt_value or populated >= 2:
                candidate_lines.append({
                    "page": int(page.get("page", 1)), "center_y": line["center_y"],
                    "cells": cells, "stt": stt_value,
                })

        # Wrapped descriptions often form a text-only baseline just below the
        # row's numeric cells. Attach only to a nearby row on the same page.
        for line in table_lines:
            cells = _line_cells(line["regions"], centers)
            if set(cells) != {"item_description"}:
                continue
            previous = [row for row in candidate_lines if row["center_y"] < line["center_y"]]
            following = [row for row in candidate_lines if row["center_y"] > line["center_y"]]
            choices = []
            if previous:
                choices.append(max(previous, key=lambda row: row["center_y"]))
            if following:
                choices.append(min(following, key=lambda row: row["center_y"]))
            if not choices:
                continue
            nearest = min(choices, key=lambda row: abs(row["center_y"] - line["center_y"]))
            if abs(nearest["center_y"] - line["center_y"]) <= max(16.0, median_height * 2.5):
                nearest["cells"].setdefault("item_description", []).extend(cells["item_description"])

        for row in candidate_lines:
            row["page_width"] = page_width
            row["cells"] = {
                field: sorted(regions, key=lambda region: (_box(region)[1], _box(region)[0]))
                for field, regions in row["cells"].items()
            }
            rows.append(row)
        if table_ended:
            active_centers = None

    predictions = []
    global_line = 0
    previous_stt = None
    ordered_rows = sorted(rows, key=lambda item: (item["page"], item["center_y"]))
    parsed_stt = {}
    for index, row in enumerate(ordered_rows):
        match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", str(row.get("stt", "")))
        if match:
            parsed_stt[index] = int(match.group(1))

    # A row without a readable STT gets a trusted global line only when
    # readable anchors on both sides leave exactly one slot per detected row.
    # A larger STT gap means OCR may have missed one or more complete rows.
    safely_inferred_rows = set()
    for index, row in enumerate(ordered_rows):
        if row.get("stt"):
            continue
        previous_anchor = max((i for i in parsed_stt if i < index), default=None)
        next_anchor = min((i for i in parsed_stt if i > index), default=None)
        if previous_anchor is None or next_anchor is None:
            continue
        previous_number = parsed_stt[previous_anchor]
        next_number = parsed_stt[next_anchor]
        detected_between = next_anchor - previous_anchor - 1
        stt_slots_between = next_number - previous_number - 1
        if stt_slots_between == detected_between and stt_slots_between > 0:
            safely_inferred_rows.add(index)

    for row_index, row in enumerate(ordered_rows):
        try:
            printed_stt = int(row["stt"]) if row["stt"] else None
        except ValueError:
            printed_stt = None
        row_number_confident = (
            printed_stt is not None or row_index in safely_inferred_rows
        )
        if not row_number_confident:
            # Do not publish a guessed line number or let an uncertain row
            # shift the next readable STT anchor.
            line_number = ""
        elif printed_stt is not None and (
            previous_stt is None or printed_stt > previous_stt
        ):
            line_number = max(printed_stt, global_line + 1)
        else:
            line_number = global_line + 1
        if line_number != "":
            global_line = int(line_number)
        if printed_stt is not None:
            previous_stt = printed_stt
        source = _cell_text([region for values in row["cells"].values() for region in values])
        for field in ITEM_FIELDS:
            cell = row["cells"].get(field, [])
            value = _cell_text(cell)
            if not value:
                continue
            predictions.append({
                "field": field, "line": str(line_number), "predicted_value": value,
                "source_ocr_line": source,
                "assignment_status": "assigned" if row_number_confident else "ambiguous",
                "assignment_reason": (
                    "header_columns_and_row_geometry"
                    if row_number_confident
                    else "row_number_not_confident_missing_or_unanchored_rows"
                ),
                "source_page": row["page"], "bbox": _cell_bbox(cell),
            })
    return predictions


def _date_value(value: str) -> str:
    for pattern, order in ((DATE_DMY, "dmy"), (DATE_YMD, "ymd")):
        match = pattern.search(value)
        if match:
            a, b, c = (int(part) for part in match.groups())
            try:
                day, month, year = (a, b, c) if order == "dmy" else (c, b, a)
                return date(year, month, day).isoformat()
            except ValueError:
                pass
    match = DATE_WORDS.search(_fold(value))
    if match:
        try:
            day, month, year = (int(part) for part in match.groups())
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    return ""


def _header_value(field: str, suffix: str) -> str:
    suffix = re.sub(r"^\s*(?:\([^)]{1,40}\)\s*)?[:：–—-]?\s*", "", suffix).strip()
    if field == "invoice_date":
        return _date_value(suffix)
    if field == "invoice_number":
        match = INVOICE_NUMBER.search(suffix)
        return match.group(0) if match else ""
    if field == "seller_tax_code":
        match = TAX_CODE.search(suffix)
        return match.group(0) if match else ""
    if field == "total_amount":
        match = AMOUNT.search(suffix)
        return match.group(0).strip() if match else ""
    if field == "seller_name":
        other = LABELS["seller_tax_code"].search(_fold(suffix))
        if other:
            suffix = suffix[:other.start()].strip()
        suffix = suffix.rstrip(" ;,.")
        return suffix if len(suffix) >= 3 else ""
    return ""


def extract_fields(ocr_text: str, ocr_pages: list[dict] | None = None) -> list[dict]:
    """Extract labeled header values and table cells from text and OCR geometry."""
    lines = [unicodedata.normalize("NFC", line).strip() for line in str(ocr_text or "").splitlines()]
    predictions: dict[tuple[str, str], dict] = {}
    buyer_started = False

    def add(field: str, line: str, value: str, source: str, **metadata) -> None:
        if value:
            predictions.setdefault((field, line), {
                "field": field, "line": line, "predicted_value": value,
                "source_ocr_line": source, "assignment_status": "text_rule",
                "assignment_reason": "labeled_text", "source_page": "", "bbox": [],
                **metadata,
            })

    for index, source in enumerate(lines):
        if not source:
            continue
        folded = _fold(source)
        if BUYER_MARKER.search(folded):
            buyer_started = True
        for field in HEADER_FIELDS:
            if buyer_started and field in ("seller_name", "seller_tax_code"):
                continue
            match = LABELS[field].search(folded)
            if not match or (field, "") in predictions:
                continue
            value = _header_value(field, source[match.end():])
            value_source = source
            if not value and index + 1 < len(lines):
                following = lines[index + 1]
                following_folded = _fold(following)
                if following and not any(rule.search(following_folded) for rule in LABELS.values()):
                    value = _header_value(field, following)
                    value_source = following
            add(field, "", value, value_source)

        # Legacy text outputs can still expose a fully delimited item row.
        # When geometry exists, use the coordinate-based parser below instead.
        if ocr_pages:
            continue
        cells = [cell.strip() for cell in re.split(r"[|│\t]", source)]
        if len(cells) != 7 or not re.fullmatch(r"\d{1,3}", cells[0]):
            continue
        item_line = str(int(cells[0]))
        for field, value in zip(ITEM_FIELDS, cells[1:]):
            add(field, item_line, value, source,
                assignment_status="legacy_text", assignment_reason="explicit_delimited_row")
    if ocr_pages:
        for row in _extract_layout_items(ocr_pages):
            predictions[(row["field"], row["line"])] = row
    return list(predictions.values())


def _same_value(field: str, expected: str, predicted: str) -> bool:
    if field == "invoice_date":
        return bool(canonical_date(expected)) and canonical_date(expected) == canonical_date(predicted)
    if field in NUMERIC_FIELDS:
        return bool(normalize_number(expected)) and normalize_number(expected) == normalize_number(predicted)
    if field in ("invoice_number", "seller_tax_code"):
        clean = lambda value: re.sub(r"[^A-Za-z0-9]", "", value).casefold()
        return bool(clean(expected)) and clean(expected) == clean(predicted)
    return bool(normalize_phrase(expected)) and normalize_phrase(expected) == normalize_phrase(predicted)


def _metric_row(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn"))
    if tp + fp + fn == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": "", "recall": "", "f1": ""}
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _all_slots(dataset: dict) -> list[dict]:
    slots = []
    for invoice in dataset.get("invoices", []):
        for field, label in invoice.get("header", {}).items():
            slots.append({
                "invoice_id": invoice["id"], "pdf": invoice["pdf"], "field": field,
                "field_label": FIELD_LABELS.get(field, field), "line": "",
                "reference_status": label.get("status", ""),
                "expected_value": str(label.get("normalized") or label.get("raw") or "").strip(),
            })
        for item in invoice.get("items", []):
            for field, label in item.get("fields", {}).items():
                slots.append({
                    "invoice_id": invoice["id"], "pdf": invoice["pdf"], "field": field,
                    "field_label": FIELD_LABELS.get(field, field),
                    "line": str(item.get("line", "")),
                    "reference_status": label.get("status", ""),
                    "expected_value": str(label.get("normalized") or label.get("raw") or "").strip(),
                })
    return slots


def _score_slot(slot: dict, prediction: dict | None) -> dict:
    status = slot["reference_status"]
    expected = slot["expected_value"]
    predicted = str((prediction or {}).get("predicted_value", "")).strip()
    assignment_status = str((prediction or {}).get("assignment_status", ""))
    if assignment_status in ("ambiguous", "unassigned"):
        predicted = ""
    source = (prediction or {}).get("source_ocr_line", "")
    tp = fp = fn = 0
    if status == CONFIRMED_STATUS and expected:
        if not predicted:
            outcome, fn = "FN", 1
        elif _same_value(slot["field"], expected, predicted):
            outcome, tp = "TP", 1
        else:
            outcome, fp, fn = "FP+FN", 1, 1
    elif status == ABSENT_STATUS:
        if predicted:
            outcome, fp = "FP", 1
        else:
            outcome = "TN"
    else:
        outcome = "UNSCORED"
    return {
        **slot, "predicted_value": predicted, "source_ocr_line": source,
        "assignment_status": assignment_status,
        "assignment_reason": (prediction or {}).get("assignment_reason", ""),
        "source_page": (prediction or {}).get("source_page", ""),
        "bbox": json.dumps((prediction or {}).get("bbox", []), ensure_ascii=False),
        "outcome": outcome, "tp": tp, "fp": fp, "fn": fn,
    }


def _diagnose_slot(slot: dict, prediction: dict | None, parsed_by_key: dict,
                   raw_ocr_text: str, ambiguous_predictions: list[dict] | None = None,
                   ocr_pages: list[dict] | None = None) -> dict:
    expected = slot["expected_value"]
    field = slot["field"]
    key = (field, str(slot["line"]))
    prediction = prediction or {}
    raw_found = bool(expected and (
        value_found(field, expected, raw_ocr_text)
        or value_found_in_regions(field, expected, ocr_pages or [])
    ))
    linked = []
    if expected:
        for other_key, other in parsed_by_key.items():
            if other_key == key or not str(other.get("predicted_value", "")).strip():
                continue
            if _same_value(field, expected, str(other.get("predicted_value", ""))):
                linked.append(other_key)
    ambiguous_predictions = ambiguous_predictions or []
    ambiguous_matches = [
        row for row in ambiguous_predictions
        if row.get("field") == field
        and _same_value(field, expected, str(row.get("predicted_value", "")))
    ]

    assignment = str(prediction.get("assignment_status", ""))
    predicted = str(prediction.get("predicted_value", "")).strip()
    if slot["reference_status"] != CONFIRMED_STATUS or not expected:
        diagnostic = "unscored"
    elif assignment in ("ambiguous", "unassigned"):
        diagnostic = "ambiguous"
    elif predicted and _same_value(field, expected, predicted):
        diagnostic = "correct"
    elif len(linked) == 1:
        diagnostic = "wrong_assignment"
    elif len(linked) > 1:
        diagnostic = "ambiguous"
    elif ambiguous_matches:
        diagnostic = "ambiguous"
    elif predicted:
        diagnostic = "wrong_value"
    elif raw_found:
        diagnostic = "read_but_unassigned"
    else:
        diagnostic = "not_read"
    return {
        "diagnostic_status": diagnostic,
        "raw_value_found": int(raw_found),
        "linked_prediction_field": linked[0][0] if len(linked) == 1 else "",
        "linked_prediction_line": linked[0][1] if len(linked) == 1 else "",
        "_ambiguous_prediction": ambiguous_matches[0] if len(ambiguous_matches) == 1 else None,
    }


def _raw_value_source(field: str, expected: str, pages: list[dict]) -> dict | None:
    """Locate a raw OCR hit for diagnostics without turning it into a prediction."""
    matches = []
    for page in pages:
        for line in _cluster_regions_by_y(page.get("regions", [])):
            regions = line["regions"]
            text = _cell_text(regions)
            line_page = {"ocr_text": text, "regions": regions}
            if not (value_found(field, expected, text)
                    or value_found_in_regions(field, expected, [line_page])):
                continue
            exact_regions = [
                region for region in regions
                if (value_found(field, expected, str(region.get("text", "")))
                    or value_found_in_regions(field, expected, [{"regions": [region]}]))
            ]
            source_regions = exact_regions or regions
            matches.append({
                "source_page": int(page.get("page", 1)),
                "bbox": _cell_bbox(source_regions),
                "source_ocr_line": text,
            })
    return matches[0] if len(matches) == 1 else None


def evaluate_run(predictions_csv: Path, dataset_json: Path, output_dir: Path,
                 regions_jsonl: Path | None = None) -> Path:
    """Apply a common parser, then score values and field/row assignments.

    Extraction consumes OCR text and optional boxes only. Labels are loaded
    afterward and are used only to score and diagnose the predictions.
    """
    predictions_csv = Path(predictions_csv)
    dataset_json = Path(dataset_json)
    output_dir = Path(output_dir)
    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    with predictions_csv.open(newline="", encoding="utf-8-sig") as stream:
        prediction_rows = list(csv.DictReader(stream))

    pages_by_invoice: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if regions_jsonl and Path(regions_jsonl).is_file():
        with Path(regions_jsonl).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    page = json.loads(line)
                    pages_by_invoice[(page["engine"], page["invoice_id"])].append(page)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_slots = _all_slots(dataset)
    by_invoice = {invoice["id"]: invoice for invoice in dataset.get("invoices", [])}
    scored_rows: list[dict] = []
    extracted_rows: list[dict] = []

    for source in prediction_rows:
        engine = source["engine"]
        invoice_id = source["invoice_id"]
        invoice = by_invoice[invoice_id]
        invoice_slots = [slot for slot in all_slots if slot["invoice_id"] == invoice_id]
        raw_ocr_text = source.get("ocr_text", "")
        parsed = extract_fields(
            raw_ocr_text, pages_by_invoice.get((engine, invoice_id))
        )
        ambiguous_predictions = [
            row for row in parsed
            if row.get("assignment_status") in ("ambiguous", "unassigned")
        ]
        parsed_by_key = {
            (row["field"], str(row["line"])): row
            for row in parsed
            if row.get("assignment_status") not in ("ambiguous", "unassigned")
        }
        for row in parsed:
            extracted_rows.append({
                "engine": engine, "invoice_id": invoice_id, "pdf": source["pdf"],
                "field": row["field"], "field_label": FIELD_LABELS.get(row["field"], row["field"]),
                "line": row["line"], "predicted_value": row["predicted_value"],
                "source_ocr_line": row["source_ocr_line"],
                "assignment_status": row.get("assignment_status", "text_rule"),
                "assignment_reason": row.get("assignment_reason", ""),
                "source_page": row.get("source_page", ""),
                "bbox": json.dumps(row.get("bbox", []), ensure_ascii=False),
            })

        known_keys = {(slot["field"], str(slot["line"])) for slot in invoice_slots}
        for slot in invoice_slots:
            key = (slot["field"], str(slot["line"]))
            prediction = parsed_by_key.get(key)
            diagnostic = _diagnose_slot(
                slot, prediction, parsed_by_key, raw_ocr_text, ambiguous_predictions,
                pages_by_invoice.get((engine, invoice_id), []),
            )
            ambiguous_prediction = diagnostic.pop("_ambiguous_prediction", None)
            scored = {
                "engine": engine, "pdf": source["pdf"],
                **_score_slot(slot, prediction),
                **diagnostic,
            }
            if ambiguous_prediction:
                scored.update({
                    "assignment_status": "ambiguous",
                    "assignment_reason": ambiguous_prediction.get("assignment_reason", ""),
                    "source_page": ambiguous_prediction.get("source_page", ""),
                    "bbox": json.dumps(ambiguous_prediction.get("bbox", []), ensure_ascii=False),
                    "source_ocr_line": ambiguous_prediction.get("source_ocr_line", ""),
                })
            if diagnostic["diagnostic_status"] == "read_but_unassigned":
                scored["assignment_status"] = "unassigned"
                scored["assignment_reason"] = "raw_value_found_but_no_confident_cell"
                raw_source = _raw_value_source(
                    slot["field"], slot["expected_value"],
                    pages_by_invoice.get((engine, invoice_id), []),
                )
                if raw_source:
                    scored.update(raw_source)
                    scored["bbox"] = json.dumps(raw_source["bbox"], ensure_ascii=False)
            scored_rows.append(scored)
        # A parser-produced field/line with no corresponding annotation is an
        # extra prediction and therefore a false positive.
        for key, parsed_slot in parsed_by_key.items():
            if key in known_keys:
                continue
            extra_slot = {
                "invoice_id": invoice_id, "pdf": source["pdf"], "field": key[0],
                "field_label": FIELD_LABELS.get(key[0], key[0]), "line": key[1],
                "reference_status": "Không có nhãn trường", "expected_value": "",
            }
            scored_rows.append({"engine": engine, **extra_slot,
                                "predicted_value": parsed_slot["predicted_value"],
                                "source_ocr_line": parsed_slot["source_ocr_line"],
                                "assignment_status": parsed_slot.get("assignment_status", "text_rule"),
                                "assignment_reason": parsed_slot.get("assignment_reason", ""),
                                "source_page": parsed_slot.get("source_page", ""),
                                "bbox": json.dumps(parsed_slot.get("bbox", []), ensure_ascii=False),
                                "diagnostic_status": "unscored",
                                "raw_value_found": 0,
                                "linked_prediction_field": "",
                                "linked_prediction_line": "",
                                "outcome": "FP", "tp": 0, "fp": 1, "fn": 0})

    _write_csv(output_dir / "structured_field_predictions.csv", PREDICTION_COLUMNS, extracted_rows)
    _write_csv(output_dir / "structured_field_scores.csv", SCORE_COLUMNS, scored_rows)

    time_by_engine_invoice = {
        (row["engine"], row["invoice_id"]): float(row.get("inference_seconds") or 0)
        for row in prediction_rows
    }
    invoice_csv = output_dir / "invoice_summary.csv"
    if invoice_csv.exists():
        with invoice_csv.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                time_by_engine_invoice[(row["engine"], row["invoice_id"])] = float(
                    row.get("inference_seconds") or 0
                )
    engines = list(dict.fromkeys(row["engine"] for row in prediction_rows))
    invoice_summaries = []
    field_summaries = []
    summaries = []
    for engine in engines:
        engine_rows = [row for row in scored_rows if row["engine"] == engine]
        overall = _metric_row(engine_rows)
        for field in FIELD_LABELS:
            rows = [row for row in engine_rows if row["field"] == field]
            if rows:
                metric = _metric_row(rows)
                field_summaries.append({
                    "engine": engine, "field": field, "field_label": FIELD_LABELS[field], **metric,
                })
        for invoice_id, invoice in by_invoice.items():
            rows = [row for row in engine_rows if row["invoice_id"] == invoice_id]
            metric = _metric_row(rows)
            invoice_summaries.append({
                "engine": engine, "invoice_id": invoice_id, "pdf": invoice["pdf"],
                **metric, "ocr_seconds": time_by_engine_invoice.get((engine, invoice_id), 0),
            })
        seconds = [row["ocr_seconds"] for row in invoice_summaries if row["engine"] == engine]
        summaries.append({
            "engine": engine, "invoices": len(by_invoice), **overall,
            "avg_ocr_seconds_per_invoice": sum(seconds) / len(seconds) if seconds else 0,
        })

    _write_csv(output_dir / "structured_summary.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(output_dir / "structured_field_summary.csv", FIELD_SUMMARY_COLUMNS, field_summaries)
    _write_csv(output_dir / "structured_invoice_summary.csv", INVOICE_SUMMARY_COLUMNS, invoice_summaries)
    outcome_counts = {}
    for row in scored_rows:
        outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1
    (output_dir / "structured_evaluation.json").write_text(
        json.dumps({
            "method": "Same deterministic text and optional OCR-box parser applied to each OCR engine; labels are used only after extraction.",
            "wrong_value_rule": "One FP for the wrong prediction and one FN for the missed reference.",
            "unverified_label_rule": "Unverified and not-applicable slots are excluded from TP/FP/FN.",
            "item_row_rule": "Use detected header-column geometry and page/y row order; abstain when assignment is ambiguous.",
            "diagnostic_statuses": ["correct", "wrong_value", "wrong_assignment", "read_but_unassigned", "not_read", "ambiguous", "unscored"],
            "metrics": "Micro-averaged field-slot precision, recall and F1; true negatives excluded.",
            "outcome_counts_across_engines": outcome_counts,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
