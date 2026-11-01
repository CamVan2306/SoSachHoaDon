"""Compare Tesseract, EasyOCR, and PaddleOCR on the reviewed 40-invoice set.

The primary metric is exact normalized value presence somewhere in the invoice
OCR text. It does not measure field assignment or invoice-line association.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import struct
import tempfile
import time
import unicodedata
import warnings
import zipfile
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


CONFIRMED_STATUS = "Đã kiểm tra PDF"
HEADER_FIELDS = (
    "invoice_number",
    "invoice_date",
    "seller_tax_code",
    "seller_name",
    "total_amount",
)
ITEM_FIELDS = (
    "item_description",
    "unit",
    "quantity",
    "unit_price",
    "tax_rate",
    "line_total",
)
FIELD_LABELS = {
    "invoice_number": "Số hóa đơn",
    "invoice_date": "Ngày hóa đơn",
    "seller_tax_code": "MST bên bán",
    "seller_name": "Tên nhà cung cấp",
    "total_amount": "Tổng tiền hóa đơn",
    "item_description": "Tên mặt hàng",
    "unit": "Đơn vị tính",
    "quantity": "Số lượng",
    "unit_price": "Đơn giá",
    "tax_rate": "Thuế suất",
    "line_total": "Thành tiền dòng",
}
NUMERIC_FIELDS = {"total_amount", "quantity", "unit_price", "tax_rate", "line_total"}
IDENTIFIER_FIELDS = {"invoice_number", "seller_tax_code"}
ENGINES = ("tesseract", "easyocr", "paddleocr")
OCR_ADAPTER_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2


def iter_confirmed_values(dataset: dict):
    """Yield only non-empty values manually confirmed against a source PDF."""
    for invoice in dataset.get("invoices", []):
        for field, label in invoice.get("header", {}).items():
            value = str(label.get("normalized") or label.get("raw") or "").strip()
            if label.get("status") == CONFIRMED_STATUS and value:
                yield {
                    "invoice_id": invoice["id"],
                    "pdf": invoice["pdf"],
                    "field": field,
                    "field_label": FIELD_LABELS.get(field, field),
                    "line": "",
                    "normalized": value,
                }
        for item in invoice.get("items", []):
            for field, label in item.get("fields", {}).items():
                value = str(label.get("normalized") or label.get("raw") or "").strip()
                if label.get("status") == CONFIRMED_STATUS and value:
                    yield {
                        "invoice_id": invoice["id"],
                        "pdf": invoice["pdf"],
                        "field": field,
                        "field_label": FIELD_LABELS.get(field, field),
                        "line": item.get("line", ""),
                        "normalized": value,
                    }


def normalize_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def canonical_date(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def date_candidates(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    found: set[str] = set()
    patterns = (
        (r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})(?!\d)", "dmy"),
        (r"(?<!\d)(\d{4})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{1,2})(?!\d)", "ymd"),
    )
    for pattern, order in patterns:
        for parts in re.findall(pattern, text):
            try:
                if order == "dmy":
                    day, month, year = map(int, parts)
                else:
                    year, month, day = map(int, parts)
                found.add(date(year, month, day).isoformat())
            except ValueError:
                continue
    vietnamese = re.compile(
        r"(?i)ng[aà]y\D{0,32}(\d{1,2})\D{0,32}th[aá]ng\D{0,32}(\d{1,2})\D{0,32}n[aă]m\D{0,32}(\d{4})"
    )
    for day, month, year in vietnamese.findall(text):
        try:
            found.add(date(int(year), int(month), int(day)).isoformat())
        except ValueError:
            continue
    return found


def normalize_number(value: str) -> str:
    """Normalize Vietnamese/English grouping and decimal punctuation to a number."""
    value = unicodedata.normalize("NFKC", str(value or "")).replace("\u00a0", " ").strip()
    value = value.replace("−", "-").replace("%", "")
    value = re.sub(r"[^\d.,\s+-]", "", value)
    value = re.sub(r"\s+", "", value)
    if not value or not re.search(r"\d", value):
        return ""
    sign = "-" if value.startswith("-") else ""
    value = value.lstrip("+-")
    if not value:
        return ""

    separators = [index for index, char in enumerate(value) if char in ".,"]
    decimal_index = None
    if separators:
        last = separators[-1]
        suffix_length = len(value) - last - 1
        # A final one- or two-digit group is a decimal fraction. Three-digit
        # groups are thousands separators, as on Vietnamese invoices.
        if suffix_length in (1, 2, 4):
            decimal_index = last

    if decimal_index is None:
        digits = re.sub(r"\D", "", value)
        numeric = digits or "0"
    else:
        integer = re.sub(r"\D", "", value[:decimal_index]) or "0"
        fraction = re.sub(r"\D", "", value[decimal_index + 1 :])
        numeric = integer + ("." + fraction if fraction else "")
    try:
        number = Decimal(sign + numeric)
    except InvalidOperation:
        return ""
    normalized = format(number.normalize(), "f")
    return "0" if normalized in ("-0", "") else normalized


def number_candidates(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", str(text or "")).replace("\u00a0", " ")
    # Whitespace often separates OCR cells (for example quantity ``2`` and
    # price ``445,454.55``). Treating it as a thousands separator can merge
    # those cells into ``2445454.55`` and create both misses and false hits.
    pattern = r"(?<!\w)-?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d{1,4})?(?:\s*(?:đ|vnd))?%?(?!\w)"
    return {
        normalize_number(token)
        for token in re.findall(pattern, text, flags=re.IGNORECASE)
        if normalize_number(token)
    }


def identifier_candidates(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    pattern = r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:\s*[-./]\s*[A-Za-z0-9]+)*(?![A-Za-z0-9])"
    return {
        re.sub(r"[^A-Za-z0-9]", "", token).casefold()
        for token in re.findall(pattern, text)
    }


def value_found(field: str, reference: str, text: str) -> bool:
    if not str(reference or "").strip():
        return False
    if field == "invoice_date":
        expected = canonical_date(reference)
        return bool(expected) and expected in date_candidates(text)
    if field in NUMERIC_FIELDS:
        expected = normalize_number(reference)
        return bool(expected) and expected in number_candidates(text)
    if field in IDENTIFIER_FIELDS:
        expected = re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKC", reference)).casefold()
        return bool(expected) and expected in identifier_candidates(text)
    expected = normalize_phrase(reference)
    actual = normalize_phrase(text)
    return bool(expected) and expected in actual


def _space_group_number_candidates(text: str) -> set[str]:
    """Read numbers whose thousands groups are separated by whitespace."""
    text = unicodedata.normalize("NFKC", str(text or "")).replace("\u00a0", " ")
    pattern = r"(?<!\w)-?\d{1,3}(?:\s+\d{3})+(?:[.,]\d{1,4})?(?!\w)"
    return {
        normalize_number(token)
        for token in re.findall(pattern, text)
        if normalize_number(token)
    }


def value_found_in_regions(field: str, reference: str, pages: list[dict]) -> bool:
    """Match OCR values, using box proximity to resolve space-grouped numbers.

    Plain OCR text treats whitespace as a token boundary. When boxes are
    available, groups such as ``1`` + ``234`` + ``567,89`` may be reconstructed
    only when the numeric tokens are on one line and physically adjacent.
    """
    if field not in NUMERIC_FIELDS:
        text = "\n".join(
            page.get("ocr_text") or regions_to_text(page.get("regions", []))
            for page in pages
        )
        return value_found(field, reference, text)

    expected = normalize_number(reference)
    if not expected:
        return False

    for page in pages:
        regions = [
            region for region in page.get("regions", [])
            if str(region.get("text", "")).strip()
        ]
        page_text = page.get("ocr_text") or regions_to_text(regions)
        if value_found(field, reference, page_text):
            return True

        for region in regions:
            if expected in _space_group_number_candidates(region.get("text", "")):
                return True

        numeric_regions = []
        for region in regions:
            text = unicodedata.normalize("NFKC", str(region.get("text", ""))).strip()
            bbox = region.get("bbox")
            if (not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", text)
                    or not isinstance(bbox, (list, tuple)) or len(bbox) < 4):
                continue
            try:
                x0, y0, x1, y1 = (float(value) for value in bbox[:4])
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            numeric_regions.append({"text": text, "bbox": [x0, y0, x1, y1]})

        lines = []
        for region in sorted(numeric_regions, key=lambda row: (row["bbox"][1], row["bbox"][0])):
            x0, y0, x1, y1 = region["bbox"]
            center_y = (y0 + y1) / 2
            height = max(1.0, y1 - y0)
            choices = [
                (abs(center_y - line["center_y"]), line)
                for line in lines
                if abs(center_y - line["center_y"])
                <= max(4.0, min(height, line["height"]) * 0.6)
            ]
            if choices:
                _, line = min(choices, key=lambda candidate: candidate[0])
                line["regions"].append(region)
            else:
                lines.append({"center_y": center_y, "height": height, "regions": [region]})

        for line in lines:
            ordered = sorted(line["regions"], key=lambda row: row["bbox"][0])
            for start in range(len(ordered)):
                joined = ordered[start]["text"]
                if expected in _space_group_number_candidates(joined):
                    return True
                previous = ordered[start]
                for following in ordered[start + 1:]:
                    gap = following["bbox"][0] - previous["bbox"][2]
                    height = min(previous["bbox"][3] - previous["bbox"][1],
                                 following["bbox"][3] - following["bbox"][1])
                    if gap > max(4.0, height * 0.45):
                        break
                    joined += " " + following["text"]
                    if expected in _space_group_number_candidates(joined):
                        return True
                    previous = following
    return False


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract a benchmark ZIP while rejecting path traversal entries."""
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            target = (dest / item.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in dataset ZIP: {item.filename}")
        archive.extractall(dest)
    json_files = list(dest.rglob("40HD.json"))
    if len(json_files) != 1:
        raise FileNotFoundError("Dataset ZIP must contain exactly one 40HD.json")
    return json_files[0]


def load_dataset(json_path: Path, pdf_root: Path, limit: int | None = None):
    dataset = json.loads(json_path.read_text(encoding="utf-8"))
    invoices = dataset.get("invoices")
    if not isinstance(invoices, list) or not invoices:
        raise ValueError("Ground-truth JSON has no invoices")
    if len(invoices) != 40:
        raise ValueError(f"Expected the 40-invoice dataset; found {len(invoices)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        invoices = invoices[:limit]
    missing = [invoice.get("pdf", "<missing name>") for invoice in invoices
               if not (pdf_root / str(invoice.get("pdf", ""))).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} PDF(s): {', '.join(missing[:5])}")
    selected = {**dataset, "invoices": invoices}
    values = list(iter_confirmed_values(selected))
    if not values:
        raise ValueError("No non-empty PDF-confirmed values are available for scoring")
    return selected, values


def render_pdf(pdf_path: Path, dpi: int, output_dir: Path) -> list[Path]:
    import pypdfium2 as pdfium

    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    paths: list[Path] = []
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil().convert("RGB")
            target = output_dir / f"{pdf_path.stem}_page_{index + 1:03d}.png"
            image.save(target, format="PNG")
            paths.append(target)
            image.close()
            bitmap.close()
            page.close()
    finally:
        document.close()
    if not paths:
        raise ValueError(f"PDF has no pages: {pdf_path.name}")
    return paths


def normalize_engine_regions(engine_name: str, raw_result: Any) -> list[dict]:
    """Convert one engine's result into text, confidence, and xyxy boxes."""
    regions: list[dict] = []

    def add(text: Any, confidence: Any, box: Any) -> None:
        value = str(text or "").strip()
        if not value:
            return
        if hasattr(box, "tolist"):
            box = box.tolist()
        points = list(box or [])
        if not points:
            return
        if isinstance(points[0], (list, tuple)) or hasattr(points[0], "tolist"):
            xy = [point.tolist() if hasattr(point, "tolist") else point for point in points]
            xs = [float(point[0]) for point in xy]
            ys = [float(point[1]) for point in xy]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        else:
            if len(points) < 4:
                return
            x0, y0, x1, y1 = (float(item) for item in points[:4])
        try:
            score = float(confidence)
            if score < 0 or not math.isfinite(score):
                score = None
        except (TypeError, ValueError):
            score = None
        regions.append({
            "text": value,
            "confidence": score,
            "bbox": [x0, y0, x1, y1],
        })

    if engine_name == "tesseract":
        texts = raw_result.get("text", [])
        for index, text in enumerate(texts):
            try:
                box = [
                    raw_result["left"][index], raw_result["top"][index],
                    float(raw_result["left"][index]) + float(raw_result["width"][index]),
                    float(raw_result["top"][index]) + float(raw_result["height"][index]),
                ]
                confidence = raw_result.get("conf", [None] * len(texts))[index]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            add(text, confidence, box)
    elif engine_name == "easyocr":
        for item in raw_result or []:
            try:
                box, text, confidence = item
            except (TypeError, ValueError):
                continue
            add(text, confidence, box)
    elif engine_name == "paddleocr":
        texts = raw_result.get("rec_texts", [])
        boxes = raw_result.get("rec_boxes", [])
        scores = raw_result.get("rec_scores", [None] * len(texts))
        for index, text in enumerate(texts):
            try:
                add(text, scores[index], boxes[index])
            except (IndexError, TypeError):
                continue
    else:
        raise ValueError(f"Unknown OCR engine: {engine_name}")

    return sorted(regions, key=lambda row: (row["bbox"][1], row["bbox"][0]))


def regions_to_text(regions: list[dict]) -> str:
    """Rebuild readable lines from OCR boxes ordered by their page geometry."""
    lines: list[dict] = []
    for region in sorted(regions, key=lambda row: (row["bbox"][1], row["bbox"][0])):
        x0, y0, x1, y1 = region["bbox"]
        center_y = (y0 + y1) / 2
        height = max(1.0, y1 - y0)
        candidates = [
            (abs(center_y - line["center_y"]), line)
            for line in lines
            if abs(center_y - line["center_y"]) <= max(4.0, min(height, line["height"]) * 0.6)
        ]
        if candidates:
            _, line = min(candidates, key=lambda candidate: candidate[0])
            line["regions"].append(region)
            centers = [(item["bbox"][1] + item["bbox"][3]) / 2 for item in line["regions"]]
            heights = [max(1.0, item["bbox"][3] - item["bbox"][1]) for item in line["regions"]]
            line["center_y"] = sum(centers) / len(centers)
            line["height"] = sorted(heights)[len(heights) // 2]
        else:
            lines.append({"center_y": center_y, "height": height, "regions": [region]})

    text_lines = []
    for line in lines:
        ordered = sorted(line["regions"], key=lambda row: row["bbox"][0])
        text_lines.append(" ".join(region["text"].strip() for region in ordered if region["text"].strip()))
    return "\n".join(line for line in text_lines if line)


def resolve_ocr_devices(requested: str, *, easyocr_gpu_available: bool,
                        paddle_gpu_available: bool) -> dict[str, str]:
    if requested not in {"cpu", "auto", "gpu"}:
        raise ValueError("device must be one of: cpu, auto, gpu")
    available = {"easyocr": easyocr_gpu_available, "paddleocr": paddle_gpu_available}
    if requested == "gpu" and not all(available.values()):
        raise RuntimeError("GPU mode requires CUDA-enabled EasyOCR and PaddleOCR")
    return {
        "tesseract": "cpu",
        "easyocr": "cuda" if requested != "cpu" and easyocr_gpu_available else "cpu",
        "paddleocr": "gpu:0" if requested != "cpu" and paddle_gpu_available else "cpu",
    }


def _easyocr_gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _paddle_gpu_available() -> bool:
    try:
        import paddle
        return bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


def _lazy_engine(factory, engine_name: str, selected_device: str, allow_gpu_fallback: bool):
    state: dict[str, Any] = {"recognizer": None, "device": selected_device}

    def clear_gpu_cache() -> None:
        try:
            if engine_name == "easyocr":
                import torch
                torch.cuda.empty_cache()
            elif engine_name == "paddleocr":
                import paddle
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def recognize(image_path: Path) -> list[dict]:
        if state["recognizer"] is None:
            try:
                state["recognizer"] = factory(state["device"])
            except Exception as exc:
                if not allow_gpu_fallback or state["device"] == "cpu":
                    raise
                clear_gpu_cache()
                warnings.warn(
                    f"{engine_name} GPU initialization failed; falling back to CPU: {exc}",
                    RuntimeWarning,
                )
                state["device"] = "cpu"
                state["recognizer"] = factory("cpu")
            recognize.device = state["device"]
        return state["recognizer"](image_path)

    def release() -> None:
        state["recognizer"] = None
        clear_gpu_cache()

    recognize.device = selected_device
    recognize.release = release
    return recognize


def load_engines(tesseract_lang: str, tesseract_cmd: str | None = None,
                 device: str = "cpu"):
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Missing pytesseract; install requirements_40hd.txt first") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    languages = set(pytesseract.get_languages(config=""))
    missing = set(tesseract_lang.split("+")) - languages
    if missing:
        raise RuntimeError(f"Tesseract language data missing: {sorted(missing)}")

    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("Missing EasyOCR; install requirements_40hd.txt first") from exc

    try:
        import paddle
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("Missing PaddleOCR/PaddlePaddle; install requirements_40hd.txt first") from exc

    devices = resolve_ocr_devices(
        device,
        easyocr_gpu_available=_easyocr_gpu_available(),
        paddle_gpu_available=_paddle_gpu_available(),
    )

    def build_easy(selected_device: str):
        reader = easyocr.Reader(["vi", "en"], gpu=selected_device == "cuda", verbose=False)

        def recognize(image_path: Path) -> list[dict]:
            blocks = reader.readtext(str(image_path), detail=1, paragraph=False)
            return normalize_engine_regions("easyocr", blocks)

        return recognize

    def build_paddle(selected_device: str):
        reader = PaddleOCR(
            lang="vi",
            device=selected_device,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        def recognize(image_path: Path) -> list[dict]:
            regions: list[dict] = []
            for result in reader.predict(str(image_path)):
                regions.extend(normalize_engine_regions("paddleocr", result))
            return sorted(regions, key=lambda row: (row["bbox"][1], row["bbox"][0]))

        return recognize

    def tesseract(image_path: Path) -> list[dict]:
        raw = pytesseract.image_to_data(
            str(image_path), lang=tesseract_lang, config="--oem 1 --psm 3",
            output_type=pytesseract.Output.DICT,
        )
        return normalize_engine_regions("tesseract", raw)

    tesseract.device = "cpu"
    tesseract.release = lambda: None
    return {
        "tesseract": tesseract,
        "easyocr": _lazy_engine(build_easy, "easyocr", devices["easyocr"], device == "auto"),
        "paddleocr": _lazy_engine(build_paddle, "paddleocr", devices["paddleocr"], device == "auto"),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def tesseract_runtime_version(tesseract_cmd: str | None = None) -> str:
    try:
        import pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return str(pytesseract.get_tesseract_version()).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    with Path(image_path).open("rb") as image:
        header = image.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Rendered page is not a valid PNG image: {image_path}")
    return struct.unpack(">II", header[16:24])


def _checkpoint_file(output_dir: Path, engine: str, invoice_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", invoice_id)
    return output_dir / "checkpoints" / engine / f"{safe_id}.json"


def _write_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    envelope = {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "record": record,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")


def _read_checkpoint(path: Path) -> dict | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        record = envelope["record"]
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if envelope.get("sha256") != hashlib.sha256(payload.encode("utf-8")).hexdigest():
            return None
        return record
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_fingerprint(engine_name: str, tesseract_lang: str,
                        tesseract_cmd: str | None, device: str = "cpu") -> str:
    package_names = {
        "tesseract": ("pytesseract",),
        "easyocr": ("easyocr", "torch", "torchvision"),
        "paddleocr": ("paddleocr", "paddlepaddle"),
    }
    settings = {
        "tesseract": {
            "binary": tesseract_cmd or shutil.which("tesseract") or "tesseract",
            "runtime": tesseract_runtime_version(tesseract_cmd),
            "languages": tesseract_lang,
            "config": "--oem 1 --psm 3",
        },
        "easyocr": {"languages": ["vi", "en"], "device": device,
                    "detail": 1, "paragraph": False, "verbose": False},
        "paddleocr": {"language": "vi", "device": device, "enable_mkldnn": False,
                      "use_doc_orientation_classify": False,
                      "use_doc_unwarping": False, "use_textline_orientation": False},
    }
    payload = {
        "engine": engine_name,
        "adapter_version": OCR_ADAPTER_VERSION,
        "packages": {name: package_version(name) for name in package_names[engine_name]},
        "pdf_renderer": {"name": "pypdfium2", "version": package_version("pypdfium2")},
        "settings": settings[engine_name],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_benchmark(dataset: dict, values: list[dict], pdf_root: Path, output_dir: Path,
                  dpi: int, tesseract_lang: str, tesseract_cmd: str | None = None,
                  resume: bool = False, device: str = "cpu") -> Path:
    engines = load_engines(tesseract_lang, tesseract_cmd, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    invoice_values = defaultdict(list)
    for value in values:
        invoice_values[value["invoice_id"]].append(value)

    prediction_rows: list[dict] = []
    score_rows: list[dict] = []
    invoice_times: dict[tuple[str, str], float] = defaultdict(float)
    page_counts: dict[str, int] = {}
    result_by_pair: dict[tuple[str, str], dict] = {}
    texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    region_records: list[dict] = []
    engine_devices: dict[str, str] = {}
    pdf_hashes = {
        invoice["pdf"]: _file_sha256(Path(pdf_root) / invoice["pdf"])
        for invoice in dataset["invoices"]
    }
    engine_fingerprints: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="40hd-ocr-render-") as temp_name:
        render_root = Path(temp_name)
        rendered: dict[str, list[Path]] = {}
        for invoice in dataset["invoices"]:
            pdf_path = pdf_root / invoice["pdf"]
            rendered[invoice["pdf"]] = render_pdf(
                pdf_path, dpi, render_root / Path(invoice["pdf"]).stem
            )
            page_counts[invoice["pdf"]] = len(rendered[invoice["pdf"]])

        first_image = rendered[dataset["invoices"][0]["pdf"]][0]
        for engine_name, engine in engines.items():
            try:
                engine(first_image)  # warm-up; not included in inference timings
                engine_device = getattr(engine, "device", "cpu")
                engine_devices[engine_name] = engine_device
                engine_fingerprints[engine_name] = _engine_fingerprint(
                    engine_name, tesseract_lang, tesseract_cmd, engine_device
                )

                for invoice in dataset["invoices"]:
                    filename = invoice["pdf"]
                    checkpoint = _checkpoint_file(output_dir, engine_name, invoice["id"])
                    cached = None
                    if resume and checkpoint.is_file():
                        candidate = _read_checkpoint(checkpoint)
                        if (candidate and candidate.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
                                and candidate.get("engine") == engine_name
                                and candidate.get("invoice_id") == invoice["id"]
                                and candidate.get("pdf") == filename
                                and candidate.get("dpi") == dpi
                                and candidate.get("tesseract_lang") == tesseract_lang
                                and candidate.get("source_pdf_sha256") == pdf_hashes[filename]
                                and candidate.get("engine_fingerprint") == engine_fingerprints[engine_name]
                                and len(candidate.get("pages", [])) == len(rendered[filename])):
                            cached = candidate

                    if cached is None:
                        pages: list[dict] = []
                        elapsed_total = 0.0
                        for page_number, image_path in enumerate(rendered[filename], start=1):
                            started = time.perf_counter()
                            regions = engine(image_path)
                            elapsed_total += time.perf_counter() - started
                            width, height = _image_dimensions(image_path)
                            page_text = regions_to_text(regions)
                            pages.append({
                                "engine": engine_name, "invoice_id": invoice["id"],
                                "pdf": filename, "page": page_number,
                                "width": width, "height": height, "regions": regions,
                                "ocr_text": page_text,
                            })
                        result = {
                            "schema_version": CHECKPOINT_SCHEMA_VERSION, "engine": engine_name,
                            "invoice_id": invoice["id"], "pdf": filename,
                            "dpi": dpi, "tesseract_lang": tesseract_lang,
                            "source_pdf_sha256": pdf_hashes[filename],
                            "engine_fingerprint": engine_fingerprints[engine_name],
                            "inference_seconds": round(elapsed_total, 4),
                            "pages": pages,
                            "ocr_text": "\n\n".join(page["ocr_text"] for page in pages),
                        }
                        _write_checkpoint(checkpoint, result)
                    else:
                        result = cached
                    result_by_pair[(engine_name, filename)] = result
                    invoice_times[(engine_name, filename)] = float(result["inference_seconds"])
                    texts[(engine_name, filename)] = [page["ocr_text"] for page in result["pages"]]
            finally:
                release = getattr(engine, "release", None)
                if callable(release):
                    release()

    for engine_name in ENGINES:
        for invoice in dataset["invoices"]:
            result = result_by_pair[(engine_name, invoice["pdf"])]
            prediction_rows.append({
                "engine": engine_name, "invoice_id": invoice["id"],
                "device": engine_devices.get(engine_name, "cpu"),
                "pdf": invoice["pdf"], "pages": len(result["pages"]),
                "inference_seconds": result["inference_seconds"],
                "ocr_text": result["ocr_text"],
            })
            region_records.extend(result["pages"])

    with (output_dir / "ocr_regions.jsonl").open("w", encoding="utf-8") as stream:
        for page in region_records:
            stream.write(json.dumps(page, ensure_ascii=False) + "\n")

    for engine_name in ENGINES:
        for value in values:
            invoice_text = "\n".join(texts[(engine_name, value["pdf"])])
            pages = result_by_pair[(engine_name, value["pdf"])].get("pages", [])
            found = (
                value_found(value["field"], value["normalized"], invoice_text)
                or value_found_in_regions(value["field"], value["normalized"], pages)
            )
            score_rows.append({
                "engine": engine_name,
                **value,
                "exact_value_found": int(found),
            })

    field_summary: list[dict] = []
    for engine_name in ENGINES:
        for field in FIELD_LABELS:
            subset = [row for row in score_rows if row["engine"] == engine_name and row["field"] == field]
            hits = sum(row["exact_value_found"] for row in subset)
            if subset:
                field_summary.append({
                    "engine": engine_name,
                    "field": field,
                    "field_label": FIELD_LABELS[field],
                    "confirmed_values": len(subset),
                    "values_found": hits,
                    "value_presence_rate": round(hits / len(subset), 6),
                })

    invoice_summary: list[dict] = []
    for engine_name in ENGINES:
        for invoice in dataset["invoices"]:
            subset = [row for row in score_rows
                      if row["engine"] == engine_name and row["invoice_id"] == invoice["id"]]
            hits = sum(row["exact_value_found"] for row in subset)
            invoice_summary.append({
                "engine": engine_name,
                "device": engine_devices.get(engine_name, "cpu"),
                "invoice_id": invoice["id"],
                "pdf": invoice["pdf"],
                "confirmed_values": len(subset),
                "values_found": hits,
                "value_presence_rate": round(hits / len(subset), 6) if subset else 0,
                "inference_seconds": round(invoice_times[(engine_name, invoice["pdf"])], 4),
            })

    summary: list[dict] = []
    for engine_name in ENGINES:
        subset = [row for row in score_rows if row["engine"] == engine_name]
        total_time = sum(invoice_times[(engine_name, invoice["pdf"])] for invoice in dataset["invoices"])
        total_pages = sum(page_counts.values())
        hits = sum(row["exact_value_found"] for row in subset)
        invoice_rates = [row["value_presence_rate"] for row in invoice_summary if row["engine"] == engine_name]
        fully_found = sum(rate == 1 for rate in invoice_rates)
        summary.append({
            "engine": engine_name,
            "device": engine_devices.get(engine_name, "cpu"),
            "invoices": len(dataset["invoices"]),
            "pages": total_pages,
            "confirmed_values": len(subset),
            "values_found": hits,
            "overall_value_presence_rate": round(hits / len(subset), 6) if subset else 0,
            "invoices_with_all_confirmed_values_found": fully_found,
            "all_values_found_invoice_rate": round(fully_found / len(dataset["invoices"]), 6),
            "inference_seconds_total": round(total_time, 3),
            "inference_seconds_per_invoice": round(total_time / len(dataset["invoices"]), 3),
            "inference_seconds_per_page": round(total_time / total_pages, 3),
        })

    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "field_summary.csv", field_summary)
    write_csv(output_dir / "invoice_summary.csv", invoice_summary)
    write_csv(output_dir / "field_scores.csv", score_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)

    status_counts = defaultdict(int)
    for invoice in dataset["invoices"]:
        for label in list(invoice.get("header", {}).values()) + [
            field for item in invoice.get("items", []) for field in item.get("fields", {}).values()
        ]:
            status_counts[label.get("status", "")] += 1
    metadata = {
        "dataset": "40HD",
        "invoice_count": len(dataset["invoices"]),
        "positive_confirmed_values_scored": len(values),
        "requested_device": device,
        "engine_devices": engine_devices,
        "status_counts": dict(status_counts),
        "engines": {
            "tesseract": {"package": package_version("pytesseract"),
                          "runtime_version": tesseract_runtime_version(tesseract_cmd),
                          "language": tesseract_lang,
                          "binary": tesseract_cmd or shutil.which("tesseract") or "tesseract"},
            "easyocr": {"package": package_version("easyocr"), "languages": ["vi", "en"],
                        "device": engine_devices.get("easyocr", "cpu")},
            "paddleocr": {"package": package_version("paddleocr"), "language": "vi",
                          "device": engine_devices.get("paddleocr", "cpu")},
            "paddlepaddle": {"package": package_version("paddlepaddle-gpu")
                             if package_version("paddlepaddle-gpu") != "not-installed"
                             else package_version("paddlepaddle")},
        },
        "rendering": {"dpi": dpi, "library": "pypdfium2", "library_version": package_version("pypdfium2")},
        "ocr_region_schema": {"version": 1, "bbox": "pixel xyxy relative to the rendered page"},
        "score_definition": "Exact normalized value appears somewhere in full-page OCR text; field assignment is evaluated separately.",
        "field_value_counts": {field: sum(row["field"] == field for row in values) for field in FIELD_LABELS},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Kết quả so sánh OCR — bộ 40 hóa đơn",
        "",
        f"- Hóa đơn: {len(dataset['invoices'])}",
        f"- Giá trị chuẩn đã xác nhận và được chấm: {len(values)}",
        f"- Tiền xử lý: rasterize cùng PDF ở {dpi} DPI bằng pypdfium2.",
        "- Thiết bị OCR: " + "; ".join(
            f"{name}={engine_devices.get(name, 'cpu')}" for name in ENGINES
        ) + ".",
        "- Điểm đo trong phần này: tỷ lệ giá trị chuẩn hóa xuất hiện trong OCR toàn trang; kết quả gán trường và dòng được báo riêng.",
        "- Các nhãn 'Không có trên hóa đơn', 'Không áp dụng', nhãn trống và 'Cần đọc từ PDF' không được tính là giá trị đúng.",
        "",
        "## Tổng hợp",
        "",
        "| Kỹ thuật | Giá trị tìm thấy | Tỷ lệ | Hóa đơn đủ toàn bộ giá trị đã xác nhận | Giây/trang |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['engine']} | {row['values_found']}/{row['confirmed_values']} | "
            f"{row['overall_value_presence_rate']:.2%} | "
            f"{row['invoices_with_all_confirmed_values_found']}/{row['invoices']} | "
            f"{row['inference_seconds_per_page']:.2f} |"
        )
    report += ["", "## Theo trường", "", "| Kỹ thuật | Trường | Tìm thấy | Tỷ lệ |", "|---|---|---:|---:|"]
    for row in field_summary:
        report.append(
            f"| {row['engine']} | {row['field_label']} | {row['values_found']}/{row['confirmed_values']} | "
            f"{row['value_presence_rate']:.2%} |"
        )
    report += [
        "",
        "Kết quả chi tiết từng giá trị nằm trong `field_scores.csv`; văn bản OCR đầy đủ nằm trong `predictions.csv`; hộp OCR mỗi trang nằm trong `ocr_regions.jsonl`; checkpoint theo hóa đơn và engine nằm trong `checkpoints/`.",
    ]
    (output_dir / "BAO_CAO.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output_dir


def default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    return (
        project_root / "1113HoaDonVietNam" / "CSV" / "40HD.json",
        project_root / "Nhan_Dien_Truong_HD" / "pdfs",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, help="Path to 40HD.json")
    parser.add_argument("--pdf-root", type=Path, help="Folder containing the 40 source PDFs")
    parser.add_argument("--dataset-zip", type=Path, help="ZIP containing 40HD.json and pdfs/")
    parser.add_argument("--output-dir", type=Path, default=Path("results/40hd_run"))
    parser.add_argument("--limit", type=int, help="Run only the first N invoices as a smoke test")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Reuse completed engine/invoice checkpoints in the output directory")
    parser.add_argument("--tesseract-lang", default="vie+eng")
    parser.add_argument("--tesseract-cmd", help="Optional path to the Tesseract executable")
    parser.add_argument(
        "--device", choices=("cpu", "auto", "gpu"), default="cpu",
        help="OCR device for EasyOCR/PaddleOCR; Tesseract always uses CPU",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    default_json, default_pdf_root = default_paths()
    temp = tempfile.TemporaryDirectory(prefix="40hd-dataset-") if args.dataset_zip else None
    try:
        if args.dataset_zip:
            json_path = safe_extract(args.dataset_zip, Path(temp.name))
            pdf_root = json_path.parent / "pdfs"
            if not pdf_root.is_dir():
                pdf_root = next((path for path in json_path.parent.rglob("pdfs") if path.is_dir()), None)
            if pdf_root is None:
                raise FileNotFoundError("Dataset ZIP has no pdfs/ folder")
        else:
            json_path = args.dataset_json or default_json
            pdf_root = args.pdf_root or default_pdf_root
        dataset, values = load_dataset(json_path, pdf_root, args.limit)
        out = run_benchmark(
            dataset, values, pdf_root, args.output_dir, args.dpi,
            args.tesseract_lang, args.tesseract_cmd, resume=args.resume,
            device=args.device,
        )
        print(f"Completed: {len(dataset['invoices'])} invoices, {len(values)} confirmed values")
        print(f"Results: {out.resolve()}")
    finally:
        if temp:
            temp.cleanup()


if __name__ == "__main__":
    main()
