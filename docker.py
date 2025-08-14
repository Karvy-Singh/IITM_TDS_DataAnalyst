Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Enterprise
Pricing



Spaces:

KarvySingh
/
DataAnalyst


like
0

Logs
App
Files
Community
Settings
DataAnalyst
/
app.py

KarvySingh's picture
KarvySingh
Update app.py
d0e99a4
verified
less than a minute ago
raw

Copy download link
history
blame
edit
delete

45.2 kB
import os
import re
import json
import asyncio
import tempfile
import sys
import io
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import magic, io, csv, json, pandas as pd, numpy as np
from charset_normalizer import from_bytes
import pdfplumber, xmltodict
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError, Field

# ---- LLM client (OpenAI-compatible) -----------------------------------------
from openai import OpenAI

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

# Defaults tuned for small models; override via env
PLANNER_MODEL    = os.getenv("LLM_PLANNER_MODEL",    "gpt-4o-mini")
CODER_MODEL      = os.getenv("LLM_CODER_MODEL",      "gpt-4o-mini")
FORMATTER_MODEL  = os.getenv("LLM_FORMATTER_MODEL",  "gpt-4o-mini")

if not LLM_API_KEY:
    # /health and / will work; /api will return a clear error.
    pass
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

app = FastAPI(title="Universal Data Analyst (speed-optimized)", version="4.2.0")

# ---- Prompts ----------------------------------------------------------------
# Simplified planner for faster processing
PLANNER_SYSTEM = """You are a planning engine. Output STRICT JSON that validates this schema:
{
  "type": "object",
  "properties": {
    "question": { "type": "string" },
    "parameters": { "type": "object" },
    "steps": { "type": "array", "items": { "type": "string" } },
    "final_variables": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["question", "parameters", "steps", "final_variables"],
  "additionalProperties": false
}
Keep steps minimal (max 3). Return JSON only. No commentary."""

# Streamlined coder prompt for faster generation
CODER_SYSTEM = """You write ONLY STRICT JSON analysis specs. Be concise and efficient.
Schema (must validate exactly):
{
  "type": "object",
  "properties": {
    "inputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type":"string"},
          "source": {"type":"string", "enum": ["html","csv","json","inline"]},
          "url": {"type":["string","null"]},
          "data": {"type":["string","object","array","null"]},
          "table_index": {"type":["integer","null"]}
        },
        "required": ["name","source"],
        "additionalProperties": false
      }
    },
    "transforms": {
      "type": "array",
      "items": {
        "type":"object",
        "properties": {
          "target": {"type":"string"},
          "op": {"type":"string", "enum": [
            "select_columns","rename","dropna","head","sort_values",
            "filter_query","groupby_agg","join","add_column","parse_dates"
          ]},
          "args": {"type":"object"}
        },
        "required": ["target","op","args"],
        "additionalProperties": false
      }
    },
    "charts": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "table": {"type":"string"},
          "kind": {"type":"string","enum":["line","bar","scatter","hist"]},
          "x": {"type":["string","null"]},
          "y": {"type":["string","array","null"]},
          "title": {"type":["string","null"]},
          "bins": {"type":["integer","null"]}
        },
        "required": ["table","kind"],
        "additionalProperties": false
      }
    },
    "answer": {
      "type":"object",
      "properties": {
        "type": {"type":"string","enum":["text_summary","basic_stats","none"]},
        "table": {"type":["string","null"]},
        "columns": {"type":["array","null"], "items":{"type":"string"}}
      },
      "required": ["type"],
      "additionalProperties": false
    },
    "result_table": {"type":"string"}
  },
  "required": ["inputs","transforms","charts","answer","result_table"],
  "additionalProperties": false
}
Minimize transforms. Prefer simple operations. Return JSON only."""

# Simplified formatter for speed — respects requested output format
OUTPUT_FORMATTER_SYSTEM = """
Your task is to produce the final result in the exact format requested by the user.
Rules:
- Read the user request carefully and identify any explicit or implicit output format or schema, no matter what it is called or how it is described.
- Follow that format or schema exactly, including structure, syntax, spacing, and punctuation.
- If the user describes the format in words, deduce and apply it precisely.
- Do not add extra explanations, headers, or commentary outside the requested format.
- If no format is specified at all, default to concise plain text.
- Always return a single string as the final output.
"""


# ---- Helper code (optimized for speed) --------------------------------------
HELPERS = r"""
import json, base64, io, time, math, statistics, re, datetime
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
def fetch_text(url, timeout=8, retries=1):
    last = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/4.0)"}
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            if _ < retries:
                time.sleep(0.5)  # Reduced sleep
    raise last
def fetch_json(url, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/4.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()
def read_table_html(html):
    try:
        dfs = pd.read_html(html)
        return dfs
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        if not tables:
            return []
        out = []
        for t in tables[:3]:  # Limit to first 3 tables for speed
            try:
                out.append(pd.read_html(str(t))[0])
            except Exception:
                pass
        return out
def df_to_records(df):
    return df.replace({pd.NA: None, np.nan: None}).to_dict(orient="records")
def fig_to_data_uri(fig):
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=72)  # Lower DPI for speed
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
"""

# ---- Models / Validators -----------------------------------------------------
class Plan(BaseModel):
    question: str
    parameters: Dict[str, Any]
    steps: List[str]
    final_variables: List[str]

REQUIRED_FINAL_KEYS = {"answer", "tables", "images", "logs"}

class AnalysisInput(BaseModel):
    name: str
    source: str  # "html" | "csv" | "json" | "inline"
    url: Optional[str] = None
    data: Optional[Any] = None
    table_index: Optional[int] = None

class Transform(BaseModel):
    target: str
    op: str
    args: Dict[str, Any]

class ChartSpec(BaseModel):
    table: str
    kind: str
    x: Optional[str] = None
    y: Optional[Any] = None
    title: Optional[str] = None
    bins: Optional[int] = None

class AnswerSpec(BaseModel):
    type: str  # "text_summary" | "basic_stats" | "none"
    table: Optional[str] = None
    columns: Optional[List[str]] = None

class AnalysisSpec(BaseModel):
    inputs: List[AnalysisInput]
    transforms: List[Transform]
    charts: List[ChartSpec]
    answer: AnswerSpec
    result_table: str

# ---- Utilities ---------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_payload(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    payload = m.group(1) if m else text
    return payload.strip()

def coerce_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise

# Optimized LLM call with shorter timeouts and aggressive JSON mode
def llm_call_raw(system_prompt: str, user_prompt: str, model: str, temperature: float = 0, max_tokens: int = 1000) -> str:
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY/LLM_API_KEY is not set.")
    
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,  # Limit tokens for speed
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback without JSON mode
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

# Async LLM call for parallel processing
async def llm_call_async(system_prompt: str, user_prompt: str, model: str, temperature: float = 0, max_tokens: int = 1000) -> str:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor, 
            llm_call_raw, 
            system_prompt, user_prompt, model, temperature, max_tokens
        )

# Fast planning with reduced retries
def plan_question(question: str) -> Plan:
    out = llm_call_raw(PLANNER_SYSTEM, question, PLANNER_MODEL, temperature=0, max_tokens=500)
    try:
        data = coerce_json(extract_payload(out))
        return Plan(**data)
    except Exception:
        # Single retry only
        out2 = llm_call_raw(PLANNER_SYSTEM + "\n\nFAST MODE: Return minimal valid JSON.", question, PLANNER_MODEL, temperature=0, max_tokens=500)
        data2 = coerce_json(extract_payload(out2))
        return Plan(**data2)

# ---- Speed-optimized spec generation ----------------------------------------
def make_coder_prompt(plan: Plan, attachments_summary: str, debug_hint: Optional[str]=None) -> str:
    hint = f"\nDEBUG_HINT: {debug_hint}" if debug_hint else ""
    return (
        f"Question: {plan.question}\n"
        f"Plan: {plan.model_dump_json()}\n"
        f"Attachments available (prefer using them with source='inline' and their table names):\n{attachments_summary}\n"
        "When referencing attachments, set inputs[].source='inline' and inputs[].data to a SMALL sample (we will provide full data at execution time)."
        f"{hint}\n"
        "Return analysis_spec JSON. Be minimal and efficient."
    )

# Single-shot spec generation (no best-of-N for speed)
def generate_analysis_spec(plan: Plan, attachments_summary: str, debug_hint: Optional[str]=None) -> AnalysisSpec:
    raw = llm_call_raw(
        CODER_SYSTEM,
        make_coder_prompt(plan, attachments_summary, debug_hint),
        CODER_MODEL,
        temperature=0,
        max_tokens=800
    )
    
    try:
        data = coerce_json(extract_payload(raw))
        return AnalysisSpec(**data)
    except Exception:
        # Single retry with stricter prompt
        fixer_prompt = f"Fix to valid JSON schema:\n{raw}\n\nReturn only valid JSON."
        fixed = llm_call_raw(CODER_SYSTEM, fixer_prompt, CODER_MODEL, temperature=0, max_tokens=800)
        data2 = coerce_json(extract_payload(fixed))
        return AnalysisSpec(**data2)

# ---- Attachment ingestion & helpers -----------------------------------------
TEXT_EXTS = {".txt", ".md", ".rst"}
TABULAR_EXTS = {".csv", ".tsv", ".json", ".ndjson", ".parquet", ".pq", ".feather", ".xlsx", ".xls"}
HTML_EXTS = {".html", ".htm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

def _is_probably_text(content: bytes) -> bool:
    if not content:
        return False
    # Heuristic: if >90% printable (incl. whitespace), treat as text
    try:
        decoded = content.decode("utf-8", errors="ignore")
    except Exception:
        return False
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in decoded)
    return (printable / max(1, len(decoded))) > 0.9

def _sanitize_name(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_")
    return cleaned or "table"

class AttachmentPreview(BaseModel):
    name: str
    table_name: Optional[str] = None
    kind: str  # 'csv','json','parquet','excel','html','image','text','unknown'
    sample_text: Optional[str] = None
    sample_records: Optional[List[Dict[str, Any]]] = None
    shape: Optional[Tuple[int, int]] = None
    columns: Optional[List[str]] = None

# def _read_tabular_to_df(name: str, data: bytes, content_type: Optional[str]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[Tuple[int,int]], Optional[List[str]]]:
#     import pandas as pd
#     ext = Path(name).suffix.lower()
#     buf = io.BytesIO(data)
#     df = None
#     if ext in {".csv", ".tsv"}:
#         sep = "\t" if ext == ".tsv" else ","
#         df = pd.read_csv(buf, sep=sep)
#     elif ext in {".json", ".ndjson"}:
#         try:
#             text = data.decode("utf-8")
#             # try ndjson first
#             if "\n" in text and text.strip().startswith("{") and text.strip().endswith("}"):
#                 df = pd.read_json(io.StringIO(text), lines=True)
#             else:
#                 j = json.loads(text)
#                 df = pd.json_normalize(j)
#         except Exception:
#             df = pd.read_json(buf, lines=True)
#     elif ext in {".parquet", ".pq", ".feather"}:
#         df = pd.read_parquet(buf)
#     elif ext in {".xlsx", ".xls"}:
#         df = pd.read_excel(buf)
#     else:
#         # last resort: try csv
#         try:
#             df = pd.read_csv(io.BytesIO(data))
#         except Exception:
#             df = None
#     if df is None:
#         return "unknown", None, None, None
#     df = df.head(1000)
#     return "tabular", df_to_records(df.head(50)), df.shape, list(df.columns)[:50]
# 

# pip install python-magic-bin charset-normalizer openpyxl pyarrow pdfplumber pillow lxml xmltodict
def sniff_bytes(b: bytes, content_type: Optional[str] = None, filename: str = ""):
    mt = content_type or magic.from_buffer(b, mime=True) or ""
    # fallback by ext
    ext = Path(filename).suffix.lower()
    return mt, ext

def decode_text(b: bytes) -> str:
    best = from_bytes(b).best()
    return (best.output() if best else b.decode("utf-8", "ignore"))

def read_any(name: str, data: bytes, content_type: Optional[str]):
    mime, ext = sniff_bytes(data, name, content_type)

    # ---- TABLES
    try:
        if "csv" in mime or ext in {".csv", ".tsv"}:
            txt = decode_text(data)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(txt.splitlines(True)[0]) if txt else csv.excel
            df = pd.read_csv(io.StringIO(txt), dialect=dialect)
            return {"kind":"table","table_name":_sanitize_name(name), "df": df}
        if "parquet" in mime or ext in {".parquet", ".pq"}:
            import pyarrow.parquet as pq; import pyarrow as pa
            table = pq.read_table(io.BytesIO(data))
            return {"kind":"table","table_name":_sanitize_name(name), "df": table.to_pandas()}
        if ext in {".feather"}:
            import pyarrow.feather as feather
            table = feather.read_feather(io.BytesIO(data))
            return {"kind":"table","table_name":_sanitize_name(name), "df": table.to_pandas()}
        if "excel" in mime or ext in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(data))
            return {"kind":"table","table_name":_sanitize_name(name), "df": df}
        if "json" in mime or ext in {".json", ".ndjson"}:
            text = decode_text(data)
            # try ndjson first
            if any(line.strip().startswith("{") for line in text.splitlines()[:5]):
                try:
                    df = pd.read_json(io.StringIO(text), lines=True)
                    return {"kind":"table","table_name":_sanitize_name(name), "df": df}
                except Exception:
                    pass
            obj = json.loads(text)
            # normalize list-of-objects or nested dicts
            df = pd.json_normalize(obj)
            return {"kind":"table","table_name":_sanitize_name(name), "df": df}
        if "html" in mime or ext in {".html", ".htm"}:
            html = decode_text(data)
            dfs = pd.read_html(html)
            if dfs:
                return {"kind":"table","table_name":_sanitize_name(name), "df": dfs[0]}
            return {"kind":"html","text": html}
        if "xml" in mime or ext in {".xml"}:
            text = decode_text(data)
            obj = xmltodict.parse(text)
            df = pd.json_normalize(obj)
            return {"kind":"table","table_name":_sanitize_name(name), "df": df}
    except Exception:
        pass

    # ---- PDF (try tables)
    if "pdf" in mime or ext == ".pdf":
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages[:2]:
                    tbl = page.extract_table()
                    if tbl:
                        df = pd.DataFrame(tbl[1:], columns=tbl[0])
                        return {"kind":"table","table_name":_sanitize_name(name), "df": df}
            return {"kind":"binary","bytes": data, "mime": mime}
        except Exception:
            return {"kind":"binary","bytes": data, "mime": mime}

    # ---- Images
    if mime.startswith("image/") or ext in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
        try:
            Image.open(io.BytesIO(data))  # validate
            return {"kind":"image","bytes": data, "mime": mime}
        except Exception:
            return {"kind":"binary","bytes": data, "mime": mime}

    # ---- Text fallback
    try:
        text = decode_text(data)
        if text.strip():
            return {"kind":"text","text": text}
    except Exception:
        pass

    # ---- Binary fallback
    return {"kind":"binary","bytes": data, "mime": mime or "application/octet-stream"}

def _ensure_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    # last resort – stringify other types
    return str(x)

def build_attachment_previews(files: List[UploadFile]) -> Tuple[Optional[str], List[AttachmentPreview], Dict[str, Any]]:
    """
    Returns (question_text, previews, attachment_tables)
    - question_text: the detected question/prompt text (from one uploaded file)
    - previews: lightweight AttachmentPreview objects for UI/LLM context
    - attachment_tables: dict mapping table_name -> FULL pandas.DataFrame (no sampling)
    This version relies on `read_any(name: str, data: bytes, content_type: Optional[str])`
    to robustly sniff/parse inputs of many formats.
    """
    question_text: Optional[str] = None
    previews: List[AttachmentPreview] = []
    attachment_tables: Dict[str, Any] = {}

    # ---- Read all files into memory once (same as before)
    loaded: List[Tuple[UploadFile, bytes]] = []
    for f in files:
        content = f.file.read()
        f.file.seek(0)
        loaded.append((f, content))

    # ---- Heuristics to pick the question file (prefer explicit names)
    candidates = []
    for f, b in loaded:
        name = (f.filename or "").lower()
        ext = Path(name).suffix.lower()
        # Strong signal by filename
        if any(k in name for k in ("question", "questions", "prompt")) and _is_probably_text(b):
            candidates.append((0, f, b))
        # Text-like and not too large
        elif ext in TEXT_EXTS and len(b) <= 256_000 and _is_probably_text(b):
            candidates.append((1, f, b))
        elif _is_probably_text(b) and len(b) <= 128_000:
            candidates.append((2, f, b))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, fq, bq = candidates[0]
        try:
            question_text = bq.decode("utf-8", errors="ignore").strip()
        except Exception:
            question_text = None
        question_file = fq
    else:
        question_file = None

    # ---- Build previews using the universal sniffer; keep FULL tables
    for f, b in loaded:
        # Skip the chosen question file
        if question_file is not None and f is question_file:
            continue

        name = f.filename or "file"
        content_type = getattr(f, "content_type", None)

        try:
            parsed = read_any(name, b, content_type)  # <-- universal loader
        except Exception:
            parsed = {"kind": "binary", "bytes": b, "mime": content_type or "application/octet-stream"}

        kind = parsed.get("kind", "unknown")

        # Tabular: store the FULL DataFrame, and prepare a small preview for LLM/UI
        if kind == "table":
            df = parsed["df"]
            table_name = parsed.get("table_name") or _sanitize_name(name)
            attachment_tables[table_name] = df  # store FULL DF

            previews.append(AttachmentPreview(
                name=name,
                table_name=table_name,
                kind="tabular",
                sample_text=None,  # not needed for tables
                sample_records=df_to_records(df.head(50)),
                shape=getattr(df, "shape", None),
                columns=list(df.columns)[:50] if hasattr(df, "columns") else None
            ))
            continue

        # ---- HTML detected but no table extracted
        if kind == "html":
            html_text = _ensure_str(parsed.get("text", ""))
            mime = "text/html"
            previews.append(AttachmentPreview(
                name=name,
                table_name=None,
                kind="html",
                sample_text=f"[mime: {mime}] " + (html_text[:500] if html_text else ""),
                sample_records=None,
                shape=None,
                columns=None
            ))
            continue
        
        # ---- Plain text
        if kind == "text":
            text = _ensure_str(parsed.get("text", ""))
            mime = "text/plain"
            previews.append(AttachmentPreview(
                name=name,
                table_name=None,
                kind="text",
                sample_text=f"[mime: {mime}] " + (text[:500] if text else ""),
                sample_records=None,
                shape=None,
                columns=None
            ))
            continue


        # Image
        if kind == "image":
            mime = parsed.get("mime", "image/*")
            previews.append(AttachmentPreview(
                name=name,
                table_name=None,
                kind="image",
                sample_text=f"[mime: {mime}]",
                sample_records=None,
                shape=None,
                columns=None
            ))
            continue

        # Fallback: unknown/binary
        mime = parsed.get("mime", content_type or "application/octet-stream")
        previews.append(AttachmentPreview(
            name=name,
            table_name=None,
            kind="unknown",
            sample_text=f"[mime: {mime}] (binary)",
            sample_records=None,
            shape=None,
            columns=None
        ))

    return question_text, previews, attachment_tables

def attachments_summary_for_llm(previews: List[AttachmentPreview]) -> str:
    lines = []
    for p in previews:
        if p.kind == "tabular":
            lines.append(f"- {p.name} -> table '{p.table_name}' (rows≈{p.shape[0] if p.shape else '?'}, cols={len(p.columns) if p.columns else '?'}, columns sample={p.columns[:5] if p.columns else []})")
        elif p.kind == "html":
            lines.append(f"- {p.name} (html) sample: {repr((p.sample_text or '')[:120])}")
        elif p.kind == "text":
            lines.append(f"- {p.name} (text) sample: {repr((p.sample_text or '')[:120])}")
        elif p.kind == "image":
            lines.append(f"- {p.name} (image)")
        else:
            lines.append(f"- {p.name} (unknown)")
    return "\n".join(lines) if lines else "(none)"

def build_attachment_tables(previews: List[AttachmentPreview]) -> Dict[str, Any]:
    """
    Build DataFrame tables (as dict: name->DataFrame-like records will be created later).
    We'll store as pandas DataFrames to merge into pipeline tables.
    """
    import pandas as pd
    tables: Dict[str, Any] = {}
    for p in previews:
        if p.kind == "tabular" and p.table_name and p.sample_records is not None:
            # We'll reconstruct small DataFrame from sample records for spec guidance,
            # but also keep a larger placeholder to be replaced during execution if needed.
            try:
                df = pd.DataFrame(p.sample_records)
                tables[p.table_name] = df
            except Exception:
                continue
    return tables

IO_INTENT_SYSTEM = """You output STRICT JSON:
{"output_format": "string"}
you are to provide a valid file format or valid output structure that best matches the user's wording. No commentary."""

# ---- Output format detection -------------------------------------------------
def detect_requested_format(question_text: str) -> Optional[str]:
    try:
        raw = llm_call_raw(IO_INTENT_SYSTEM, question_text, FORMATTER_MODEL, temperature=0, max_tokens=120)
        return coerce_json(extract_payload(raw))
    except Exception:
        # fallback: simple regex rules
        fmt = "txt"
        ql = (question_text or "").lower()
        if "csv" in ql: fmt = "csv"
        elif "json" in ql: fmt = "json"
        elif "yaml" in ql or "yml" in ql: fmt = "yaml"
        elif "excel" in ql or "xlsx" in ql: fmt = "xlsx"
        elif "parquet" in ql: fmt = "parquet"
        elif "markdown" in ql or "md" in ql: fmt = "markdown"
        return {"output_format": fmt, "filename": None}
# ---- Optimized executor -----------------------------------------------------
exec(HELPERS, globals(), globals())

ALLOWED_OPS = {
    "select_columns","rename","dropna","head","sort_values",
    "filter_query","groupby_agg","join","add_column","parse_dates"
}

def load_inputs(inputs: List[AnalysisInput], logs: List[str]) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    
    # Parallel loading for multiple inputs
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_input = {}
        
        for inp in inputs:
            if inp.source == "html" and inp.url:
                future = executor.submit(fetch_text, inp.url, 8, 1)  # Reduced timeout/retries
                future_to_input[future] = inp
            elif inp.source == "csv" and inp.url:
                future = executor.submit(fetch_text, inp.url, 8, 1)
                future_to_input[future] = inp
            elif inp.source == "json" and inp.url:
                future = executor.submit(fetch_json, inp.url, 8)
                future_to_input[future] = inp
            else:
                # Handle non-URL inputs immediately
                _load_single_input(inp, tables, logs)
        
        # Process completed futures
        for future in as_completed(future_to_input, timeout=15):  # Global timeout
            inp = future_to_input[future]
            try:
                data = future.result()
                _process_fetched_data(inp, data, tables, logs)
            except Exception as e:
                logs.append(f"[inputs] {inp.name}: fetch error {e}")
    
    return tables

def _load_single_input(inp: AnalysisInput, tables: Dict[str, Any], logs: List[str]):
    import pandas as pd
    
    if inp.source == "csv" and isinstance(inp.data, str):
        from io import StringIO
        tables[inp.name] = pd.read_csv(StringIO(inp.data))
        logs.append(f"[inputs] {inp.name}: inline csv shape={tables[inp.name].shape}")
    elif inp.source == "json" and inp.data:
        tables[inp.name] = pd.json_normalize(inp.data)
        logs.append(f"[inputs] {inp.name}: inline json shape={tables[inp.name].shape}")
    elif inp.source == "inline":
        tables[inp.name] = pd.DataFrame(inp.data or [])
        logs.append(f"[inputs] {inp.name}: inline records shape={tables[inp.name].shape}")

def _process_fetched_data(inp: AnalysisInput, data: Any, tables: Dict[str, Any], logs: List[str]):
    import pandas as pd
    
    if inp.source == "html":
        dfs = read_table_html(data)
        if dfs:
            idx = min(inp.table_index or 0, len(dfs) - 1)
            tables[inp.name] = dfs[idx].head(1000)  # Limit rows for speed
            logs.append(f"[inputs] {inp.name}: html table shape={tables[inp.name].shape}")
    elif inp.source == "csv":
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        tables[inp.name] = df.head(1000)  # Limit rows for speed
        logs.append(f"[inputs] {inp.name}: csv shape={tables[inp.name].shape}")
    elif inp.source == "json":
        df = pd.json_normalize(data)
        tables[inp.name] = df.head(1000)  # Limit rows for speed
        logs.append(f"[inputs] {inp.name}: json shape={tables[inp.name].shape}")

def apply_transform(df_map: Dict[str, Any], t: Transform, logs: List[str]) -> None:
    if t.op not in ALLOWED_OPS:
        logs.append(f"[transform] unsupported op {t.op}")
        return
    if t.target not in df_map:
        logs.append(f"[transform] missing target {t.target}")
        return
    
    import pandas as pd
    df = df_map[t.target]
    
    try:
        if t.op == "select_columns":
            cols = t.args.get("columns", [])
            df_map[t.target] = df[cols]
        elif t.op == "rename":
            mapping = t.args.get("map", {})
            df_map[t.target] = df.rename(columns=mapping)
        elif t.op == "dropna":
            subset = t.args.get("subset", None)
            df_map[t.target] = df.dropna(subset=subset)
        elif t.op == "head":
            n = min(int(t.args.get("n", 10)), 100)  # Cap at 100 for speed
            df_map[t.target] = df.head(n)
        elif t.op == "sort_values":
            by = t.args.get("by")
            asc = bool(t.args.get("ascending", True))
            df_map[t.target] = df.sort_values(by=by, ascending=asc)
        elif t.op == "filter_query":
            q = t.args.get("query", "")
            df_map[t.target] = df.query(q)
        elif t.op == "groupby_agg":
            by = t.args.get("by", [])
            aggs = t.args.get("aggs", {})
            df_map[t.target] = df.groupby(by, dropna=False).agg(aggs).reset_index()
        elif t.op == "join":
            right = t.args.get("right")
            how = t.args.get("how", "left")
            on = t.args.get("on")
            if right not in df_map:
                logs.append(f"[transform] join: right table {right} missing")
            else:
                df_map[t.target] = df.merge(df_map[right], how=how, on=on)
        elif t.op == "add_column":
            name = t.args.get("name")
            expr = t.args.get("expr")
            if name and expr:
                df_map[t.target][name] = pd.eval(expr, engine="python", parser="pandas", target=df_map[t.target])
        elif t.op == "parse_dates":
            cols = t.args.get("columns", [])
            for c in cols:
                df_map[t.target][c] = pd.to_datetime(df_map[t.target][c], errors="coerce")
        logs.append(f"[transform] {t.op} on {t.target}: shape={df_map[t.target].shape}")
    except Exception as e:
        logs.append(f"[transform] {t.op} error: {e}")

def render_charts(df_map: Dict[str, Any], charts: List[ChartSpec], logs: List[str]) -> List[str]:
    if not charts:  # Skip if no charts requested
        return []
    
    images: List[str] = []
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for speed
    import matplotlib.pyplot as plt
    
    for ch in charts[:3]:  # Limit to 3 charts max for speed
        if ch.table not in df_map:
            logs.append(f"[chart] missing table {ch.table}")
            continue
        df = df_map[ch.table].head(200)  # Limit data points for speed
        try:
            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figures
            
            if ch.kind == "line":
                if isinstance(ch.y, list):
                    for col in ch.y[:3]:  # Max 3 series
                        ax.plot(df[ch.x], df[col], label=col)
                    ax.legend()
                else:
                    ax.plot(df[ch.x], df[ch.y])
            elif ch.kind == "bar":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                ax.bar(df[ch.x], df[ycol])
            elif ch.kind == "scatter":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                ax.scatter(df[ch.x], df[ycol], alpha=0.6, s=20)  # Smaller markers
            elif ch.kind == "hist":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                bins = min(ch.bins or 20, 20)  # Limit bins
                ax.hist(df[ycol], bins=bins)
            
            if ch.title: 
                ax.set_title(ch.title)
            
            images.append(fig_to_data_uri(fig))
            logs.append(f"[chart] {ch.kind} on {ch.table}")
        except Exception as e:
            logs.append(f"[chart] error: {e}")
    
    return images

def summarize(answer: AnswerSpec, df_map: Dict[str, Any], logs: List[str]) -> Any:
    if answer.type == "none":
        return None
    
    import pandas as pd
    
    if answer.type == "basic_stats":
        tname = answer.table or next(iter(df_map.keys()), None)
        if not tname or tname not in df_map:
            return {"note": "no table available for stats"}
        df = df_map[tname]
        cols = (answer.columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])[:5]  # Limit cols
        out = {}
        for c in cols:
            try:
                s = df[c].dropna()
                out[c] = {
                    "count": int(s.count()),
                    "mean": float(s.mean()) if len(s) else None,
                    "min": float(s.min()) if len(s) else None,
                    "max": float(s.max()) if len(s) else None,
                }
            except Exception:
                out[c] = {"error": "stat failed"}
        logs.append(f"[answer] basic_stats on {tname}")
        return out
    
    if answer.type == "text_summary":
        tname = answer.table or next(iter(df_map.keys()), None)
        if not tname or tname not in df_map:
            return "No data available."
        df = df_map[tname]
        return f"Rows: {len(df)}, Columns: {len(df.columns)}. Sample cols: {list(df.columns)[:3]}"
    
    return None

def validate_final_result(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object.")
    missing = REQUIRED_FINAL_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Output missing required keys: {sorted(missing)}")
    return data

def make_formatter_prompt(question: str, final_result: Dict[str, Any], requested_format: Optional[str]) -> str:
    # Simplified prompt for faster processing, includes requested format
    return f"REQUESTED OUTPUT FORMAT: {requested_format}\nQuestion: {question}\nData: {json.dumps(final_result, default=str)[:2000]}"

# ---- FRESH fallback spec generator ------------------------------------------
def make_fresh_spec_from_attachments(question: str, previews: List[AttachmentPreview]) -> AnalysisSpec:
    """
    Minimal, robust spec: pick first tabular attachment if present.
    """
    # Choose table
    table_name = None
    for p in previews:
        if p.kind == "tabular" and p.table_name:
            table_name = p.table_name
            break
    if table_name is None:
        # No tabular data; return empty minimal spec
        return AnalysisSpec(
            inputs=[AnalysisInput(name="dummy", source="inline", data=[])],
            transforms=[],
            charts=[],
            answer=AnswerSpec(type="text_summary", table="dummy"),
            result_table="dummy"
        )
    # Build simple spec
    return AnalysisSpec(
        inputs=[AnalysisInput(name=table_name, source="inline", data=[])],  # data will be ignored; tables will be injected
        transforms=[Transform(target=table_name, op="head", args={"n": 50})],
        charts=[],
        answer=AnswerSpec(type="basic_stats", table=table_name),
        result_table=table_name
    )

# ---- Optimized pipeline execution -------------------------------------------
def run_analysis_spec(spec: AnalysisSpec, attachment_tables: Dict[str, Any]) -> Dict[str, Any]:
    logs: List[str] = []
    
    # Load inputs (with parallel fetching)
    tables = load_inputs(spec.inputs, logs)

    # Merge in attachment tables (always available by table_name)
    for k, v in attachment_tables.items():
        if k not in tables:
            tables[k] = v
            logs.append(f"[inputs] attachment injected: {k} shape={getattr(v,'shape',None)}")
    
    # Apply transforms sequentially (but limited)
    for t in spec.transforms[:10]:  # Limit transforms for speed
        apply_transform(tables, t, logs)
    
    # Build images (limited)
    images = render_charts(tables, spec.charts, logs)
    
    # Choose result table
    res_name = spec.result_table
    if res_name not in tables and tables:
        res_name = next(iter(tables.keys()))
    
    # Prepare tables output (smaller previews)
    tables_out: Dict[str, List[Dict[str, Any]]] = {}
    for name, df in tables.items():
        try:
            preview = df.head(50)  # Much smaller preview
            tables_out[name] = df_to_records(preview)
        except Exception:
            tables_out[name] = []
    
    # Generate answer
    ans = summarize(spec.answer, tables, logs)
    
    final_result = {
        "answer": ans,
        "tables": {res_name: tables_out.get(res_name, [])},
        "images": images,
        "logs": logs,
    }
    
    return final_result

# ---- Parallel API execution with 2 debug tries + fresh fallback -------------
async def analyze_question_async(question: str,
                                 previews: List[AttachmentPreview],
                                 attachment_tables: Dict[str, Any],
                                 requested_format: Optional[str],
                                 debug: int = 0):

    attachments_summary = attachments_summary_for_llm(previews)
    last_err: Optional[str] = None

    # Attempt 1: normal
    for attempt in range(1, 4):  # 1=normal, 2=debug try 1, 3=debug try 2
        debug_hint = None if attempt == 1 else f"Previous error: {last_err}. Prefer using the provided attachments via inline inputs. Avoid network fetches."
        try:
            # Step 1: Plan (fast)
            plan = plan_question(question if attempt == 1 else f"{question}\n\n(You have attachments. See list below.)\n{attachments_summary}")
            # Step 2: Generate spec (single attempt)
            spec = generate_analysis_spec(plan, attachments_summary, debug_hint=debug_hint)
            # Step 3: Execute spec
            final_structured = run_analysis_spec(spec, attachment_tables)
            final_structured = validate_final_result(final_structured)
            # Step 4: Format output (with requested format)
            fmt_prompt = make_formatter_prompt(question, final_structured, requested_format)
            formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0, max_tokens=500)
            if debug == 1:
                return JSONResponse({
                    "ok": True,
                    "formatted_output": formatted,
                    "internal_final_result": final_structured,
                    "debug": {
                        "plan": plan.model_dump(),
                        "analysis_spec": spec.model_dump(),
                        "attempt": attempt,
                        "performance_mode": "speed_optimized",
                        "attachments": [p.model_dump() for p in previews]
                    }
                })
            return Response(formatted, media_type="text/plain")
        except Exception as e:
            last_err = str(e)

    # Fresh fallback (programmatic minimal spec)
    try:
        spec = make_fresh_spec_from_attachments(question, previews)
        final_structured = run_analysis_spec(spec, attachment_tables)
        final_structured = validate_final_result(final_structured)
        fmt_prompt = make_formatter_prompt(question, final_structured, requested_format)
        formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0, max_tokens=500)
        if debug == 1:
            return JSONResponse({
                "ok": True,
                "formatted_output": formatted,
                "internal_final_result": final_structured,
                "debug": {
                    "plan": "(fresh-fallback)",
                    "analysis_spec": spec.model_dump(),
                    "attempt": "fresh_fallback",
                    "previous_error": last_err,
                    "attachments": [p.model_dump() for p in previews]
                }
            })
        return Response(formatted, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"All attempts failed. Last error: {last_err} | Fresh fallback error: {e}")

from fastapi import FastAPI, Request, UploadFile, HTTPException

@app.post("/api/")
async def analyze_question(
    request: Request,
    debug: int = Query(0, description="Set to 1 to include debug payloads (JSON)."),
):
    
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {e}")

    seen_keys = list(form.keys())

    files: List[UploadFile] = []
    for key, value in form.multi_items():
        if hasattr(value, "filename") and hasattr(value, "read"):
            files.append(value)

    if not files:
        if debug == 1:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "No files uploaded",
                    "observed_form_keys": seen_keys,
                    "content_type": request.headers.get("content-type"),
                    "hint": "Ensure you're using -F with curl so Content-Type is multipart/form-data."
                },
            )
        raise HTTPException(status_code=400, detail="No files uploaded")
        
    question_text, previews, attachment_tables = build_attachment_previews(files)
    if not question_text:
        raise HTTPException(status_code=400, detail="Could not detect a question file.")

    requested_format = detect_requested_format(question_text)
    return await analyze_question_async(
    question_text, previews, attachment_tables, requested_format, debug
)  
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "4.2.0-speed-optimized"}

@app.get("/")
async def root():
    return {
        "message": "Universal Data Analyst API v4.2 (Speed Optimized)",
        "usage": [
            "curl -s -X POST 'http://localhost:8000/api/?debug=1' -F 'questions.txt=@question.txt' -F 'data.csv=@data.csv'",
            "curl -s -X POST 'http://localhost:8000/api/' -F 'prompt.md=@prompt.md' -F 'dataset.parquet=@dataset.parquet'"
        ],
        "optimizations": [
            "Reduced network timeouts (8s vs 20s)",
            "Parallel input fetching",
            "Single-shot spec generation (no best-of-N)",
            "Limited data processing (1000 rows max per input)",
            "Optimized chart rendering (lower DPI, smaller figures)",
            "Truncated LLM responses (max_tokens limits)",
            "Async processing pipeline",
            "Arbitrary multipart field names (auto-detect question vs attachments)",
            "Requested output format inferred from question",
            "Two debug retries + fresh fallback"
        ],
        "notes": [
            "Tabular attachments (csv/tsv/json/ndjson/parquet/feather/xlsx) are auto-exposed as tables by sanitized filename.",
            "Images and unknown binaries are accepted as context but not parsed into tables."
        ],
        "target_performance": "< 3 minutes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

