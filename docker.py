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

from typing import Literal
from pydantic import Field

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError, Field

# ---- LLM client (OpenAI-compatible) -----------------------------------------
from openai import OpenAI

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

# Defaults tuned for small models; override via env
PLANNER_MODEL    = os.getenv("LLM_PLANNER_MODEL",    "gpt-5-mini")
CODER_MODEL      = os.getenv("LLM_CODER_MODEL",      "gpt-5-mini")
FORMATTER_MODEL  = os.getenv("LLM_FORMATTER_MODEL",  "gpt-5-mini")

if not LLM_API_KEY:
    # /health and / will work; /api will return a clear error.
    pass
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

app = FastAPI(title="Universal Data Analyst (speed-optimized)", version="4.2.0")

import copy

def strip_payload_for_llm(final_structured: dict) -> dict:
    slim = copy.deepcopy(final_structured)

    # Replace image URIs with lightweight placeholders
    if "images" in slim:
        slim["images"] = {k: f"[image:{k}]" for k in slim["images"].keys()}

    # (Optional) shrink tables so you don’t leak lots of rows into the prompt
    if "tables" in slim:
        slim["tables"] = {
            name: {"preview_sample_cols": list(rows[0].keys()) if rows else [], "rows_previewed": len(rows)}
            for name, rows in slim["tables"].items()
        }
    return slim

def inject_images(text: str, images: Dict[str, str]) -> str:
    # Make sure the data URIs are single-line and well-formed
    def sanitize(uri: str) -> str:
        uri = uri.replace("\n", "").replace("\r", "").strip()
        return uri if uri.startswith("data:image/") else f"data:image/png;base64,{uri}"
    for k, uri in (images or {}).items():
        text = text.replace(f"[image:{k}]", sanitize(uri))
    return text


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
          "bins": {"type":["integer","null"]},
          "result_key": {"type":["string","null"]},
          "color": {"type":["string","null"]}
        },
        "required": ["table","kind"],
        "additionalProperties": false
      }
    },
    "metrics": {
      "type":"array",
      "items": {
        "type":"object",
        "properties": {
          "table": {"type":"string"},
          "op": {"type":"string","enum":[
            "mean","min","max","sum","median","std","nunique","count",
            "corr","argmax_return","argmin_return","expr"
          ]},
          "args": {"type":"object"},
          "result_key": {"type":"string"}
        },
        "required": ["table","op","args","result_key"],
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
  "required": ["inputs","transforms","charts","metrics","answer","result_table"],
  "additionalProperties": false
}
Guidelines:
- Keep transforms minimal.
- If the user specifies exact output keys (e.g., "Return a JSON with keys ..."), create metrics with matching `result_key` names and charts with matching `result_key` names for any image keys.
- Use simple, robust ops (mean/min/max/sum/corr/argmax_return/argmin_return). Use `expr` only when necessary for a scalar result.
Return JSON only."""

# Simplified formatter for speed — respects requested output format
OUTPUT_FORMATTER_SYSTEM = """
Your task is to produce the final result in the exact format requested by the user.
Rules:
- Read the user request carefully and identify any explicit or implicit output format or schema, no matter what it is called or how it is described.
- Follow that format or schema exactly, including structure, syntax, spacing, and punctuation.
-Use ONLY the values already present in Data.metrics (scalars) and Data.images (data URIs).
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
def fig_to_data_uri_sized(fig, size_candidates=(72, 60, 48), max_bytes=100*1024):
    import io, base64
    for dpi in size_candidates:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            plt_close = True
            break
    else:
        # fallback: accept the smallest even if > max_bytes
        data = buf.getvalue()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
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
    kind: str  # "line" | "bar" | "scatter" | "hist"
    x: Optional[str] = None
    y: Optional[Any] = None
    title: Optional[str] = None
    bins: Optional[int] = None
    # NEW: allow the LLM to name the output image key and optionally a color
    result_key: Optional[str] = None
    color: Optional[str] = None

class MetricSpec(BaseModel):
    table: str
    op: Literal[
        "mean","min","max","sum","median","std","nunique","count",
        "corr","argmax_return","argmin_return","expr"
    ]
    # args:
    # - column: for mean/min/max/sum/median/std/nunique/count
    # - x,y: for corr
    # - by, return_col: for argmax_return/argmin_return
    # - expr: pandas-compatible scalar expression for "expr" (e.g., "temperature_c.mean()")
    # - type: "float"|"int"|"str" (coercion for expr output)
    args: Dict[str, Any] = Field(default_factory=dict)
    # Where to place the computed value in final_result["metrics"]
    result_key: str

class AnswerSpec(BaseModel):
    type: str  # "text_summary" | "basic_stats" | "none"
    table: Optional[str] = None
    columns: Optional[List[str]] = None

class AnalysisSpec(BaseModel):
    inputs: List[AnalysisInput]
    transforms: List[Transform]
    charts: List[ChartSpec]
    # NEW: optional list of metrics (defaults to [])
    metrics: List[MetricSpec] = Field(default_factory=list)
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
def make_coder_prompt(plan: Plan, attachments_summary: str, attachments_present: bool, debug_hint: Optional[str]=None) -> str:
    hint = f"\nDEBUG_HINT: {debug_hint}" if debug_hint else ""

    if attachments_present:
        attach_guidance = (
            "When referencing attachments, set inputs[].source='inline' and inputs[].data to a SMALL sample "
            "(we will provide full data at execution time)."
        )
    else:
        attach_guidance = (
            "No attachments are available. Do NOT fabricate inline samples. "
            "If data must be fetched, create URL-based inputs with source='html'|'csv'|'json' and a real url. "
            "If nothing is fetchable, set answer.type='none'."
        )
    return (
         f"Question: {plan.question}\n"
         f"Plan: {plan.json()}\n"
         f"Attachments available (prefer using them with source='inline' and their table names):\n{attachments_summary}\n"
         f"{attach_guidance}"
         f"{hint}\n"
         "Return analysis_spec JSON. Be minimal and efficient."
     )

# Single-shot spec generation (no best-of-N for speed)
def generate_analysis_spec(plan: Plan, attachments_summary: str, attachments_present: bool, debug_hint: Optional[str]=None) -> AnalysisSpec:
    raw = llm_call_raw(
        CODER_SYSTEM,
        make_coder_prompt(plan, attachments_summary, attachments_present,debug_hint),
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
    try:
        # Try simple UTF-8 first
        simple = b.decode("utf-8")
        print(f"[DEBUG] UTF-8 decode: {simple[:1000]}...")
        return simple
    except:
        best = from_bytes(b).best()
        result = (best.output() if best else b.decode("utf-8", "ignore"))
        print(f"[DEBUG] Charset-normalizer result: {result[:1000]}...")
        return result


def read_any(name: str, data: bytes, content_type: Optional[str]):
    """
    Robust file sniffer/loader. Returns one of:
      - {"kind":"table", "table_name": <safe>, "df": <pd.DataFrame>}
      - {"kind":"image", "bytes": data, "mime": mime}
      - {"kind":"html", "text": html}
      - {"kind":"text", "text": text}
      - {"kind":"binary","bytes": data, "mime": mime}
    """
    import io, json
    from typing import Optional
    import pandas as pd

    # --- helpers
    def _lower(s): return (s or "").lower()
    def _has_ext(n, exts): 
        n = (n or "").lower()
        return any(n.endswith(e) for e in exts)
    def _safe_name(n):
        return _sanitize_name(n) if "_sanitize_name" in globals() else (n or "table").replace(".", "_")

    mime, ext = sniff_bytes(data, content_type, name)
    mime_l = _lower(mime)
    ext_l  = _lower(ext)
# ============= 1) CSV/TSV — do this FIRST and return early  =============
    # Conditions: explicit CSV/TSV extension, CSV-y mime, OR the bytes look csv-ish.
    looks_csvish = False
    try:
        # Peek just a little to avoid decoding huge blobs; decode defensively.
        head_text = decode_text(data[:4096] if isinstance(data, (bytes, bytearray)) else data)
        if head_text:
            # simple heuristic for delimited text
            first_line = head_text.splitlines()[0] if head_text.splitlines() else ""
            looks_csvish = ("," in first_line or "\t" in first_line) and len(first_line.split(",")) + len(first_line.split("\t")) > 1
    except Exception:
        pass

    if ext_l in {".csv", ".tsv"} or "csv" in mime_l or _has_ext(name, [".csv", ".tsv"]) or looks_csvish:
        try:
            # decode to text (handles BOM); NEVER feed text into BytesIO
            txt = decode_text(data)
            if not txt.strip():
                raise ValueError("Empty CSV text.")

            # Prefer auto-sniff unless it's clearly TSV
            # pandas: sep=None requires engine='python' for sniffing
            if ext_l == ".tsv":
                df = pd.read_csv(io.StringIO(txt.lstrip("\ufeff")), sep="\t")
            else:
                try:
                    df = pd.read_csv(io.StringIO(txt.lstrip("\ufeff")), sep=None, engine="python")
                except Exception:
                    # fallback to comma, then tab
                    try:
                        df = pd.read_csv(io.StringIO(txt.lstrip("\ufeff")), sep=",")
                    except Exception:
                        df = pd.read_csv(io.StringIO(txt.lstrip("\ufeff")), sep="\t")

            # sanity: require at least 1 column
            if getattr(df, "shape", (0, 0))[1] == 0:
                raise ValueError("Parsed CSV had 0 columns.")

            return {"kind": "table", "table_name": _safe_name(name), "df": df}
        except Exception as e:
            print(f"CSV parsing failed for {name}: {e}")  # keep your debug, but don't abort
    # ============= 2) Columnar & spreadsheets  =============
    try:
        if "parquet" in mime_l or ext_l in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            table = pq.read_table(io.BytesIO(data))
            return {"kind":"table","table_name":_safe_name(name), "df": table.to_pandas()}

        if ext_l in {".feather"}:
            import pyarrow.feather as feather
            table = feather.read_feather(io.BytesIO(data))
            return {"kind":"table","table_name":_safe_name(name), "df": table.to_pandas()}

        if "excel" in mime_l or ext_l in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(data))
            return {"kind":"table","table_name":_safe_name(name), "df": df}

        if "json" in mime_l or ext_l in {".json", ".ndjson"}:
            import pandas as pd, json
            text = decode_text(data)
            # Try NDJSON first (lines of JSON objects)
            head = "\n".join(text.splitlines()[:5])
            if any(line.strip().startswith("{") for line in head.splitlines()):
                try:
                    df = pd.read_json(io.StringIO(text), lines=True)
                    return {"kind":"table","table_name":_safe_name(name), "df": df}
                except Exception:
                    pass
            obj = json.loads(text)
            df = pd.json_normalize(obj)
            return {"kind":"table","table_name":_safe_name(name), "df": df}

        if "html" in mime_l or ext_l in {".html", ".htm"}:
            html = decode_text(data)
            try:
                dfs = pd.read_html(html)
                if dfs:
                    return {"kind":"table","table_name":_safe_name(name), "df": dfs[0]}
            except Exception:
                pass
            return {"kind":"html","text": html}

        if "xml" in mime_l or ext_l in {".xml"}:
            import xmltodict
            text = decode_text(data)
            obj = xmltodict.parse(text)
            df = pd.json_normalize(obj)
            return {"kind":"table","table_name":_safe_name(name), "df": df}
    except Exception:
        # fall through to other handlers
        pass

    # ============= 3) PDF (try to extract simple tables)  =============
    if "pdf" in mime_l or ext_l == ".pdf":
        try:
            import pdfplumber, pandas as pd
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages[:2]:
                    # try multiple tables per page
                    tables = page.extract_tables() or []
                    for tbl in tables:
                        if tbl and len(tbl) > 1:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            return {"kind":"table","table_name":_safe_name(name), "df": df}
            return {"kind":"binary","bytes": data, "mime": mime}
        except Exception:
            return {"kind":"binary","bytes": data, "mime": mime}

    # ============= 4) Images  =============
    if mime_l.startswith("image/") or ext_l in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
        try:
            from PIL import Image
            Image.open(io.BytesIO(data))  # validate
            return {"kind":"image","bytes": data, "mime": mime}
        except Exception:
            return {"kind":"binary","bytes": data, "mime": mime}

    # ============= 5) Text fallback  =============
    try:
        text = decode_text(data)
        if text.strip():
            return {"kind":"text","text": text}
    except Exception:
        pass

    # ============= 6) Binary fallback  =============
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
    Fixed version that properly handles file reading and processing
    """
    question_text: Optional[str] = None
    previews: List[AttachmentPreview] = []
    attachment_tables: Dict[str, Any] = {}

    # Read all files into memory once
    loaded: List[Tuple[UploadFile, bytes]] = [] 
    for f in files: 
        content = f.file.read()
        # Ensure content is bytes
        if isinstance(content, str):
            content = content.encode("utf-8")
        f.file.seek(0) 
        loaded.append((f, content))

    # Heuristics to pick the question file
    candidates = []
    for f, b in loaded:
        name = (f.filename or "").lower()
        ext = Path(name).suffix.lower()
        if any(k in name for k in ("question", "questions", "prompt")) and _is_probably_text(b):
            candidates.append((0, f, b))
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

    # Process each file and store FULL DataFrames
    for f, b in loaded:
        if question_file is not None and f is question_file:
            continue

        name = f.filename or "file"
        content_type = getattr(f, "content_type", None)

        try:
            # CRITICAL FIX: Ensure b is bytes, not string
            if isinstance(b, str):
                b = b.encode("utf-8")
            parsed = read_any(name, b, content_type)
        except Exception as e:
            print(f"Error parsing {name}: {e}")  # Debug print
            parsed = {"kind": "binary", "bytes": b, "mime": content_type or "application/octet-stream"}

        kind = parsed.get("kind", "unknown")

        # CRITICAL FIX: Store FULL DataFrame in attachment_tables
        if kind == "table":
            df = parsed["df"]
            table_name = parsed.get("table_name") or _sanitize_name(name)
            
            # Store the FULL DataFrame (not just preview)
            attachment_tables[table_name] = df
            
            # Create preview for LLM
            previews.append(AttachmentPreview(
                name=name,
                table_name=table_name,
                kind="tabular",  # Use "tabular" not "table" to match the enum
                sample_text=None,
                sample_records=df_to_records(df.head(50)),  # Just preview
                shape=df.shape,
                columns=list(df.columns)[:50] if hasattr(df, "columns") else None
            ))
            continue
        if kind != "table":
            name_l = (name or "").lower()
            looks_csvish = False
            try:
                head_text = decode_text(b[:4096] if isinstance(b, (bytes, bytearray)) else b)
                if head_text:
                    first_line = head_text.splitlines()[0] if head_text.splitlines() else ""
                    looks_csvish = ("," in first_line or "\t" in first_line) and (
                        len(first_line.split(",")) + len(first_line.split("\t")) > 1
                    )
            except Exception:
                pass
            if name_l.endswith((".csv", ".tsv")) or looks_csvish:
                try:
                    import pandas as pd, io as _io
                    txt = decode_text(b)
                    sep = "\t" if name_l.endswith(".tsv") or (txt.count("\t") > txt.count(",")) else ","
                    df = pd.read_csv(_io.StringIO(txt.lstrip("\ufeff")), sep=sep)
                    table_name = _sanitize_name(name)
                    attachment_tables[table_name] = df
                    previews.append(AttachmentPreview(
                        name=name, table_name=table_name, kind="tabular",
                        sample_text=None, sample_records=df_to_records(df.head(50)),
                        shape=df.shape, columns=list(df.columns)[:50]
                    ))
                    continue
                except Exception:
                    pass

        # Handle other file types (HTML, text, images, etc.)
        if kind == "html":
            html_text = _ensure_str(parsed.get("text", ""))
            previews.append(AttachmentPreview(
                name=name, table_name=None, kind="html",
                sample_text=f"[HTML] " + (html_text[:500] if html_text else ""),
                sample_records=None, shape=None, columns=None
            ))
        elif kind == "text":
            text = _ensure_str(parsed.get("text", ""))
            previews.append(AttachmentPreview(
                name=name, table_name=None, kind="text",
                sample_text=f"[TEXT] " + (text[:500] if text else ""),
                sample_records=None, shape=None, columns=None
            ))
        elif kind == "image":
            mime = parsed.get("mime", "image/*")
            previews.append(AttachmentPreview(
                name=name, table_name=None, kind="image",
                sample_text=f"[IMAGE: {mime}]",
                sample_records=None, shape=None, columns=None
            ))
        else:
            mime = parsed.get("mime", content_type or "application/octet-stream")
            previews.append(AttachmentPreview(
                name=name, table_name=None, kind="unknown",
                sample_text=f"[BINARY: {mime}]",
                sample_records=None, shape=None, columns=None
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
    
    # Separate URL-based inputs for parallel processing
    url_inputs = []
    inline_inputs = []
    
    for inp in inputs:
        if inp.url and inp.source in ["html", "csv", "json"]:
            url_inputs.append(inp)
        else:
            inline_inputs.append(inp)
    
    # Process inline inputs immediately
    for inp in inline_inputs:
        _load_single_input(inp, tables, logs)
    
    # Process URL inputs in parallel (if any)
    if url_inputs:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_input = {}
            
            for inp in url_inputs:
                if inp.source == "html":
                    future = executor.submit(fetch_text, inp.url, 8, 1)
                elif inp.source == "csv":
                    future = executor.submit(fetch_text, inp.url, 8, 1)
                elif inp.source == "json":
                    future = executor.submit(fetch_json, inp.url, 8)
                else:
                    continue
                future_to_input[future] = inp
            
            # Process results
            for future in as_completed(future_to_input, timeout=15):
                inp = future_to_input[future]
                try:
                    data = future.result()
                    _process_fetched_data(inp, data, tables, logs)
                except Exception as e:
                    logs.append(f"[inputs] {inp.name}: fetch error {e}")
    
    return tables

def debug_attachment_flow(attachment_tables: Dict[str, Any], logs: List[str]):
    """Helper function to debug attachment data flow"""
    logs.append(f"[DEBUG] attachment_tables keys: {list(attachment_tables.keys())}")
    for k, v in attachment_tables.items():
        if hasattr(v, 'shape'):
            logs.append(f"[DEBUG] {k}: DataFrame shape={v.shape}, columns={list(v.columns) if hasattr(v, 'columns') else 'no columns'}")
        else:
            logs.append(f"[DEBUG] {k}: type={type(v)}")

def _load_single_input(inp: AnalysisInput, tables: Dict[str, Any], logs: List[str]):
    import pandas as pd
    
    try:
        if inp.source == "csv" and isinstance(inp.data, str):
            from io import StringIO
            tables[inp.name] = pd.read_csv(StringIO(inp.data))
            logs.append(f"[inputs] {inp.name}: inline csv shape={tables[inp.name].shape}")
        elif inp.source == "json" and inp.data:
            if isinstance(inp.data, list):
                # List of records
                tables[inp.name] = pd.DataFrame(inp.data)
            else:
                # Nested JSON - normalize it
                tables[inp.name] = pd.json_normalize(inp.data)
            logs.append(f"[inputs] {inp.name}: inline json shape={tables[inp.name].shape}")
        elif inp.source == "inline":
            # Handle inline data properly - could be CSV text or records
            if isinstance(inp.data, str):
                # Check if it's CSV text (starts with column names and has delimiters)
                data_str = inp.data
                # Remove any [TEXT] prefix that might be added
                if data_str.startswith("[TEXT] "):
                    data_str = data_str[7:]
                
                # Try to parse as CSV if it looks like CSV
                lines = data_str.strip().splitlines()
                if len(lines) >= 2 and (',' in data_str or '\t' in data_str):
                    try:
                        from io import StringIO
                        # Determine separator
                        if data_str.count('\t') > data_str.count(','):
                            df = pd.read_csv(StringIO(data_str), sep='\t')
                        else:
                            df = pd.read_csv(StringIO(data_str), sep=',')
                        tables[inp.name] = df
                        logs.append(f"[inputs] {inp.name}: parsed CSV text shape={tables[inp.name].shape}")
                    except Exception as e:
                        logs.append(f"[inputs] {inp.name}: failed to parse as CSV: {e}")
                        tables[inp.name] = pd.DataFrame()
                else:
                    # Not CSV, treat as empty
                    tables[inp.name] = pd.DataFrame()
                    logs.append(f"[inputs] {inp.name}: text data not CSV format")
            elif isinstance(inp.data, list) and len(inp.data) > 0:
                tables[inp.name] = pd.DataFrame(inp.data)
                logs.append(f"[inputs] {inp.name}: inline records shape={tables[inp.name].shape}")
            else:
                # Create empty DataFrame if no data
                tables[inp.name] = pd.DataFrame()
                logs.append(f"[inputs] {inp.name}: no valid data, created empty DataFrame")
        else:
            logs.append(f"[inputs] {inp.name}: unsupported source {inp.source}")
            tables[inp.name] = pd.DataFrame()
    except Exception as e:
        logs.append(f"[inputs] {inp.name}: error loading - {e}")
        tables[inp.name] = pd.DataFrame()  # Fallback empty DataFrame

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
            # accept string or list
            if isinstance(by, str):
                by = [by]
            # accept 'aggs' or 'agg'
            aggs = t.args.get("aggs", t.args.get("agg", {}))
            if not isinstance(aggs, dict) or not by:
                raise ValueError("groupby_agg requires by (str|list) and agg(s) dict")
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

def render_charts(df_map: Dict[str, Any], charts: List[ChartSpec], logs: List[str]) -> Dict[str, str]:
    if not charts:
        return {}

    images: Dict[str, str] = {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for ch in charts[:5]:  # small cap
        if ch.table not in df_map:
            logs.append(f"[chart] missing table {ch.table}")
            continue
        df = df_map[ch.table].head(500)
        try:
            fig, ax = plt.subplots(figsize=(5, 3))
            color = ch.color  # may be None
            if ch.kind == "line":
                if isinstance(ch.y, list):
                    for col in ch.y[:3]:
                        ax.plot(df[ch.x], df[col], label=col, color=color)
                    ax.legend()
                else:
                    ax.plot(df[ch.x], df[ch.y], color=color)
                ax.set_xlabel(ch.x or "")
                ax.set_ylabel(",".join(ch.y) if isinstance(ch.y, list) else (ch.y or ""))
            elif ch.kind == "bar":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                ax.bar(df[ch.x], df[ycol], color=color)
                ax.set_xlabel(ch.x or "")
                ax.set_ylabel(ycol or "")
            elif ch.kind == "scatter":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                ax.scatter(df[ch.x], df[ycol], alpha=0.6, s=20, color=color)
                ax.set_xlabel(ch.x or "")
                ax.set_ylabel(ycol or "")
            elif ch.kind == "hist":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                bins = min(ch.bins or 20, 50)
                ax.hist(df[ycol], bins=bins, color=color)
                ax.set_xlabel(ycol or "")
                ax.set_ylabel("count")
            if ch.title:
                ax.set_title(ch.title)

            uri = fig_to_data_uri(fig)
            key = ch.result_key or f"chart_{len(images)+1}"
            images[key] = uri
            logs.append(f"[chart] {ch.kind} -> {key}")
        except Exception as e:
            logs.append(f"[chart] error: {e}")
    return images

def compute_metrics(df_map: Dict[str, Any], metrics: List[MetricSpec], logs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    import pandas as pd
    for m in metrics[:50]:
        if m.table not in df_map:
            logs.append(f"[metric] missing table {m.table}")
            continue
        df = df_map[m.table]
        try:
            if m.op in {"mean","min","max","sum","median","std","nunique","count"}:
                col = m.args.get("column")
                s = pd.to_numeric(df[col], errors="coerce") if col else None
                if m.op == "mean": val = float(s.mean())
                elif m.op == "min": val = float(s.min())
                elif m.op == "max": val = float(s.max())
                elif m.op == "sum": val = float(s.sum())
                elif m.op == "median": val = float(s.median())
                elif m.op == "std": val = float(s.std())
                elif m.op == "nunique": val = int(s.nunique(dropna=True))
                elif m.op == "count": val = int(s.count())
                out[m.result_key] = val
            elif m.op == "corr":
                x_col = m.args.get("x") or m.args.get("column1")
                y_col = m.args.get("y") or m.args.get("column2")
                
                if not x_col or not y_col:
                    logs.append(f"[metric] corr: missing column arguments")
                    continue
                    
                x = pd.to_numeric(df[x_col], errors="coerce")
                y = pd.to_numeric(df[y_col], errors="coerce")
                out[m.result_key] = float(x.corr(y))     

            elif m.op in {"argmax_return","argmin_return"}:
    # Handle different argument formats
                by_col = m.args.get("by") or m.args.get("column")
                ret_col = m.args.get("return_col") or "date"  # Default return column
                
                if not by_col:
                    logs.append(f"[metric] {m.op}: missing by/column argument")
                    continue
                    
                by = pd.to_numeric(df[by_col], errors="coerce")
                idx = by.idxmax() if m.op == "argmax_return" else by.idxmin()
                out[m.result_key] = None if (idx is None or (isinstance(idx, float) and pd.isna(idx))) else _ensure_str(df.loc[idx, ret_col])
            elif m.op == "expr":
                expr = m.args.get("expr", "")
                out_type = (m.args.get("type") or "float").lower()
                # safe eval context
                local = {c: pd.to_numeric(df[c], errors="coerce") if pd.api.types.is_numeric_dtype(df[c]) else df[c] for c in df.columns}
                val = pd.eval(expr, engine="python", parser="pandas", target=local)
                if hasattr(val, "item"):
                    val = val.item()
                if out_type == "int":
                    try: val = int(val)
                    except: val = None
                elif out_type == "float":
                    try: val = float(val)
                    except: val = None
                else:
                    val = _ensure_str(val)
                out[m.result_key] = val
            logs.append(f"[metric] {m.op} -> {m.result_key}")
        except Exception as e:
            logs.append(f"[metric] error on {m.result_key}: {e}")
    return out


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

    # CRITICAL FIX: Inject attachments FIRST, then load inputs
    tables = {}
    
    # Start with attachment tables (full data)
    for k, v in attachment_tables.items():
        tables[k] = v.copy() if hasattr(v, 'copy') else v  # Make a copy to avoid modifying original
        logs.append(f"[inputs] attachment loaded: {k} shape={getattr(v,'shape','unknown')}")

    # Then load additional inputs from spec
    spec_tables = load_inputs(spec.inputs, logs)
    
    # Merge spec tables, but don't overwrite attachments unless explicitly intended
    for k, v in spec_tables.items():
        if k not in tables:  # Only add if not already present from attachments
            tables[k] = v
            logs.append(f"[inputs] spec input added: {k} shape={getattr(v,'shape','unknown')}")
        else:
            logs.append(f"[inputs] skipped spec input {k} - attachment data takes precedence")

    if not tables:
        logs.append("[inputs] WARNING: No tables loaded!")
        return {
            "answer": "No data tables found",
            "tables": {},
            "images": {},
            "metrics": {},
            "logs": logs,
        }

    # Apply transforms
    for i, t in enumerate(spec.transforms[:10]):
        logs.append(f"[transform] {i+1}: {t.op} on {t.target}")
        apply_transform(tables, t, logs)

    # Generate charts
    images_by_key = render_charts(tables, spec.charts, logs)

    # Compute metrics
    metrics = compute_metrics(tables, spec.metrics, logs)

    # Pick result table
    res_name = spec.result_table if spec.result_table in tables else (next(iter(tables.keys())) if tables else "result")
    
    if res_name not in tables:
        logs.append(f"[result] WARNING: result_table '{res_name}' not found, using first available")
        res_name = next(iter(tables.keys())) if tables else "empty"
        if res_name == "empty":
            tables["empty"] = pd.DataFrame()

    # Create output tables (preview only)
    tables_out: Dict[str, List[Dict[str, Any]]] = {}
    for name, df in tables.items():
        try:
            # Limit output size but ensure we have the result table
            preview_size = 100 if name == res_name else 50
            tables_out[name] = df_to_records(df.head(preview_size))
            logs.append(f"[output] table {name}: {len(tables_out[name])} rows in output")
        except Exception as e:
            logs.append(f"[output] error converting table {name}: {e}")
            tables_out[name] = []

    # Generate answer
    ans = summarize(spec.answer, tables, logs)

    return {
        "answer": ans,
        "tables": {res_name: tables_out.get(res_name, [])} if res_name in tables_out else tables_out,
        "images": images_by_key,
        "metrics": metrics,
        "logs": logs,
    }
# ---- Parallel API execution with 2 debug tries + fresh fallback -------------
async def analyze_question_async_fixed(question: str,
                                      previews: List[AttachmentPreview],
                                      attachment_tables: Dict[str, Any],
                                      requested_format: Optional[str],
                                      debug: int = 0):
    
    attachments_summary = attachments_summary_for_llm(previews)
    attachments_present = bool(attachment_tables) or any(p.kind == "tabular" for p in previews)
    last_err: Optional[str] = None
    
    # Debug logging
    debug_logs = []
    if debug:
        debug_logs.append(f"Question: {question[:200]}...")
        debug_logs.append(f"Attachments summary: {attachments_summary}")
        debug_logs.append(f"Attachment tables: {list(attachment_tables.keys())}")
        for k, v in attachment_tables.items():
            if hasattr(v, 'shape'):
                debug_logs.append(f"  {k}: shape={v.shape}")
    
    # Attempt normal processing with debug info
    for attempt in range(1, 4):
        debug_hint = None if attempt == 1 else f"Previous error: {last_err}. Use attachment tables: {list(attachment_tables.keys())}"
        
        try:
            # Step 1: Plan
            plan = plan_question(f"{question}\n\nAvailable attachments:\n{attachments_summary}")
            if debug:
                debug_logs.append(f"Plan generated: {plan.dict()}")
            
            # Step 2: Generate spec
            spec = generate_analysis_spec(plan, attachments_summary,attachments_present, debug_hint=debug_hint)
            if debug:
                debug_logs.append(f"Spec generated: {spec.dict()}")
            
            # Step 3: Execute with proper attachment data
            if debug:
                debug_logs.append(f"About to execute with attachments: {list(attachment_tables.keys())}")
            
            final_structured = run_analysis_spec(spec, attachment_tables)
            final_structured = validate_final_result(final_structured)
            
            if debug:
                debug_logs.append(f"Execution completed. Tables in result: {list(final_structured.get('tables', {}).keys())}")
            
            # Format output
            structured_for_llm = strip_payload_for_llm(final_structured)
            fmt_prompt = make_formatter_prompt(question, structured_for_llm, requested_format)
            formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0, max_tokens=500)
            
            if debug == 1:
                return JSONResponse({
                    "ok": True,
                    "formatted_output": formatted,
                    "internal_final_result": final_structured,
                    "debug": {
                        "plan": plan.dict(),
                        "analysis_spec": spec.dict(),
                        "attempt": attempt,
                        "debug_logs": debug_logs,
                        "attachments": [p.dict() for p in previews]
                    }
                })
            
            formatted = inject_images(formatted, final_structured.get("images", {}))
            return Response(formatted, media_type="text/plain")
            
        except Exception as e:
            last_err = str(e)
            if debug:
                debug_logs.append(f"Attempt {attempt} failed: {e}")
    
    # Fallback handling
    try:
        spec = make_fresh_spec_from_attachments(question, previews)
        final_structured = run_analysis_spec(spec, attachment_tables)
        final_structured = validate_final_result(final_structured)
        
        structured_for_llm = strip_payload_for_llm(final_structured)
        fmt_prompt = make_formatter_prompt(question, structured_for_llm, requested_format)
        formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0, max_tokens=500)
        
        if debug == 1:
            return JSONResponse({
                "ok": True,
                "formatted_output": formatted,
                "internal_final_result": final_structured,
                "debug": {
                    "plan": "(fresh-fallback)",
                    "analysis_spec": spec.dict(),
                    "attempt": "fresh_fallback",
                    "previous_error": last_err,
                    "debug_logs": debug_logs,
                    "attachments": [p.dict() for p in previews]
                }
            })
        
        formatted = inject_images(formatted, final_structured.get("images", {}))
        return Response(formatted, media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"All attempts failed. Last error: {last_err} | Fresh fallback error: {e} | Debug logs: {debug_logs}")

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
    return await analyze_question_async_fixed(
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
