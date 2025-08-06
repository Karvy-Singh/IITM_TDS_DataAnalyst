import os
import re
import json
import asyncio
import tempfile
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError, Field

# ---- LLM client (OpenAI-compatible) -----------------------------------------
from openai import OpenAI

LLM_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDExMjdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.xJoqjnl3dcuRcZfdQ46I7sEH7N--qvq0pprsqunP5P8"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openrouter/v1")

# Defaults tuned for small models; override via env
PLANNER_MODEL    = os.getenv("LLM_PLANNER_MODEL",    "gpt-4o-mini")
CODER_MODEL      = os.getenv("LLM_CODER_MODEL",      "gpt-4o-mini")
FORMATTER_MODEL  = os.getenv("LLM_FORMATTER_MODEL",  "gpt-4o-mini")

if not LLM_API_KEY:
    # /health and / will work; /api will return a clear error.
    pass
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

app = FastAPI(title="Universal Data Analyst (small-model optimized)", version="4.0.0")

# ---- Prompts ----------------------------------------------------------------
# Planner still provides a thin task plan (kept simple for small models)
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

Return JSON only. No commentary, no code, no formatting hints.
"""

# The "coder" no longer writes Python. It returns a STRICT JSON analysis spec.
# The backend executes this DSL safely.
CODER_SYSTEM = """You write ONLY STRICT JSON that describes an analysis plan ("analysis_spec") the backend will execute.
Do NOT write Python. Do NOT include code blocks. Return ONLY JSON.

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
          "table_index": {"type":["integer","null"]}  // used for html tables
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
          "target": {"type":"string"},           // which input/result table to operate on
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
          "bins": {"type":["integer","null"]}   // used for hist
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
    "result_table": {"type":"string"}  // which table to return as main output
  },
  "required": ["inputs","transforms","charts","answer","result_table"],
  "additionalProperties": false
}

Guidelines:
- Prefer simple operations; if unsure, keep transforms minimal.
- Use "filter_query" with pandas-query-compatible strings (e.g., "value > 0 and category == 'A'").
- For HTML pages with tables, set source="html", include "url", and specify "table_index" (0-based).
- If the question doesn't require charts, return an empty array for "charts".
- For "answer.type":
  - "text_summary" for quick textual insight
  - "basic_stats" to compute mean/min/max/count on chosen columns
  - "none" if no summary is needed
Return JSON only. No prose.
"""

# Debugging small models: if JSON invalid, we ask for corrected JSON only.
DEBUGGER_SYSTEM = """You are a precise JSON fixer.
Return ONLY STRICT JSON that validates the provided schema.
No prose. No code fences. No comments."""

# The formatter stays: produce ONE string based on question + final_result.
OUTPUT_FORMATTER_SYSTEM = """You are an output formatter.
Given the original user question and a JSON object `final_result` (with fields answer/tables/images/logs),
produce a SINGLE plain-text string that matches the output requested by the question.
If no format is specified, output a short, clear Markdown report.

Rules:
- Output ONE string only (no fences).
- Use only the data in `final_result`.
- For CSV: header row + comma-separated values.
- For JSON: compact valid JSON.
- For HTML tables: minimal valid HTML.
- For images-only: output data URI(s), one per line.
"""

# ---- Helper code (same capabilities, server-controlled) ---------------------
HELPERS = r"""
import json, base64, io, time, math, statistics, re, datetime
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

def fetch_text(url, timeout=20, retries=2):
    last = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/4.0)"}
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(1)
    raise last

def fetch_json(url, timeout=20):
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
        for t in tables:
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
    fig.savefig(buf, format="png", bbox_inches="tight")
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

def llm_call_raw(system_prompt: str, user_prompt: str, model: str, temperature: float = 0) -> str:
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY/LLM_API_KEY is not set.")
    # Try to request JSON when supported; fall back gracefully
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

def plan_question(question: str) -> Plan:
    out = llm_call_raw(PLANNER_SYSTEM, question, PLANNER_MODEL, temperature=0)
    try:
        data = coerce_json(extract_payload(out))
        return Plan(**data)
    except Exception:
        # Re-ask with stronger instruction
        question2 = question + "\n\nReturn VALID JSON only. No commentary."
        out2 = llm_call_raw(PLANNER_SYSTEM, question2, PLANNER_MODEL, temperature=0)
        data2 = coerce_json(extract_payload(out2))
        return Plan(**data2)

# ---- Analysis Spec generation (best-of-N) -----------------------------------
SPEC_SCHEMA_HINT = """Strict schema reminder:
- inputs: array[{name, source(html|csv|json|inline), url|null, data|null, table_index|null}]
- transforms: array[{target, op(select_columns|rename|dropna|head|sort_values|filter_query|groupby_agg|join|add_column|parse_dates), args{...}}]
- charts: array[{table, kind(line|bar|scatter|hist), x|null, y|null|array, title|null, bins|null}]
- answer: {type(text_summary|basic_stats|none), table|null, columns|null}
- result_table: string
Return ONLY JSON. No code, no prose.
"""

def make_coder_prompt(plan: Plan) -> str:
    return (
        f"Original question:\n{plan.question}\n\n"
        f"High-level plan (JSON):\n{plan.model_dump_json(indent=2)}\n\n"
        f"{SPEC_SCHEMA_HINT}\n"
        "Produce the analysis_spec now."
    )

def generate_spec_candidates(plan: Plan, n: int = 2) -> List[AnalysisSpec]:
    specs: List[AnalysisSpec] = []
    for i in range(n):
        # 1) first pass with CODER_SYSTEM
        raw = llm_call_raw(
            CODER_SYSTEM,
            make_coder_prompt(plan),
            CODER_MODEL,
            temperature=0.0 if i == 0 else 0.2
        )

        try:
            # try to parse the raw response
            data = coerce_json(extract_payload(raw))
            specs.append(AnalysisSpec(**data))
        except Exception:
            if i < n - 1:
                # 2a) for all but the last try, run the one-shot fixer
                fixer_input = (
                    "Fix this to match the schema. Return JSON only.\n\n"
                    f"---\n{raw}\n---\n"
                )
                fixed = llm_call_raw(DEBUGGER_SYSTEM, fixer_input, CODER_MODEL, temperature=0)
                try:
                    data2 = coerce_json(extract_payload(fixed))
                    specs.append(AnalysisSpec(**data2))
                except Exception:
                    # still bad? move on to next iteration
                    continue
            else:
                # 2b) on the last retry, do a fresh CODER_SYSTEM call instead of debugging
                raw_retry = llm_call_raw(
                    CODER_SYSTEM,
                    make_coder_prompt(plan),
                    CODER_MODEL,
                    temperature=0.2
                )
                try:
                    data3 = coerce_json(extract_payload(raw_retry))
                    specs.append(AnalysisSpec(**data3))
                except Exception:
                    # give up on this slot
                    continue

    return specs

# ---- Executor for the JSON DSL ----------------------------------------------
# Import helper functions into this module's namespace
exec(HELPERS, globals(), globals())  # safe: static, server-controlled

ALLOWED_OPS = {
    "select_columns","rename","dropna","head","sort_values",
    "filter_query","groupby_agg","join","add_column","parse_dates"
}

def load_inputs(inputs: List[AnalysisInput], logs: List[str]) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    for inp in inputs:
        if inp.source == "html":
            if not inp.url:
                logs.append(f"[inputs] {inp.name}: missing url for html")
                continue
            html = fetch_text(inp.url)
            dfs = read_table_html(html)
            if not dfs:
                logs.append(f"[inputs] {inp.name}: no tables found")
                continue
            idx = inp.table_index or 0
            if idx < 0 or idx >= len(dfs):
                idx = 0
            tables[inp.name] = dfs[idx]
            logs.append(f"[inputs] {inp.name}: html table index {idx} shape={dfs[idx].shape}")
        elif inp.source == "csv":
            if inp.url:
                text = fetch_text(inp.url)
                import pandas as pd
                from io import StringIO
                tables[inp.name] = pd.read_csv(StringIO(text))
                logs.append(f"[inputs] {inp.name}: csv loaded shape={tables[inp.name].shape}")
            elif isinstance(inp.data, str):
                from io import StringIO
                import pandas as pd
                tables[inp.name] = pd.read_csv(StringIO(inp.data))
                logs.append(f"[inputs] {inp.name}: inline csv shape={tables[inp.name].shape}")
        elif inp.source == "json":
            if inp.url:
                js = fetch_json(inp.url)
                import pandas as pd
                tables[inp.name] = pd.json_normalize(js)
                logs.append(f"[inputs] {inp.name}: json loaded shape={tables[inp.name].shape}")
            else:
                import pandas as pd
                tables[inp.name] = pd.json_normalize(inp.data or {})
                logs.append(f"[inputs] {inp.name}: inline json shape={tables[inp.name].shape}")
        elif inp.source == "inline":
            # allow pre-constructed table (list of dicts)
            import pandas as pd
            try:
                tables[inp.name] = pd.DataFrame(inp.data or [])
                logs.append(f"[inputs] {inp.name}: inline records shape={tables[inp.name].shape}")
            except Exception as e:
                logs.append(f"[inputs] {inp.name}: inline parse error {e}")
        else:
            logs.append(f"[inputs] {inp.name}: unsupported source {inp.source}")
    return tables

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
            n = int(t.args.get("n", 10))
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
            expr = t.args.get("expr")  # simple expr using columns, e.g., "colA + colB"
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
    images: List[str] = []
    for ch in charts:
        if ch.table not in df_map:
            logs.append(f"[chart] missing table {ch.table}")
            continue
        df = df_map[ch.table]
        try:
            import matplotlib.pyplot as plt
            fig = None
            if ch.kind == "line":
                fig = plt.figure()
                if isinstance(ch.y, list):
                    for col in ch.y:
                        plt.plot(df[ch.x], df[col], label=col)
                else:
                    plt.plot(df[ch.x], df[ch.y])
                if ch.title: plt.title(ch.title)
                if isinstance(ch.y, list): plt.legend()
            elif ch.kind == "bar":
                fig = plt.figure()
                plt.bar(df[ch.x], df[ch.y] if isinstance(ch.y, str) else df[ch.y[0]])
                if ch.title: plt.title(ch.title)
            elif ch.kind == "scatter":
                fig = plt.figure()
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                plt.scatter(df[ch.x], df[ycol])
                if ch.title: plt.title(ch.title)
            elif ch.kind == "hist":
                fig = plt.figure()
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                bins = ch.bins or 30
                plt.hist(df[ycol], bins=bins)
                if ch.title: plt.title(ch.title)
            if fig is not None:
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
        cols = answer.columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
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
        logs.append(f"[answer] basic_stats on {tname} cols={cols}")
        return out
    if answer.type == "text_summary":
        # Simple heuristic summary
        tname = answer.table or next(iter(df_map.keys()), None)
        if not tname or tname not in df_map:
            return "No data available."
        df = df_map[tname]
        return f"Rows: {len(df)}, Columns: {len(df.columns)}. Preview columns: {list(df.columns)[:6]}"
    return None

def validate_final_result(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object.")
    missing = REQUIRED_FINAL_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Output missing required keys: {sorted(missing)}")
    if not isinstance(data["tables"], dict):
        raise ValueError("final_result.tables must be a dict of name -> list[dict].")
    if not isinstance(data["images"], list):
        raise ValueError("final_result.images must be a list.")
    if not isinstance(data["logs"], list):
        raise ValueError("final_result.logs must be a list.")
    return data

def make_formatter_prompt(question: str, final_result: Dict[str, Any]) -> str:
    payload = {
        "question": question,
        "final_result": final_result,
    }
    return json.dumps(payload, indent=2)

# ---- Pipeline: plan -> spec -> execute -> format ----------------------------
def run_analysis_spec(spec: AnalysisSpec) -> Dict[str, Any]:
    logs: List[str] = []
    tables = load_inputs(spec.inputs, logs)
    # Apply transforms in order
    for t in spec.transforms:
        apply_transform(tables, t, logs)
    # Build images
    images = render_charts(tables, spec.charts, logs)
    # Choose result table
    res_name = spec.result_table
    if res_name not in tables and tables:
        res_name = next(iter(tables.keys()))
    # Prepare tables output (preview trims)
    tables_out: Dict[str, List[Dict[str, Any]]] = {}
    for name, df in tables.items():
        try:
            preview = df.head(500)  # cap to keep payloads reasonable
            tables_out[name] = df_to_records(preview)
        except Exception:
            tables_out[name] = []
    # Answer
    ans = summarize(spec.answer, tables, logs)
    final_result = {
        "answer": ans,
        "tables": {res_name: tables_out.get(res_name, [])},
        "images": images,
        "logs": logs,
    }
    return final_result

# ---- API --------------------------------------------------------------------
@app.post("/api/")
async def analyze_question(
    file: UploadFile = File(...),
    debug: int = Query(0, description="Set to 1 to include debug payloads (JSON)."),
):
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY/LLM_API_KEY is not set.")

    content = await file.read()
    question = content.decode("utf-8").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question file")

    # 1) Plan
    try:
        plan = plan_question(question)
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"Planner JSON invalid: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planner failed: {e}")

    # 2) Generate analysis spec (best-of-N for small models)
    candidates = generate_spec_candidates(plan, n=3)
    if not candidates:
        raise HTTPException(status_code=500, detail="Could not generate a valid analysis spec.")

    # 3) Execute the first viable spec
    final_structured: Optional[Dict[str, Any]] = None
    chosen_spec = None
    errors: List[str] = []
    for spec in candidates:
        try:
            fr = run_analysis_spec(spec)
            # Sanity checks (cheap correctness filters)
            if (fr.get("tables") or fr.get("images") or fr.get("answer") is not None):
                final_structured = validate_final_result(fr)
                chosen_spec = spec
                break
        except Exception as e:
            errors.append(str(e))
            continue

    if final_structured is None:
        if debug == 1:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "Execution failed for all spec candidates.",
                    "errors": errors,
                    "plan": plan.model_dump(),
                },
            )
        return Response(
            "Execution failed for all spec candidates.\n" + "\n".join(errors),
            media_type="text/plain",
            status_code=500,
        )

    # 4) Format the outgoing payload based on the question
    fmt_prompt = make_formatter_prompt(question, final_structured)
    formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0)

    if debug == 1:
        return JSONResponse(
            {
                "ok": True,
                "formatted_output": formatted,
                "internal_final_result": final_structured,
                "debug": {
                    "plan": plan.model_dump(),
                    "analysis_spec": chosen_spec.model_dump() if chosen_spec else None,
                    "models": {
                        "planner": PLANNER_MODEL,
                        "coder": CODER_MODEL,
                        "formatter": FORMATTER_MODEL,
                        "base_url": LLM_BASE_URL,
                    },
                },
            }
        )

    # Default to plain text so clients can render/parse as they need.
    return Response(formatted, media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Universal Data Analyst API v4 (small-model optimized: JSON DSL executor)",
        "usage": "curl -s -X POST 'http://localhost:8000/api/?debug=1' -F 'file=@question.txt'",
        "notes": [
            "No free-form Python generation. The model emits a strict JSON analysis spec.",
            "Executor performs IO, transforms, charts safely with Pandas/Matplotlib.",
            "Best-of-N spec generation for robustness with small models.",
            "Formatter infers outward format from the question and final_result."
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

