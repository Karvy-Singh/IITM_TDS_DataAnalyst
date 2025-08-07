import os
import re
import json
import asyncio
import tempfile
import sys
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

app = FastAPI(title="Universal Data Analyst (speed-optimized)", version="4.1.0")

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

# Simplified formatter for speed
OUTPUT_FORMATTER_SYSTEM = """Format the result concisely. Return ONE string only.
For CSV: header+rows. For JSON: compact. For markdown: brief summary.
No extra formatting or explanations."""

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
def make_coder_prompt(plan: Plan) -> str:
    return (
        f"Question: {plan.question}\n"
        f"Plan: {plan.model_dump_json()}\n"
        "Return analysis_spec JSON. Be minimal and efficient."
    )

# Single-shot spec generation (no best-of-N for speed)
def generate_analysis_spec(plan: Plan) -> AnalysisSpec:
    raw = llm_call_raw(
        CODER_SYSTEM,
        make_coder_prompt(plan),
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
                ax.bar(df[ch.x], df[ch.y] if isinstance(ch.y, str) else df[ch.y[0]])
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

def make_formatter_prompt(question: str, final_result: Dict[str, Any]) -> str:
    # Simplified prompt for faster processing
    return f"Question: {question}\nData: {json.dumps(final_result, default=str)[:2000]}"  # Truncate

# ---- Optimized pipeline execution -------------------------------------------
def run_analysis_spec(spec: AnalysisSpec) -> Dict[str, Any]:
    logs: List[str] = []
    
    # Load inputs (with parallel fetching)
    tables = load_inputs(spec.inputs, logs)
    
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

# ---- Parallel API execution -------------------------------------------------
async def analyze_question_async(question: str, debug: int = 0):
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY/LLM_API_KEY is not set.")

    # Step 1: Plan (fast)
    try:
        plan = plan_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planner failed: {e}")

    # Step 2: Generate spec (single attempt)
    try:
        spec = generate_analysis_spec(plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spec generation failed: {e}")

    # Step 3: Execute spec
    try:
        final_structured = run_analysis_spec(spec)
        final_structured = validate_final_result(final_structured)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    # Step 4: Format output (with timeout)
    try:
        fmt_prompt = make_formatter_prompt(question, final_structured)
        formatted = llm_call_raw(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, temperature=0, max_tokens=500)
    except Exception:
        # Fallback to basic formatting
        formatted = f"Analysis complete. Found {len(final_structured.get('tables', {}))} tables, {len(final_structured.get('images', []))} charts."

    if debug == 1:
        return JSONResponse({
            "ok": True,
            "formatted_output": formatted,
            "internal_final_result": final_structured,
            "debug": {
                "plan": plan.model_dump(),
                "analysis_spec": spec.model_dump(),
                "performance_mode": "speed_optimized"
            }
        })

    return Response(formatted, media_type="text/plain")

# ---- API endpoints -----------------------------------------------------------
@app.post("/api/")
async def analyze_question(
    file: UploadFile = File(...),
    debug: int = Query(0, description="Set to 1 to include debug payloads (JSON)."),
):
    content = await file.read()
    question = content.decode("utf-8").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question file")
    
    return await analyze_question_async(question, debug)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "4.1.0-speed-optimized"}

@app.get("/")
async def root():
    return {
        "message": "Universal Data Analyst API v4.1 (Speed Optimized)",
        "usage": "curl -s -X POST 'http://localhost:8000/api/?debug=1' -F 'file=@question.txt'",
        "optimizations": [
            "Reduced network timeouts (8s vs 20s)",
            "Parallel input fetching",
            "Single-shot spec generation (no best-of-N)",
            "Limited data processing (1000 rows max per input)",
            "Optimized chart rendering (lower DPI, smaller figures)",
            "Truncated LLM responses (max_tokens limits)",
            "Async processing pipeline"
        ],
        "target_performance": "< 3 minutes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
