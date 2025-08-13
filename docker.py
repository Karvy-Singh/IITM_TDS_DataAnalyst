import os
import re
import json
import asyncio
import tempfile
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
import base64
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

# ---- LLM client (OpenAI-compatible, async) ----------------------------------
from openai import AsyncOpenAI

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = "https://api.openai.com/v1"
PLANNER_MODEL ="gpt-4o-mini"
CODER_MODEL = "gpt-4o-mini"
DEBUGGER_MODEL = "gpt-4o-mini"
INTERPRETER_MODEL = "gpt-4o-mini"
FORMATTER_MODEL = "gpt-4o-mini"

if not LLM_API_KEY:
    # You can still run /health and / endpoints; /api/ will error clearly.
    pass

client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

app = FastAPI(title="Universal Data Analyst (no explicit format/datasets)", version="3.0.0")

# ---- Prompts ----------------------------------------------------------------
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
Do NOT include any output format hints (e.g., json, yaml, html, markdown) and do NOT list datasets/URLs.
Return JSON only. No prose.
"""

CODER_SYSTEM = """Write ONLY Python code. No comments. Do not print anything except one final print(json.dumps(final_result)).
Use ONLY the provided helpers for network I/O and visualization.

Local files do not exist. Use these helpers for uploads:
- list_attachments()
- read_text_attachment(name, encoding="utf-8")
- read_json_attachment(name)
- read_csv_attachment(name, **pandas_read_csv_kwargs)
- read_excel_attachment(name, **pandas_kwargs)
- read_parquet_attachment(name, **pandas_kwargs)  # requires pyarrow/fastparquet in env
- read_image_attachment(name)  # returns PIL.Image.Image if Pillow available, else numpy array
- load_dataframe_auto(name)  # infers by extension
- is_image_name(name) -> bool

Helpers available (already imported in the runtime):
- fetch_text(url: str, timeout: int = 20, retries: int = 2) -> str
- fetch_json(url: str, timeout: int = 20) -> dict
- read_table_html(html: str) -> list[pandas.DataFrame]
- df_to_records(df: pandas.DataFrame) -> list[dict]
- fig_to_data_uri(fig) -> str  # returns a data:image/png;base64,... string

Allowed imports: json, math, statistics, re, datetime, io, base64, pandas as pd, numpy as np, matplotlib.pyplot as plt.

Contract (internal only):
- Compute results with Python (not the LLM).
- If you make charts, convert them with fig_to_data_uri(fig).
- Produce: final_result = {
    "answer": <any JSON-serializable value>,
    "tables": { "<name>": [ {<record>}, ... ] },
    "images": [ "data:image/png;base64,...", ... ],
    "logs": [ "string messages with dataset shapes, sanity checks, etc." ]
  }
- End with: print(json.dumps(final_result))
"""

DEBUGGER_SYSTEM = """You are a precise code fixer.
Given:
- A plan (JSON)
- The previous code
- Its stderr and stdout
- The required output contract (final_result as JSON printed once)
Return ONLY corrected Python code that:
- Uses the provided helpers for network I/O
- Avoids disallowed imports
- Prints exactly one line: json.dumps(final_result)
- Produces valid JSON per the contract
No commentary. Code only.
"""

OUTPUT_FORMATTER_SYSTEM = """You are an output formatter.
Goal: Given the original user question and a JSON object `final_result` (with fields answer/tables/images/logs),
produce a SINGLE plain-text string that matches the output/reponse format explicitly requested by the question.
If the question requests a very specific format (e.g., "CSV with columns ...", "a single data URI image", "raw JSON", "HTML table"),
produce exactly that format. Strictly follow column order and field names if provided.
If the question does NOT specify any particular format, produce a short, clear Markdown report.
Rules:
- Output ONE string only. No surrounding code fences. No explanations.
- Do not invent data; use the provided `final_result`.
- For CSV: include a header row. Use commas, newline-separated rows.
- For JSON: output compact valid JSON (minimize whitespace).
- For HTML table(s): output minimal valid HTML (no external CSS/JS).
- For images-only requests: output the data URI(s), one per line, nothing else.
"""

# ---- Helper code injected into the execution sandbox ------------------------
HELPERS = r"""
import json, base64, io, time, math, statistics, re, datetime
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
def fetch_text(url, timeout=20, retries=2):
    last = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/3.0)"}
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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/3.0)"}
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

# ---- Utilities ---------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)

async def extract_code(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    code = m.group(1) if m else text
    return code.strip()

async def coerce_json(text: str) -> Dict[str, Any]:
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

async def llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 1,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM_API_KEY/OPENAI_API_KEY is not set.")

    if attachments:
        attach_block = json.dumps(
            [{"filename": a.get("filename"),
              "content_type": a.get("content_type"),
              "size": a.get("size"),
              "data_b64": a.get("data_b64")} for a in attachments],
            separators=(",", ":")
        )
        user_content = f"{user_prompt}\n\n[ATTACHMENTS_JSON]\n{attach_block}"
    else:
        user_content = user_prompt

    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()

async def plan_question(question: str) -> Plan:
    out = await llm_call(PLANNER_SYSTEM, question, PLANNER_MODEL)
    try:
        data = await coerce_json(out)
        return Plan(**data)
    except Exception:
        question2 = question + "\n\nReturn VALID JSON only. No commentary."
        out2 = await llm_call(PLANNER_SYSTEM, question2, PLANNER_MODEL)
        data2 = await coerce_json(out2)
        return Plan(**data2)

DANGEROUS_IMPORTS = {
    "subprocess","os","sys","shutil","socket","pathlib","multiprocessing",
    "asyncio","http.server","flask","fastapi","openai","shlex","pexpect","uvicorn"
}

async def contains_dangerous_imports(code: str) -> Optional[str]:
    lines = code.splitlines()
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            for bad in DANGEROUS_IMPORTS:
                if re.search(rf"\b{re.escape(bad)}\b", s):
                    return f"Line {i}: disallowed import: {s}"
    return None

async def run_python_code(code: str, timeout: int = 120) -> Tuple[str, str, int]:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        proc = await asyncio.create_subprocess_exec(
            sys.executable, temp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode(), stderr.decode(), proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "", "Process timed out", -1
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    except Exception as e:
        return "", f"Execution error: {str(e)}", -1

async def validate_final_result(stdout_text: str) -> Dict[str, Any]:
    data = await coerce_json(stdout_text)
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

async def make_coder_prompt(plan: Plan) -> str:
    schema_hint = """Internal final_result schema (for reliability):
{
  "answer": <any JSON-serializable>,
  "tables": { "<name>": [ {<record>}, ... ] },
  "images": [ "data:image/png;base64,...", ... ],
  "logs": [ "string", ... ]
}
"""
    return (
        f"Original question:\n{plan.question}\n\n"
        f"Plan JSON (no output format hints, no datasets):\n{plan.json(indent=2)}\n\n"
        f"{schema_hint}\n"
        "Write the analysis code now."
    )

async def make_debugger_prompt(plan: Plan, prev_code: str, stdout_text: str, stderr_text: str) -> str:
    s_out = stdout_text[-2000:]
    s_err = stderr_text[-2000:]
    s_code = prev_code[-4000:]
    return (
        "PLAN (JSON):\n"
        + plan.json(indent=2)
        + "\n\nPREVIOUS CODE:\n"
        + s_code
        + "\n\nSTDOUT (tail):\n"
        + s_out
        + "\n\nSTDERR (tail):\n"
        + s_err
        + "\n\nPlease return corrected Python code only."
    )

async def build_executable_source(generated_code: str, attachments: List[Dict[str, Any]]) -> str:
    attachments_py_json = json.dumps(attachments, separators=(",", ":"))
    prelude = f"""
ATTACHMENTS = {attachments_py_json}

import base64, io, json, pandas as pd, numpy as np

try:
    import pyarrow  # noqa: F401
    _HAS_ARROW = True
except Exception:
    _HAS_ARROW = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def _get_bytes(name):
    for a in ATTACHMENTS:
        if a["filename"] == name:
            return base64.b64decode(a["data_b64"])
    raise FileNotFoundError(name)

def list_attachments():
    return [{{"filename": a["filename"], "content_type": a["content_type"], "size": a["size"]}} for a in ATTACHMENTS]

def read_text_attachment(name, encoding="utf-8"):
    return _get_bytes(name).decode(encoding, errors="replace")

def read_json_attachment(name):
    return json.loads(read_text_attachment(name))

def read_csv_attachment(name, **kwargs):
    return pd.read_csv(io.BytesIO(_get_bytes(name)), **kwargs)

def read_excel_attachment(name, **kwargs):
    return pd.read_excel(io.BytesIO(_get_bytes(name)), **kwargs)

def read_parquet_attachment(name, **kwargs):
    if not _HAS_ARROW:
        raise RuntimeError("Parquet requires pyarrow or fastparquet.")
    return pd.read_parquet(io.BytesIO(_get_bytes(name)), **kwargs)

def read_image_attachment(name):
    b = _get_bytes(name)
    if _HAS_PIL:
        return Image.open(io.BytesIO(b))
    import matplotlib.image as mpimg
    return mpimg.imread(io.BytesIO(b))

def load_dataframe_auto(name):
    low = name.lower()
    if low.endswith(".csv"):
        return read_csv_attachment(name)
    if low.endswith(".json"):
        js = read_json_attachment(name)
        if isinstance(js, list):
            return pd.DataFrame(js)
        if isinstance(js, dict):
            return pd.json_normalize(js)
        raise ValueError("JSON not table-like")
    if low.endswith(".xlsx") or low.endswith(".xls"):
        return read_excel_attachment(name)
    if low.endswith(".parquet") or low.endswith(".pq"):
        return read_parquet_attachment(name)
    try:
        return read_csv_attachment(name)
    except Exception:
        try:
            js = read_json_attachment(name)
            return pd.json_normalize(js) if isinstance(js, dict) else pd.DataFrame(js)
        except Exception:
            raise ValueError(f"Unsupported tabular file: {{name}}")

def is_image_name(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))
"""
    return HELPERS + "\n" + prelude + "\n" + generated_code + "\n"

async def make_formatter_prompt(question: str, final_result: Dict[str, Any]) -> str:
    payload = {
        "question": question,
        "final_result": final_result,
    }
    return json.dumps(payload, indent=2)

# ---- API --------------------------------------------------------------------
@app.post("/api/")
async def analyze_question(
    request: Request,
    debug: int = Query(0, description="Set to 1 to include debug payloads (JSON)."),
):
    print("analyze_question: got a request")
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM_API_KEY/OPENAI_API_KEY is not set.")

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

    async def read_bytes(f: UploadFile) -> bytes:
        return await f.read()

    async def is_text_like(upload: UploadFile, data: bytes) -> bool:
        ct = (getattr(upload, "content_type", "") or "").lower()
        if ct.startswith("text/") or ct in {"application/json", "application/xml"}:
            return True
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    file_blobs: List[Dict[str, Any]] = []
    first_text_payload: Optional[str] = None
    first_text_filename: Optional[str] = None

    for f in files:
        blob = await read_bytes(f)

        file_blobs.append({
            "filename": f.filename,
            "content_type": (getattr(f, "content_type", "") or "application/octet-stream"),
            "data_b64": base64.b64encode(blob).decode("ascii"),
            "size": len(blob),
        })

        if first_text_payload is None and await is_text_like(f, blob):
            try:
                first_text_payload = blob.decode("utf-8").strip()
                first_text_filename = f.filename
            except UnicodeDecodeError:
                pass

    if not first_text_payload:
        if debug == 1:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "No text-like file found for the question",
                    "observed_files": [
                        {"filename": a["filename"], "content_type": a["content_type"], "size": a["size"]}
                        for a in file_blobs
                    ],
                    "observed_form_keys": seen_keys,
                    "hint": "Upload at least one UTF-8 text file (e.g., .txt, .md, .json).",
                },
            )
        raise HTTPException(
            status_code=400,
            detail="No text-like file found for the question. Upload at least one UTF-8 text file (e.g., .txt, .md, JSON).",
        )

    question = first_text_payload

    # === 1) Plan ===
    try:
        plan = await plan_question(question)
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"Planner JSON invalid: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planner failed: {e}")

    # === 2) Generate code ===
    coder_prompt = await make_coder_prompt(plan)
    raw_code = await llm_call(CODER_SYSTEM, coder_prompt, CODER_MODEL, attachments=file_blobs)
    generated_code = await extract_code(raw_code)

    bad = await contains_dangerous_imports(generated_code)
    if bad:
        dbg_prompt = await make_debugger_prompt(plan, generated_code, "", bad)
        raw_fixed = await llm_call(DEBUGGER_SYSTEM, dbg_prompt, DEBUGGER_MODEL, attachments=file_blobs)
        generated_code = await extract_code(raw_fixed)

    # === 3) Execute with retries + debugger (with async LLM calls) ===
    MAX_RETRIES = 4
    debug_attempts = 0
    last_stdout = ""
    last_stderr = ""
    last_code = generated_code
    final_structured: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_RETRIES + 1):
        print("starting attempting...")
        to_run = await build_executable_source(last_code, file_blobs)
        stdout, stderr, returncode = await run_python_code(to_run)
        last_stdout, last_stderr = stdout, stderr

        if returncode == 0:
            try:
                final_structured = await validate_final_result(stdout.strip())
                print("successful first try...")
                break
            except Exception as ve:
                if debug_attempts < 2:
                    dbg_prompt = await make_debugger_prompt(plan, last_code, stdout, f"ValidationError: {ve}")
                    raw_fixed = await llm_call(DEBUGGER_SYSTEM, dbg_prompt, DEBUGGER_MODEL, attachments=file_blobs)
                    last_code = await extract_code(raw_fixed)
                    debug_attempts += 1
                    continue
                else:
                    fresh_raw = await llm_call(CODER_SYSTEM, coder_prompt, CODER_MODEL, attachments=file_blobs)
                    fresh_code = await extract_code(fresh_raw)
                    bad2 = await contains_dangerous_imports(fresh_code)
                    if bad2:
                        fix_prompt = await make_debugger_prompt(plan, fresh_code, "", bad2)
                        fixed_fresh = await llm_call(DEBUGGER_SYSTEM, fix_prompt, DEBUGGER_MODEL, attachments=file_blobs)
                        fresh_code = await extract_code(fixed_fresh)
                    last_code = fresh_code
                    continue
        else:
            print("else try...")
            if debug_attempts < 2:
                dbg_prompt = await make_debugger_prompt(plan, last_code, stdout, stderr)
                raw_fixed = await llm_call(DEBUGGER_SYSTEM, dbg_prompt, DEBUGGER_MODEL, attachments=file_blobs)
                last_code = await extract_code(raw_fixed)
                debug_attempts += 1
                continue
            else:
                fresh_raw = await llm_call(CODER_SYSTEM, coder_prompt, CODER_MODEL, attachments=file_blobs)
                fresh_code = await extract_code(fresh_raw)
                bad2 = await contains_dangerous_imports(fresh_code)
                if bad2:
                    fix_prompt = await make_debugger_prompt(plan, fresh_code, "", bad2)
                    fixed_fresh = await llm_call(DEBUGGER_SYSTEM, fix_prompt, DEBUGGER_MODEL, attachments=file_blobs)
                    fresh_code = await extract_code(fixed_fresh)
                last_code = fresh_code
                continue

    if final_structured is None:
        if debug == 1:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "Execution failed after retries.",
                    "stderr_tail": last_stderr[-1000:],
                    "stdout_tail": last_stdout[-1000:],
                    "plan": plan.model_dump(),
                    "generated_code": last_code,
                    "question_file": first_text_filename,
                    "attachments": [
                        {"filename": a["filename"], "content_type": a["content_type"], "size": a["size"]}
                        for a in file_blobs
                    ],
                },
            )
        return Response(
            "Execution failed after retries.\n"
            f"STDERR (tail):\n{last_stderr[-1000:]}\n\nSTDOUT (tail):\n{last_stdout[-1000:]}\n",
            media_type="text/plain",
            status_code=500,
        )

    # === 4) Format ===
    fmt_prompt = await make_formatter_prompt(question, final_structured)
    formatted = await llm_call(OUTPUT_FORMATTER_SYSTEM, fmt_prompt, FORMATTER_MODEL, attachments=file_blobs)

    if debug == 1:
        return JSONResponse(
            {
                "ok": True,
                "formatted_output": formatted,
                "internal_final_result": final_structured,
                "debug": {
                    "plan": plan.model_dump(),
                    "generated_code": last_code,
                    "stdout_tail": last_stdout[-2000:],
                    "stderr_tail": last_stderr[-2000:],
                    "question_file": first_text_filename,
                    "observed_form_keys": seen_keys,
                    "attachments": [
                        {"filename": a["filename"], "content_type": a["content_type"], "size": a["size"]}
                        for a in file_blobs
                    ],
                    "models": {
                        "planner": PLANNER_MODEL,
                        "coder": CODER_MODEL,
                        "debugger": DEBUGGER_MODEL,
                        "formatter": FORMATTER_MODEL,
                        "base_url": LLM_BASE_URL,
                    },
                },
            }
        )

    return Response(formatted, media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Universal Data Analyst API v3 (no explicit output type / no datasets in plan)",
        "usage": "curl -s -X POST 'http://localhost:8000/api/?debug=1' -F 'file=@question.txt'",
        "notes": [
            "The planner never specifies output format or datasets.",
            "The coder produces a reliable JSON object internally.",
            "The formatter infers the outward format from the question and returns plain text."
        ]
    }
