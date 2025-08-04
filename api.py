import os
import json
import requests
import tempfile
import subprocess
import re
import textwrap
import base64
from typing import List, Optional

from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

# -------------------- OpenAI Client Configuration --------------------
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDExMjdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Y74CCboue-31JAlwbwCpT-AFlYvinglf636FmsfmLTE"
client = OpenAI(api_key=API_KEY,
                base_url="http://aiproxy.sanand.workers.dev/openai/v1")

# -------------------- Prompt Templates --------------------
PLANNER_SYSTEM = """
You are a master data-analysis planner. I will give you a user's request and any attachments or URLs.
Produce a JSON plan with keys:
- steps: an ordered list of {name, goal, inputs, outputs, needs_code (bool)}
- requirements: (IF APPLICABLE) a list of pip‐installable package names/libraries your code will need 
- final_outputs: list of names of variables for the final report
- respone_type: expected structure of answer/reponse to EXTRACTED from the question itself. (e.g. json,yaml,db etc.)
Only emit valid JSON.
"""

CODER_SYSTEM = """
You are a reliable code generator.

Rules:
- Write ONLY Python code (no prose, no comments, no print/logging).
- Assign EXACTLY the requested output variable names. Do not rename or add others.
- use full library/package name do not use short forms like pandas as pd etc.
- Do not read/write local files unless explicitly requested; fetch via provided helpers.
- Be robust:
    - example:
      - When scraping HTML, handle missing nodes.
      - Handle empty/NaN rows sensibly.
    - Do whatever sensible data cleaning is required to achieve proper results.
- If a library may be missing, prefer pure-Python or built-in alternatives when feasible.
"""

INTERPRETER_SYSTEM = """
You are a data-analysis interpreter. Given the original user request, the plan, and the final output variables, produce a response.

If the plan includes a 'response_type', return the result in that format — structured accordingly (e.g., dict, JSON, YAML, database row, etc. — depending on the specified type). Do not default to Markdown in this case.

If no response_type is specified, return a Markdown report with the following structure:
1. Summary of what was done
2. Key findings (tables, charts, numbers)
3. Conclusions or next steps
4. Caveats if any
"""

# -------------------- Helpers --------------------
def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
RESULT_START = "<<RESULT_JSON_START>>"
RESULT_END   = "<<RESULT_JSON_END>>"

class StepModel(BaseModel):
    name: str = ""
    goal: str = ""
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    needs_code: bool = True


class PlanModel(BaseModel):
    steps: List[StepModel] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    final_outputs: List[str] = Field(default_factory=list)
    response_type: Optional[str] = None


def generate_plan(user_text: str) -> dict:
    plan_json = call_llm(PLANNER_SYSTEM, user_text)
    try:
        try:
            plan = PlanModel.model_validate_json(plan_json)
        except AttributeError:
            plan = PlanModel.parse_raw(plan_json)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Planner produced invalid JSON: {e}\n{plan_json}")
    return plan.model_dump()

def extract_code(code_text: str) -> str:
    m = _CODE_FENCE_RE.search(code_text)
    code = m.group(1) if m else code_text
    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]
    return "\n".join(lines).strip()


def normalize_outputs(outputs):
    if outputs is None:
        return []
    if isinstance(outputs, str):
        return [outputs]
    if isinstance(outputs, dict):
        return list(outputs.keys())
    if isinstance(outputs, list):
        return [x for x in outputs if isinstance(x, str)]
    return []

# -------------------- Core Execution --------------------
def execute_plan(plan: dict) -> dict:
    import sys
    import subprocess
    # 1) install any extra requirements the planner specified
    for pkg in plan.get("requirements", []):
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # 2) execution environment
    env = {"fetch_text": lambda url: requests.get(url, timeout=30).text}
    for step in plan.get("steps", []):
        if not step.get("needs_code", True):
            continue
        name = step.get("name", "")
        outputs = normalize_outputs(step.get("outputs"))
        # Build initial prompt
        prompt_base = (
            f"GOAL: {step.get('goal','')}\n"
            f"INPUTS: {step.get('inputs', [])}\n"
            f"OUTPUTS: {outputs}\n"
        )
        last_error = None
        last_code = ""
        max_retries = 5
        for attempt in range(1, max_retries+1):
            # If error on previous attempt, ask to fix
            prompt = prompt_base
            if last_error:
                prompt += (
                    f"Error on attempt {attempt-1}: {last_error}\n"
                    f"Previous code:\n{last_code}\n"
                    "Please correct the code."
                )
            code = call_llm(CODER_SYSTEM, prompt)
            code = extract_code(code)
            last_code = code
            # Write temp script
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
                tmp.write(textwrap.dedent("""
                    import json, datetime, base64
                    # any imports your generated code needs should be included in the code itself
                    def _json_safe(x, _depth=0):
                        if _depth>5: return str(type(x).__name__)
                        if x is None or isinstance(x,(bool,int,float,str)): return x
                        try:
                            # DataFrame/Series support if pandas was imported by user code
                            import pandas as _pd
                            if isinstance(x, _pd.DataFrame): return x.to_dict(orient='records')
                            if isinstance(x, _pd.Series): return x.tolist()
                        except ImportError:
                            pass
                        try:
                            import numpy as _np
                            if isinstance(x, _np.generic): return x.item()
                            if isinstance(x, _np.ndarray): return x.tolist()
                        except ImportError:
                            pass
                        if isinstance(x, (datetime.datetime, datetime.date)): return x.isoformat()
                        if isinstance(x,(list,tuple,set)):
                            return [_json_safe(v,_depth+1) for v in x]
                        if isinstance(x, dict):
                            return {str(k): _json_safe(v,_depth+1) for k,v in x.items()}
                        try: json.dumps(x); return x
                        except: return str(x)
                        """))                # preload simple env
                for var,val in env.items():
                    if isinstance(val,(str,int,float,bool,list,dict)):
                        tmp.write(f"{var} = {json.dumps(val)}\n")
                tmp.write("\n# USER CODE\n")
                tmp.write(code + "\n")
                # serialize outputs
                tmp.write(f"_req={json.dumps(outputs)}\n")
                tmp.write("out={}\n")
                tmp.write("for v in _req:\n")
                tmp.write("    out[v]=_json_safe(globals().get(v))\n")
                tmp.write(f"print('{RESULT_START}')\n")
                tmp.write("print(json.dumps(out, ensure_ascii=False))\n")
                tmp.write(f"print('{RESULT_END}')\n")
                script=tmp.name
            try:
                res = subprocess.run(["python3", script], capture_output=True, text=True, timeout=90, check=True)
                payload = res.stdout
                # extract between markers
                start = payload.find(RESULT_START)
                end = payload.rfind(RESULT_END)
                if start!=-1 and end!=-1:
                    payload = payload[start+len(RESULT_START):end]
                data = json.loads(payload)
                env.update(data)
                break  # success
            except subprocess.CalledProcessError as err:
                last_error = err.stderr or err.stdout
                if attempt==max_retries:
                    raise HTTPException(status_code=500, detail=f"Step '{name}' failed after {max_retries} attempts. Last error:\n{last_error}")
            except subprocess.TimeoutExpired:
                last_error = "Timeout"
                if attempt==max_retries:
                    raise HTTPException(status_code=504, detail=f"Step '{name}' timed out after {max_retries} attempts.")
            finally:
                try: os.unlink(script)
                except: pass
    return env

# -------------------- Reporting --------------------
def generate_report(original_request: str, plan: dict, env: dict) -> str:
    prompt = (
        f"Original request:\n{original_request}\n\n"
        f"Plan:\n{json.dumps(plan,indent=2)}\n\n"
        f"Outputs vars:\n{list(env.keys())}\n"
    )
    return call_llm(INTERPRETER_SYSTEM, prompt)

# -------------------- FastAPI App --------------------
app = FastAPI()

@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    try:
        text = (await file.read()).decode('utf-8', errors='replace')
    except:
        raise HTTPException(status_code=400, detail="Invalid UTF-8 in uploaded file.")
    plan = generate_plan(text)
    env = execute_plan(plan)
    report = generate_report(text, plan, env)
    return PlainTextResponse(report)

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT",8000)))

