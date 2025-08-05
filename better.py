from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import google.generativeai as genai
import subprocess
import sys
import tempfile
import os
import json
import asyncio
import re
from typing import Optional
from openai import OpenAI

API_KEY = ""
client = OpenAI(api_key=API_KEY,
                base_url="https://aipipe.org/openrouter/v1/")

app = FastAPI(title="Universal Data Analyst", version="1.0.0")

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

def extract_code(code_text: str) -> str:
    m = _CODE_FENCE_RE.search(code_text)
    code = m.group(1) if m else code_text
    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]
    return "\n".join(lines).strip()

# Prompt Templates
PLANNER_SYSTEM = """Role: You are an AI assistant that specializes in breaking down complex questions into programmable Python code and extracting parameters from the question for implementation.

Task: Given a specific question regarding data scraping OR analysis, create a plan that accomplishes the following:
1. Scrapes data from the provided URL (IF APPLICABLE)
2. Derives parameters necessary for the analysis from the question
3. Outputs answers to the specified sub-questions in a structured format

Output Format:
- The response should include:
- Clearly defined variables for parameters extracted from the question
- The expected output type as extracted from the question itself (e.g., JSON, image data URI, yaml, OR anything else possible)
- Variable names that should be assigned in the final code

Tone: Formal and technical."""

LIBRARY_PLANNER = """Role: You are a senior software developer.

Task: You will receive a plan, after analyzing, you will give out all proper python library names for pip installing them, which will be required to execute the plan, in a JSON format with field called "requirements"."""

CODER_SYSTEM = """You are a reliable code generator.

Rules:
- Write ONLY Python code (no prose, no comments, no print/logging statements)
- The code MUST print ONLY the final result to stdout using print() at the end
- Assign variables as needed but ensure the final result is printed
- Do not read/write local files unless explicitly requested; fetch via provided helpers
- Be robust:
  - When scraping HTML, handle missing nodes
  - Handle empty/NaN rows sensibly
  - Do whatever sensible data cleaning is required to achieve proper results
- If a library may be missing, prefer pure-Python or built-in alternatives when feasible
- The final line of code should always be: print(final_result)"""

INTERPRETER_SYSTEM = """
You are a data-analysis interpreter. Given the original user request, the plan, and the final output variables, produce a response in structure/expected output type as specified by the plan.

If no response_type is specified, return a Markdown report with the following structure:
1. Summary of what was done
2. Key findings (tables, charts, numbers)
3. Conclusions or next steps
4. Caveats if any
"""
DEBUG_SYSTEM ="""
Role: You are an intelligent debugger.
You need to debug/modify/update code in any way possible to get the original results as intended by the code with errors.
"""

def install_libraries(requirements: list) -> bool:
    """Install required Python libraries"""
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return False
    return True

async def run_python_code(code: str, timeout: int = 300) -> tuple[str, str, int]:
    """Run Python code as subprocess and return stdout, stderr, returncode"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        process = await asyncio.create_subprocess_exec(
            sys.executable, temp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return stdout.decode(), stderr.decode(), process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "", "Process timed out", -1
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        return "", f"Execution error: {str(e)}", -1

@app.post("/api/", response_class=PlainTextResponse)
async def analyze_question(file: UploadFile = File(...)):
    """Main endpoint that accepts question file and returns analysis result"""
    try:
        # Read the question from uploaded file
        content = await file.read()
        question = content.decode('utf-8').strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Empty question file")
        
        # Step 1: Create analysis plan
        plan = call_llm(PLANNER_SYSTEM, question)
        
        # Step 2: Determine required libraries
        library_response = call_llm(LIBRARY_PLANNER, plan)
        
        try:
            library_data = json.loads(library_response)
            requirements = library_data.get("requirements", [])
        except json.JSONDecodeError:
            # Fallback common libraries
            requirements = ["requests", "beautifulsoup4", "pandas", "numpy", "matplotlib", "seaborn"]
        
        # Step 3: Install libraries
        install_success = install_libraries(requirements)
        if not install_success:
            return "Error: Failed to install required libraries"
        
        # Step 4: Generate code with retry logic
        code_prompt = f"Plan: {plan}\n\nGenerate Python code to execute this plan. Remember to print the final result."
        generated_code = call_llm(CODER_SYSTEM, code_prompt)
        generated_code = extract_code(generated_code)

        max_retries = 5
        last_error = None
        for attempt in range(1, max_retries + 1):
            # Step 5: Execute code as subprocess
            
            stdout, stderr, returncode = await run_python_code(generated_code)
            if returncode == 0:
                break
            last_error = stderr.strip()
            # Correction prompt for LLM
            if last_error==None:
                break
            correction_prompt = (
                f"{code_prompt}\n\n"
                f"Previous code resulted in error (attempt {attempt}):\n{last_error}\n"
                "Please correct the code and ensure it prints the final result."
            )
            generated_code = call_llm(DEBUG_SYSTEM, correction_prompt)
            generated_code = extract_code(generated_code)
        else:
            return f"Execution Error after {max_retries} attempts:\n{last_error}"

        if not stdout.strip():
            return "No output generated from analysis"

        # Step 6: Use interpreter to structure the final answer
        raw_output = stdout.strip()
        interpreter_prompt = f"""Plan: {plan}

Generated Code Output:
{raw_output}

Please interpret these results and provide a structured response according to the plan's expected output format."""
        
        final_structured_result = call_llm(INTERPRETER_SYSTEM, interpreter_prompt)
        
        return final_structured_result
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with usage information"""
    return {
        "message": "Universal Data Analyst API",
        "usage": "curl 'https://app.example.com/api/' -F '@question.txt'",
        "endpoint": "/api/ (POST with file upload)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
