import os
import uuid
import shutil
import subprocess
import json
import logging
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from typing import List
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
import re

# --- Configuration ---
# It's recommended to use environment variables for sensitive data like API keys.
# For deployment, set the GROQ_API_KEY environment variable on your hosting platform.
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
logging.basicConfig(level=logging.INFO)
MODEL_NAME = "llama3-70b-8192" # Using Llama 3 on Groq
EXECUTION_TIMEOUT = 170  # Seconds (3 minutes is 180s, leave a buffer)

app = FastAPI()

@app.get("/")
async def health_check():
    """A simple health check endpoint that Render can use."""
    return {"status": "ok"}
   
# === REPLACE THE SYSTEM_PROMPT WITH THIS ===

SYSTEM_PROMPT = """
You are an expert data analyst AI. Your task is to write a single, self-contained Python script to answer a user's questions.

**CRITICAL SCRIPTING RULES:**

1.  **JSON Output Schema (MANDATORY):** The user's question will specify the exact keys required in the final JSON object (e.g., `average_temp_c`, `total_sales`, `edge_count`). You MUST use these exact keys in your output. Do not invent your own. This is the most important rule.

2.  **Data Loading & Inspection:** Load data from provided files (`.csv`) or by scraping, then immediately print the `.dtypes` and `.head()` to `stderr` to understand the data's structure before proceeding.

3.  **Conditional Data Cleaning:** Before calculations, you MUST check if a column's `dtype` is `object`. If it is a string that should be numeric (like price or temperature), you MUST clean it by removing non-numeric characters and using `pd.to_numeric`. If it is already `int64` or `float64`, do not attempt to clean it with string methods.

4.  **JSON Serialization:** Before the final `json.dumps()`, you MUST convert all NumPy numeric types (`int64`, `float64`) and other special objects into native Python types (e.g., `int()`, `float()`, `str()`). This prevents `TypeError`.

5.  **SQL Queries:** For large, remote datasets, generate the SQL query text as the answer; do not execute the full query as it will time out.

6.  **Output:** The final result MUST be a single line of valid JSON printed to `stdout`. All debugging prints (like `.dtypes` and `.head()`) MUST go to `stderr`.
"""
# === REPLACE THE ENTIRE FUNCTION WITH THIS ===

@app.post("/api/")
async def data_analyst_agent(request: Request):
    session_id = str(uuid.uuid4())
    work_dir = os.path.join("/tmp", session_id)
    os.makedirs(work_dir)

    try:
        # 1. Parse multipart form data and save files
        form = await request.form()
        filenames = []
        questions_content = ""

        # Add extra logging to see exactly what files are being received
        received_items = [(item.filename, getattr(item, 'name', 'N/A')) for item in form.values()]
        logging.info(f"[{session_id}] Received form items (filename, name): {received_items}")

        for file in form.values():
            if not file.filename:
                continue
            
            filepath = os.path.join(work_dir, file.filename)
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            filenames.append(file.filename)
            
            # Make the check case-insensitive to be safer
            if "questions.txt" in file.filename.lower():
                with open(filepath, "r") as f:
                    questions_content = f.read()

        if not questions_content:
            logging.error(f"[{session_id}] 'questions.txt' content was not found in any of the received files.")
            raise HTTPException(status_code=400, detail="400: questions.txt is missing.")

        # 2. Construct the prompt for the LLM
        file_list_str = ", ".join(filenames)
        user_prompt = f"User questions are in 'questions.txt'.\n\nContent of questions.txt:\n---\n{questions_content}\n---\n\nAvailable data files in the working directory: [{file_list_str}]"
        
        logging.info(f"[{session_id}] Generating script for files: {file_list_str}")

        # 3. Call the LLM to generate the Python script
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        
        # 4. Clean up the LLM's response to get raw Python code
        script_content = response.choices[0].message.content
        code_match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if code_match:
            script_content = code_match.group(1).strip()
        else:
            lines = script_content.strip().split('\n')
            start_index = 0
            end_index = len(lines) -1
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    start_index = i
                    break
            for i in range(len(lines) - 1, start_index - 1, -1):
                if lines[i].strip().startswith('print(json.dumps'):
                    end_index = i
                    break
            script_content = '\n'.join(lines[start_index : end_index + 1])

        # 5. Apply hard-coded patches for known LLM errors
        script_content = script_content.replace("'Release year'", "'Year'")
        script_content = script_content.replace('["Release year"]', '["Year"]')
        script_content = script_content.replace("INSTALL s3fs", "INSTALL httpfs")
        script_content = script_content.replace("LOAD s3fs", "LOAD httpfs")
        
        script_path = os.path.join(work_dir, "agent_script.py")
        with open(script_path, "w") as f:
            f.write(script_content)

        logging.info(f"[{session_id}] Executing generated script.")

        # 6. Execute the generated script in a sandboxed environment
        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
                cwd=work_dir,
                check=True 
            )
            
            stdout = result.stdout.strip()
            
            try:
                json_output = json.loads(stdout)
                return JSONResponse(content=json_output)
            except json.JSONDecodeError:
                logging.error(f"[{session_id}] Failed to decode JSON from script output: {stdout}")
                raise HTTPException(status_code=500, detail="Agent script produced invalid JSON.")

        except subprocess.TimeoutExpired:
            logging.error(f"[{session_id}] Script execution timed out.")
            raise HTTPException(status_code=504, detail="Analysis task timed out.")
        except subprocess.CalledProcessError as e:
            logging.error(f"[{session_id}] Script execution failed with stderr:\n{e.stderr}")
            error_response = { "error": "Script execution failed", "stderr": e.stderr }
            return JSONResponse(content=error_response, status_code=500)

    except Exception as e:
        logging.error(f"[{session_id}] An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        logging.info(f"[{session_id}] Cleanup complete.")
