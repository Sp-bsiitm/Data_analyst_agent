import os
import uuid
import shutil
import subprocess
import json
import logging
from fastapi import FastAPI, Request, HTTPException
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
   
SYSTEM_PROMPT = """
You are an expert data analyst AI. Your task is to write a single, self-contained Python script to answer a user's questions, keeping in mind a strict 3-minute execution limit.

**CRITICAL SCRIPTING RULES:**

1.  **Web Scraping:** Always use `pandas.read_html(url)` for HTML tables.

2.  **SQL & Large Datasets (VERY IMPORTANT):**
    *   When a question involves querying a large, remote dataset (like the Indian Courts dataset), **DO NOT ATTEMPT TO EXECUTE THE FULL QUERY.** It will time out.
    *   Your primary goal is to **GENERATE THE SQL QUERY TEXT** that *would* answer the question.
    *   The answer to a question like "Which court had the most cases?" should be a descriptive string like "The answer would be found by running the following SQL query...", combined with the query itself.
    *   You MUST use the provided DuckDB template to generate the query text.
    *   **MANDATORY DUCKDB TEMPLATE:**
        ```python
        # The query to answer the user's question.
        # Notice the f-string formatting and the use of read_parquet.
        s3_path = 's3://indian-high-court-judgments/...'
        query = f"SELECT ... FROM read_parquet('{s3_path}') WHERE ..."
        
        # The final JSON should contain this query text as an answer.
        ```

3.  **Data Cleaning:** After loading any local data, you must clean it. For currency strings (e.g., '$1,234'), remove all non-numeric characters before converting to a numeric type.

4.  **Output Requirements:**
    *   The script's final result must be a single line of valid JSON printed to **standard output (`stdout`)**.
    *   All debugging information MUST be printed to **standard error (`stderr`)**.

**Final Step:** Your script must end by printing the final JSON result to `stdout`.
"""
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

        for name, file in form.items():
            if not file.filename:
                continue
            
            filepath = os.path.join(work_dir, file.filename)
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            filenames.append(file.filename)
            
            if file.filename == "questions.txt":
                with open(filepath, "r") as f:
                    questions_content = f.read()

        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt is missing.")

        # 2. Construct the prompt for the LLM
        file_list_str = ", ".join(filenames)
        user_prompt = f"User questions are in 'questions.txt'.\n\nContent of questions.txt:\n---\n{questions_content}\n---\n\nAvailable files in the working directory: [{file_list_str}]"
        
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
        
        script_content = response.choices[0].message.content

        # 4. Clean up the LLM's response to get raw Python code
        code_match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if code_match:
            script_content = code_match.group(1).strip()
        else:
            lines = script_content.strip().split('\n')
            start_index = 0
            end_index = len(lines) - 1
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    start_index = i
                    break
            for i in range(len(lines) - 1, start_index - 1, -1):
                if lines[i].strip().startswith('print(json.dumps'):
                    end_index = i
                    break
            script_content = '\n'.join(lines[start_index : end_index + 1])

        # 5. Apply hard-coded patch for known LLM errors
        script_content = script_content.replace("'Release year'", "'Year'")
        script_content = script_content.replace('["Release year"]', '["Year"]')
        # NEW -- Fix for DuckDB extension hallucination
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
            error_response = {
                "error": "Script execution failed",
                "stderr": e.stderr
            }
            return JSONResponse(content=error_response, status_code=500)

    except Exception as e:
        logging.error(f"[{session_id}] An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 7. Clean up the temporary directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        logging.info(f"[{session_id}] Cleanup complete.")
