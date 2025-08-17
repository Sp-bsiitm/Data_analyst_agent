import os
import uuid
import shutil
import subprocess
import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

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
   
# --- System Prompt for the LLM Agent ---
SYSTEM_PROMPT = """
You are an expert data analyst AI. Your task is to write a single, self-contained Python script to answer a user's questions based on the provided text and files.

**Instructions:**
1.  **Analyze the Request:** Carefully read the user's questions from `questions.txt`.
2.  **Access Files:** The user may provide additional files (e.g., `.csv`, `.png`). Your script will be executed in a directory containing all these files. You can access them directly by their filenames (e.g., `pd.read_csv('data.csv')`).
3.  **Required Libraries:** You have access to a pre-installed environment with the following libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`, `beautifulsoup4`, `lxml`, `duckdb`, `pyarrow`. Do NOT include installation commands.
4.  **Write a Python Script:** Generate a complete Python script that performs all the necessary steps:
    *   Sourcing data (reading files, scraping URLs).
    *   Data preparation and cleaning.
    *   Analysis and calculations.
    *   Generating visualizations if requested.
5.  **Output Requirements:**
    *   The script's **final output** MUST be a single line of valid JSON printed to standard output.
    *   Do NOT print any other logs, comments, or intermediate results. The only thing printed to stdout should be the final JSON string.
    *   The JSON structure (array or object) must match what is requested in `questions.txt`.
    *   **Visualizations:** If a plot is requested:
        *   Generate it using Matplotlib/Seaborn.
        *   Save it to an in-memory buffer (`io.BytesIO`).
        *   Encode it as a Base64 data URI string (`data:image/png;base64,...`).
        *   The data URI string must be less than 100,000 bytes. Use techniques like lowering DPI (`dpi=75`) or changing format (`format='webp'`) if necessary.
        *   Include this string as a value in the final JSON output.
6.  **Final Step:** Your script must end by printing the JSON. For example:
    `import json; print(json.dumps({"answer1": 42, "plot": "data:image/png;base64,..."}))`
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
        # Clean up the response to get raw Python code
        if script_content.startswith("```python"):
            script_content = script_content[9:]
        if script_content.endswith("```"):
            script_content = script_content[:-3]
        
        script_path = os.path.join(work_dir, "agent_script.py")
        with open(script_path, "w") as f:
            f.write(script_content)

        logging.info(f"[{session_id}] Executing generated script.")

        # 4. Execute the generated script in a sandboxed environment
        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
                cwd=work_dir,
                check=True # Will raise CalledProcessError if return code is non-zero
            )
            
            # The script should only print the final JSON to stdout
            stdout = result.stdout.strip()
            
            # 5. Parse and return the result
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
            # To meet the requirement of always returning a JSON structure, return an error object.
            # You could also try to guess the expected format (list/dict) and return an empty one.
            error_response = {
                "error": "Script execution failed",
                "stderr": e.stderr
            }
            return JSONResponse(content=error_response, status_code=500)

    except Exception as e:
        logging.error(f"[{session_id}] An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 6. Clean up the temporary directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        logging.info(f"[{session_id}] Cleanup complete.")
