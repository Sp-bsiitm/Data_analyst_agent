# Data_analyst_agent

This project is a fully autonomous AI Data Analyst Agent. It exposes a single API endpoint that can accept any data analysis task described in plain English, along with optional data files. The agent leverages a Large Language Model (LLM) to generate a Python script on the fly, which is then executed in a sandboxed environment to produce the final result.

## Features

- **Flexible & Autonomous:** Can handle a wide variety of data analysis tasks without pre-written code for specific questions.
- **Multi-Modal Input:** Accepts a `questions.txt` file and optional data files like `.csv`, `.json`, etc.
- **Versatile Data Sourcing:** Capable of scraping web pages, querying conceptual databases, and analyzing user-provided files.
- **On-the-Fly Code Generation:** Uses the Groq API with the Llama 3 model to write Python scripts tailored to each specific request.
- **Secure Execution:** Runs the generated code in an isolated, sandboxed environment with a strict timeout to ensure security and stability.
- **Dynamic Output:** Responds with a correctly formatted JSON object or array, as specified by the user's request.

## Architecture

The system follows a modern agentic architecture:

1.  **FastAPI Server:** A lightweight web server receives the incoming `POST` request and files.
2.  **Orchestrator Logic:** The main application logic prepares a detailed prompt for the LLM, including the user's questions and a list of provided files.
3.  **LLM Code Generation (Groq):** The prompt is sent to the Groq API, which uses the `llama3-70b-8192` model to generate a Python script. The system prompt is engineered to guide the AI in writing robust, clean, and correct code.
4.  **Sandboxed Execution:** The generated Python script is executed using a `subprocess` with a strict timeout. The application captures the script's standard output (`stdout`) and standard error (`stderr`).
5.  **Response Formatting:** The captured `stdout`, which is expected to be a single line of JSON, is parsed and returned to the user.

## API Endpoint

The agent is deployed and can be accessed at the following endpoint.

- **URL:** `https://data-analyst-agent-5bah.onrender.com/api/`
- **Method:** `POST`
- **Body:** `multipart/form-data`

### How to Use

You must send a `questions.txt` file and can optionally include other data files.

#### Example 1: Web Scraping and Plotting

**`questions.txt`:**
```txt
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI.
```

**cURL Command:**
```bash
curl -F "questions.txt=@questions.txt" https://data-analyst-agent-5bah.onrender.com/api/
```

#### Example 2: Analyzing a Provided CSV File

**`questions.txt`:**
```txt
You have been provided with a file named sales_data.csv. Please perform the following analysis.

Answer the questions and respond with a JSON object.

1. What was the total revenue for the 'Electronics' category? (Revenue = Price * Quantity)
2. Which product had the highest single transaction value?
```

**`sales_data.csv`:**
```csv
Date,Product,Category,Price,Quantity
2024-01-15,Laptop,Electronics,1200,5
2024-01-16,Mouse,Electronics,25,30
...
```

**cURL Command:**
```bash
curl -F "questions.txt=@questions.txt" -F "sales_data.csv=@sales_data.csv" https://data-analyst-agent-5bah.onrender.com/api/
```

## Local Setup and Deployment

### Prerequisites

- Python 3.11+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application requires an API key from [Groq](https://console.groq.com/keys).

Set the following environment variable:

| Name          | Description                  |
|---------------|------------------------------|
| `GROQ_API_KEY`| Your API key for the Groq API. |

You can create a `.env` file for local development:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### Running Locally

Use `uvicorn` to run the FastAPI server:
```bash
uvicorn main:app --reload
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
