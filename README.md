# Financial Document Analyzer

A comprehensive financial document analysis system that processes corporate reports, financial statements, and investment documents using AI-powered analysis agents. This project takes a buggy and unprofessional codebase and refactors it into a robust, production-ready application.

## Bugs Found and How They Were Fixed

The original codebase (`@copy/` files) was plagued with deterministic bugs, inefficient and unprofessional prompts, and a lack of robust architecture. Here's a summary of the key issues and their solutions.

### 1. Inefficient and Unprofessional Prompts (`agents.py`, `task.py`)

-   **Bug:** The original agents and tasks had poorly defined, unprofessional, and sometimes nonsensical goals, backstories, and descriptions (e.g., "Make up investment advice," "Just say yes to everything"). This would have resulted in unpredictable and low-quality output.
-   **Fix:** A complete rewrite of `agents.py` and `task.py` was performed.
    -   **Agents (`agents.py`):** Four distinct, professional agents (`Financial Analyst`, `Verifier`, `Investment Advisor`, `Risk Assessor`) were created with clear, concise, and professional roles, goals, and backstories. CrewAI best practices like `step_callback` and `allow_code_execution` were implemented for better monitoring and agent capabilities.
    -   **Tasks (`task.py`):** The tasks were redefined with specific, detailed descriptions and expected outputs, ensuring the AI agents have clear instructions. Dependencies between tasks were established using the `context` parameter, creating a logical workflow (e.g., analysis depends on verification).

### 2. Broken and Inefficient Tools (`tools-c.py`)

-   **Bug:** The original `tools.py` contained broken, incomplete, and inefficient tools. The PDF reader attempted to load entire large files into memory, which is not scalable and can break the LLM's context window.
-   **Fix:**
 TCrewAI's built-in `PDFSearchTool`. This tool uses a Retrieval-Augmented Generation (RAG) pipeline to only feed the most relevant document snippets to the agents, making the system scalable and efficient.
    -   **Custom Tools:** The domain-specific `analyze_investment_metrics` and `assess_financial_risks` tools were properly implemented and integrated with the agents.

### 3. Flawed Application Logic and Architecture (`main.py`)

-   **Bug:** The original `main.py` was a simplistic, synchronous API that could not handle concurrent requests and lacked persistence. Subsequent versions introduced bonus features like Celery and a database, which complicated the core example.
-   **Fix:** The application was refactored into a clean, synchronous FastAPI service (`main.py`). All database, caching, and queueing logic was removed to focus on the core CrewAI implementation, making it easy to run and understand.

### 4. Enhanced PDF Parsing and Financial Analysis 

-   **Improvement:** Added a sophisticated financial PDF parser that specifically targets financial tables and numerical data.
    -   **Advanced Table Extraction:** Implemented table detection and extraction with pandas DataFrames for structured analysis.
    -   **Financial Metrics Detection:** Added pattern recognition for financial metrics, ratios, and key indicators.
    -   **Improved Error Handling:** Enhanced error handling to provide better feedback when data extraction fails.
    -   **Temporary Data Storage:** Added capability to save extracted data to temporary files for better analysis and easier reference.

### 5. **Integrated Parser:** Moved the financial PDF parser directly into tools.py to reduce file count and simplify imports.

**Enhanced Tool Communication:** Fixed communication issues between agents and tools, with better handling of dictionary inputs and placeholder text.

 **Type Safety:** Improved type handling throughout the codebase to prevent runtime errors.

### 6. Agent Workflow Improvements

-   **Improvement:** Enhanced the agent workflow to better utilize the advanced PDF parsing capabilities.
    -   **Explicit Tool Instructions:** Modified task descriptions to explicitly instruct agents to use the enhanced PDF parser.
    -   **Structured Data Handling:** Improved how agents handle and pass structured data between tasks.

## Setup and Usage Instructions

### 1. Prerequisites

-   Python 3.11.9
-   An environment with `venv` is recommended.

### 2. Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd financial-document-analyzer-debug
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Unix/Mac:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your API keys:
    ```
    # Required: Google Gemini API key
    GEMINI_API_KEY=your_actual_gemini_api_key_here
    
    # Optional: For web search functionality
    SERPER_API_KEY=your_serper_api_key_here
    ```

    **How to get a Gemini API Key:**
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Create a new API key
    3. Copy it to your `.env` file

### 3. Running the Application

You have two options for running the application:

#### Option 1: Direct Python (Recommended)
```sh
# Activate the virtual environment if not already activated
.\venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Unix/Mac

# Run the application
python main.py
```

#### Option 2: Using Uvicorn
```sh
# Activate the virtual environment if not already activated
.\venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Unix/Mac

# Run with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### 4. Using the Command Line Interface

For direct document analysis without the API, you can use the CLI:

```sh
# Activate the virtual environment
.\venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Unix/Mac

# Run the CLI with a PDF file and query
python cli.py data/TSLA-Q2-2025-Update.pdf --query "Analyze financial health and risks"
```

### 5. Running Tests

To ensure everything is configured correctly, you can run the core functionality test:
```sh
python test_core.py
```

## API Documentation

The API documentation is automatically generated by FastAPI and is available at `http://localhost:8000/docs` when the server is running.

### Endpoints

-   **`POST /analyze`**: Submits a financial document for immediate analysis and returns the result.
    -   **Body:** `multipart/form-data`
        -   `file`: The PDF document to analyze.
        -   `query`: (Optional) A specific query for the analysis.
    -   **Response:**
        ```json
        {
          "status": "success",
          "result": {
            // ... detailed analysis from the crew ...
          }
        }
        ```

-   **`GET /`**: Health check endpoint.
