from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import uuid
import logging

from crewai import Crew, Process
from agents import financial_analyst, verifier, investment_advisor, risk_assessor
from task import (
    document_verification,
    analyze_financial_document,
    investment_analysis,
    risk_assessment,
)
from tools import validate_financial_document, extract_pdf_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Document Analyzer API",
    description="A simplified, synchronous API for financial document analysis using AI agents",
    version="1.0.0",
)

class AnalysisResponse(BaseModel):
    status: str
    result: Dict[str, Any]

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "service": "Financial Document Analyzer API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/analyze": "POST - Submit document for immediate analysis",
            "/docs": "API documentation",
        },
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    query: str = Form(default="Provide comprehensive financial analysis"),
    include_investment_advice: bool = Form(default=True),
    include_risk_assessment: bool = Form(default=True),
):
    """
    Submit a financial document for immediate, synchronous analysis.
    """
    # 1. File validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    # 2. Save file temporarily
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        # 3. Pre-analysis validation with the tool
        validation = validate_financial_document(file_path)
        if not validation.get("is_valid", False):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid financial document: {validation.get('error', 'Unknown error')}",
            )

        # 4. Configure and run the Crew
        agents = [verifier, financial_analyst]
        tasks = [document_verification, analyze_financial_document]

        if include_investment_advice:
            agents.append(investment_advisor)
            tasks.append(investment_analysis)

        if include_risk_assessment:
            agents.append(risk_assessor)
            tasks.append(risk_assessment)
        
        financial_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = financial_crew.kickoff(
            {"query": query, "file_path": file_path}
        )

        return AnalysisResponse(status="success", result=result)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5. Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)