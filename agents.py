## Importing libraries and files
import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, LLM
import os
from tools import (
    validate_financial_document, 
    search_tool, 
    file_tool, 
    analyze_investment_metrics, 
    assess_financial_risks, 
    extract_pdf_content, 
    extract_financial_tables
)

### Loading LLM - Using modern CrewAI LLM class
llm = LLM(
    model="gemini/gemini-1.5-pro",
    temperature=0.1,
    api_key=os.getenv("GEMINI_API_KEY")
)

# Instantiate the tools
validation_tool = validate_financial_document

# Define a list of all tools used for easier maintenance
all_tools = [
    validate_financial_document, 
    search_tool, 
    file_tool, 
    analyze_investment_metrics, 
    assess_financial_risks, 
    extract_pdf_content,
    extract_financial_tables
]

# Ensure all tools have __name__ attribute
for tool in all_tools:
    if tool and not hasattr(tool, '__name__'):
        tool.__name__ = getattr(tool, 'name', tool.__class__.__name__)

# Creating an Experienced Financial Analyst agent with professional backstory
financial_analyst = Agent(
    role="Senior Financial Analyst",
    goal=(
        "Analyze the provided financial document and deliver concise, data-backed insights "
        "aligned to {query}. Prioritize accuracy, clarity, and actionable conclusions."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "Experienced equity analyst focused on ratio analysis and fundamentals. "
        "Uses advanced financial data extraction techniques to analyze complex financial documents."
    ),
    tools=[extract_financial_tables, analyze_investment_metrics, assess_financial_risks, extract_pdf_content],
    llm=llm,
    max_iter=3,
    allow_delegation=True
)

# Creating a document verifier agent with professional standards
verifier = Agent(
    role="Financial Document Compliance Officer",
    goal=(
        "Validate document type, integrity, and presence of required financial sections before analysis: {query}."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "Compliance reviewer ensuring documents are valid and complete. "
        "Extracts and validates financial data using advanced parsing techniques."
    ),
    tools=[validation_tool, extract_financial_tables, extract_pdf_content],
    llm=llm,
    max_iter=2,
    allow_delegation=False
)

# Creating a professional investment advisor
investment_advisor = Agent(
    role="Certified Investment Advisor",
    goal=(
        "Provide clear, risk-adjusted recommendations based on the analysis; explain rationale and trade-offs: {query}."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "Advisor focused on actionable strategies and risk-adjusted outcomes. "
        "Analyzes structured financial data to provide evidence-based investment recommendations."
    ),
    tools=[analyze_investment_metrics, extract_financial_tables],
    llm=llm,
    max_iter=2,
    allow_delegation=False
)

# Creating a professional risk assessment expert
risk_assessor = Agent(
    role="Risk Management Expert",
    goal=(
        "Identify material risks, their likelihood and impact, and propose mitigations relevant to {query}."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "Quantitative risk analyst producing practical mitigation guidance. "
        "Specializes in extracting and analyzing financial risk data from complex documents."
    ),
    tools=[assess_financial_risks, extract_financial_tables],
    llm=llm,
    max_iter=2,
    allow_delegation=False
)