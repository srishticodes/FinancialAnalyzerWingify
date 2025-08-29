#!/usr/bin/env python3
"""
Financial Document Analyzer CLI
A command-line interface for analyzing financial documents using CrewAI agents
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from crewai import Crew, Process

# Load environment variables
load_dotenv()

# Import agents and tasks
from agents import financial_analyst, verifier, investment_advisor, risk_assessor
from task import (
    document_verification,
    analyze_financial_document,
    investment_analysis,
    risk_assessment,
)
from tools import validate_financial_document, extract_pdf_content

def analyze_financial_document_cli(
    file_path: str, 
    query: str = "Provide comprehensive financial analysis", 
    output_file: Optional[str] = None,
    include_investment: bool = True,
    include_risk: bool = True
) -> str:
    """
    Analyze a financial document using CrewAI agents
    
    Args:
        file_path: Path to the financial document
        query: Analysis query
        output_file: Path to save the analysis results
        include_investment: Whether to include investment advice
        include_risk: Whether to include risk assessment
        
    Returns:
        Analysis results
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    # Generate default output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"financial_analysis_{Path(file_path).stem}_{timestamp}.txt"
    
    # Configure the agents and tasks
    agents = [verifier, financial_analyst]
    tasks = [document_verification, analyze_financial_document]
    
    # Add optional analysis
    if include_investment:
        agents.append(investment_advisor)
        tasks.append(investment_analysis)
        
    if include_risk:
        agents.append(risk_assessor)
        tasks.append(risk_assessment)
    
    # Create and run the crew
    print(f"\n{'='*60}")
    print(f"Analyzing document: {file_path}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    # Start the analysis
    result = crew.kickoff(
        inputs={"query": query, "file_path": file_path}
    )
    
    elapsed_time = time.time() - start_time
    
    # Save results to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"FINANCIAL ANALYSIS REPORT\n")
        f.write(f"Document: {file_path}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis time: {elapsed_time:.2f} seconds\n")
        f.write("="*80 + "\n\n")
        f.write(str(result))
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Analysis took {elapsed_time:.2f} seconds")
    
    return result

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Financial Document Analyzer CLI")
    parser.add_argument("file", help="Path to the financial document (PDF)")
    parser.add_argument("--query", "-q", 
                      default="Provide comprehensive financial analysis and investment recommendations", 
                      help="Specific analysis query")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generated)")
    parser.add_argument("--no-investment", action="store_true", 
                      help="Skip investment advice analysis")
    parser.add_argument("--no-risk", action="store_true", 
                      help="Skip risk assessment")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is missing!")
        print("Please add it to your .env file")
        sys.exit(1)
    
    # Run analysis
    analyze_financial_document_cli(
        file_path=args.file,
        query=args.query,
        output_file=args.output,
        include_investment=not args.no_investment,
        include_risk=not args.no_risk
    )

if __name__ == "__main__":
    main()
