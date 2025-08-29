## Importing libraries and files
from crewai import Task
from agents import financial_analyst, verifier, investment_advisor, risk_assessor

## Document verification task
document_verification = Task(
    description="""Verify and validate the uploaded financial document.
    Check for:
    1. Document type and format validity
    2. Completeness of financial data
    3. Time period covered
    4. Company/entity identification
    5. Any data quality issues or missing sections
    
    IMPORTANT: First use the extract_financial_tables tool to extract structured financial data including tables and metrics.
    If that doesn't yield sufficient information, then use the extract_pdf_content tool for full text extraction.
    
    User query: {query}
    Document path: {file_path}""",
    
    expected_output="""A verification report including:
    - Document type and validity confirmation
    - Key sections identified (balance sheet, income statement, cash flow, etc.)
    - Time period and entity details
    - Data completeness assessment
    - Any identified issues or limitations""",
    
    agent=verifier,
    async_execution=False,
)

## Creating the main financial analysis task
analyze_financial_document = Task(
    description="""Conduct comprehensive financial analysis based on the user's query: {query}
    
    Analyze the financial document at: {file_path}
    
    IMPORTANT: First use the extract_financial_tables tool to extract structured financial data including tables, metrics and ratios.
    This will provide you with well-structured financial data for analysis. If needed, you can then use analyze_investment_metrics
    with the PDF path to get detailed investment metrics.
    
    Provide detailed analysis including:
    1. Key financial metrics and ratios (liquidity, profitability, efficiency, leverage)
    2. Trend analysis and year-over-year comparisons
    3. Strengths and weaknesses identified in the financials
    4. Industry benchmarking where applicable
    5. Cash flow analysis and working capital assessment
    6. Quality of earnings assessment
    7. Any red flags or areas of concern
    
    If the financial tables extraction doesn't yield enough information, use extract_pdf_content as a fallback.""",
    
    expected_output="""A comprehensive financial analysis report containing:
    - Executive summary of financial health
    - Detailed ratio analysis with interpretations
    - Trend analysis with visual indicators
    - Key strengths and weaknesses
    - Comparison to industry standards (if applicable)
    - Cash flow and liquidity assessment
    - Specific insights related to the user's query
    - Data-driven conclusions and observations""",
    
    agent=financial_analyst,
    async_execution=False,
    context=[document_verification]  # Depends on verification
)

## Investment recommendations task
investment_analysis = Task(
    description="""Based on the financial analysis, provide investment recommendations.
    
    User query: {query}
    
    IMPORTANT: First check if the financial analysis already includes structured financial data. If not, use extract_financial_tables
    tool on {file_path} to get structured financial data including tables, metrics and ratios, then use analyze_investment_metrics
    to get detailed investment metrics.
    
    Consider the financial analysis results and provide:
    1. Investment thesis (bull/bear case)
    2. Valuation assessment
    3. Risk-adjusted return potential
    4. Suitable investment strategies
    5. Portfolio allocation suggestions
    6. Time horizon considerations
    7. Alternative investment options""",
    
    expected_output="""Investment recommendation report including:
    - Clear investment thesis with supporting data
    - Valuation metrics and price targets (if applicable)
    - Recommended investment strategies
    - Portfolio allocation suggestions based on risk tolerance
    - Time horizon and exit strategy considerations
    - Alternative investment options
    - Disclaimer about investment risks and need for personal due diligence""",
    
    agent=investment_advisor,
    async_execution=False,
    context=[analyze_financial_document]  # Depends on financial analysis
)

## Risk assessment task
risk_assessment = Task(
    description="""Conduct comprehensive risk assessment based on the financial analysis.
    
    User query: {query}
    
    IMPORTANT: First check if the financial analysis already includes structured risk data. If not, use extract_financial_tables
    tool on {file_path} to get structured financial data, then use assess_financial_risks to perform a detailed risk assessment.
    
    Evaluate:
    1. Financial risks (credit, liquidity, solvency)
    2. Operational risks
    3. Market risks and sensitivity analysis
    4. Industry-specific risks
    5. Regulatory and compliance risks
    6. ESG (Environmental, Social, Governance) considerations
    7. Scenario analysis and stress testing""",
    
    expected_output="""Risk assessment report including:
    - Risk matrix with probability and impact ratings
    - Detailed analysis of identified risks
    - Quantitative risk metrics (VaR, Beta, Sharpe ratio where applicable)
    - Stress test scenarios and outcomes
    - Risk mitigation strategies
    - Risk-adjusted performance metrics
    - Recommendations for risk management""",
    
    agent=risk_assessor,
    async_execution=False,
    context=[analyze_financial_document]  # Depends on financial analysis
)