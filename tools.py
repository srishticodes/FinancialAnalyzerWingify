## Importing libraries and files
import os
import sys
import re
from typing import Dict, Any, Optional, Union, List
import pdfplumber
from dotenv import load_dotenv
load_dotenv()

try:
    from crewai.tools import tool
    from crewai_tools import (
        BaseTool, 
        SerperDevTool, 
        FileReadTool,
        DirectoryReadTool
    )
    
    orig_tool_init = BaseTool.__init__
    def patched_init(self, *args, **kwargs):
        orig_tool_init(self, *args, **kwargs)
        if not hasattr(self, '__name__'):
            self.__name__ = self.name if hasattr(self, 'name') else self.__class__.__name__
    BaseTool.__init__ = patched_init
    
    # Creating search tool
    search_tool = None
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key and "your_serper_api_key" not in serper_api_key:
        search_tool = SerperDevTool(api_key=serper_api_key)

    # Set Google API key for tools that might need it
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    
    # File reading tool for document processing
    file_tool = FileReadTool()
    
except ImportError:
    # Fallback stubs if crewai_tools is unavailable
    class BaseTool:
        name: str = "Base Tool"
        description: str = "Base tool description"
        
        def __init__(self, *args, **kwargs):
            self.name = getattr(self, 'name', 'Base Tool')
            self.description = getattr(self, 'description', 'Base tool description')
            
        def _run(self, *args, **kwargs):
            raise NotImplementedError("crewai_tools is not installed.")
            
        def run(self, *args, **kwargs):
            return self._run(*args, **kwargs)

    class SerperDevTool(BaseTool):
        name: str = "Serper Search Tool"
        description: str = "Search tool using Serper API"
        
        def __init__(self, api_key: str):
            super().__init__()
            self.api_key = api_key
            
        def search(self, *args, **kwargs):
            return "Fallback search: No crewai_tools installed."
            
        def _run(self, query: str) -> str:
            return self.search(query)
    
    def tool(fn):
        return fn
    
    # Fallback search tool
    search_tool = SerperDevTool(api_key="fallback")
    
    # Fallback file tool
    file_tool = BaseTool()

## Document validation tool
@tool
def validate_financial_document(file_path: str) -> Dict[str, Any]:
        """
        Validate that the uploaded file is a proper financial document.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            Dict containing validation results
        """
        try:
            validation_result = {
                "is_valid": False,
                "document_type": "Unknown",
                "has_financial_data": False,
                "pages": 0,
                "file_size_mb": 0
            }
            
            # Check file exists and get size
            if not os.path.exists(file_path):
                validation_result["error"] = "File not found"
                return validation_result
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            validation_result["file_size_mb"] = round(file_size, 2)
            
            # Check if it's a PDF
            if not file_path.lower().endswith('.pdf'):
                validation_result["error"] = "Not a PDF file"
                return validation_result
            
            # Try to read the PDF using pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    validation_result["pages"] = len(pdf.pages)
                    
                    # Extract some text to check for financial keywords
                    sample_text = ""
                    for i in range(min(3, len(pdf.pages))):
                        page_text = pdf.pages[i].extract_text()
                        if page_text:
                            sample_text += page_text
                    
                    # Check for financial keywords
                    financial_keywords = ['revenue', 'income', 'balance', 'cash flow', 
                                        'assets', 'liabilities', 'equity', 'profit', 
                                        'loss', 'statement', 'financial']
                    
                    sample_text_lower = sample_text.lower()
                    keyword_count = sum(1 for keyword in financial_keywords 
                                      if keyword in sample_text_lower)
                    
                    if keyword_count >= 3:
                        validation_result["has_financial_data"] = True
                        validation_result["is_valid"] = True
                        
                        # Try to identify document type
                        if 'balance sheet' in sample_text_lower:
                            validation_result["document_type"] = "Balance Sheet"
                        elif 'income statement' in sample_text_lower:
                            validation_result["document_type"] = "Income Statement"
                        elif 'cash flow' in sample_text_lower:
                            validation_result["document_type"] = "Cash Flow Statement"
                        elif 'annual report' in sample_text_lower:
                            validation_result["document_type"] = "Annual Report"
                        else:
                            validation_result["document_type"] = "Financial Document"
                    
            except ImportError:
                validation_result["error"] = "pdfplumber not available for validation"
            except Exception as e:
                validation_result["error"] = f"Error reading PDF: {str(e)}"
            
            return validation_result
            
        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

## Advanced financial PDF parsing
# Integrated directly into tools.py for code consolidation
import pandas as pd
import numpy as np
import logging

# Set up logging for financial PDF parser
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialPDFParser:
    """
    Advanced parser for extracting financial data from PDF documents.
    Focuses on detecting and extracting tables, financial metrics, and numerical data.
    """
    
    def __init__(self, file_path: str):
        """Initialize parser with the PDF file path."""
        self.file_path = file_path
        self.pdf = None
        self.text_content = ""
        self.tables = []
        self.financial_metrics = {}
        
        # Common financial terms for identification (simplified)
        self.financial_terms = {
            'income_statement': ['revenue', 'sales', 'income', 'earnings', 'profit', 'expenses'],
            'balance_sheet': ['assets', 'liabilities', 'equity', 'cash', 'debt'],
            'cash_flow': ['cash flow', 'operating activities', 'investing', 'financing'],
            'ratios': ['margin', 'ratio', 'return', 'roe', 'roa']
        }
        
    def load_pdf(self) -> bool:
        """Load the PDF file and initialize pdfplumber."""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return False
                
            self.pdf = pdfplumber.open(self.file_path)
            return True
        except Exception as e:
            logger.error(f"Error opening PDF: {str(e)}")
            return False
    
    def extract_text(self) -> str:
        """Extract all text from the PDF."""
        if not self.pdf:
            if not self.load_pdf():
                return ""
        
        text_content = ""
        try:
            for page in self.pdf.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n\n"
            
            self.text_content = text_content
            return text_content
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_tables(self) -> List[pd.DataFrame]:
        """Extract tables from PDF and convert to pandas DataFrames."""
        if not self.pdf:
            if not self.load_pdf():
                return []
        
        tables = []
        try:
            for page_num, page in enumerate(self.pdf.pages):
                page_tables = page.extract_tables()
                
                if page_tables:
                    for i, table in enumerate(page_tables):
                        if table:
                            # Convert to pandas DataFrame
                            df = pd.DataFrame(table)
                            
                            if len(df) > 0:
                                # Use first row as header if it contains string values
                                if df.iloc[0].astype(str).str.contains('[a-zA-Z]').any():
                                    headers = df.iloc[0].tolist()
                                    df = df[1:]
                                    df.columns = headers
                                
                                # Clean up DataFrame
                                df = self._clean_dataframe(df)
                                
                                # Add metadata
                                df.attrs['page_num'] = page_num + 1
                                df.attrs['table_num'] = i + 1
                                df.attrs['table_type'] = self._identify_table_type(df)
                                
                                tables.append(df)
            
            self.tables = tables
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up the DataFrame by handling NaN values and converting numeric data."""
        # Replace None and empty strings with NaN
        df = df.replace(['', 'None', None], np.nan)
        
        # Attempt to convert columns with numeric data
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df
    
    def _identify_table_type(self, df: pd.DataFrame) -> str:
        """Identify the type of financial table based on keywords."""
        # Convert headers and first column to lowercase for matching
        headers = [str(col).lower() for col in df.columns]
        first_col = [str(item).lower() for item in df.iloc[:, 0]] if len(df) > 0 else []
        
        # Check for keywords
        text_to_check = " ".join(headers + first_col)
        
        for table_type, keywords in self.financial_terms.items():
            if any(term in text_to_check for term in keywords):
                return table_type
        
        return 'unknown'
    
    def extract_financial_metrics(self) -> Dict[str, Any]:
        """Extract key financial metrics from the text."""
        if not self.text_content:
            self.extract_text()
        
        metrics = {'profitability': {}, 'liquidity': {}, 'leverage': {}}
        
        # Extract revenue
        revenue_pattern = r'revenue[s]?[\s:]+\$?([0-9.,]+)\s?(million|billion|m|b|M|B)?'
        match = re.search(revenue_pattern, self.text_content, re.IGNORECASE)
        if match:
            value = match.group(1).replace(',', '')
            try:
                amount = float(value)
                unit = match.group(2).lower() if match.group(2) else ''
                if 'b' in unit or 'billion' in unit:
                    amount *= 1_000_000_000
                elif 'm' in unit or 'million' in unit:
                    amount *= 1_000_000
                
                metrics['profitability']['revenue'] = amount
            except:
                pass
        
        # Extract net income
        income_pattern = r'net income[\s:]+\$?([0-9.,]+)\s?(million|billion|m|b|M|B)?'
        match = re.search(income_pattern, self.text_content, re.IGNORECASE)
        if match:
            value = match.group(1).replace(',', '')
            try:
                amount = float(value)
                unit = match.group(2).lower() if match.group(2) else ''
                if 'b' in unit or 'billion' in unit:
                    amount *= 1_000_000_000
                elif 'm' in unit or 'million' in unit:
                    amount *= 1_000_000
                
                metrics['profitability']['net_income'] = amount
            except:
                pass
        
        self.financial_metrics = metrics
        return metrics
    
    def extract_financial_ratios(self) -> Dict[str, float]:
        """Extract financial ratios from text."""
        if not self.text_content:
            self.extract_text()
        
        ratios = {}
        
        # Common ratio patterns
        ratio_patterns = {
            'current_ratio': r'current ratio[\s:]+([0-9.]+)',
            'quick_ratio': r'quick ratio[\s:]+([0-9.]+)',
            'debt_to_equity': r'debt[- ]to[- ]equity[\s:]+([0-9.]+)',
            'operating_margin': r'operating margin[\s:]+([0-9.]+)[%]?'
        }
        
        for ratio_name, pattern in ratio_patterns.items():
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    ratios[ratio_name] = value
                except:
                    continue
        
        return ratios
    
    def extract_all(self) -> Dict[str, Any]:
        """Extract all financial data from the PDF."""
        results = {
            'file_path': self.file_path,
            'file_name': os.path.basename(self.file_path),
            'text_content': self.extract_text(),
            'tables': self.extract_tables(),
            'metrics': self.extract_financial_metrics(),
            'ratios': self.extract_financial_ratios()
        }
        
        # Convert tables to dict for JSON serialization
        table_data = []
        for i, df in enumerate(results['tables']):
            table_data.append({
                'table_id': i + 1,
                'page_num': df.attrs.get('page_num', 0),
                'table_type': df.attrs.get('table_type', 'unknown'),
                'data': df.to_dict('records'),
                'columns': df.columns.tolist()
            })
        results['tables'] = table_data
        
        return results
    
    def close(self):
        """Close the PDF file."""
        if self.pdf:
            self.pdf.close()

def extract_financial_data(file_path: str) -> Dict[str, Any]:
    """
    Main function to extract financial data from a PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted financial data
    """
    parser = FinancialPDFParser(file_path)
    try:
        results = parser.extract_all()
        return results
    except Exception as e:
        logger.error(f"Error extracting financial data: {str(e)}")
        return {"error": str(e)}
    finally:
        parser.close()

def extract_financial_summary(file_path: str) -> Dict[str, Any]:
    """
    Extract a summary of key financial metrics from a PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing summary of key financial metrics
    """
    parser = FinancialPDFParser(file_path)
    try:
        parser.extract_text()
        metrics = parser.extract_financial_metrics()
        ratios = parser.extract_financial_ratios()
        
        # Combine metrics and ratios into a summary
        summary = {
            'profitability': metrics.get('profitability', {}),
            'liquidity': metrics.get('liquidity', {}),
            'leverage': metrics.get('leverage', {}),
            'ratios': ratios
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error extracting financial summary: {str(e)}")
        return {"error": str(e)}
    finally:
        parser.close()

def format_financial_data_as_text(data: Dict[str, Any]) -> str:
    """
    Format extracted financial data as a structured text for LLM processing.
    
    Args:
        data: Dictionary containing extracted financial data
        
    Returns:
        Structured text representation of financial data
    """
    text_output = []
    
    # Add basic document info
    text_output.append(f"# Financial Analysis: {data.get('file_name', 'Unknown Document')}")
    text_output.append("")
    
    # Add metrics section
    text_output.append("## Key Financial Metrics")
    metrics = data.get('metrics', {})
    
    if 'profitability' in metrics and metrics['profitability']:
        text_output.append("\n### Profitability Metrics")
        for key, value in metrics['profitability'].items():
            text_output.append(f"- {key.replace('_', ' ').title()}: ${value:,.2f}")
    
    # Add ratios section
    ratios = data.get('ratios', {})
    if ratios:
        text_output.append("\n### Financial Ratios")
        for key, value in ratios.items():
            if 'margin' in key or 'roe' in key or 'roa' in key:
                text_output.append(f"- {key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                text_output.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
    
    # Add tables section
    tables = data.get('tables', [])
    if tables:
        text_output.append("\n## Financial Tables")
        for table in tables:
            table_type = table.get('table_type', 'Unknown').replace('_', ' ').title()
            text_output.append(f"\n### {table_type} (Page {table.get('page_num', 0)})")
            
            # Format table data
            columns = table.get('columns', [])
            table_data = table.get('data', [])
            
            if columns and table_data:
                # Format header
                header = " | ".join(str(col) for col in columns)
                text_output.append(header)
                text_output.append("-" * len(header))
                
                # Format rows (limit to first 10 rows for brevity)
                for row in table_data[:10]:
                    row_values = []
                    for col in columns:
                        value = row.get(col, "")
                        if isinstance(value, (int, float)):
                            row_values.append(f"{value:,.2f}")
                        else:
                            row_values.append(str(value))
                    text_output.append(" | ".join(row_values))
                
                # Show ellipsis if there are more rows
                if len(table_data) > 10:
                    text_output.append("... (additional rows omitted for brevity)")
    
    return "\n".join(text_output)

has_financial_pdf_parser = True

## PDF content extraction tool
@tool
def extract_pdf_content(file_path: str) -> str:
    """
    Extract text content from a PDF file using PDFPlumber.
    
    Args:
        file_path (str): Path to the PDF document
        
    Returns:
        Extracted text content as a string
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
            
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_content += text + "\n\n"
        
        return text_content
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@tool
def extract_financial_tables(file_path: str) -> Dict[str, Any]:
    """
    Extract financial tables and structured data from a PDF document.
    Uses advanced parsing techniques to identify and extract financial tables.
    
    Args:
        file_path (str): Path to the financial document (PDF)
        
    Returns:
        Dictionary containing structured financial data including tables and metrics
    """
    if not has_financial_pdf_parser:
        return {"error": "Advanced financial PDF parser is not available. Please install it first."}
    
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        
        data = extract_financial_data(file_path)
        
        # Format the data for better readability
        if "error" not in data:
            data["formatted_text"] = format_financial_data_as_text(data)
            
        return data
    except Exception as e:
        return {"error": f"Failed to extract financial tables: {str(e)}"}

## Investment Analysis Tool with proper implementation
@tool
def analyze_investment_metrics(financial_data: str) -> Dict[str, Any]:
    """
    Analyze financial data to extract key investment metrics.
    
    Args:
        financial_data (str): Text containing financial information or path to a PDF file
        
    Returns:
        Dict containing analyzed investment metrics
    """
    try:
        # Handle empty or placeholder input
        if not financial_data or financial_data in ["[extracted text]", "{extracted text}"]:
            return {
                "error": "No valid financial data provided. Please provide actual financial data or a valid PDF path.",
                "profitability_indicators": [],
                "liquidity_metrics": [],
                "leverage_ratios": [],
                "efficiency_metrics": [],
                "valuation_multiples": []
            }
        
        # Check if input is a file path to a PDF
        if isinstance(financial_data, str) and financial_data.endswith('.pdf') and os.path.exists(financial_data):
          
            if has_financial_pdf_parser:
                # Extract structured financial data
                data = extract_financial_data(financial_data)
                
                # Save the extracted data to a temporary file for reference
                temp_data_path = os.path.join("temp_data", f"{os.path.basename(financial_data)}_extracted_data.json")
                os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)
                
                with open(temp_data_path, 'w') as f:
                    import json
                    json.dump(data, f, indent=2, default=str)
                
                metrics = data.get('metrics', {})
                ratios = data.get('ratios', {})
                
                # Format metrics into the expected structure
                result = {
                    "profitability_indicators": [],
                    "liquidity_metrics": [],
                    "leverage_ratios": [],
                    "efficiency_metrics": [],
                    "valuation_multiples": [],
                    "extracted_data_path": temp_data_path  # Include path to saved data
                }
                
                # Add profitability metrics
                for key, value in metrics.get('profitability', {}).items():
                    result["profitability_indicators"].append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                
                # Add liquidity metrics
                for key in ['current_ratio', 'quick_ratio', 'cash_ratio']:
                    if key in ratios:
                        result["liquidity_metrics"].append(f"{key.replace('_', ' ').title()}: {ratios[key]:.2f}")
                
                # Add leverage ratios
                for key in ['debt_to_equity', 'debt_to_asset', 'interest_coverage']:
                    if key in ratios:
                        result["leverage_ratios"].append(f"{key.replace('_', ' ').title()}: {ratios[key]:.2f}")
                
                # Add efficiency metrics
                for key in ['asset_turnover', 'inventory_turnover', 'receivables_turnover']:
                    if key in ratios:
                        result["efficiency_metrics"].append(f"{key.replace('_', ' ').title()}: {ratios[key]:.2f}")
                
                # Add valuation multiples
                for key in ['pe_ratio', 'price_to_sales', 'price_to_book']:
                    if key in ratios:
                        result["valuation_multiples"].append(f"{key.replace('_', ' ').title()}: {ratios[key]:.2f}")
                
                # Add formatted data if available
                if "formatted_text" in data:
                    result["formatted_data"] = data["formatted_text"]
                
                return result
            
            # Fallback to basic extraction if enhanced parser not available
            text_content = extract_pdf_content(financial_data)
            financial_data = text_content
        
        # Process as text (original implementation)
        metrics = {
            "profitability_indicators": [],
            "liquidity_metrics": [],
            "leverage_ratios": [],
            "efficiency_metrics": [],
            "valuation_multiples": []
        }
        
        # Parse financial data for common metrics (only if it's a string)
        if isinstance(financial_data, str):
            lines = financial_data.lower().split('\n')
            for line in lines:
                # Check for common financial metrics
                if 'revenue' in line or 'sales' in line:
                    metrics["profitability_indicators"].append(line.strip())
                elif 'current ratio' in line or 'quick ratio' in line:
                    metrics["liquidity_metrics"].append(line.strip())
                elif 'debt' in line or 'leverage' in line:
                    metrics["leverage_ratios"].append(line.strip())
                elif 'turnover' in line or 'efficiency' in line:
                    metrics["efficiency_metrics"].append(line.strip())
                elif 'p/e' in line or 'earnings' in line:
                    metrics["valuation_multiples"].append(line.strip())
        
        return metrics
        
    except Exception as e:
        return {
            "error": f"Error analyzing investment metrics: {str(e)}",
            "profitability_indicators": [],
            "liquidity_metrics": [],
            "leverage_ratios": [],
            "efficiency_metrics": [],
            "valuation_multiples": []
        }



## Risk Assessment Tool with proper implementation
@tool
def assess_financial_risks(financial_data: Any) -> Dict[str, Any]:
    """
    Assess various financial risks from the document data.
    
    Args:
        financial_data: Text containing financial information, path to a PDF file,
                      or dictionary with financial data
        
    Returns:
        Dict containing risk assessment results
    """
    try:
        # Handle empty input
        if not financial_data:
            return {
                "error": "No valid financial data provided. Please provide actual financial data or a valid PDF path.",
                "credit_risk": {"level": "Unknown", "factors": ["Insufficient data to assess credit risk"]},
                "liquidity_risk": {"level": "Unknown", "factors": ["Insufficient data to assess liquidity risk"]},
                "market_risk": {"level": "Unknown", "factors": ["Insufficient data to assess market risk"]},
                "operational_risk": {"level": "Unknown", "factors": ["Insufficient data to assess operational risk"]},
                "risk_factors": ["No data available for risk assessment"],
                "risk_score": 0
            }
        
        # Handle dictionary input 
        if isinstance(financial_data, dict):
            # Try different common keys that might contain the actual financial text
            for key in ['description', 'content', 'text', 'financial_data', 'data', 'text_content']:
                if key in financial_data and financial_data[key]:
                    financial_data = financial_data[key]
                    break
            
            # If we still have a dict, try to convert it to string
            if isinstance(financial_data, dict):
                try:
                    import json
                    financial_data = json.dumps(financial_data)
                except:
                    financial_data = str(financial_data)
        
        # Handle placeholder text
        if isinstance(financial_data, str) and financial_data in ["[extracted text]", "{extracted text}"]:
            return {
                "error": "No valid financial data provided. Please provide actual financial data or a valid PDF path.",
                "credit_risk": {"level": "Unknown", "factors": ["Insufficient data to assess credit risk"]},
                "liquidity_risk": {"level": "Unknown", "factors": ["Insufficient data to assess liquidity risk"]},
                "market_risk": {"level": "Unknown", "factors": ["Insufficient data to assess market risk"]},
                "operational_risk": {"level": "Unknown", "factors": ["Insufficient data to assess operational risk"]},
                "risk_factors": ["No data available for risk assessment"],
                "risk_score": 0
            }
        
        # Check if input is a file path to a PDF
        if isinstance(financial_data, str) and financial_data.endswith('.pdf') and os.path.exists(financial_data):
            # Use the enhanced financial parser if available
            if has_financial_pdf_parser:
                # Extract structured financial data
                data = extract_financial_data(financial_data)
                
                # Save the extracted data to a temporary file for reference
                temp_data_path = os.path.join("temp_data", f"{os.path.basename(financial_data)}_risk_data.json")
                os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)
                
                with open(temp_data_path, 'w') as f:
                    import json
                    json.dump(data, f, indent=2, default=str)
                
                metrics = data.get('metrics', {})
                ratios = data.get('ratios', {})
                
                # Initialize risk assessment with more detailed structure
                risk_assessment = {
                    "credit_risk": {"level": "Low", "factors": []},
                    "liquidity_risk": {"level": "Low", "factors": []},
                    "market_risk": {"level": "Medium", "factors": []},
                    "operational_risk": {"level": "Low", "factors": []},
                    "regulatory_risk": {"level": "Low", "factors": []},
                    "esg_risk": {"level": "Low", "factors": []},
                    "risk_factors": [],
                    "risk_score": 0,
                    "risk_summary": "",
                    "extracted_data_path": temp_data_path  # Include path to saved data
                }
                
                # Analyze tables for financial metrics
                profitability = metrics.get('profitability', {})
                
                # Assess liquidity risk based on ratios
                if 'current_ratio' in ratios:
                    if ratios['current_ratio'] < 1.0:
                        risk_assessment["liquidity_risk"]["level"] = "High"
                        risk_assessment["liquidity_risk"]["factors"].append(
                            f"Current ratio ({ratios['current_ratio']:.2f}) is below 1.0, indicating potential liquidity issues"
                        )
                    elif ratios['current_ratio'] < 1.5:
                        risk_assessment["liquidity_risk"]["level"] = "Medium"
                        risk_assessment["liquidity_risk"]["factors"].append(
                            f"Current ratio ({ratios['current_ratio']:.2f}) is below ideal level of 1.5"
                        )
                else:
                    risk_assessment["liquidity_risk"]["factors"].append(
                        "Current ratio not found in the extracted data"
                    )
                
                # Assess credit risk based on leverage ratios
                if 'debt_to_equity' in ratios:
                    if ratios['debt_to_equity'] > 2.0:
                        risk_assessment["credit_risk"]["level"] = "High"
                        risk_assessment["credit_risk"]["factors"].append(
                            f"Debt-to-equity ratio ({ratios['debt_to_equity']:.2f}) exceeds 2.0, indicating high leverage"
                        )
                    elif ratios['debt_to_equity'] > 1.0:
                        risk_assessment["credit_risk"]["level"] = "Medium"
                        risk_assessment["credit_risk"]["factors"].append(
                            f"Debt-to-equity ratio ({ratios['debt_to_equity']:.2f}) exceeds 1.0"
                        )
                else:
                    risk_assessment["credit_risk"]["factors"].append(
                        "Debt-to-equity ratio not found in the extracted data"
                    )
                
                # Extract text for textual risk analysis
                text_content = data.get('text_content', '')
                
                # Continue with keyword-based analysis on text content
                financial_data = text_content
            else:
                # Fallback 
                text_content = extract_pdf_content(financial_data)
                financial_data = text_content
                
                # Initialize with standard risk assessment
                risk_assessment = {
                    "credit_risk": "Low",
                    "liquidity_risk": "Low",
                    "market_risk": "Medium",
                    "operational_risk": "Low",
                    "risk_factors": [],
                    "risk_score": 0
                }
        else:
            # Initialize with standard risk assessment for text input
            risk_assessment = {
                "credit_risk": "Low",
                "liquidity_risk": "Low",
                "market_risk": "Medium",
                "operational_risk": "Low",
                "risk_factors": [],
                "risk_score": 0
            }
        
        # Analyze text for risk indicators 
        risk_keywords = {
            "high_risk": [
                "loss", "deficit", "negative", "decline", "impairment", "default", "bankruptcy",
                "litigation", "lawsuit", "regulatory action", "severe", "critical", "violation"
            ],
            "medium_risk": [
                "volatile", "uncertainty", "fluctuation", "exposure", "concern", "challenging",
                "competitive pressure", "disruption", "moderate", "potential risk"
            ],
            "low_risk": [
                "stable", "strong", "positive", "growth", "improvement", "robust", "resilient",
                "conservative", "compliant", "well-positioned", "sustainable"
            ]
        }
        
        # Ensure financial_data is a string
        if not isinstance(financial_data, str):
            financial_data = str(financial_data)
            
        financial_data_lower = financial_data.lower()
        
        # Count risk keywords
        high_risk_count = sum(1 for keyword in risk_keywords["high_risk"] 
                            if keyword in financial_data_lower)
        medium_risk_count = sum(1 for keyword in risk_keywords["medium_risk"] 
                              if keyword in financial_data_lower)
        low_risk_count = sum(1 for keyword in risk_keywords["low_risk"] 
                           if keyword in financial_data_lower)
        
        # Calculate overall risk score with more sophisticated weighting
        risk_score = (high_risk_count * 3) + (medium_risk_count * 2) - (low_risk_count * 0.8)
        risk_score = max(0, risk_score)  # Ensure non-negative
        
        # Determine risk levels
        if risk_score > 12:
            # Update risk levels in the appropriate format based on risk_assessment structure
            if isinstance(risk_assessment["credit_risk"], dict):
                risk_assessment["credit_risk"]["level"] = "High"
                risk_assessment["liquidity_risk"]["level"] = "High"
            else:
                risk_assessment["credit_risk"] = "High"
                risk_assessment["liquidity_risk"] = "High"
        elif risk_score > 6:
            if isinstance(risk_assessment["credit_risk"], dict):
                risk_assessment["credit_risk"]["level"] = "Medium"
                risk_assessment["liquidity_risk"]["level"] = "Medium"
            else:
                risk_assessment["credit_risk"] = "Medium"
                risk_assessment["liquidity_risk"] = "Medium"
        
        # Set normalized risk score (0-100)
        risk_assessment["risk_score"] = max(0, min(100, risk_score * 8))
        
        # Add summary of risk findings
        if isinstance(risk_assessment.get("risk_summary"), str):
            if risk_score > 12:
                risk_assessment["risk_summary"] = "High overall risk detected. Multiple risk factors identified."
            elif risk_score > 6:
                risk_assessment["risk_summary"] = "Moderate overall risk detected. Several risk factors identified."
            else:
                risk_assessment["risk_summary"] = "Low overall risk detected. Few risk factors identified."
        
        # Extract specific risk factors mentioned
        lines = financial_data.split('\n')
        risk_factor_pattern = re.compile(r'.*risk.*', re.IGNORECASE)
        
        for line in lines:
            if risk_factor_pattern.match(line):
                # Clean up and limit length
                risk_factor = line.strip()
                if len(risk_factor) > 20:  # Minimum length to filter out noise
                    risk_assessment["risk_factors"].append(risk_factor[:200])
        
        # Add ESG risk factors if we can identify them
        esg_keywords = {
            "environmental": ["emissions", "carbon", "climate", "pollution", "waste", "environmental"],
            "social": ["labor", "diversity", "human rights", "community", "privacy", "health", "safety"],
            "governance": ["board", "compensation", "ethics", "compliance", "shareholder", "voting", "disclosure"]
        }
        
        # Check for ESG factors
        for category, keywords in esg_keywords.items():
            for keyword in keywords:
                if keyword in financial_data_lower:
                    pattern = re.compile(r'.*{}.*'.format(keyword), re.IGNORECASE)
                    for line in lines:
                        if pattern.match(line):
                            if isinstance(risk_assessment.get("esg_risk"), dict):
                                risk_assessment["esg_risk"]["factors"].append(
                                    f"{category.capitalize()}: {line.strip()[:150]}"
                                )
                            break
        
        return risk_assessment
        
    except Exception as e:
        return {
            "error": f"Error assessing risks: {str(e)}",
            "credit_risk": "Unknown",
            "liquidity_risk": "Unknown", 
            "market_risk": "Unknown",
            "operational_risk": "Unknown",
            "risk_factors": [f"Error during assessment: {str(e)}"],
            "risk_score": 0
        }

