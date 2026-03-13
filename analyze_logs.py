#!/usr/bin/env python3
"""
Log Analysis Script: Parse VS Code/Copilot logs with Gemini AI.
Extracts errors, actions, and suggests improvements.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# UPDATED: Import the new GenAI SDK
from google import genai

logger = logging.getLogger(__name__)

# ============================================================================
# LOG FILE DISCOVERY
# ============================================================================

def find_vscode_logs() -> List[str]:
    """Find VS Code log files on Windows."""
    log_paths = []
    
    if sys.platform == 'win32':
        # Windows VS Code logs
        appdata = os.getenv('APPDATA')
        if appdata:
            log_dir = Path(appdata) / 'Code' / 'logs'
            if log_dir.exists():
                log_paths.extend(log_dir.glob('**/*.log'))
    elif sys.platform == 'darwin':
        # macOS VS Code logs
        log_dir = Path.home() / 'Library' / 'Application Support' / 'Code' / 'logs'
        if log_dir.exists():
            log_paths.extend(log_dir.glob('**/*.log'))
    else:
        # Linux VS Code logs
        log_dir = Path.home() / '.config' / 'Code' / 'logs'
        if log_dir.exists():
            log_paths.extend(log_dir.glob('**/*.log'))
    
    return [str(p) for p in log_paths]

def find_app_logs() -> List[str]:
    """Find SP-LineBot app logs."""
    log_paths = []
    log_dir = Path('logs')
    
    if log_dir.exists():
        log_paths.extend(log_dir.glob('*.log'))
        
    return [str(p) for p in log_paths]

# ============================================================================
# LOG ANALYSIS (GEMINI AI)
# ============================================================================

def read_log_tail(file_path: str, lines: int = 200) -> str:
    """Read the last N lines of a log file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
            return ''.join(content[-lines:])
    except Exception as e:
        logger.error(f"Read log failed for {file_path}: {e}")
        return ""

def analyze_log_with_gemini(log_text: str) -> Dict[str, Any]:
    """Analyze log text with Gemini AI using the new SDK."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GEMINI_API_KEY not set")
            return {"error": "API key not set"}
        
        # UPDATED: Initialize the new Client
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Analyze this application log. Identify:
        1. Any errors or exceptions
        2. Key actions performed (e.g., users added, files embedded)
        3. Potential security or performance issues
        4. Summary of health status
        
        Log:
        {log_text[:5000]} # Limit size for token constraints
        
        Provide JSON output strictly with these keys: errors, actions, issues, status. Do not include markdown formatting like ```json.
        """
        
        logger.info("🔮 Sending log to Gemini 2.5 Flash for analysis...")
        
        # UPDATED: Use the new generation syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        result = response.text.strip()
        
        # Strip potential markdown formatting that Gemini might add
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
            
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning("Gemini output was not valid JSON. Returning raw text.")
            return {"raw_analysis": result}
            
    except Exception as e:
        logger.error(f"Gemini log analysis failed: {e}")
        return {"error": str(e)}

def analyze_all_logs() -> Dict[str, Any]:
    """Find and analyze all relevant logs."""
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'logs_analyzed': [],
        'summary': {'total_errors': 0}
    }
    
    # Analyze App Logs (Priority)
    app_logs = find_app_logs()
    for log_file in app_logs:
        logger.info(f"📄 Analyzing app log: {log_file}")
        log_text = read_log_tail(log_file, lines=100)
        
        if log_text:
            analysis = analyze_log_with_gemini(log_text)
            
            # Count errors if available
            errors = analysis.get('errors', [])
            if isinstance(errors, list):
                results['summary']['total_errors'] += len(errors)
                
            results['logs_analyzed'].append({
                'file': os.path.basename(log_file),
                'type': 'app_log',
                'analysis': analysis,
                'metadata': {
                    'size_bytes': os.path.getsize(log_file),
                    'errors_found': len(errors) if isinstance(errors, list) else 0
                }
            })
    
    return results

def save_analysis(results: Dict[str, Any], output_file: str = 'logs/analysis_report.json'):
    """Save analysis results to JSON."""
    try:
        Path(output_file).parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis saved to {output_file}")
    except Exception as e:
        logger.error(f"Save failed: {e}")

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 SP-LineBot Log Analyzer")
    print("=" * 60)
    
    # Run analysis
    results = analyze_all_logs()
    
    # Display summary
    print("\nAnalysis Summary:")
    print(json.dumps(results['summary'], indent=2))
    
    # Save report
    save_analysis(results)
    
    print("\nAnalysis complete! Report saved to logs/analysis_report.json")
    
    # Display first analysis
    if results['logs_analyzed']:
        first_log = results['logs_analyzed'][0]
        print(f"\nFirst Log: {first_log['file']}")
        print(f"Errors Found: {first_log['metadata'].get('errors_found', 0)}")
        
        if 'analysis' in first_log:
            print("\nAnalysis Status:")
            print(first_log['analysis'].get('status', 'N/A'))