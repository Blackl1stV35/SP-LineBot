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

import google.generativeai as genai

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
    
    # SP-LineBot logs
    local_logs = Path('logs')
    if local_logs.exists():
        log_paths.extend(local_logs.glob('**/*.log'))
    
    return [str(p) for p in log_paths if p.is_file()]

# ============================================================================
# LOG PARSING
# ============================================================================

class LogParser:
    """Parse and structure logs."""
    
    ERROR_PATTERNS = {
        'error': r'\[ERROR\]|❌|Exception|Traceback',
        'warning': r'\[WARN\]|⚠️|Warning',
        'timeout': r'timeout|Timeout|TIMEOUT',
        'auth': r'auth|Auth|AUTH|permission|Permission',
        'memory': r'Memory|memory|OutOfMemory|OOM',
        'network': r'Connection|connection|ConnectionError|timeout'
    }
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lines = []
        self.errors = []
        self.warnings = []
        self.actions = []
        self.metadata = {}
    
    def load(self) -> bool:
        """Load log file."""
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                self.lines = f.readlines()
            
            logger.info(f"✅ Loaded {len(self.lines)} lines from {self.log_file}")
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
    
    def parse(self) -> Dict[str, Any]:
        """Parse log into structured format."""
        try:
            # Extract errors
            for line in self.lines:
                for error_type, pattern in self.ERROR_PATTERNS.items():
                    if re.search(pattern, line):
                        self.errors.append({'type': error_type, 'line': line.strip()})
                        break
            
            # Extract timestamps
            timestamps = []
            for line in self.lines:
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', line)
                if match:
                    timestamps.append(match.group())
            
            # Extract API calls (Gemini, Ollama, Drive, etc.)
            api_calls = []
            for line in self.lines:
                if any(api in line for api in ['gemini', 'ollama', 'drive', 'api', 'http']):
                    api_calls.append(line.strip())
            
            self.metadata = {
                'file': self.log_file,
                'lines_total': len(self.lines),
                'errors_found': len(self.errors),
                'timestamps': sorted(set(timestamps)),
                'api_calls_count': len(api_calls),
                'parsed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Parsed log: {len(self.errors)} errors, {len(api_calls)} API calls")
            return {
                'metadata': self.metadata,
                'errors': self.errors[:20],  # Top 20 errors
                'api_calls': api_calls[:10],
                'full_content': ''.join(self.lines[-1000:])  # Last 1000 lines for Gemini
            }
        except Exception as e:
            logger.error(f"Parse failed: {e}")
            return {'error': str(e)}

# ============================================================================
# GEMINI ANALYSIS
# ============================================================================

class GeminiLogAnalyzer:
    """Analyze logs with Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            logger.warning("GEMINI_API_KEY not set")
            self.model = None
    
    def analyze(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log with Gemini."""
        if not self.model:
            return {'error': 'Gemini not configured'}
        
        try:
            metadata = log_data.get('metadata', {})
            errors = log_data.get('errors', [])
            api_calls = log_data.get('api_calls', [])
            content = log_data.get('full_content', '')
            
            # Build prompt
            prompt = f"""
Analyze this SP-LineBot application log and provide:
1. **Critical Issues**: List the most severe errors
2. **Root Causes**: What likely caused each issue?
3. **Recommendations**: How to fix each issue
4. **Performance Notes**: Any bottlenecks or slow operations?
5. **Security Notes**: Any exposed credentials or risky patterns?

Metadata:
- File: {metadata.get('file')}
- Total Lines: {metadata.get('lines_total')}
- Errors Found: {metadata.get('errors_found')}
- API Calls: {metadata.get('api_calls_count')}

Errors Summary:
{json.dumps(errors[:5], indent=2)}

API Operations:
{json.dumps(api_calls[:3], indent=2)}

Log Content (last 500 chars):
{content[-500:] if content else 'N/A'}

Please be concise and actionable. Format as JSON.
"""
            
            logger.info("🔮 Sending log to Gemini for analysis...")
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': 1024,
                    'temperature': 0.3  # Low temperature for analytical output
                }
            )
            
            analysis_text = response.text.strip()
            
            # Try to parse as JSON
            try:
                analysis = json.loads(analysis_text)
            except:
                # If not JSON, wrap in dict
                analysis = {
                    'analysis': analysis_text,
                    'format': 'text'
                }
            
            logger.info("✅ Gemini analysis complete")
            return analysis
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {'error': str(e)}

# ============================================================================
# MAIN ANALYSIS WORKFLOW
# ============================================================================

def analyze_all_logs() -> Dict[str, Any]:
    """Analyze all available logs."""
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'logs_analyzed': [],
        'summary': {}
    }
    
    try:
        # Find logs
        log_files = find_vscode_logs()
        if not log_files:
            logger.warning("No log files found")
            return results
        
        logger.info(f"Found {len(log_files)} log files")
        
        # Parse logs
        analyzer = GeminiLogAnalyzer()
        
        for log_file in log_files[:5]:  # Analyze top 5 logs only
            try:
                parser = LogParser(log_file)
                if not parser.load():
                    continue
                
                log_data = parser.parse()
                
                # Analyze with Gemini
                analysis = analyzer.analyze(log_data)
                
                results['logs_analyzed'].append({
                    'file': log_file,
                    'metadata': parser.metadata,
                    'analysis': analysis
                })
            except Exception as e:
                logger.error(f"Process log {log_file} failed: {e}")
        
        # Summary
        results['summary'] = {
            'logs_processed': len(results['logs_analyzed']),
            'total_errors': sum(len(log.get('metadata', {}).get('errors_found', 0)) 
                               for log in results['logs_analyzed']),
            'status': 'complete'
        }
        
        return results
    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        results['error'] = str(e)
        return results

def save_analysis(results: Dict[str, Any], output_file: str = 'logs/analysis_report.json'):
    """Save analysis report."""
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Analysis saved to {output_file}")
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
    print("\n📊 Analysis Summary:")
    print(json.dumps(results['summary'], indent=2))
    
    # Save report
    save_analysis(results)
    
    print("\n✅ Analysis complete! Report saved to logs/analysis_report.json")
    
    # Display first analysis
    if results['logs_analyzed']:
        first_log = results['logs_analyzed'][0]
        print(f"\n📄 First Log: {first_log['file']}")
        print(f"Errors Found: {first_log['metadata'].get('errors_found', 0)}")
        
        if 'analysis' in first_log and isinstance(first_log['analysis'], dict):
            print("\n💡 AI Analysis Highlights:")
            for key, value in list(first_log['analysis'].items())[:3]:
                print(f"• {key}: {str(value)[:100]}...")
