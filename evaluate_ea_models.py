#!/usr/bin/env python3
"""
EA (Eventual Accuracy) Model Evaluator with LangGraph
======================================================

Evaluates LLM models with agentic self-correction using execution feedback.
Uses LangGraph for state management and iterative refinement.

Metrics:
- FS (First-Shot): Initial accuracy without feedback
- EA (Eventual Accuracy): Accuracy after iterative correction
- SC (Self-Correction Rate): % of failures corrected
- Avg Iterations: Average attempts needed for success

EA Scoring:
- First-shot success: 1.0
- Corrected in N iterations: 1.0 - 0.15 * (N - 1)
- Failed after max iterations: 0.0

Usage:
    python evaluate_ea_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
        --model hf:/path/to/model \
        --mode Q2SQL \
        --model_type llama \
        --max_iterations 5 \
        --use_finetuned_schema

Author: Ali Taherdoust
Date: November 2025
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated
import time
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import numpy as np
from tqdm import tqdm
import requests

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Import utility functions from evaluate_ftv2_models
def sanitize_model_name(model_spec: str) -> str:
    """Convert model spec to filesystem-safe string."""
    safe_chars = []
    for char in model_spec:
        if char.isalnum() or char in ('_', '-'):
            safe_chars.append(char)
        elif char in ('.', '/', ':'):
            safe_chars.append('_')
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)


def json_serializer(obj):
    """Custom JSON serializer for non-standard types."""
    from uuid import UUID
    from decimal import Decimal
    from datetime import datetime, date
    
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    raise TypeError(f"Type {type(obj)} not serializable")


def remap_home_path(path: Path) -> Path:
    """Remap /home/<user> to actual HOME if mounted elsewhere."""
    home = Path.home()
    user = os.getenv('USER', home.name)
    canonical_home = Path('/home') / user
    if not path.is_absolute():
        return path
    if canonical_home == home:
        return path
    try:
        relative = path.relative_to(canonical_home)
    except ValueError:
        return path
    return home / relative


def resolve_cli_path(path_value: str, ensure_exists: bool = False, description: str = "path") -> Path:
    """Normalize CLI paths with home remapping."""
    if path_value is None:
        raise ValueError(f"{description} is required.")
    original_display = str(path_value)
    candidate = Path(path_value).expanduser()
    before_remap = candidate
    candidate = remap_home_path(candidate)
    if candidate != before_remap and candidate != Path(original_display):
        print(f"[path] Remapped {description} from {original_display} to {candidate}")
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    if ensure_exists and not candidate.exists():
        print(f"Error: {description} not found: {candidate}")
        sys.exit(1)
    return candidate


def normalize_hf_model_name(model_name: str) -> str:
    """Detect and resolve local HF model paths."""
    candidate = Path(model_name).expanduser()
    is_windows_drive = len(model_name) > 1 and model_name[1] == ':' and model_name[0].isalpha()
    looks_like_path = (
        model_name.startswith('/')
        or model_name.startswith('./')
        or model_name.startswith('../')
        or model_name.startswith('~')
        or is_windows_drive
        or candidate.exists()
    )
    if not looks_like_path:
        return model_name
    resolved = resolve_cli_path(str(candidate), ensure_exists=True, description='Model path')
    return str(resolved)


def load_database_schema() -> str:
    """Load CIM database schema for context."""
    schema_file = Path(__file__).parent / "CIM_DATABASE_SCHEMA.txt"
    if schema_file.exists():
        with open(schema_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Warning: Schema file not found at {schema_file}")
        return "# CIM Database Schema\nSchema information not available."


def get_minimal_schema() -> str:
    """Return the minimal schema summary used during fine-tuning."""
    return """- cim_vector: Building geometries, project scenarios, grid infrastructure
- cim_census: Italian census demographic data (ISTAT 2011)
- cim_raster: DTM/DSM raster data
- cim_network: Electrical grid network data"""


def load_benchmark(benchmark_file: Path) -> List[Dict[str, Any]]:
    """Load FTv2 evaluation benchmark."""
    print(f"Loading benchmark from: {benchmark_file}")
    samples = []
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} benchmark items")
    return samples


def load_model(model_spec: str, openrouter_api_key: Optional[str] = None):
    """Load model from HuggingFace, Ollama, or OpenRouter."""
    print(f"\nLoading model: {model_spec}")
    
    if model_spec.startswith('hf:'):
        model_name = model_spec[3:]
        model_name = normalize_hf_model_name(model_name)
        print(f"Loading HuggingFace model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return {'type': 'hf', 'model': model, 'tokenizer': tokenizer}
    
    elif model_spec.startswith('ollama:'):
        model_name = model_spec[7:]
        print(f"Ollama model: {model_name}")
        return {'type': 'ollama', 'model_name': model_name}
    
    elif model_spec.startswith('openrouter:'):
        model_name = model_spec[11:]
        print(f"OpenRouter API model: {model_name}")
        if not openrouter_api_key:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or use --openrouter_api_key")
        else:
            api_key = openrouter_api_key
        return {'type': 'openrouter', 'model_name': model_name, 'api_key': api_key}
    
    else:
        raise ValueError(f"Unknown model spec: {model_spec}")


def generate_hf(model_dict: Dict, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from HuggingFace model."""
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response[len(prompt):].strip()
    return response


def generate_ollama(model_name: str, prompt: str) -> str:
    """Generate response from Ollama model."""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={'model': model_name, 'prompt': prompt, 'stream': False}
    )
    return response.json()['response']


def generate_openrouter(model_name: str, api_key: str, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from OpenRouter API model."""
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/taherdoust/coesi',
            'X-Title': 'CIM Wizard EA Evaluation'
        },
        json={
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.1
        },
        timeout=60
    )
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
    return response.json()['choices'][0]['message']['content']


def extract_sql_from_response(response: str) -> str:
    """Extract clean SQL from model response."""
    import re
    for marker in ['<|im_start|>', '<|im_end|>', '<|begin_of_text|>', '<|eot_id|>', 
                   '<|start_header_id|>', '<|end_header_id|>']:
        response = response.replace(marker, '')
    assistant_match = re.search(r'assistant\s*\n+(.*)', response, re.DOTALL | re.IGNORECASE)
    if assistant_match:
        response = assistant_match.group(1).strip()
    else:
        response = re.sub(r'^(system|user|assistant)\s*\n', '', response, flags=re.MULTILINE | re.IGNORECASE)
    code_block_match = re.search(r'```(?:sql)?\s*\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()
    sql_match = re.search(
        r'\b((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*?)(?:\n\n|$)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if sql_match:
        candidate = sql_match.group(1).strip()
        candidate = re.sub(r'\n\s*(?:Note|Explanation|This query).*$', '', candidate, flags=re.IGNORECASE | re.DOTALL)
        return candidate
    return response.strip()


def fix_common_sql_errors(sql: str) -> str:
    """Fix common SQL syntax errors from fine-tuned models."""
    import re
    if ';' in sql:
        sql = sql.split(';')[0] + ';'
    sql = re.sub(
        r'(LIMIT\s+\d+)\s+(GROUP\s+BY|ORDER\s+BY).*?(?=LIMIT|$)',
        r'\1',
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
    limit_matches = list(re.finditer(r'LIMIT\s+\d+', sql, re.IGNORECASE))
    if len(limit_matches) > 1:
        first_limit_end = limit_matches[0].end()
        sql = sql[:first_limit_end]
        if not sql.rstrip().endswith(';'):
            sql += ';'
    return sql.strip()


def execute_sql(sql: str, db_uri: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL and return results with metadata."""
    start_time = time.time()
    try:
        engine = create_engine(db_uri, poolclass=NullPool, echo=False)
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            duration_ms = (time.time() - start_time) * 1000
            return {
                'success': True,
                'result': [list(row) for row in rows],
                'rowcount': len(rows),
                'duration_ms': duration_ms,
                'error': None
            }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            'success': False,
            'result': None,
            'rowcount': 0,
            'duration_ms': duration_ms,
            'error': str(e)
        }


def create_prompt_q2sql(question: str, model_type: str, include_schema: bool = False, use_finetuned_schema: bool = False) -> str:
    """Create prompt for Q2SQL task."""
    schema_text = ""
    if use_finetuned_schema:
        schema_text = get_minimal_schema()
    elif include_schema:
        schema_text = load_database_schema()
    
    if schema_text:
        system_msg = f"""You are an expert in PostGIS spatial SQL for City Information Modeling (CIM).
Your task is to generate precise PostGIS spatial SQL queries for the CIM Wizard database.

Database Schema:
{schema_text}

Generate only the SQL query without explanations."""
    else:
        system_msg = """You are an expert in PostGIS spatial SQL for City Information Modeling.
Generate only the SQL query without explanations."""
    
    if model_type == 'qwen':
        return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    elif model_type == 'llama':
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        return f"""{system_msg}

Question: {question}

SQL Query:"""


def create_correction_prompt(question: str, failed_sql: str, error_msg: str, model_type: str, 
                            include_schema: bool = False, use_finetuned_schema: bool = False) -> str:
    """Create prompt for SQL correction with error feedback."""
    schema_text = ""
    if use_finetuned_schema:
        schema_text = get_minimal_schema()
    elif include_schema:
        schema_text = load_database_schema()
    
    schema_section = f"""
Database Schema:
{schema_text}
""" if schema_text else ""
    
    system_msg = f"""You are an expert in PostGIS spatial SQL for City Information Modeling (CIM).
Your previous SQL query had an error. Fix it based on the error message.
{schema_section}
Generate only the corrected SQL query without explanations."""
    
    if model_type == 'qwen':
        return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
Question: {question}

Previous SQL (FAILED):
{failed_sql}

Error:
{error_msg}

Please provide the corrected SQL query.<|im_end|>
<|im_start|>assistant
"""
    elif model_type == 'llama':
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Previous SQL (FAILED):
{failed_sql}

Error:
{error_msg}

Please provide the corrected SQL query.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        return f"""{system_msg}

Question: {question}

Previous SQL (FAILED):
{failed_sql}

Error:
{error_msg}

Corrected SQL Query:"""


# LangGraph State Definition
class AgentState(TypedDict):
    """State for the SQL generation agent."""
    question: str
    ground_truth_sql: str
    ground_truth_result: Any
    current_sql: str
    current_execution: Dict[str, Any]
    iteration: int
    max_iterations: int
    success: bool
    history: List[Dict[str, Any]]
    model_dict: Dict
    model_type: str
    db_uri: str
    include_schema: bool
    use_finetuned_schema: bool


def generate_sql_node(state: AgentState) -> AgentState:
    """Generate initial SQL from question."""
    question = state['question']
    model_dict = state['model_dict']
    model_type = state['model_type']
    
    prompt = create_prompt_q2sql(
        question,
        model_type,
        include_schema=state['include_schema'],
        use_finetuned_schema=state['use_finetuned_schema']
    )
    
    if model_dict['type'] == 'hf':
        raw_response = generate_hf(model_dict, prompt, max_tokens=512)
    elif model_dict['type'] == 'ollama':
        raw_response = generate_ollama(model_dict['model_name'], prompt)
    elif model_dict['type'] == 'openrouter':
        raw_response = generate_openrouter(model_dict['model_name'], model_dict['api_key'], prompt, max_tokens=512)
    
    generated_sql = extract_sql_from_response(raw_response)
    generated_sql = fix_common_sql_errors(generated_sql)
    
    state['current_sql'] = generated_sql
    state['iteration'] = 1
    state['history'].append({
        'iteration': 1,
        'raw_response': raw_response,
        'sql': generated_sql,
        'type': 'initial'
    })
    
    return state


def is_subset_match(generated_result: list, ground_truth_result: list) -> bool:
    """
    Check if ground truth results are a subset of generated results.
    This allows for:
    - Model returning more rows than ground truth (e.g., no LIMIT or higher LIMIT)
    - Different ordering (ORDER BY variations)
    
    Returns True if all ground truth rows are present in generated results.
    """
    if not ground_truth_result:
        # Empty ground truth matches empty or any result
        return True
    
    if not generated_result:
        # Non-empty ground truth doesn't match empty result
        return False
    
    # Convert to sets of tuples for efficient lookup
    # Handle nested lists by converting to tuples recursively
    def to_hashable(row):
        if isinstance(row, list):
            return tuple(to_hashable(item) if isinstance(item, list) else item for item in row)
        return row
    
    try:
        ground_truth_set = {to_hashable(row) for row in ground_truth_result}
        generated_set = {to_hashable(row) for row in generated_result}
        
        # Check if ground truth is subset of generated
        return ground_truth_set.issubset(generated_set)
    except (TypeError, ValueError):
        # Fallback to exact match if unhashable types
        return generated_result == ground_truth_result


def execute_sql_node(state: AgentState) -> AgentState:
    """Execute current SQL and update state."""
    execution_output = execute_sql(state['current_sql'], state['db_uri'])
    state['current_execution'] = execution_output
    
    # Check if execution matches ground truth (using subset matching)
    if execution_output['success'] and state['ground_truth_result'] is not None:
        if is_subset_match(execution_output['result'], state['ground_truth_result']):
            state['success'] = True
    
    return state


def correction_node(state: AgentState) -> AgentState:
    """Generate corrected SQL based on error feedback."""
    question = state['question']
    failed_sql = state['current_sql']
    error_msg = state['current_execution']['error']
    model_dict = state['model_dict']
    model_type = state['model_type']
    
    prompt = create_correction_prompt(
        question,
        failed_sql,
        error_msg,
        model_type,
        include_schema=state['include_schema'],
        use_finetuned_schema=state['use_finetuned_schema']
    )
    
    if model_dict['type'] == 'hf':
        raw_response = generate_hf(model_dict, prompt, max_tokens=512)
    elif model_dict['type'] == 'ollama':
        raw_response = generate_ollama(model_dict['model_name'], prompt)
    elif model_dict['type'] == 'openrouter':
        raw_response = generate_openrouter(model_dict['model_name'], model_dict['api_key'], prompt, max_tokens=512)
    
    generated_sql = extract_sql_from_response(raw_response)
    generated_sql = fix_common_sql_errors(generated_sql)
    
    state['current_sql'] = generated_sql
    state['iteration'] += 1
    state['history'].append({
        'iteration': state['iteration'],
        'raw_response': raw_response,
        'sql': generated_sql,
        'type': 'correction',
        'previous_error': error_msg
    })
    
    return state


def should_continue(state: AgentState) -> str:
    """Decide whether to continue iteration or end."""
    if state['success']:
        return "end"
    if state['iteration'] >= state['max_iterations']:
        return "end"
    if not state['current_execution']['success']:
        return "correct"
    return "end"


def build_agent_graph() -> StateGraph:
    """Build LangGraph workflow for iterative SQL generation."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate", generate_sql_node)
    workflow.add_node("execute", execute_sql_node)
    workflow.add_node("correct", correction_node)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "execute")
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "correct": "correct",
            "end": END
        }
    )
    workflow.add_edge("correct", "execute")
    
    return workflow.compile()


def calculate_ea_score(iteration: int, success: bool, max_iterations: int) -> float:
    """
    Calculate EA score with penalty for iterations.
    - First-shot success: 1.0
    - Success after N iterations: 1.0 - 0.15 * (N - 1)
    - Failed: 0.0
    """
    if not success:
        return 0.0
    if iteration == 1:
        return 1.0
    penalty = 0.15 * (iteration - 1)
    return max(0.0, 1.0 - penalty)


def evaluate_ea_q2sql(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    max_iterations: int,
    include_schema: bool,
    use_finetuned_schema: bool,
    artifact_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate Q2SQL with EA (Eventual Accuracy) using LangGraph agent."""
    print("\nEvaluating Q2SQL with EA (Eventual Accuracy)...")
    print(f"Max iterations: {max_iterations}")
    if use_finetuned_schema:
        print("Schema context: FINE-TUNED MINI-SCHEMA")
    elif include_schema:
        print("Schema context: FULL DATABASE SCHEMA")
    
    agent = build_agent_graph()
    
    results = []
    fs_correct = 0  # First-shot correct
    ea_correct = 0  # Eventually correct
    total_iterations = 0
    self_corrections = 0
    
    for item in tqdm(benchmark, desc="Evaluating EA"):
        question = item['question']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        # Initialize agent state
        initial_state = {
            'question': question,
            'ground_truth_sql': ground_truth_sql,
            'ground_truth_result': ground_truth_result,
            'current_sql': '',
            'current_execution': {},
            'iteration': 0,
            'max_iterations': max_iterations,
            'success': False,
            'history': [],
            'model_dict': model_dict,
            'model_type': model_type,
            'db_uri': db_uri,
            'include_schema': include_schema,
            'use_finetuned_schema': use_finetuned_schema
        }
        
        # Run agent
        final_state = agent.invoke(initial_state)
        
        # Extract results
        final_sql = final_state['current_sql']
        final_execution = final_state['current_execution']
        iterations_used = final_state['iteration']
        success = final_state['success']
        
        total_iterations += iterations_used
        
        # Check first-shot accuracy
        first_shot_success = False
        if len(final_state['history']) > 0:
            first_execution = final_state['history'][0]
            if success and iterations_used == 1:
                first_shot_success = True
                fs_correct += 1
        
        # Check eventual accuracy
        if success:
            ea_correct += 1
            if iterations_used > 1:
                self_corrections += 1
        
        # Calculate EA score
        ea_score = calculate_ea_score(iterations_used, success, max_iterations)
        
        sample_record = {
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'ground_truth_sql': ground_truth_sql,
            'final_sql': final_sql,
            'final_execution': final_execution,
            'iterations_used': iterations_used,
            'first_shot_success': first_shot_success,
            'eventual_success': success,
            'ea_score': ea_score,
            'history': final_state['history']
        }
        results.append(sample_record)
        
        # Save artifact
        if artifact_file:
            with artifact_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(sample_record, ensure_ascii=False, default=json_serializer) + '\n')
    
    total = len(benchmark)
    avg_iterations = total_iterations / total if total > 0 else 0
    fs_accuracy = fs_correct / total if total > 0 else 0
    ea_accuracy = ea_correct / total if total > 0 else 0
    sc_rate = self_corrections / (total - fs_correct) if (total - fs_correct) > 0 else 0
    avg_ea_score = sum(r['ea_score'] for r in results) / total if total > 0 else 0
    
    return {
        'mode': 'Q2SQL_EA',
        'max_iterations': max_iterations,
        'total_samples': total,
        'first_shot_correct': fs_correct,
        'eventual_correct': ea_correct,
        'self_corrections': self_corrections,
        'first_shot_accuracy': fs_accuracy,
        'eventual_accuracy': ea_accuracy,
        'self_correction_rate': sc_rate,
        'average_iterations': avg_iterations,
        'average_ea_score': avg_ea_score,
        'results': results
    }


def prepare_artifact_file(artifacts_dir: Optional[Path], mode: str, model_spec: str) -> Optional[Path]:
    """Prepare artifact file for logging."""
    if artifacts_dir is None:
        return None
    safe_model_name = sanitize_model_name(model_spec)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_dir = artifacts_dir / mode.lower()
    mode_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = mode_dir / f"{safe_model_name}_{timestamp}.jsonl"
    if artifact_file.exists():
        artifact_file.unlink()
    artifact_file.touch()
    return artifact_file


def save_results(evaluation_results: Dict[str, Any], output_file: Path):
    """Save evaluation results."""
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=json_serializer)
    print(f"Results saved")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models with EA (Eventual Accuracy) using LangGraph',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--benchmark', type=str, required=True,
                       help='FTv2 benchmark JSONL file')
    parser.add_argument('--model', type=str, required=True,
                       help='Model spec (hf:model_name, ollama:model_name, or openrouter:provider/model_name)')
    parser.add_argument('--mode', type=str, default='Q2SQL',
                       choices=['Q2SQL'],
                       help='Evaluation mode (currently only Q2SQL supported)')
    parser.add_argument('--model_type', type=str, default='qwen',
                       choices=['qwen', 'llama', 'plain'],
                       help='Model type for prompt formatting (default: qwen)')
    parser.add_argument('--max_iterations', type=int, default=5,
                       help='Maximum correction iterations (default: 5)')
    parser.add_argument('--db_uri', type=str,
                       default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
                       help='Database URI')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')
    parser.add_argument('--artifacts_dir', type=str,
                       default='ea_model_artifacts',
                       help='Directory for per-sample artifacts')
    parser.add_argument('--no_artifacts', action='store_true',
                       help='Disable saving per-sample artifacts')
    parser.add_argument('--include_schema', action='store_true',
                       help='Include full database schema in prompts')
    parser.add_argument('--use_finetuned_schema', action='store_true',
                       help='Use mini-schema from fine-tuning')
    parser.add_argument('--openrouter_api_key', type=str,
                       help='OpenRouter API key (can also use OPENROUTER_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Load API key from environment if needed
    if args.model.startswith('openrouter:') and not args.openrouter_api_key:
        args.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if args.openrouter_api_key:
            print("[env] Loaded OPENROUTER_API_KEY from environment")
    
    benchmark_path = resolve_cli_path(args.benchmark, ensure_exists=True, description='Benchmark file')
    benchmark = load_benchmark(benchmark_path)
    
    model_dict = load_model(args.model, args.openrouter_api_key)
    
    artifact_file = None
    if not args.no_artifacts:
        artifact_base = resolve_cli_path(args.artifacts_dir, ensure_exists=False, description='Artifacts directory')
        artifact_file = prepare_artifact_file(artifact_base, f"{args.mode}_EA", args.model)
    
    results = evaluate_ea_q2sql(
        benchmark,
        model_dict,
        args.db_uri,
        args.model_type,
        args.max_iterations,
        args.include_schema,
        args.use_finetuned_schema,
        artifact_file
    )
    
    results['evaluation_timestamp'] = datetime.now().isoformat()
    results['model'] = args.model
    results['benchmark_file'] = str(benchmark_path)
    
    if args.output:
        output_file = resolve_cli_path(args.output, ensure_exists=False, description='Output file')
    else:
        mode_lower = args.mode.lower()
        model_name = args.model.split('/')[-1].replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(f'ea_results_{mode_lower}_{model_name}_{timestamp}.json')
    
    save_results(results, output_file)
    
    print("\n" + "="*70)
    print(f"EA EVALUATION COMPLETE - {args.mode} MODE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  First-shot accuracy: {results['first_shot_accuracy']*100:.2f}%")
    print(f"  Eventual accuracy (EA): {results['eventual_accuracy']*100:.2f}%")
    print(f"  Self-correction rate: {results['self_correction_rate']*100:.2f}%")
    print(f"  Average iterations: {results['average_iterations']:.2f}")
    print(f"  Average EA score: {results['average_ea_score']:.3f}")
    print(f"\nImprovement: {(results['eventual_accuracy'] - results['first_shot_accuracy'])*100:.2f} percentage points")


if __name__ == '__main__':
    main()

