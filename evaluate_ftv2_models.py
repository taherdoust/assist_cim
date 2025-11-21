#!/usr/bin/env python3
"""
FTv2 Model Evaluator
====================

Evaluates LLM models on FTv2 benchmark across three training modes:
1. Q2SQL: Question → SQL (EM, EX metrics)
2. QInst2SQL: Question + Instruction → SQL (EM, EX metrics)
3. Q2Inst: Question → Instruction (Semantic Similarity, Downstream Accuracy)

Metrics:
- EM (Exact Match): Generated output exactly matches ground truth
- EX (Execution Accuracy): Generated SQL produces same results
- EA (Eventual Accuracy): Accuracy with iteration and feedback (agent mode)
- Semantic Similarity: Cosine similarity of instruction embeddings
- Downstream Accuracy: Does generated instruction lead to correct SQL?

Model Sources:
- HuggingFace fine-tuned models: hf:model_name
- Ollama local models: ollama:model_name
- OpenRouter API models: openrouter:provider/model_name

EA (Agent Mode):
- Enables iteration with execution feedback (max iterations configurable)
- Model sees error message and tries again
- Calculates: first-shot accuracy, eventual accuracy, self-correction rate
- EA score: 1.0 (first-shot) or 1.0 - 0.15*(iterations-1) (corrected)

Usage:
    # Q2SQL mode (first-shot)
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model hf:taherdoust/qwen25-14b-cim-q2sql \
        --mode Q2SQL

    # Q2SQL mode (EA with agent)
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model hf:taherdoust/qwen25-14b-cim-q2sql \
        --mode Q2SQL \
        --agent_mode \
        --max_iterations 5

    # Plain model with schema context
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model ollama:qwen2.5-coder:14b \
        --mode Q2SQL \
        --include_schema

    # OpenRouter API model
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model openrouter:anthropic/claude-3.5-sonnet \
        --mode Q2SQL \
        --include_schema \
        --openrouter_api_key YOUR_KEY

Author: Ali Taherdoust
Date: November 2025
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import numpy as np
from tqdm import tqdm
import requests

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly


def sanitize_model_name(model_spec: str) -> str:
    """
    Convert a model spec into a filesystem-safe string.
    Examples:
        hf:taherdoust/qwen25 -> hf_taherdoust_qwen25
        ollama:qwen2.5 -> ollama_qwen2_5
    """
    safe_chars = []
    for char in model_spec:
        if char.isalnum() or char in ('_', '-'):
            safe_chars.append(char)
        elif char in ('.', '/', ':'):
            safe_chars.append('_')
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)


def prepare_artifact_file(
    artifacts_dir: Optional[Path],
    mode: str,
    model_spec: str
) -> Optional[Path]:
    """Return path to artifacts JSONL file (per model/mode) and reset it."""
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


def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code."""
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


def append_artifact(artifact_file: Optional[Path], record: Dict[str, Any]) -> None:
    """Append a JSON record to the artifact file."""
    if artifact_file is None:
        return
    with artifact_file.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_serializer) + '\n')


def remap_home_path(path: Path) -> Path:
    """
    If a path points to /home/<user>/... but the actual home is mounted elsewhere
    (e.g., /media/space/<user>), remap it automatically.
    """
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


def resolve_cli_path(
    path_value: str,
    ensure_exists: bool = False,
    description: str = "path"
) -> Path:
    """
    Normalize CLI-provided paths:
    - Expand user (~)
    - Remap legacy /home/<user> prefixes when HOME is elsewhere
    - Resolve relative paths against current working directory
    """
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
    """
    Detect if the provided HuggingFace model name is actually a local path.
    If so, resolve/remap it; otherwise return the original repo id.
    """
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
    resolved = resolve_cli_path(
        str(candidate),
        ensure_exists=True,
        description='Model path'
    )
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
        raise ValueError(f"Unknown model spec: {model_spec}. Use hf:model_name, ollama:model_name, or openrouter:provider/model_name")


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
    
    # Try to remove prompt if present
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return response


def extract_sql_from_response(response: str) -> str:
    """
    Extract clean SQL from model response, removing chat tags and explanations.
    Handles various formats: plain SQL, SQL with markdown, SQL with chat tags.
    """
    import re
    
    # Step 1: Remove ALL chat template tokens
    for marker in ['<|im_start|>', '<|im_end|>', '<|begin_of_text|>', '<|eot_id|>', 
                   '<|start_header_id|>', '<|end_header_id|>']:
        response = response.replace(marker, '')
    
    # Step 2: Remove role labels (assistant, user, system) as standalone words
    response = re.sub(r'\b(assistant|user|system)\b', '', response, flags=re.IGNORECASE)
    
    # Step 3: Split by "assistant" marker if present
    assistant_match = re.search(r'assistant\s*\n+(.*)', response, re.DOTALL | re.IGNORECASE)
    if assistant_match:
        response = assistant_match.group(1).strip()
    else:
        # Remove system/user/assistant labels at start of lines as fallback
        response = re.sub(r'^(system|user|assistant)\s*\n', '', response, flags=re.MULTILINE | re.IGNORECASE)
    
    # Step 4: Extract SQL from markdown code blocks if present
    code_block_match = re.search(r'```(?:sql)?\s*\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        # Step 5: Look for SELECT/WITH/INSERT/UPDATE/DELETE statements
        sql_match = re.search(
            r'\b((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*?)(?:\n\n|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if sql_match:
            sql = sql_match.group(1).strip()
        else:
            sql = response.strip()
    
    # Step 6: Remove trailing role labels
    sql = re.sub(r'\s+(assistant|user|system)\s*$', '', sql, flags=re.IGNORECASE)
    
    # Step 7: Remove trailing explanatory text after SQL
    sql = re.sub(r'\n\s*(?:Note|Explanation|This query).*$', '', sql, flags=re.IGNORECASE | re.DOTALL)
    
    return sql.strip()


def fix_common_sql_errors(sql: str) -> str:
    """
    Fix common SQL syntax errors from fine-tuned models.
    - Remove GROUP BY/ORDER BY after LIMIT
    - Remove duplicate LIMIT clauses
    - Stop at first semicolon
    """
    import re
    
    # Stop at first semicolon (model sometimes continues generating)
    if ';' in sql:
        sql = sql.split(';')[0] + ';'
    
    # Fix: GROUP BY/ORDER BY after LIMIT (invalid syntax)
    # Pattern: LIMIT <number> followed by GROUP BY or ORDER BY
    sql = re.sub(
        r'(LIMIT\s+\d+)\s+(GROUP\s+BY|ORDER\s+BY).*?(?=LIMIT|$)',
        r'\1',
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Remove duplicate LIMIT clauses (keep only the first one)
    limit_matches = list(re.finditer(r'LIMIT\s+\d+', sql, re.IGNORECASE))
    if len(limit_matches) > 1:
        # Keep everything up to and including the first LIMIT
        first_limit_end = limit_matches[0].end()
        sql = sql[:first_limit_end]
        if not sql.rstrip().endswith(';'):
            sql += ';'
    
    return sql.strip()


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
            'X-Title': 'CIM Wizard FTv2 Evaluation'
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


def get_minimal_schema() -> str:
    """Return the minimal schema summary used during fine-tuning."""
    return """- cim_vector: Building geometries, project scenarios, grid infrastructure
- cim_census: Italian census demographic data (ISTAT 2011)
- cim_raster: DTM/DSM raster data
- cim_network: Electrical grid network data"""


def create_prompt_q2sql(question: str, model_type: str, schema_context: Optional[str] = None, include_schema: bool = False, use_finetuned_schema: bool = False) -> str:
    """Create prompt for Q2SQL task."""
    schema_text = ""
    if use_finetuned_schema:
        # Use the exact mini-schema from training
        schema_text = get_minimal_schema()
    elif include_schema or schema_context:
        if schema_context:
            schema_text = schema_context
        else:
            # Load full schema for context
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
    else:  # openrouter or plain
        return f"""{system_msg}

Question: {question}

SQL Query:"""


def create_prompt_qinst2sql(question: str, instruction: str, model_type: str) -> str:
    """Create prompt for QInst2SQL task."""
    system_msg = """You are an expert in PostGIS spatial SQL for City Information Modeling.
Generate only the SQL query based on the question and instruction."""
    
    if model_type == 'qwen':
        return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
Question: {question}

Instruction: {instruction}<|im_end|>
<|im_start|>assistant
"""
    else:  # llama
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Instruction: {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def create_prompt_q2inst(question: str, model_type: str) -> str:
    """Create prompt for Q2Inst task."""
    system_msg = """You are an expert in spatial SQL reasoning for City Information Modeling.
Generate detailed reasoning instructions for converting the question to PostGIS spatial SQL."""
    
    if model_type == 'qwen':
        return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    else:  # llama
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def execute_sql(sql: str, db_uri: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL and return results along with metadata."""
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


def compute_em(generated: str, ground_truth: str) -> bool:
    """Compute Exact Match metric."""
    gen_clean = ' '.join(generated.strip().split())
    gt_clean = ' '.join(ground_truth.strip().split())
    return gen_clean.lower() == gt_clean.lower()


def is_subset_match(generated_result: list, ground_truth_result: list) -> bool:
    """
    Check if GENERATED results are a subset of GROUND TRUTH results.
    This allows for:
    - Model returning FEWER rows (e.g., with LIMIT)
    - Different ordering (ORDER BY variations)
    
    Returns True if all generated rows are present in ground truth.
    """
    if not generated_result:
        # Empty generated result is acceptable
        return True
    
    if not ground_truth_result:
        # Non-empty generated but empty ground truth: mismatch
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
        
        # FIX: Check if GENERATED is subset of GROUND TRUTH (not the other way!)
        return generated_set.issubset(ground_truth_set)
    except (TypeError, ValueError):
        # Fallback to exact match if unhashable types
        return generated_result == ground_truth_result


def calculate_deep_em(generated_sql: str, benchmark_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate deep Exact Match by comparing SQL structural features.
    
    Compares:
    - Spatial functions used
    - Spatial function count
    - Function count
    - Join count
    - Table count
    
    Returns dict with individual scores and overall deep_EM score.
    """
    import re
    
    # Extract features from generated SQL
    gen_spatial_funcs = re.findall(r'ST_\w+', generated_sql, re.IGNORECASE)
    gen_spatial_func_count = len(gen_spatial_funcs)
    
    # Extract tables
    gen_tables = []
    from_matches = re.findall(r'FROM\s+(\w+\.\w+|\w+)', generated_sql, re.IGNORECASE)
    gen_tables.extend(from_matches)
    join_matches = re.findall(r'JOIN\s+(\w+\.\w+|\w+)', generated_sql, re.IGNORECASE)
    gen_tables.extend(join_matches)
    gen_table_count = len(set(gen_tables))
    
    # Count joins
    gen_join_count = generated_sql.upper().count('JOIN')
    
    # Get ground truth values
    gt_spatial_funcs = set(benchmark_item.get('spatial_functions', []))
    gt_spatial_func_count = benchmark_item.get('spatial_function_count', 0)
    gt_function_count = benchmark_item.get('function_count', '0')
    gt_join_count = benchmark_item.get('join_count', '0')
    gt_table_count = benchmark_item.get('table_count', 0)
    
    # Convert join_count to int for comparison
    try:
        gt_join_count_int = int(gt_join_count.replace('+', ''))
    except:
        gt_join_count_int = 0
    
    # Calculate individual scores
    scores = {}
    
    # Spatial functions match (set comparison)
    gen_spatial_funcs_set = set(f.upper() for f in gen_spatial_funcs)
    gt_spatial_funcs_upper = set(f.upper() for f in gt_spatial_funcs)
    scores['spatial_functions_match'] = gen_spatial_funcs_set == gt_spatial_funcs_upper
    
    # Spatial function count match (exact or close)
    scores['spatial_func_count_match'] = abs(gen_spatial_func_count - gt_spatial_func_count) <= 1
    
    # Join count match
    if '+' in str(gt_join_count):
        scores['join_count_match'] = gen_join_count >= gt_join_count_int
    else:
        scores['join_count_match'] = abs(gen_join_count - gt_join_count_int) <= 1
    
    # Table count match
    scores['table_count_match'] = abs(gen_table_count - gt_table_count) <= 1
    
    # Calculate overall deep_EM score (average of individual scores)
    score_values = [1.0 if v else 0.0 for v in scores.values()]
    overall_score = sum(score_values) / len(score_values) if score_values else 0.0
    
    return {
        'deep_em_score': overall_score,
        'deep_em_pass': overall_score >= 0.75,  # Pass if 75% of features match
        'spatial_functions_match': scores['spatial_functions_match'],
        'spatial_func_count_match': scores['spatial_func_count_match'],
        'join_count_match': scores['join_count_match'],
        'table_count_match': scores['table_count_match'],
        'generated_spatial_funcs': list(gen_spatial_funcs_set),
        'generated_table_count': gen_table_count,
        'generated_join_count': gen_join_count
    }


def compute_ex(execution_output: Dict[str, Any], ground_truth_result: Any) -> bool:
    """
    Compute Execution Accuracy metric from an execution output.
    Uses subset matching: ground truth must be a subset of generated results.
    """
    if ground_truth_result is None:
        return False
    if not execution_output['success']:
        return False
    return is_subset_match(execution_output['result'], ground_truth_result)


def compute_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Compute semantic similarity between two texts."""
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


def calculate_performance_breakdowns(results: List[Dict[str, Any]], benchmark: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance breakdowns across difficulty dimensions.
    
    Dimensions analyzed:
    - Query Complexity (EASY, MEDIUM, HARD)
    - Spatial Complexity (NONE, BASIC, INTERMEDIATE, ADVANCED)
    - Schema Complexity (SINGLE_TABLE, SINGLE_SCHEMA, MULTI_SCHEMA)
    - Complexity Level (A, B, C)
    - SQL Type
    - Function Count (0, 1, 2, 3+)
    - Join Count (0, 1, 2+)
    - Question Tone
    """
    from collections import defaultdict
    
    breakdowns = {
        'query_complexity': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'spatial_complexity': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'schema_complexity': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'complexity_level': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'sql_type': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'function_count': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'join_count': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0}),
        'question_tone': defaultdict(lambda: {'total': 0, 'em_correct': 0, 'ex_correct': 0})
    }
    
    # Match results with benchmark metadata
    for result in results:
        benchmark_id = result['benchmark_id']
        # Find corresponding benchmark item
        bench_item = next((item for item in benchmark if item['benchmark_id'] == benchmark_id), None)
        
        if not bench_item:
            continue
        
        em = result.get('em', False)
        ex = result.get('ex', False)
        
        # Query complexity
        query_complexity = bench_item.get('query_complexity', 'UNKNOWN')
        breakdowns['query_complexity'][query_complexity]['total'] += 1
        if em:
            breakdowns['query_complexity'][query_complexity]['em_correct'] += 1
        if ex:
            breakdowns['query_complexity'][query_complexity]['ex_correct'] += 1
        
        # Spatial complexity
        spatial_complexity = bench_item.get('spatial_complexity', 'UNKNOWN')
        breakdowns['spatial_complexity'][spatial_complexity]['total'] += 1
        if em:
            breakdowns['spatial_complexity'][spatial_complexity]['em_correct'] += 1
        if ex:
            breakdowns['spatial_complexity'][spatial_complexity]['ex_correct'] += 1
        
        # Schema complexity
        schema_complexity = bench_item.get('schema_complexity', 'UNKNOWN')
        breakdowns['schema_complexity'][schema_complexity]['total'] += 1
        if em:
            breakdowns['schema_complexity'][schema_complexity]['em_correct'] += 1
        if ex:
            breakdowns['schema_complexity'][schema_complexity]['ex_correct'] += 1
        
        # Complexity level
        complexity_level = bench_item.get('complexity_level', 'UNKNOWN')
        breakdowns['complexity_level'][complexity_level]['total'] += 1
        if em:
            breakdowns['complexity_level'][complexity_level]['em_correct'] += 1
        if ex:
            breakdowns['complexity_level'][complexity_level]['ex_correct'] += 1
        
        # SQL type
        sql_type = bench_item.get('sql_type', 'UNKNOWN')
        breakdowns['sql_type'][sql_type]['total'] += 1
        if em:
            breakdowns['sql_type'][sql_type]['em_correct'] += 1
        if ex:
            breakdowns['sql_type'][sql_type]['ex_correct'] += 1
        
        # Function count
        function_count = bench_item.get('function_count', '0')
        breakdowns['function_count'][function_count]['total'] += 1
        if em:
            breakdowns['function_count'][function_count]['em_correct'] += 1
        if ex:
            breakdowns['function_count'][function_count]['ex_correct'] += 1
        
        # Join count
        join_count = bench_item.get('join_count', '0')
        breakdowns['join_count'][join_count]['total'] += 1
        if em:
            breakdowns['join_count'][join_count]['em_correct'] += 1
        if ex:
            breakdowns['join_count'][join_count]['ex_correct'] += 1
        
        # Question tone
        tone = (bench_item.get('question_tone') or 'UNKNOWN').upper()
        breakdowns['question_tone'][tone]['total'] += 1
        if em:
            breakdowns['question_tone'][tone]['em_correct'] += 1
        if ex:
            breakdowns['question_tone'][tone]['ex_correct'] += 1
    
    # Calculate accuracy percentages
    breakdown_results = {}
    for dimension, stats in breakdowns.items():
        breakdown_results[dimension] = {}
        for category, counts in stats.items():
            total = counts['total']
            if total > 0:
                breakdown_results[dimension][category] = {
                    'total': total,
                    'em_correct': counts['em_correct'],
                    'ex_correct': counts['ex_correct'],
                    'em_accuracy': counts['em_correct'] / total,
                    'ex_accuracy': counts['ex_correct'] / total
                }
    
    return breakdown_results


def compute_tone_robustness(
    breakdowns: Dict[str, Any],
    primary_tone: str
) -> Optional[Dict[str, float]]:
    """Compute robustness metrics for question tones."""
    tone_stats = breakdowns.get('question_tone')
    if not tone_stats:
        return None
    
    primary = (primary_tone or 'INTERROGATIVE').upper()
    
    def aggregate(stats_items):
        total = sum(item['total'] for item in stats_items)
        ex_correct = sum(item['ex_correct'] for item in stats_items)
        em_correct = sum(item['em_correct'] for item in stats_items)
        return {
            'total': total,
            'em_accuracy': (em_correct / total) if total else 0.0,
            'ex_accuracy': (ex_correct / total) if total else 0.0
        }
    
    primary_stats = tone_stats.get(primary)
    variants_stats = [v for k, v in tone_stats.items() if k != primary]
    
    primary_summary = aggregate([primary_stats]) if primary_stats else {'total': 0, 'em_accuracy': 0.0, 'ex_accuracy': 0.0}
    variant_summary = aggregate(variants_stats) if variants_stats else {'total': 0, 'em_accuracy': 0.0, 'ex_accuracy': 0.0}
    
    ex_gap = primary_summary['ex_accuracy'] - variant_summary['ex_accuracy']
    
    robustness_score = 0.0
    if primary_summary['ex_accuracy'] > 0:
        robustness_score = variant_summary['ex_accuracy'] / primary_summary['ex_accuracy']
    
    return {
        'primary_tone': primary,
        'primary_ex_accuracy': primary_summary['ex_accuracy'],
        'variant_ex_accuracy': variant_summary['ex_accuracy'],
        'ex_gap': ex_gap,
        'robustness_score': robustness_score
    }


def evaluate_q2sql(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    primary_tone: str,
    artifact_file: Optional[Path] = None,
    include_schema: bool = False,
    use_finetuned_schema: bool = False
) -> Dict[str, Any]:
    """Evaluate Q2SQL mode: Question → SQL."""
    print("\nEvaluating Q2SQL mode...")
    if use_finetuned_schema:
        print("Schema context: FINE-TUNED MINI-SCHEMA (from training)")
    elif include_schema:
        print("Schema context: FULL DATABASE SCHEMA")
    
    results = []
    em_correct = 0
    ex_correct = 0
    deep_em_correct = 0
    
    for item in tqdm(benchmark, desc="Evaluating"):
        question = item['question']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        prompt = create_prompt_q2sql(question, model_type, include_schema=include_schema, use_finetuned_schema=use_finetuned_schema)
        
        if model_dict['type'] == 'hf':
            raw_response = generate_hf(model_dict, prompt, max_tokens=512)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        elif model_dict['type'] == 'ollama':
            raw_response = generate_ollama(model_dict['model_name'], prompt)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        elif model_dict['type'] == 'openrouter':
            raw_response = generate_openrouter(model_dict['model_name'], model_dict['api_key'], prompt, max_tokens=512)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        
        execution_output = execute_sql(generated_sql, db_uri)
        em = compute_em(generated_sql, ground_truth_sql)
        ex = compute_ex(execution_output, ground_truth_result)
        deep_em_result = calculate_deep_em(generated_sql, item)
        
        if em:
            em_correct += 1
        if ex:
            ex_correct += 1
        if deep_em_result['deep_em_pass']:
            deep_em_correct += 1
        
        sample_record = {
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'ground_truth_sql': ground_truth_sql,
            'raw_response': raw_response,
            'generated_sql': generated_sql,
            'execution': execution_output,
            'em': em,
            'ex': ex,
            'deep_em': deep_em_result
        }
        results.append(sample_record)
        append_artifact(artifact_file, sample_record)
    
    total = len(benchmark)
    
    # Calculate performance breakdowns
    breakdowns = calculate_performance_breakdowns(results, benchmark)
    tone_robustness = compute_tone_robustness(breakdowns, primary_tone)
    
    # Calculate average deep_EM score
    avg_deep_em_score = sum(r['deep_em']['deep_em_score'] for r in results) / total if total > 0 else 0
    
    return {
        'mode': 'Q2SQL',
        'total_samples': total,
        'em_correct': em_correct,
        'ex_correct': ex_correct,
        'deep_em_correct': deep_em_correct,
        'em_accuracy': em_correct / total if total > 0 else 0,
        'ex_accuracy': ex_correct / total if total > 0 else 0,
        'deep_em_accuracy': deep_em_correct / total if total > 0 else 0,
        'average_deep_em_score': avg_deep_em_score,
        'tone_robustness': tone_robustness,
        'performance_breakdowns': breakdowns,
        'results': results
    }


def evaluate_qinst2sql(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    primary_tone: str,
    artifact_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate QInst2SQL mode: Question + Instruction → SQL."""
    print("\nEvaluating QInst2SQL mode...")
    
    results = []
    em_correct = 0
    ex_correct = 0
    
    for item in tqdm(benchmark, desc="Evaluating"):
        question = item['question']
        instruction = item['instruction']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        prompt = create_prompt_qinst2sql(question, instruction, model_type)
        
        if model_dict['type'] == 'hf':
            raw_response = generate_hf(model_dict, prompt, max_tokens=512)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        elif model_dict['type'] == 'ollama':
            raw_response = generate_ollama(model_dict['model_name'], prompt)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        elif model_dict['type'] == 'openrouter':
            raw_response = generate_openrouter(model_dict['model_name'], model_dict['api_key'], prompt, max_tokens=512)
            generated_sql = extract_sql_from_response(raw_response)
            generated_sql = fix_common_sql_errors(generated_sql)
        
        em = compute_em(generated_sql, ground_truth_sql)
        execution_output = execute_sql(generated_sql, db_uri)
        ex = compute_ex(execution_output, ground_truth_result)
        
        if em:
            em_correct += 1
        if ex:
            ex_correct += 1
        
        sample_record = {
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'instruction': instruction,
            'ground_truth_sql': ground_truth_sql,
            'raw_response': raw_response,
            'generated_sql': generated_sql,
            'execution': execution_output,
            'em': em,
            'ex': ex
        }
        results.append(sample_record)
        append_artifact(artifact_file, sample_record)
    
    total = len(benchmark)
    
    # Calculate performance breakdowns
    breakdowns = calculate_performance_breakdowns(results, benchmark)
    tone_robustness = compute_tone_robustness(breakdowns, primary_tone)
    
    return {
        'mode': 'QInst2SQL',
        'total_samples': total,
        'em_correct': em_correct,
        'ex_correct': ex_correct,
        'em_accuracy': em_correct / total if total > 0 else 0,
        'ex_accuracy': ex_correct / total if total > 0 else 0,
        'tone_robustness': tone_robustness,
        'performance_breakdowns': breakdowns,
        'results': results
    }


def evaluate_q2inst(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    downstream_model_dict: Optional[Dict] = None,
    downstream_model_type: Optional[str] = None,
    artifact_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate Q2Inst mode: Question → Instruction (Hybrid evaluation)."""
    print("\nEvaluating Q2Inst mode (Hybrid)...")
    print("Metrics: Semantic Similarity + Downstream SQL Accuracy")
    
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded semantic similarity model")
    
    results = []
    similarities = []
    downstream_correct = 0
    
    for item in tqdm(benchmark, desc="Evaluating"):
        question = item['question']
        ground_truth_instruction = item['instruction']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        prompt = create_prompt_q2inst(question, model_type)
        
        if model_dict['type'] == 'hf':
            generated_instruction = generate_hf(model_dict, prompt, max_tokens=512)
        elif model_dict['type'] == 'ollama':
            generated_instruction = generate_ollama(model_dict['model_name'], prompt)
        elif model_dict['type'] == 'openrouter':
            generated_instruction = generate_openrouter(model_dict['model_name'], model_dict['api_key'], prompt, max_tokens=512)
        
        similarity = compute_semantic_similarity(
            generated_instruction,
            ground_truth_instruction,
            sim_model
        )
        similarities.append(similarity)
        
        downstream_sql_correct = False
        generated_downstream_sql = None
        downstream_execution = None
        
        if downstream_model_dict is not None:
            prompt_downstream = create_prompt_qinst2sql(
                question,
                generated_instruction,
                downstream_model_type
            )
            
            if downstream_model_dict['type'] == 'hf':
                generated_downstream_sql = generate_hf(
                    downstream_model_dict,
                    prompt_downstream,
                    max_tokens=512
                )
            elif downstream_model_dict['type'] == 'ollama':
                generated_downstream_sql = generate_ollama(
                    downstream_model_dict['model_name'],
                    prompt_downstream
                )
            elif downstream_model_dict['type'] == 'openrouter':
                generated_downstream_sql = generate_openrouter(
                    downstream_model_dict['model_name'],
                    downstream_model_dict['api_key'],
                    prompt_downstream,
                    max_tokens=512
                )
            
            downstream_execution = execute_sql(
                generated_downstream_sql,
                db_uri
            )
            downstream_sql_correct = compute_ex(
                downstream_execution,
                ground_truth_result
            )
            if downstream_sql_correct:
                downstream_correct += 1
        
        sample_record = {
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'ground_truth_instruction': ground_truth_instruction,
            'generated_instruction': generated_instruction,
            'semantic_similarity': similarity,
            'downstream_sql': generated_downstream_sql,
            'downstream_execution': downstream_execution,
            'downstream_sql_correct': downstream_sql_correct
        }
        results.append(sample_record)
        append_artifact(artifact_file, sample_record)
    
    total = len(benchmark)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    return {
        'mode': 'Q2Inst',
        'total_samples': total,
        'average_semantic_similarity': avg_similarity,
        'downstream_accuracy': downstream_correct / total if total > 0 and downstream_model_dict else None,
        'results': results
    }


def print_breakdowns(breakdowns: Dict[str, Any]):
    """Print performance breakdowns in a formatted way."""
    print("\n" + "="*70)
    print("PERFORMANCE BREAKDOWN BY DIFFICULTY DIMENSIONS")
    print("="*70)
    
    dimension_names = {
        'query_complexity': 'Query Complexity',
        'spatial_complexity': 'Spatial Complexity',
        'schema_complexity': 'Schema Complexity',
        'complexity_level': 'Complexity Level',
        'sql_type': 'SQL Type',
        'function_count': 'Function Count',
        'join_count': 'Join Count',
        'question_tone': 'Question Tone'
    }
    
    for dimension, name in dimension_names.items():
        if dimension in breakdowns and breakdowns[dimension]:
            print(f"\n{name}:")
            print(f"  {'Category':<20} {'Total':<8} {'EM %':<10} {'EX %':<10}")
            print(f"  {'-'*50}")
            for category, stats in sorted(breakdowns[dimension].items()):
                total = stats['total']
                em_pct = stats['em_accuracy'] * 100
                ex_pct = stats['ex_accuracy'] * 100
                print(f"  {category:<20} {total:<8} {em_pct:<10.1f} {ex_pct:<10.1f}")


def save_results(evaluation_results: Dict[str, Any], output_file: Path):
    """Save evaluation results."""
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=json_serializer)
    
    print(f"Results saved")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FTv2 models on multi-task benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--benchmark', type=str, required=True,
                       help='FTv2 benchmark JSONL file')
    parser.add_argument('--model', type=str, required=True,
                       help='Model spec (hf:model_name or ollama:model_name)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['Q2SQL', 'QInst2SQL', 'Q2Inst'],
                       help='Evaluation mode')
    parser.add_argument('--model_type', type=str, default='qwen',
                       choices=['qwen', 'llama'],
                       help='Model type for prompt formatting (default: qwen)')
    parser.add_argument('--downstream_model', type=str,
                       help='Downstream model for Q2Inst evaluation (QInst2SQL model)')
    parser.add_argument('--downstream_model_type', type=str, default='qwen',
                       choices=['qwen', 'llama'],
                       help='Downstream model type (default: qwen)')
    parser.add_argument('--db_uri', type=str,
                       default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
                       help='Database URI')
    parser.add_argument('--output', type=str,
                       help='Output file for results (default: auto-generated)')
    parser.add_argument('--primary_tone', type=str,
                       default='INTERROGATIVE',
                       help='Primary question tone for robustness analysis (default: INTERROGATIVE)')
    parser.add_argument('--artifacts_dir', type=str,
                       default='ftv2_model_artifacts',
                       help='Directory for per-sample SQL/results artifacts')
    parser.add_argument('--no_artifacts', action='store_true',
                       help='Disable saving per-sample artifacts')
    parser.add_argument('--include_schema', action='store_true',
                       help='Include full database schema in prompts (for non-fine-tuned/frontier models)')
    parser.add_argument('--use_finetuned_schema', action='store_true',
                       help='Use the mini-schema from fine-tuning (for FTv2/FTv3 fine-tuned models)')
    parser.add_argument('--openrouter_api_key', type=str,
                       help='OpenRouter API key (can also use OPENROUTER_API_KEY env var)')
    
    args = parser.parse_args()
    
    # If using OpenRouter and no API key provided, try to load from environment
    if args.model.startswith('openrouter:') and not args.openrouter_api_key:
        args.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if args.openrouter_api_key:
            print("[env] Loaded OPENROUTER_API_KEY from environment")
    
    benchmark_path = resolve_cli_path(
        args.benchmark,
        ensure_exists=True,
        description='Benchmark file'
    )
    benchmark = load_benchmark(benchmark_path)
    
    model_dict = load_model(args.model, args.openrouter_api_key)
    
    downstream_model_dict = None
    if args.mode == 'Q2Inst' and args.downstream_model:
        print("\nLoading downstream model for Q2Inst evaluation...")
        downstream_model_dict = load_model(args.downstream_model, args.openrouter_api_key)
    
    artifact_file = None
    if not args.no_artifacts:
        artifact_base = resolve_cli_path(
            args.artifacts_dir,
            ensure_exists=False,
            description='Artifacts directory'
        )
        artifact_file = prepare_artifact_file(artifact_base, args.mode, args.model)
    
    if args.mode == 'Q2SQL':
        results = evaluate_q2sql(
            benchmark,
            model_dict,
            args.db_uri,
            args.model_type,
            args.primary_tone,
            artifact_file,
            args.include_schema,
            args.use_finetuned_schema
        )
    elif args.mode == 'QInst2SQL':
        results = evaluate_qinst2sql(
            benchmark,
            model_dict,
            args.db_uri,
            args.model_type,
            args.primary_tone,
            artifact_file
        )
    elif args.mode == 'Q2Inst':
        results = evaluate_q2inst(
            benchmark,
            model_dict,
            args.db_uri,
            args.model_type,
            downstream_model_dict,
            args.downstream_model_type,
            artifact_file
        )
    
    results['evaluation_timestamp'] = datetime.now().isoformat()
    results['model'] = args.model
    results['benchmark_file'] = str(benchmark_path)
    
    if args.output:
        output_file = resolve_cli_path(
            args.output,
            ensure_exists=False,
            description='Output file'
        )
    else:
        mode_lower = args.mode.lower()
        model_name = args.model.split('/')[-1].replace(':', '_')
        output_file = Path(f'ftv2_results_{mode_lower}_{model_name}.json')
    
    save_results(results, output_file)
    
    print("\n" + "="*70)
    print(f"FTv2 EVALUATION COMPLETE - {args.mode} MODE")
    print("="*70)
    
    if args.mode in ['Q2SQL', 'QInst2SQL']:
        print(f"\nResults:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  EM accuracy: {results['em_accuracy']*100:.2f}%")
        print(f"  EX accuracy: {results['ex_accuracy']*100:.2f}%")
        print(f"  Deep EM accuracy: {results['deep_em_accuracy']*100:.2f}%")
        print(f"  Avg deep EM score: {results['average_deep_em_score']:.3f}")
        if results.get('tone_robustness'):
            rb = results['tone_robustness']
            print(f"  Tone robustness (EX): primary {rb['primary_ex_accuracy']*100:.1f}%, "
                  f"variants {rb['variant_ex_accuracy']*100:.1f}%, gap {rb['ex_gap']*100:.1f}pp, "
                  f"score {rb['robustness_score']:.2f}")
        
        # Print performance breakdowns
        if 'performance_breakdowns' in results:
            print_breakdowns(results['performance_breakdowns'])
            sql_type_stats = results['performance_breakdowns'].get('sql_type', {})
            if sql_type_stats:
                ranked = sorted(sql_type_stats.items(), key=lambda x: x[1]['ex_accuracy'], reverse=True)
                if ranked:
                    best = ranked[0]
                    worst = ranked[-1]
                    print(f"\nSQL type EX accuracy range: best {best[0]} "
                          f"{best[1]['ex_accuracy']*100:.1f}%, worst {worst[0]} "
                          f"{worst[1]['ex_accuracy']*100:.1f}%")
    elif args.mode == 'Q2Inst':
        print(f"\nResults:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Average semantic similarity: {results['average_semantic_similarity']:.3f}")
        if results['downstream_accuracy'] is not None:
            print(f"  Downstream SQL accuracy: {results['downstream_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()

