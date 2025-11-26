#!/usr/bin/env python3
"""
Unified Model Evaluator v2 with Taxonomy-Based Analysis
========================================================

Evaluates LLM models on spatial SQL benchmarks with comprehensive metrics
and taxonomy-based breakdown for detailed performance analysis.

Taxonomies:
- Task Taxonomy: SQL operation types (spatial/non-spatial)
- Domain Taxonomy: Schema usage patterns (single/multi-schema)

Metrics:
- EM (Exact Match): String-based SQL matching
- EX (Execution Accuracy): Result-based matching with subset logic
- Deep EM: Structural correctness (spatial functions, joins, tables)
- SC (Semantic Correctness): Execution + structure + non-empty results
- EA (Eventual Accuracy): Accuracy after iterative self-correction

Reports include:
- Overall metrics
- Metrics by task type
- Metrics by domain type
- Metrics by complexity level (1=Easy, 2=Medium, 3=Hard)
- Metrics by frequency level (1=Very Frequent, 2=Frequent, 3=Rare)

Usage:
    python evaluator_v2.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
        --model openrouter:openai/gpt-4o-mini \
        --mode Q2SQL \
        --max_iterations 5 \
        --include_schema

Author: Ali Taherdoust
Date: November 2025
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


# ============================================================================
# TASK TAXONOMY
# Complexity: 1=Easy, 2=Medium, 3=Hard
# Frequency: 1=Very Frequent, 2=Frequent, 3=Rare
# ============================================================================

TASK_TAXONOMY = {
    # Non-spatial SQL operations
    "SIMPLE_SELECT": {"complexity": 1, "frequency": 1, "description": "Simple SELECT with WHERE, no spatial operations"},
    "SQL_AGGREGATION": {"complexity": 1, "frequency": 1, "description": "Aggregation (COUNT, SUM, AVG) with GROUP BY, no spatial"},
    "SQL_JOIN": {"complexity": 2, "frequency": 2, "description": "Standard table join without spatial predicates"},
    "MULTI_SQL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple table joins (3+ tables)"},
    "NESTED_QUERY": {"complexity": 3, "frequency": 3, "description": "CTEs, subqueries, nested SELECT"},
    
    # Spatial predicates (relationships) - most_frequent
    "SPATIAL_PREDICATE": {"complexity": 1, "frequency": 1, "description": "Spatial predicates (ST_Intersects, ST_Contains, ST_Within, ST_Touches)"},
    "SPATIAL_PREDICATE_DISTANCE": {"complexity": 2, "frequency": 2, "description": "Distance-based predicates (ST_DWithin, ST_Overlaps, ST_Crosses, ST_Disjoint)"},
    
    # Spatial measurements - most_frequent
    "SPATIAL_MEASUREMENT": {"complexity": 1, "frequency": 1, "description": "Basic measurements (ST_Area, ST_Distance, ST_Length, ST_Perimeter)"},
    
    # Spatial processing - frequent to most_frequent
    "SPATIAL_PROCESSING": {"complexity": 2, "frequency": 1, "description": "Spatial processing (ST_Buffer, ST_Union, ST_Intersection, ST_Difference)"},
    
    # Spatial accessors - frequent
    "SPATIAL_ACCESSOR": {"complexity": 1, "frequency": 2, "description": "Coordinate extraction (ST_X, ST_Y, ST_Centroid, ST_Envelope)"},
    
    # Spatial constructors - most_frequent
    "SPATIAL_CONSTRUCTOR": {"complexity": 1, "frequency": 1, "description": "Geometry construction (ST_MakePoint, ST_GeomFromText, ST_Collect)"},
    
    # Spatial transforms - most_frequent
    "SPATIAL_TRANSFORM": {"complexity": 2, "frequency": 1, "description": "Coordinate transformation (ST_Transform, ST_SetSRID)"},
    
    # Spatial validation - most_frequent
    "SPATIAL_VALIDATION": {"complexity": 2, "frequency": 1, "description": "Geometry validation (ST_IsValid, ST_MakeValid)"},
    
    # Spatial joins
    "SPATIAL_JOIN": {"complexity": 2, "frequency": 1, "description": "Join using spatial predicates"},
    "MULTI_SPATIAL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple spatial joins with complex predicates"},
    
    # Advanced spatial operations - low_frequent
    "SPATIAL_CLUSTERING": {"complexity": 3, "frequency": 3, "description": "Spatial clustering (ST_ClusterDBSCAN, ST_ClusterKMeans)"},
    
    # Raster operations - frequent
    "RASTER_ANALYSIS": {"complexity": 3, "frequency": 2, "description": "Raster analysis and raster_accessor functions (ST_Value, ST_SummaryStats)"},
    "RASTER_VECTOR": {"complexity": 3, "frequency": 3, "description": "Raster-vector integration (ST_Clip, ST_Intersection with raster) and raster_processing functions like ST_Intersection_Raster"}
}

# ============================================================================
# DOMAIN TAXONOMY (CIM Wizard Schema Complexity)
# Complexity: 1=Easy, 2=Medium, 3=Hard
# Frequency: 1=Very Frequent, 2=Frequent, 3=Rare
# ============================================================================

DOMAIN_TAXONOMY = {
    "SINGLE_SCHEMA_CIM_VECTOR": {"complexity": 1, "frequency": 1, "description": "Single schema cim_vector only"},
    "MULTI_SCHEMA_WITH_CIM_VECTOR": {"complexity": 2, "frequency": 2, "description": "cim_vector + one other schema (census/network/raster)"},
    "SINGLE_SCHEMA_OTHER": {"complexity": 1, "frequency": 2, "description": "Single non-vector schema (census/network/raster only)"},
    "MULTI_SCHEMA_WITHOUT_CIM_VECTOR": {"complexity": 2, "frequency": 3, "description": "Multiple schemas without cim_vector"},
    "MULTI_SCHEMA_COMPLEX": {"complexity": 3, "frequency": 3, "description": "Three or more schemas combined"}
}

# ============================================================================
# QUESTION TONES
# ============================================================================

QUESTION_TONES = {
    "INTERROGATIVE": 0.70,
    "DIRECT": 0.20,
    "DESCRIPTIVE": 0.10
}

# ============================================================================
# SPATIAL FUNCTION PATTERNS FOR CLASSIFICATION
# ============================================================================

SPATIAL_PATTERNS = {
    "predicates": [
        r'ST_Intersects', r'ST_Contains', r'ST_Within', r'ST_Touches',
        r'ST_Equals', r'ST_Covers', r'ST_CoveredBy'
    ],
    "predicates_distance": [
        r'ST_DWithin', r'ST_Overlaps', r'ST_Crosses', r'ST_Disjoint'
    ],
    "measurements": [
        r'ST_Area', r'ST_Distance', r'ST_Length', r'ST_Perimeter',
        r'ST_3DDistance', r'ST_MaxDistance'
    ],
    "processing": [
        r'ST_Buffer', r'ST_Union', r'ST_Intersection(?!_Raster)', r'ST_Difference',
        r'ST_SymDifference', r'ST_ConvexHull', r'ST_Simplify'
    ],
    "accessors": [
        r'ST_X', r'ST_Y', r'ST_Z', r'ST_Centroid', r'ST_Envelope',
        r'ST_StartPoint', r'ST_EndPoint', r'ST_PointN', r'ST_GeometryN',
        r'ST_NumGeometries', r'ST_NumPoints', r'ST_SRID'
    ],
    "constructors": [
        r'ST_MakePoint', r'ST_GeomFromText', r'ST_Collect', r'ST_MakeLine',
        r'ST_MakePolygon', r'ST_GeomFromGeoJSON', r'ST_Point', r'ST_Polygon'
    ],
    "transforms": [
        r'ST_Transform', r'ST_SetSRID', r'ST_FlipCoordinates'
    ],
    "validation": [
        r'ST_IsValid', r'ST_MakeValid', r'ST_IsSimple', r'ST_IsClosed'
    ],
    "clustering": [
        r'ST_ClusterDBSCAN', r'ST_ClusterKMeans', r'ST_ClusterWithin'
    ],
    "raster_analysis": [
        r'ST_Value', r'ST_SummaryStats', r'ST_Histogram', r'ST_Band',
        r'ST_BandMetaData', r'ST_RasterToWorldCoord'
    ],
    "raster_vector": [
        r'ST_Clip', r'ST_Intersection_Raster', r'ST_AsRaster', r'ST_Resample'
    ]
}

# Schema patterns
SCHEMA_PATTERNS = {
    "cim_vector": [r'cim_vector\.', r'FROM\s+cim_vector', r'JOIN\s+cim_vector'],
    "cim_census": [r'cim_census\.', r'FROM\s+cim_census', r'JOIN\s+cim_census'],
    "cim_network": [r'cim_network\.', r'FROM\s+cim_network', r'JOIN\s+cim_network'],
    "cim_raster": [r'cim_raster\.', r'FROM\s+cim_raster', r'JOIN\s+cim_raster']
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


# ============================================================================
# TAXONOMY CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_task_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify SQL query into task taxonomy.
    Returns (task_type, metadata) where metadata includes complexity and frequency.
    """
    sql_upper = sql.upper()
    
    # Check for raster operations first (most specific)
    raster_vector_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_vector"])
    raster_analysis_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_analysis"])
    
    if raster_vector_found:
        return "RASTER_VECTOR", TASK_TAXONOMY["RASTER_VECTOR"]
    if raster_analysis_found:
        return "RASTER_ANALYSIS", TASK_TAXONOMY["RASTER_ANALYSIS"]
    
    # Check for clustering
    clustering_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["clustering"])
    if clustering_found:
        return "SPATIAL_CLUSTERING", TASK_TAXONOMY["SPATIAL_CLUSTERING"]
    
    # Count spatial joins (joins with spatial predicates)
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    spatial_predicates = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates"])
    spatial_predicates_dist = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates_distance"])
    
    # Check for spatial joins
    if join_count >= 2 and (spatial_predicates or spatial_predicates_dist):
        return "MULTI_SPATIAL_JOIN", TASK_TAXONOMY["MULTI_SPATIAL_JOIN"]
    if join_count >= 1 and (spatial_predicates or spatial_predicates_dist):
        return "SPATIAL_JOIN", TASK_TAXONOMY["SPATIAL_JOIN"]
    
    # Check for validation
    validation_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["validation"])
    if validation_found:
        return "SPATIAL_VALIDATION", TASK_TAXONOMY["SPATIAL_VALIDATION"]
    
    # Check for transforms
    transform_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["transforms"])
    if transform_found:
        return "SPATIAL_TRANSFORM", TASK_TAXONOMY["SPATIAL_TRANSFORM"]
    
    # Check for constructors
    constructor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["constructors"])
    if constructor_found:
        return "SPATIAL_CONSTRUCTOR", TASK_TAXONOMY["SPATIAL_CONSTRUCTOR"]
    
    # Check for accessors
    accessor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["accessors"])
    if accessor_found:
        return "SPATIAL_ACCESSOR", TASK_TAXONOMY["SPATIAL_ACCESSOR"]
    
    # Check for processing
    processing_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["processing"])
    if processing_found:
        return "SPATIAL_PROCESSING", TASK_TAXONOMY["SPATIAL_PROCESSING"]
    
    # Check for measurements
    measurement_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["measurements"])
    if measurement_found:
        return "SPATIAL_MEASUREMENT", TASK_TAXONOMY["SPATIAL_MEASUREMENT"]
    
    # Check for distance predicates
    if spatial_predicates_dist:
        return "SPATIAL_PREDICATE_DISTANCE", TASK_TAXONOMY["SPATIAL_PREDICATE_DISTANCE"]
    
    # Check for basic predicates
    if spatial_predicates:
        return "SPATIAL_PREDICATE", TASK_TAXONOMY["SPATIAL_PREDICATE"]
    
    # Non-spatial SQL operations
    has_cte = 'WITH' in sql_upper and 'AS' in sql_upper
    has_subquery = sql_upper.count('SELECT') > 1
    if has_cte or has_subquery:
        return "NESTED_QUERY", TASK_TAXONOMY["NESTED_QUERY"]
    
    if join_count >= 2:
        return "MULTI_SQL_JOIN", TASK_TAXONOMY["MULTI_SQL_JOIN"]
    
    if join_count == 1:
        return "SQL_JOIN", TASK_TAXONOMY["SQL_JOIN"]
    
    # Check for aggregation
    aggregation_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\('
    has_aggregation = re.search(aggregation_pattern, sql_upper)
    has_group_by = 'GROUP BY' in sql_upper
    if has_aggregation or has_group_by:
        return "SQL_AGGREGATION", TASK_TAXONOMY["SQL_AGGREGATION"]
    
    # Default to simple select
    return "SIMPLE_SELECT", TASK_TAXONOMY["SIMPLE_SELECT"]


def classify_domain_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify SQL query into domain taxonomy based on schema usage.
    Returns (domain_type, metadata) where metadata includes complexity and frequency.
    """
    schemas_used = set()
    
    for schema_name, patterns in SCHEMA_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                schemas_used.add(schema_name)
                break
    
    num_schemas = len(schemas_used)
    has_cim_vector = "cim_vector" in schemas_used
    
    if num_schemas >= 3:
        return "MULTI_SCHEMA_COMPLEX", DOMAIN_TAXONOMY["MULTI_SCHEMA_COMPLEX"]
    
    if num_schemas == 2:
        if has_cim_vector:
            return "MULTI_SCHEMA_WITH_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITH_CIM_VECTOR"]
        else:
            return "MULTI_SCHEMA_WITHOUT_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITHOUT_CIM_VECTOR"]
    
    if num_schemas == 1:
        if has_cim_vector:
            return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]
        else:
            return "SINGLE_SCHEMA_OTHER", DOMAIN_TAXONOMY["SINGLE_SCHEMA_OTHER"]
    
    # Default to single schema cim_vector if no schema detected
    return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]


def classify_question_tone(question: str) -> str:
    """Classify question tone based on structure."""
    question_lower = question.lower().strip()
    
    # Interrogative patterns
    interrogative_starts = ['what', 'which', 'where', 'who', 'how', 'why', 'when', 'is', 'are', 'can', 'do', 'does']
    if any(question_lower.startswith(start) for start in interrogative_starts) or question_lower.endswith('?'):
        return "INTERROGATIVE"
    
    # Direct patterns (imperative)
    direct_starts = ['find', 'get', 'list', 'show', 'select', 'calculate', 'compute', 'return', 'retrieve', 'identify']
    if any(question_lower.startswith(start) for start in direct_starts):
        return "DIRECT"
    
    # Default to descriptive
    return "DESCRIPTIVE"


def classify_sample(sql: str, question: str) -> Dict[str, Any]:
    """
    Classify a sample with full taxonomy information.
    Returns classification dictionary with task, domain, and question tone.
    """
    task_type, task_meta = classify_task_type(sql)
    domain_type, domain_meta = classify_domain_type(sql)
    question_tone = classify_question_tone(question)
    
    return {
        "task_type": task_type,
        "task_complexity": task_meta["complexity"],
        "task_frequency": task_meta["frequency"],
        "task_description": task_meta["description"],
        "domain_type": domain_type,
        "domain_complexity": domain_meta["complexity"],
        "domain_frequency": domain_meta["frequency"],
        "domain_description": domain_meta["description"],
        "question_tone": question_tone
    }


# ============================================================================
# MODEL LOADING AND GENERATION
# ============================================================================

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
            'X-Title': 'CIM Wizard EA Evaluation v2'
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


# ============================================================================
# SQL PROCESSING FUNCTIONS
# ============================================================================

def extract_sql_from_response(response: str) -> str:
    """Extract clean SQL from model response."""
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
        response = re.sub(r'^(system|user|assistant)\s*\n', '', response, flags=re.MULTILINE | re.IGNORECASE)
    
    # Step 4: Extract from code blocks
    code_block_match = re.search(r'```(?:sql)?\s*\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        # Step 5: Extract SQL statement
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
    
    # Step 7: Remove explanatory text after SQL
    sql = re.sub(r'\n\s*(?:Note|Explanation|This query).*$', '', sql, flags=re.IGNORECASE | re.DOTALL)
    
    return sql.strip()


def fix_common_sql_errors(sql: str) -> str:
    """Fix common SQL syntax errors from fine-tuned models."""
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


# ============================================================================
# PROMPT GENERATION
# ============================================================================

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


# ============================================================================
# LANGGRAPH AGENT
# ============================================================================

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
    Flexible result matching that focuses on data values, not structure.
    """
    if not generated_result:
        return not ground_truth_result or len(ground_truth_result) <= 1
    
    if not ground_truth_result:
        return False
    
    def normalize_value(val):
        if val is None:
            return 'NULL'
        if isinstance(val, (int, float)):
            if isinstance(val, float):
                return round(val, 6)
            return val
        if isinstance(val, str):
            return val.strip().lower()
        return str(val)
    
    def normalize_row(row):
        if not isinstance(row, (list, tuple)):
            row = [row]
        normalized = tuple(sorted([normalize_value(v) for v in row], key=str))
        return normalized
    
    try:
        gt_normalized = [normalize_row(row) for row in ground_truth_result]
        gen_normalized = [normalize_row(row) for row in generated_result]
        
        gt_set = set(gt_normalized)
        gen_set = set(gen_normalized)
        
        if gen_set.issubset(gt_set):
            return True
        
        intersection = len(gen_set.intersection(gt_set))
        if intersection > 0:
            overlap_ratio = intersection / len(gen_set)
            return overlap_ratio >= 0.8
        
        return False
        
    except (TypeError, ValueError):
        try:
            if len(generated_result) <= 3 and len(ground_truth_result) <= 3:
                return generated_result == ground_truth_result
        except:
            pass
        return False


def calculate_deep_em(generated_sql: str, benchmark_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate deep Exact Match by comparing SQL structural features.
    """
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
        gt_join_count_int = int(str(gt_join_count).replace('+', ''))
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
        'deep_em_pass': overall_score >= 0.75,
        'spatial_functions_match': scores['spatial_functions_match'],
        'spatial_func_count_match': scores['spatial_func_count_match'],
        'join_count_match': scores['join_count_match'],
        'table_count_match': scores['table_count_match'],
        'generated_spatial_funcs': list(gen_spatial_funcs_set),
        'generated_table_count': gen_table_count,
        'generated_join_count': gen_join_count
    }


def execute_sql_node(state: AgentState) -> AgentState:
    """Execute current SQL and update state."""
    execution_output = execute_sql(state['current_sql'], state['db_uri'])
    state['current_execution'] = execution_output
    
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
    
    workflow.add_node("generate", generate_sql_node)
    workflow.add_node("execute", execute_sql_node)
    workflow.add_node("correct", correction_node)
    
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


# ============================================================================
# METRIC CALCULATION
# ============================================================================

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


def calculate_semantic_correctness(sample: Dict[str, Any]) -> bool:
    """
    Calculate Semantic Correctness: execution success + deep EM + non-empty results.
    """
    exec_success = sample.get('final_execution', {}).get('success', False)
    deep_em_pass = sample.get('deep_em', {}).get('deep_em_pass', False)
    has_results = sample.get('final_execution', {}).get('rowcount', 0) > 0
    
    return exec_success and deep_em_pass and has_results


# ============================================================================
# TAXONOMY-BASED METRICS AGGREGATION
# ============================================================================

class TaxonomyMetrics:
    """Aggregates metrics by taxonomy categories."""
    
    def __init__(self):
        self.task_type_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.domain_type_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.task_complexity_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.task_frequency_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.domain_complexity_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.domain_frequency_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
        self.question_tone_metrics = defaultdict(lambda: {'total': 0, 'ex': 0, 'ea': 0, 'deep_em': 0, 'sc': 0})
    
    def add_sample(self, classification: Dict[str, Any], ex: bool, ea: bool, deep_em: bool, sc: bool):
        """Add a sample's metrics to all relevant categories."""
        # Task type
        task_type = classification['task_type']
        self.task_type_metrics[task_type]['total'] += 1
        self.task_type_metrics[task_type]['ex'] += int(ex)
        self.task_type_metrics[task_type]['ea'] += int(ea)
        self.task_type_metrics[task_type]['deep_em'] += int(deep_em)
        self.task_type_metrics[task_type]['sc'] += int(sc)
        
        # Domain type
        domain_type = classification['domain_type']
        self.domain_type_metrics[domain_type]['total'] += 1
        self.domain_type_metrics[domain_type]['ex'] += int(ex)
        self.domain_type_metrics[domain_type]['ea'] += int(ea)
        self.domain_type_metrics[domain_type]['deep_em'] += int(deep_em)
        self.domain_type_metrics[domain_type]['sc'] += int(sc)
        
        # Task complexity
        task_complexity = classification['task_complexity']
        self.task_complexity_metrics[task_complexity]['total'] += 1
        self.task_complexity_metrics[task_complexity]['ex'] += int(ex)
        self.task_complexity_metrics[task_complexity]['ea'] += int(ea)
        self.task_complexity_metrics[task_complexity]['deep_em'] += int(deep_em)
        self.task_complexity_metrics[task_complexity]['sc'] += int(sc)
        
        # Task frequency
        task_frequency = classification['task_frequency']
        self.task_frequency_metrics[task_frequency]['total'] += 1
        self.task_frequency_metrics[task_frequency]['ex'] += int(ex)
        self.task_frequency_metrics[task_frequency]['ea'] += int(ea)
        self.task_frequency_metrics[task_frequency]['deep_em'] += int(deep_em)
        self.task_frequency_metrics[task_frequency]['sc'] += int(sc)
        
        # Domain complexity
        domain_complexity = classification['domain_complexity']
        self.domain_complexity_metrics[domain_complexity]['total'] += 1
        self.domain_complexity_metrics[domain_complexity]['ex'] += int(ex)
        self.domain_complexity_metrics[domain_complexity]['ea'] += int(ea)
        self.domain_complexity_metrics[domain_complexity]['deep_em'] += int(deep_em)
        self.domain_complexity_metrics[domain_complexity]['sc'] += int(sc)
        
        # Domain frequency
        domain_frequency = classification['domain_frequency']
        self.domain_frequency_metrics[domain_frequency]['total'] += 1
        self.domain_frequency_metrics[domain_frequency]['ex'] += int(ex)
        self.domain_frequency_metrics[domain_frequency]['ea'] += int(ea)
        self.domain_frequency_metrics[domain_frequency]['deep_em'] += int(deep_em)
        self.domain_frequency_metrics[domain_frequency]['sc'] += int(sc)
        
        # Question tone
        question_tone = classification['question_tone']
        self.question_tone_metrics[question_tone]['total'] += 1
        self.question_tone_metrics[question_tone]['ex'] += int(ex)
        self.question_tone_metrics[question_tone]['ea'] += int(ea)
        self.question_tone_metrics[question_tone]['deep_em'] += int(deep_em)
        self.question_tone_metrics[question_tone]['sc'] += int(sc)
    
    def _compute_rates(self, metrics_dict: Dict) -> Dict:
        """Compute rates from counts."""
        result = {}
        for key, values in metrics_dict.items():
            total = values['total']
            if total > 0:
                result[key] = {
                    'total': total,
                    'ex_count': values['ex'],
                    'ea_count': values['ea'],
                    'deep_em_count': values['deep_em'],
                    'sc_count': values['sc'],
                    'ex_rate': values['ex'] / total,
                    'ea_rate': values['ea'] / total,
                    'deep_em_rate': values['deep_em'] / total,
                    'sc_rate': values['sc'] / total
                }
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete taxonomy metrics summary."""
        complexity_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
        frequency_labels = {1: "Very Frequent", 2: "Frequent", 3: "Rare"}
        
        # Format complexity and frequency metrics with labels
        task_complexity = {}
        for level, values in self._compute_rates(self.task_complexity_metrics).items():
            label = complexity_labels.get(level, str(level))
            task_complexity[f"{level}_{label}"] = values
            
        task_frequency = {}
        for level, values in self._compute_rates(self.task_frequency_metrics).items():
            label = frequency_labels.get(level, str(level))
            task_frequency[f"{level}_{label}"] = values
            
        domain_complexity = {}
        for level, values in self._compute_rates(self.domain_complexity_metrics).items():
            label = complexity_labels.get(level, str(level))
            domain_complexity[f"{level}_{label}"] = values
            
        domain_frequency = {}
        for level, values in self._compute_rates(self.domain_frequency_metrics).items():
            label = frequency_labels.get(level, str(level))
            domain_frequency[f"{level}_{label}"] = values
        
        return {
            'by_task_type': self._compute_rates(self.task_type_metrics),
            'by_domain_type': self._compute_rates(self.domain_type_metrics),
            'by_task_complexity': task_complexity,
            'by_task_frequency': task_frequency,
            'by_domain_complexity': domain_complexity,
            'by_domain_frequency': domain_frequency,
            'by_question_tone': self._compute_rates(self.question_tone_metrics)
        }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_ea_q2sql_v2(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    max_iterations: int,
    include_schema: bool,
    use_finetuned_schema: bool,
    artifact_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Evaluate Q2SQL with EA and taxonomy-based metrics."""
    print("\nEvaluating Q2SQL with EA (Eventual Accuracy) and Taxonomy Analysis...")
    print(f"Max iterations: {max_iterations}")
    if use_finetuned_schema:
        print("Schema context: FINE-TUNED MINI-SCHEMA")
    elif include_schema:
        print("Schema context: FULL DATABASE SCHEMA")
    
    agent = build_agent_graph()
    taxonomy_metrics = TaxonomyMetrics()
    
    results = []
    fs_correct = 0  # First-shot correct (EX)
    ea_correct = 0  # Eventually correct (EA)
    deep_em_correct = 0  # Deep EM correct
    sc_correct = 0  # Semantic correctness
    total_iterations = 0
    self_corrections = 0
    
    for item in tqdm(benchmark, desc="Evaluating EA with Taxonomy"):
        question = item['question']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        # Classify sample
        classification = classify_sample(ground_truth_sql, question)
        
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
        
        # Check first-shot accuracy (EX)
        first_shot_success = False
        if len(final_state['history']) > 0:
            if success and iterations_used == 1:
                first_shot_success = True
                fs_correct += 1
        
        # Check eventual accuracy (EA)
        if success:
            ea_correct += 1
            if iterations_used > 1:
                self_corrections += 1
        
        # Calculate EA score
        ea_score = calculate_ea_score(iterations_used, success, max_iterations)
        
        # Calculate deep_EM
        deep_em_result = calculate_deep_em(final_sql, item)
        deep_em_pass = deep_em_result['deep_em_pass']
        if deep_em_pass:
            deep_em_correct += 1
        
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
            'deep_em': deep_em_result,
            'classification': classification,
            'history': final_state['history']
        }
        
        # Calculate semantic correctness
        sc_pass = calculate_semantic_correctness(sample_record)
        if sc_pass:
            sc_correct += 1
        
        # Add to taxonomy metrics
        taxonomy_metrics.add_sample(
            classification,
            ex=first_shot_success,
            ea=success,
            deep_em=deep_em_pass,
            sc=sc_pass
        )
        
        results.append(sample_record)
        
        # Save artifact
        if artifact_file:
            with artifact_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(sample_record, ensure_ascii=False, default=json_serializer) + '\n')
    
    total = len(benchmark)
    avg_iterations = total_iterations / total if total > 0 else 0
    fs_accuracy = fs_correct / total if total > 0 else 0
    ea_accuracy = ea_correct / total if total > 0 else 0
    deep_em_accuracy = deep_em_correct / total if total > 0 else 0
    sc_accuracy = sc_correct / total if total > 0 else 0
    sc_rate = self_corrections / (total - fs_correct) if (total - fs_correct) > 0 else 0
    avg_ea_score = sum(r['ea_score'] for r in results) / total if total > 0 else 0
    avg_deep_em_score = sum(r['deep_em']['deep_em_score'] for r in results) / total if total > 0 else 0
    
    return {
        'mode': 'Q2SQL_EA_v2',
        'max_iterations': max_iterations,
        
        # Overall metrics
        'overall': {
            'total_samples': total,
            'first_shot_correct': fs_correct,
            'eventual_correct': ea_correct,
            'deep_em_correct': deep_em_correct,
            'semantic_correct': sc_correct,
            'self_corrections': self_corrections,
            'ex_accuracy': fs_accuracy,
            'ea_accuracy': ea_accuracy,
            'deep_em_accuracy': deep_em_accuracy,
            'semantic_correctness': sc_accuracy,
            'self_correction_rate': sc_rate,
            'average_iterations': avg_iterations,
            'average_ea_score': avg_ea_score,
            'average_deep_em_score': avg_deep_em_score
        },
        
        # Taxonomy-based metrics
        'taxonomy_metrics': taxonomy_metrics.get_summary(),
        
        # Detailed results
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


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: Dict[str, Any], model_name: str) -> str:
    """Generate a comprehensive evaluation report."""
    overall = results['overall']
    taxonomy = results['taxonomy_metrics']
    
    # Format timestamp
    timestamp = results.get('evaluation_timestamp', datetime.now().isoformat())
    if 'T' in timestamp:
        date_part, time_part = timestamp.split('T')
        time_part = time_part.split('.')[0]  # Remove microseconds
        formatted_timestamp = f"{date_part} {time_part}"
    else:
        formatted_timestamp = timestamp
    
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE EVALUATION REPORT - Evaluator v2")
    lines.append("=" * 80)
    lines.append(f"\n  Model: {model_name}")
    lines.append(f"  Evaluation Mode: {results['mode']}")
    lines.append(f"  Max Iterations: {results['max_iterations']}")
    lines.append(f"  Date & Time: {formatted_timestamp}")
    lines.append(f"  Benchmark: {results.get('benchmark_file', 'N/A')}")
    
    # Overall Metrics
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL METRICS")
    lines.append("=" * 80)
    lines.append(f"\n  Total Samples: {overall['total_samples']}")
    lines.append(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    lines.append(f"  │ Metric                          │ Count    │ Rate          │")
    lines.append(f"  ├─────────────────────────────────┼──────────┼───────────────┤")
    lines.append(f"  │ EX (First-shot Execution)       │ {overall['first_shot_correct']:>8} │ {overall['ex_accuracy']*100:>10.2f}%   │")
    lines.append(f"  │ EA (Eventual Accuracy)          │ {overall['eventual_correct']:>8} │ {overall['ea_accuracy']*100:>10.2f}%   │")
    lines.append(f"  │ Deep EM (Structural Match)      │ {overall['deep_em_correct']:>8} │ {overall['deep_em_accuracy']*100:>10.2f}%   │")
    lines.append(f"  │ SC (Semantic Correctness)       │ {overall['semantic_correct']:>8} │ {overall['semantic_correctness']*100:>10.2f}%   │")
    lines.append(f"  └─────────────────────────────────┴──────────┴───────────────┘")
    
    lines.append(f"\n  Additional Overall Metrics:")
    lines.append(f"    • Self-correction Rate: {overall['self_correction_rate']*100:.2f}%")
    lines.append(f"    • Average Iterations: {overall['average_iterations']:.2f}")
    lines.append(f"    • Average EA Score: {overall['average_ea_score']:.3f}")
    lines.append(f"    • Average Deep EM Score: {overall['average_deep_em_score']:.3f}")
    lines.append(f"    • Improvement (EA - EX): {(overall['ea_accuracy'] - overall['ex_accuracy'])*100:.2f} percentage points")
    
    # Helper function to format taxonomy table
    def format_taxonomy_table(title: str, data: Dict, sort_key: str = None) -> List[str]:
        table_lines = []
        table_lines.append(f"\n{title}")
        table_lines.append("-" * 78)
        table_lines.append(f"  {'Category':<30} │ {'N':>5} │ {'EX':>7} │ {'EA':>7} │ {'DeepEM':>7} │ {'SC':>7}")
        table_lines.append(f"  {'-'*30}─┼{'-'*7}┼{'-'*9}┼{'-'*9}┼{'-'*9}┼{'-'*9}")
        
        items = list(data.items())
        if sort_key:
            items.sort(key=lambda x: x[0])
        
        for category, metrics in items:
            ex_pct = metrics['ex_rate'] * 100
            ea_pct = metrics['ea_rate'] * 100
            deep_em_pct = metrics['deep_em_rate'] * 100
            sc_pct = metrics['sc_rate'] * 100
            table_lines.append(
                f"  {category:<30} │ {metrics['total']:>5} │ {ex_pct:>6.1f}% │ {ea_pct:>6.1f}% │ {deep_em_pct:>6.1f}% │ {sc_pct:>6.1f}%"
            )
        
        return table_lines
    
    # Taxonomy breakdowns
    lines.append("\n" + "=" * 80)
    lines.append("TAXONOMY-BASED METRICS")
    lines.append("=" * 80)
    
    # By Task Type
    lines.extend(format_taxonomy_table("BY TASK TYPE", taxonomy['by_task_type']))
    
    # By Domain Type
    lines.extend(format_taxonomy_table("BY DOMAIN TYPE", taxonomy['by_domain_type']))
    
    # By Task Complexity
    lines.extend(format_taxonomy_table("BY TASK COMPLEXITY (1=Easy, 2=Medium, 3=Hard)", taxonomy['by_task_complexity'], sort_key='level'))
    
    # By Task Frequency
    lines.extend(format_taxonomy_table("BY TASK FREQUENCY (1=Very Frequent, 2=Frequent, 3=Rare)", taxonomy['by_task_frequency'], sort_key='level'))
    
    # By Domain Complexity
    lines.extend(format_taxonomy_table("BY DOMAIN COMPLEXITY (1=Easy, 2=Medium, 3=Hard)", taxonomy['by_domain_complexity'], sort_key='level'))
    
    # By Domain Frequency
    lines.extend(format_taxonomy_table("BY DOMAIN FREQUENCY (1=Very Frequent, 2=Frequent, 3=Rare)", taxonomy['by_domain_frequency'], sort_key='level'))
    
    # By Question Tone
    lines.extend(format_taxonomy_table("BY QUESTION TONE", taxonomy['by_question_tone']))
    
    # Key Insights
    lines.append("\n" + "=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)
    
    # Find best/worst performing categories
    task_types = taxonomy['by_task_type']
    if task_types:
        best_task = max(task_types.items(), key=lambda x: x[1]['ea_rate'] if x[1]['total'] >= 3 else 0)
        worst_task = min(task_types.items(), key=lambda x: x[1]['ea_rate'] if x[1]['total'] >= 3 else 1)
        lines.append(f"\n  Task Type Performance:")
        lines.append(f"    • Best: {best_task[0]} (EA: {best_task[1]['ea_rate']*100:.1f}%, n={best_task[1]['total']})")
        lines.append(f"    • Worst: {worst_task[0]} (EA: {worst_task[1]['ea_rate']*100:.1f}%, n={worst_task[1]['total']})")
    
    domain_types = taxonomy['by_domain_type']
    if domain_types:
        best_domain = max(domain_types.items(), key=lambda x: x[1]['ea_rate'] if x[1]['total'] >= 3 else 0)
        worst_domain = min(domain_types.items(), key=lambda x: x[1]['ea_rate'] if x[1]['total'] >= 3 else 1)
        lines.append(f"\n  Domain Type Performance:")
        lines.append(f"    • Best: {best_domain[0]} (EA: {best_domain[1]['ea_rate']*100:.1f}%, n={best_domain[1]['total']})")
        lines.append(f"    • Worst: {worst_domain[0]} (EA: {worst_domain[1]['ea_rate']*100:.1f}%, n={worst_domain[1]['total']})")
    
    # Complexity analysis
    task_complexity = taxonomy['by_task_complexity']
    if task_complexity:
        lines.append(f"\n  Complexity Analysis (Task):")
        for level in sorted(task_complexity.keys()):
            metrics = task_complexity[level]
            lines.append(f"    • {level}: EA={metrics['ea_rate']*100:.1f}%, n={metrics['total']}")
    
    lines.append("\n" + "=" * 80)
    
    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models with EA and Taxonomy Analysis (v2)',
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
                       default='ea_model_artifacts_v2',
                       help='Directory for per-sample artifacts')
    parser.add_argument('--no_artifacts', action='store_true',
                       help='Disable saving per-sample artifacts')
    parser.add_argument('--include_schema', action='store_true',
                       help='Include full database schema in prompts')
    parser.add_argument('--use_finetuned_schema', action='store_true',
                       help='Use mini-schema from fine-tuning')
    parser.add_argument('--openrouter_api_key', type=str,
                       help='OpenRouter API key (can also use OPENROUTER_API_KEY env var)')
    parser.add_argument('--report_file', type=str,
                       help='Output file for human-readable report (default: auto-generated)')
    
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
        artifact_file = prepare_artifact_file(artifact_base, f"{args.mode}_EA_v2", args.model)
    
    results = evaluate_ea_q2sql_v2(
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
    
    # Generate output filenames
    mode_lower = args.mode.lower()
    model_name = args.model.split('/')[-1].replace(':', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output:
        output_file = resolve_cli_path(args.output, ensure_exists=False, description='Output file')
    else:
        output_file = Path(f'ea_v2_results_{mode_lower}_{model_name}_{timestamp}.json')
    
    save_results(results, output_file)
    
    # Generate and save report
    report = generate_report(results, args.model)
    
    if args.report_file:
        report_file = resolve_cli_path(args.report_file, ensure_exists=False, description='Report file')
    else:
        report_file = Path(f'ea_v2_report_{mode_lower}_{model_name}_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Print report to console
    print(report)


if __name__ == '__main__':
    main()

