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
- Semantic Similarity: Cosine similarity of instruction embeddings
- Downstream Accuracy: Does generated instruction lead to correct SQL?

Usage:
    # Q2SQL mode
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model hf:taherdoust/qwen25-14b-cim-q2sql \
        --mode Q2SQL

    # QInst2SQL mode
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model hf:taherdoust/qwen25-14b-cim-qinst2sql \
        --mode QInst2SQL

    # Q2Inst mode (hybrid evaluation)
    python evaluate_ftv2_models.py \
        --benchmark ../ai4db/ftv2_evaluation_benchmark.jsonl \
        --model hf:taherdoust/qwen25-14b-cim-q2inst \
        --mode Q2Inst \
        --downstream_model hf:taherdoust/qwen25-14b-cim-qinst2sql

Author: Ali Taherdoust
Date: October 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import numpy as np
from tqdm import tqdm


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


def load_model(model_spec: str):
    """Load model from HuggingFace or Ollama."""
    print(f"\nLoading model: {model_spec}")
    
    if model_spec.startswith('hf:'):
        model_name = model_spec[3:]
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
    
    else:
        raise ValueError(f"Unknown model spec: {model_spec}. Use hf:model_name or ollama:model_name")


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
    import requests
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={'model': model_name, 'prompt': prompt, 'stream': False}
    )
    
    return response.json()['response']


def create_prompt_q2sql(question: str, model_type: str) -> str:
    """Create prompt for Q2SQL task."""
    system_msg = """You are an expert in PostGIS spatial SQL for City Information Modeling.
Generate only the SQL query without explanations."""
    
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
    """Execute SQL and return results."""
    try:
        engine = create_engine(db_uri, poolclass=NullPool, echo=False)
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            return {
                'success': True,
                'result': [tuple(row) for row in rows],
                'error': None
            }
    except Exception as e:
        return {
            'success': False,
            'result': None,
            'error': str(e)
        }


def compute_em(generated: str, ground_truth: str) -> bool:
    """Compute Exact Match metric."""
    gen_clean = ' '.join(generated.strip().split())
    gt_clean = ' '.join(ground_truth.strip().split())
    return gen_clean.lower() == gt_clean.lower()


def compute_ex(generated_sql: str, ground_truth_result: Any, db_uri: str) -> bool:
    """Compute Execution Accuracy metric."""
    exec_result = execute_sql(generated_sql, db_uri)
    
    if not exec_result['success']:
        return False
    
    return exec_result['result'] == ground_truth_result


def compute_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Compute semantic similarity between two texts."""
    emb1 = model.encode(text1, convert_to_numpy=True)
    emb2 = model.encode(text2, convert_to_numpy=True)
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


def evaluate_q2sql(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str
) -> Dict[str, Any]:
    """Evaluate Q2SQL mode: Question → SQL."""
    print("\nEvaluating Q2SQL mode...")
    
    results = []
    em_correct = 0
    ex_correct = 0
    
    for item in tqdm(benchmark, desc="Evaluating"):
        question = item['question']
        ground_truth_sql = item['sql_postgis']
        ground_truth_result = item['expected_result']
        
        prompt = create_prompt_q2sql(question, model_type)
        
        if model_dict['type'] == 'hf':
            generated_sql = generate_hf(model_dict, prompt, max_tokens=512)
        else:
            generated_sql = generate_ollama(model_dict['model_name'], prompt)
        
        em = compute_em(generated_sql, ground_truth_sql)
        ex = False
        if ground_truth_result is not None:
            ex = compute_ex(generated_sql, ground_truth_result, db_uri)
        
        if em:
            em_correct += 1
        if ex:
            ex_correct += 1
        
        results.append({
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'ground_truth_sql': ground_truth_sql,
            'generated_sql': generated_sql,
            'em': em,
            'ex': ex
        })
    
    total = len(benchmark)
    return {
        'mode': 'Q2SQL',
        'total_samples': total,
        'em_correct': em_correct,
        'ex_correct': ex_correct,
        'em_accuracy': em_correct / total if total > 0 else 0,
        'ex_accuracy': ex_correct / total if total > 0 else 0,
        'results': results
    }


def evaluate_qinst2sql(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str
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
            generated_sql = generate_hf(model_dict, prompt, max_tokens=512)
        else:
            generated_sql = generate_ollama(model_dict['model_name'], prompt)
        
        em = compute_em(generated_sql, ground_truth_sql)
        ex = False
        if ground_truth_result is not None:
            ex = compute_ex(generated_sql, ground_truth_result, db_uri)
        
        if em:
            em_correct += 1
        if ex:
            ex_correct += 1
        
        results.append({
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'instruction': instruction,
            'ground_truth_sql': ground_truth_sql,
            'generated_sql': generated_sql,
            'em': em,
            'ex': ex
        })
    
    total = len(benchmark)
    return {
        'mode': 'QInst2SQL',
        'total_samples': total,
        'em_correct': em_correct,
        'ex_correct': ex_correct,
        'em_accuracy': em_correct / total if total > 0 else 0,
        'ex_accuracy': ex_correct / total if total > 0 else 0,
        'results': results
    }


def evaluate_q2inst(
    benchmark: List[Dict[str, Any]],
    model_dict: Dict,
    db_uri: str,
    model_type: str,
    downstream_model_dict: Optional[Dict] = None,
    downstream_model_type: Optional[str] = None
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
        else:
            generated_instruction = generate_ollama(model_dict['model_name'], prompt)
        
        similarity = compute_semantic_similarity(
            generated_instruction,
            ground_truth_instruction,
            sim_model
        )
        similarities.append(similarity)
        
        downstream_sql_correct = False
        generated_downstream_sql = None
        
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
            else:
                generated_downstream_sql = generate_ollama(
                    downstream_model_dict['model_name'],
                    prompt_downstream
                )
            
            if ground_truth_result is not None:
                downstream_sql_correct = compute_ex(
                    generated_downstream_sql,
                    ground_truth_result,
                    db_uri
                )
                if downstream_sql_correct:
                    downstream_correct += 1
        
        results.append({
            'benchmark_id': item['benchmark_id'],
            'question': question,
            'ground_truth_instruction': ground_truth_instruction,
            'generated_instruction': generated_instruction,
            'semantic_similarity': similarity,
            'downstream_sql': generated_downstream_sql,
            'downstream_sql_correct': downstream_sql_correct
        })
    
    total = len(benchmark)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    return {
        'mode': 'Q2Inst',
        'total_samples': total,
        'average_semantic_similarity': avg_similarity,
        'downstream_accuracy': downstream_correct / total if total > 0 and downstream_model_dict else None,
        'results': results
    }


def save_results(evaluation_results: Dict[str, Any], output_file: Path):
    """Save evaluation results."""
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FTv2 models on multi-task benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--benchmark', type=Path, required=True,
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
    parser.add_argument('--output', type=Path,
                       help='Output file for results (default: auto-generated)')
    
    args = parser.parse_args()
    
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)
    
    benchmark = load_benchmark(args.benchmark)
    
    model_dict = load_model(args.model)
    
    downstream_model_dict = None
    if args.mode == 'Q2Inst' and args.downstream_model:
        print("\nLoading downstream model for Q2Inst evaluation...")
        downstream_model_dict = load_model(args.downstream_model)
    
    if args.mode == 'Q2SQL':
        results = evaluate_q2sql(benchmark, model_dict, args.db_uri, args.model_type)
    elif args.mode == 'QInst2SQL':
        results = evaluate_qinst2sql(benchmark, model_dict, args.db_uri, args.model_type)
    elif args.mode == 'Q2Inst':
        results = evaluate_q2inst(
            benchmark,
            model_dict,
            args.db_uri,
            args.model_type,
            downstream_model_dict,
            args.downstream_model_type
        )
    
    results['evaluation_timestamp'] = datetime.now().isoformat()
    results['model'] = args.model
    results['benchmark_file'] = str(args.benchmark)
    
    if args.output:
        output_file = args.output
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
    elif args.mode == 'Q2Inst':
        print(f"\nResults:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Average semantic similarity: {results['average_semantic_similarity']:.3f}")
        if results['downstream_accuracy'] is not None:
            print(f"  Downstream SQL accuracy: {results['downstream_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()

