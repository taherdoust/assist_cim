#!/usr/bin/env python3
"""
Model Evaluation Framework
==========================

Evaluates fine-tuned and baseline LLMs on CIM spatial SQL generation using
the evaluation benchmark. Supports models from HuggingFace, Ollama, and
remote servers (ipazia126).

Usage:
    # Standard evaluation (first-shot)
    python evaluate_models.py \
        --benchmark ../ai4db/evaluation_benchmark.jsonl \
        --model ollama:qwen2.5-coder:14b \
        --metric EX \
        --output results_qwen.json

    # Agent evaluation (with iteration and self-correction)
    python evaluate_models.py \
        --benchmark ../ai4db/evaluation_benchmark.jsonl \
        --model hf:taherdoust/llama-3.1-14b-cim-spatial-sql \
        --metric EA \
        --agent_mode \
        --max_iterations 5 \
        --output results_finetuned_agent.json

Metrics:
- EM (Exact Match): Generated SQL exactly matches ground truth (first-shot)
- EX (Execution Accuracy): Generated SQL produces same results (first-shot)
- EA (Eventual Accuracy): Model reaches correct answer with iteration and feedback
- NoErr (No Error): Generated SQL executes without errors (first-shot)

Agent Mode:
- Model can see execution results and error messages
- Can retry SQL generation with feedback from previous attempts
- Measures self-correction ability and eventual accuracy
- EA metric considers both correctness and iteration efficiency
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from collections import defaultdict

def load_benchmark(benchmark_file: Path) -> List[Dict[str, Any]]:
    """Load evaluation benchmark from JSONL file."""
    print(f"\nLoading benchmark from: {benchmark_file}")
    
    benchmark = []
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                benchmark.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(benchmark)} benchmark items")
    
    # Check if benchmark has expected results
    has_results = any(item.get('expected_result') is not None for item in benchmark)
    if not has_results:
        print("Warning: Benchmark does not contain expected results (EX/EA metrics unavailable)")
    
    return benchmark


def parse_model_spec(model_spec: str) -> Tuple[str, str]:
    """
    Parse model specification string.
    
    Format: <source>:<model_name>
    
    Sources:
    - ollama: Local Ollama model
    - hf: HuggingFace model
    - remote: Remote Ollama model
    
    Returns:
        (source, model_name)
    """
    if ':' not in model_spec:
        return ('ollama', model_spec)
    
    parts = model_spec.split(':', 1)
    source = parts[0].lower()
    model_name = parts[1]
    
    return (source, model_name)


def load_model(source: str, model_name: str, remote_url: Optional[str] = None):
    """
    Load model based on source type.
    
    Returns:
        Model object or connection info
    """
    print(f"\nLoading model: {source}:{model_name}")
    
    if source == 'ollama':
        # Load local Ollama model
        from langchain_ollama import ChatOllama
        
        model = ChatOllama(
            base_url="http://localhost:11434",
            model=model_name,
            temperature=0.0,
            request_timeout=120.0,
            num_predict=2048
        )
        
        print(f"Loaded Ollama model: {model_name}")
        return ('ollama', model)
    
    elif source == 'remote':
        # Load remote Ollama model
        if not remote_url:
            raise ValueError("remote_url must be provided for remote models")
        
        from langchain_ollama import ChatOllama
        
        model = ChatOllama(
            base_url=remote_url,
            model=model_name,
            temperature=0.0,
            request_timeout=120.0,
            num_predict=2048
        )
        
        print(f"Loaded remote Ollama model: {model_name} from {remote_url}")
        return ('ollama', model)
    
    elif source == 'hf':
        # Load HuggingFace model
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            print("Error: transformers library not installed")
            print("Install with: pip install transformers torch")
            sys.exit(1)
        
        print(f"Loading HuggingFace model: {model_name}")
        print("This may take a few minutes...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"Loaded HuggingFace model: {model_name}")
        return ('hf', (model, tokenizer))
    
    else:
        raise ValueError(f"Unknown model source: {source}")


def generate_sql_ollama(model, question: str, instruction: Optional[str] = None, 
                       feedback: Optional[str] = None) -> str:
    """Generate SQL using Ollama model."""
    
    # Build prompt
    if feedback:
        # Agent mode: provide feedback from previous attempt
        prompt = f"""You are an expert in PostGIS spatial SQL for City Information Modeling.

Previous attempt failed with the following feedback:
{feedback}

Question: {question}

Learn from the error and generate a corrected SQL query. Output ONLY the SQL query, no explanations.

SQL:"""
    elif instruction:
        # Two-stage: use instruction
        prompt = f"""You are an expert in PostGIS spatial SQL for City Information Modeling.

Instruction: {instruction}

Question: {question}

Generate the SQL query to answer this question. Output ONLY the SQL query, no explanations.

SQL:"""
    else:
        # Single-stage: direct question to SQL
        prompt = f"""You are an expert in PostGIS spatial SQL for City Information Modeling.

Question: {question}

Generate the SQL query to answer this question. Output ONLY the SQL query, no explanations.

SQL:"""
    
    # Generate
    response = model.invoke(prompt)
    sql = response.content.strip()
    
    # Clean up SQL (remove markdown code blocks if present)
    if sql.startswith('```'):
        lines = sql.split('\n')
        sql = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
    
    return sql.strip()


def generate_sql_hf(model, tokenizer, question: str, instruction: Optional[str] = None,
                   feedback: Optional[str] = None) -> str:
    """Generate SQL using HuggingFace model."""
    import torch
    
    # Build prompt (using Llama 3.1 format)
    if feedback:
        # Agent mode: provide feedback
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in PostGIS spatial SQL for City Information Modeling.<|eot_id|><|start_header_id|>user<|end_header_id|>

Previous attempt failed with the following feedback:
{feedback}

Question: {question}

Learn from the error and generate a corrected SQL query.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif instruction:
        # Two-stage: use instruction
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in PostGIS spatial SQL for City Information Modeling.<|eot_id|><|start_header_id|>user<|end_header_id|>

Instruction: {instruction}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        # Single-stage: direct question to SQL
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in PostGIS spatial SQL for City Information Modeling.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL (everything after the prompt)
    sql = generated_text[len(prompt):].strip()
    
    # Clean up SQL
    if sql.startswith('```'):
        lines = sql.split('\n')
        sql = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
    
    return sql.strip()


def generate_sql(model_info: Tuple[str, Any], question: str, instruction: Optional[str] = None,
                feedback: Optional[str] = None) -> str:
    """Generate SQL using appropriate model type."""
    source, model = model_info
    
    if source == 'ollama':
        return generate_sql_ollama(model, question, instruction, feedback)
    elif source == 'hf':
        model_obj, tokenizer = model
        return generate_sql_hf(model_obj, tokenizer, question, instruction, feedback)
    else:
        raise ValueError(f"Unknown model source: {source}")


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    # Convert to lowercase for case-insensitive comparison
    sql = sql.lower()
    return sql


def execute_sql(sql: str, engine) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
    """
    Execute SQL query.
    
    Returns:
        (success, result, error)
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout = 30000;"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            result_data = [tuple(row) for row in rows]
            return (True, result_data, None)
    except Exception as e:
        return (False, None, str(e))


def calculate_exact_match(generated_sql: str, expected_sql: str) -> bool:
    """Calculate Exact Match (EM) metric."""
    return normalize_sql(generated_sql) == normalize_sql(expected_sql)


def calculate_execution_accuracy(
    generated_sql: str, 
    expected_result: List[Tuple],
    engine
) -> bool:
    """Calculate Execution Accuracy (EX) metric."""
    success, result, error = execute_sql(generated_sql, engine)
    
    if not success:
        return False
    
    # Compare results
    return result == expected_result


def calculate_no_error(generated_sql: str, engine) -> bool:
    """Calculate No Error (NoErr) metric."""
    success, result, error = execute_sql(generated_sql, engine)
    return success


def calculate_eventual_accuracy_score(correct: bool, iterations: int, max_iterations: int) -> float:
    """
    Calculate Eventual Accuracy (EA) score.
    
    EA considers both correctness and efficiency:
    - Full credit (1.0) for correct answer on first try
    - Partial credit for correct answer after iterations
    - No credit (0.0) for incorrect final answer
    
    Score formula:
    EA = 1.0 if correct on first try
    EA = 1.0 - (0.15 * (iterations - 1)) if correct after retries
    EA = 0.0 if never correct
    
    Args:
        correct: Whether final answer is correct
        iterations: Number of iterations taken (1-indexed)
        max_iterations: Maximum allowed iterations
    
    Returns:
        EA score between 0.0 and 1.0
    """
    if not correct:
        return 0.0
    
    if iterations == 1:
        return 1.0
    
    # Penalize each retry by 15% (adjustable)
    penalty = 0.15 * (iterations - 1)
    score = max(0.0, 1.0 - penalty)
    
    return score


def evaluate_sample_agent_mode(
    model_info: Tuple[str, Any],
    question: str,
    instruction: Optional[str],
    expected_result: List[Tuple],
    engine,
    max_iterations: int
) -> Dict[str, Any]:
    """
    Evaluate single sample in agent mode with iteration.
    
    Returns:
        Dictionary with iteration history and final result
    """
    iteration_history = []
    
    for iteration in range(1, max_iterations + 1):
        # Generate SQL with feedback from previous attempt
        if iteration == 1:
            feedback = None
        else:
            prev_attempt = iteration_history[-1]
            if prev_attempt['error']:
                feedback = f"Error: {prev_attempt['error'][:500]}"
            else:
                feedback = f"Query executed but returned incorrect results. Expected {len(expected_result)} rows."
        
        # Generate SQL
        try:
            generated_sql = generate_sql(model_info, question, instruction, feedback)
        except Exception as e:
            generated_sql = ""
        
        # Execute SQL
        success, result, error = execute_sql(generated_sql, engine)
        
        # Check correctness
        is_correct = success and result == expected_result
        
        # Record this iteration
        iteration_history.append({
            'iteration': iteration,
            'sql': generated_sql,
            'success': success,
            'correct': is_correct,
            'error': error[:200] if error else None
        })
        
        # Stop if correct
        if is_correct:
            break
    
    # Calculate EA score
    final_correct = iteration_history[-1]['correct']
    final_iteration = len(iteration_history)
    ea_score = calculate_eventual_accuracy_score(final_correct, final_iteration, max_iterations)
    
    return {
        'iterations': iteration_history,
        'final_iteration': final_iteration,
        'first_shot_correct': iteration_history[0]['correct'],
        'eventually_correct': final_correct,
        'ea_score': ea_score
    }


def evaluate_model(
    model_info: Tuple[str, Any],
    benchmark: List[Dict[str, Any]],
    metric: str,
    db_uri: str,
    agent_mode: bool = False,
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Evaluate model on benchmark using specified metric.
    
    Args:
        model_info: Model information (source, model_object)
        benchmark: Evaluation benchmark
        metric: Evaluation metric (EM, EX, EA, NoErr)
        db_uri: Database connection string
        agent_mode: Enable agent mode with iteration (for EA metric)
        max_iterations: Maximum iterations in agent mode
    
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating model using {metric} metric on {len(benchmark)} samples...")
    if agent_mode:
        print(f"Agent mode enabled: max {max_iterations} iterations with feedback")
    
    # Connect to database if needed
    engine = None
    if metric in ['EX', 'EA', 'NoErr']:
        print(f"Connecting to database...")
        engine = create_engine(db_uri, poolclass=NullPool, echo=False)
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"Connected to: {version[:60]}...")
        except Exception as e:
            print(f"Error: Cannot connect to database: {e}")
            sys.exit(1)
    
    # Initialize results
    results = {
        'metric': metric,
        'agent_mode': agent_mode,
        'max_iterations': max_iterations if agent_mode else 1,
        'total_samples': len(benchmark),
        'correct': 0,
        'incorrect': 0,
        'sql_type_breakdown': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'difficulty_breakdown': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'sample_results': []
    }
    
    # Agent mode specific metrics
    if agent_mode:
        results['first_shot_correct'] = 0
        results['eventually_correct'] = 0
        results['self_correction_count'] = 0
        results['avg_iterations'] = 0.0
        results['avg_ea_score'] = 0.0
        results['total_iterations'] = 0
    
    start_time = time.time()
    
    for idx, item in enumerate(benchmark, 1):
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = idx / elapsed
            eta = (len(benchmark) - idx) / samples_per_sec
            accuracy = results['correct'] / idx * 100
            print(f"  Processed {idx}/{len(benchmark)} samples "
                  f"({accuracy:.1f}% accuracy, ETA: {eta:.0f}s)")
        
        question = item['question']
        instruction = item.get('instruction')
        expected_sql = item['sql_postgis']
        expected_result = item.get('expected_result')
        sql_type = item.get('sql_type', 'UNKNOWN')
        difficulty = item.get('difficulty_level', 'UNKNOWN')
        
        # Evaluate based on mode
        if agent_mode and metric == 'EA':
            # Agent mode evaluation
            if expected_result is None:
                # Can't do EA without expected result
                sample_result = {
                    'benchmark_id': item.get('benchmark_id', idx),
                    'question': question,
                    'error': 'No expected result in benchmark',
                    'correct': False
                }
                results['incorrect'] += 1
            else:
                agent_result = evaluate_sample_agent_mode(
                    model_info, question, instruction, expected_result,
                    engine, max_iterations
                )
                
                is_correct = agent_result['eventually_correct']
                first_shot = agent_result['first_shot_correct']
                
                # Update agent-specific metrics
                if first_shot:
                    results['first_shot_correct'] += 1
                if is_correct:
                    results['eventually_correct'] += 1
                    results['correct'] += 1
                    if not first_shot:
                        results['self_correction_count'] += 1
                else:
                    results['incorrect'] += 1
                
                results['total_iterations'] += agent_result['final_iteration']
                results['avg_ea_score'] += agent_result['ea_score']
                
                # Store sample result
                sample_result = {
                    'benchmark_id': item.get('benchmark_id', idx),
                    'question': question,
                    'expected_sql': expected_sql,
                    'iterations': agent_result['iterations'],
                    'final_iteration': agent_result['final_iteration'],
                    'first_shot_correct': first_shot,
                    'eventually_correct': is_correct,
                    'ea_score': agent_result['ea_score'],
                    'correct': is_correct,
                    'sql_type': sql_type,
                    'difficulty_level': difficulty
                }
        
        else:
            # Standard first-shot evaluation
            try:
                generated_sql = generate_sql(model_info, question, instruction)
            except Exception as e:
                print(f"\n  Warning: Generation failed for sample {idx}: {e}")
                generated_sql = ""
            
            # Evaluate based on metric
            is_correct = False
            error_msg = None
            
            if metric == 'EM':
                is_correct = calculate_exact_match(generated_sql, expected_sql)
            
            elif metric == 'EX':
                if expected_result is None:
                    error_msg = "No expected result in benchmark"
                else:
                    is_correct = calculate_execution_accuracy(generated_sql, expected_result, engine)
            
            elif metric == 'NoErr':
                is_correct = calculate_no_error(generated_sql, engine)
            
            # Update statistics
            if is_correct:
                results['correct'] += 1
            else:
                results['incorrect'] += 1
            
            # Store sample result
            sample_result = {
                'benchmark_id': item.get('benchmark_id', idx),
                'question': question,
                'expected_sql': expected_sql,
                'generated_sql': generated_sql,
                'correct': is_correct,
                'sql_type': sql_type,
                'difficulty_level': difficulty,
                'error': error_msg
            }
        
        # Update breakdown statistics
        results['sql_type_breakdown'][sql_type]['total'] += 1
        results['difficulty_breakdown'][difficulty]['total'] += 1
        if is_correct:
            results['sql_type_breakdown'][sql_type]['correct'] += 1
            results['difficulty_breakdown'][difficulty]['correct'] += 1
        
        results['sample_results'].append(sample_result)
    
    # Calculate overall accuracy
    results['accuracy'] = results['correct'] / results['total_samples']
    
    # Calculate agent-specific averages
    if agent_mode:
        results['avg_iterations'] = results['total_iterations'] / results['total_samples']
        results['avg_ea_score'] = results['avg_ea_score'] / results['total_samples']
        results['first_shot_accuracy'] = results['first_shot_correct'] / results['total_samples']
        results['eventual_accuracy'] = results['eventually_correct'] / results['total_samples']
        results['self_correction_rate'] = results['self_correction_count'] / results['total_samples']
    
    # Calculate breakdown rates
    results['sql_type_breakdown'] = dict(results['sql_type_breakdown'])
    results['difficulty_breakdown'] = dict(results['difficulty_breakdown'])
    
    for sql_type, stats in results['sql_type_breakdown'].items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    for difficulty, stats in results['difficulty_breakdown'].items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Timing
    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['time_per_sample'] = total_time / len(benchmark)
    
    return results


def print_evaluation_report(results: Dict[str, Any], model_spec: str):
    """Print evaluation report to console."""
    print("\n" + "="*80)
    print(f"MODEL EVALUATION REPORT")
    print("="*80)
    
    print(f"\nModel: {model_spec}")
    print(f"Metric: {results['metric']}")
    print(f"Mode: {'Agent (with iteration)' if results['agent_mode'] else 'Standard (first-shot)'}")
    print(f"Total samples: {results['total_samples']}")
    
    print(f"\nOverall Results:")
    if results['agent_mode']:
        print(f"  First-shot correct: {results['first_shot_correct']} ({results['first_shot_accuracy']*100:.2f}%)")
        print(f"  Eventually correct: {results['eventually_correct']} ({results['eventual_accuracy']*100:.2f}%)")
        print(f"  Self-corrections: {results['self_correction_count']} ({results['self_correction_rate']*100:.2f}%)")
        print(f"  Average EA score: {results['avg_ea_score']:.3f}")
        print(f"  Average iterations: {results['avg_iterations']:.2f}")
    else:
        print(f"  Correct: {results['correct']}")
        print(f"  Incorrect: {results['incorrect']}")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    
    print(f"\nTiming:")
    print(f"  Total time: {results['total_time']:.1f}s")
    print(f"  Time per sample: {results['time_per_sample']:.2f}s")
    
    print(f"\nAccuracy by SQL Type:")
    sorted_sql_types = sorted(results['sql_type_breakdown'].items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
    for sql_type, stats in sorted_sql_types[:10]:
        print(f"  {sql_type:25s}: {stats['correct']:3d}/{stats['total']:3d} "
              f"({stats['accuracy']*100:5.1f}%)")
    
    if len(sorted_sql_types) > 10:
        print(f"  ... and {len(sorted_sql_types) - 10} more SQL types")
    
    print(f"\nAccuracy by Difficulty:")
    sorted_difficulties = sorted(results['difficulty_breakdown'].items(), 
                                key=lambda x: x[1]['accuracy'], reverse=True)
    for difficulty, stats in sorted_difficulties:
        print(f"  {difficulty:15s}: {stats['correct']:3d}/{stats['total']:3d} "
              f"({stats['accuracy']*100:5.1f}%)")
    
    if results['agent_mode']:
        print(f"\nAgent Mode Analysis:")
        print(f"  Improvement from iteration: {(results['eventual_accuracy'] - results['first_shot_accuracy'])*100:.2f}%")
        print(f"  Self-correction success rate: {results['self_correction_rate']*100:.2f}%")
    
    print("\n" + "="*80)


def save_results(results: Dict[str, Any], output_file: Path):
    """Save evaluation results to JSON file."""
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on CIM spatial SQL generation benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--benchmark',
        type=Path,
        required=True,
        help='Evaluation benchmark JSONL file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model specification (format: source:model_name, e.g., ollama:qwen2.5-coder:14b, hf:taherdoust/llama-3.1-14b-cim-spatial-sql)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        required=True,
        choices=['EM', 'EX', 'EA', 'NoErr'],
        help='Evaluation metric (EM: Exact Match, EX: Execution Accuracy, EA: Eventual Accuracy, NoErr: No Error)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evaluation_results.json'),
        help='Output results JSON file (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--db_uri',
        type=str,
        default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
        help='Database URI (required for EX, EA, and NoErr metrics)'
    )
    
    parser.add_argument(
        '--remote_url',
        type=str,
        default=None,
        help='Remote Ollama URL (for remote: models)'
    )
    
    parser.add_argument(
        '--agent_mode',
        action='store_true',
        help='Enable agent mode with iteration and feedback (recommended for EA metric)'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=5,
        help='Maximum iterations in agent mode (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.metric == 'EA' and not args.agent_mode:
        print("Warning: EA metric is designed for agent mode. Consider using --agent_mode flag.")
    
    # Validate benchmark file
    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}")
        sys.exit(1)
    
    # Load benchmark
    benchmark = load_benchmark(args.benchmark)
    
    if len(benchmark) == 0:
        print("Error: Empty benchmark")
        sys.exit(1)
    
    # Parse model specification
    source, model_name = parse_model_spec(args.model)
    
    # Load model
    model_info = load_model(source, model_name, args.remote_url)
    
    # Evaluate model
    results = evaluate_model(
        model_info, 
        benchmark, 
        args.metric, 
        args.db_uri,
        args.agent_mode,
        args.max_iterations
    )
    
    # Add model info to results
    results['model_spec'] = args.model
    results['model_source'] = source
    results['model_name'] = model_name
    
    # Print report
    print_evaluation_report(results, args.model)
    
    # Save results
    save_results(results, args.output)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
