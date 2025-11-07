# FTv2 Evaluation Implementation Status

## Completed Updates

### 1. Database Schema Context
- Created `CIM_DATABASE_SCHEMA.txt` with complete database schema
- Includes all tables, columns, spatial functions, relationships
- Schema can be loaded and included in prompts for fair comparison

### 2. Documentation Updates
- Updated `README.md` with:
  - OpenRouter API support documentation
  - EA (Eventual Accuracy) agent mode explanation
  - Schema context usage for fair comparison
  - Example commands for all evaluation modes
  - Expected performance metrics

- Updated `thesis/sections/results.tex` with:
  - Evaluation infrastructure section
  - Model sources (HuggingFace, Ollama, OpenRouter)
  - Schema context for fair comparison
  - Agent mode (EA) evaluation methodology
  - Remote execution setup (reverse SSH tunnel)

### 3. Partial Implementation in `evaluate_ftv2_models.py`
- Added OpenRouter API support infrastructure
- Added schema loading function
- Updated model loading to support OpenRouter
- Updated prompt creation to include schema context (partially)
- Added `generate_openrouter()` function

## Remaining Implementation Tasks

### Critical: Complete `evaluate_ftv2_models.py`

The following functions need to be updated or added:

#### 1. Update Prompt Creation Functions
All prompt functions need schema context parameter:
- `create_prompt_qinst2sql()` - Add schema_context parameter
- `create_prompt_q2inst()` - Add schema_context parameter

#### 2. Add Agent Mode (EA) Support
Create new evaluation functions with iteration:
- `evaluate_q2sql_agent()` - Q2SQL with iteration and feedback
- `evaluate_qinst2sql_agent()` - QInst2SQL with iteration and feedback

Agent mode logic:
```python
def evaluate_with_agent_mode(question, model_dict, db_uri, max_iterations=5):
    for attempt in range(1, max_iterations + 1):
        # Generate SQL
        sql = generate_sql(question)
        
        # Execute and check
        result = execute_sql(sql, db_uri)
        
        if result['success'] and matches_ground_truth(result):
            return {
                'success': True,
                'attempts': attempt,
                'ea_score': 1.0 if attempt == 1 else 1.0 - 0.15 * (attempt - 1)
            }
        
        # Provide error feedback for next iteration
        error_feedback = f"Error: {result['error']}"
        # Continue with error feedback in next prompt
    
    return {'success': False, 'attempts': max_iterations, 'ea_score': 0.0}
```

#### 3. Update Generation Functions
Modify generate functions to support OpenRouter:
- `evaluate_q2sql()` - Add OpenRouter branch
- `evaluate_qinst2sql()` - Add OpenRouter branch
- `evaluate_q2inst()` - Add OpenRouter branch

Add generation dispatch:
```python
if model_dict['type'] == 'hf':
    generated = generate_hf(model_dict, prompt)
elif model_dict['type'] == 'ollama':
    generated = generate_ollama(model_dict['model_name'], prompt)
elif model_dict['type'] == 'openrouter':
    generated = generate_openrouter(
        model_dict['model_name'],
        model_dict['api_key'],
        prompt
    )
```

#### 4. Update main() Function
Add new command-line arguments:
- `--include_schema` - Boolean flag to include schema in prompts
- `--agent_mode` - Enable EA evaluation with iteration
- `--max_iterations` - Maximum attempts (default: 5)
- `--openrouter_api_key` - API key for OpenRouter

Load schema if requested:
```python
schema_context = None
if args.include_schema:
    schema_context = load_database_schema()
```

#### 5. EA Results Format
EA mode should output additional metrics:
```json
{
  "mode": "Q2SQL",
  "agent_mode": true,
  "total_samples": 100,
  "first_shot_correct": 87,
  "eventually_correct": 93,
  "first_shot_accuracy": 0.87,
  "eventual_accuracy": 0.93,
  "self_correction_count": 6,
  "self_correction_rate": 0.06,
  "average_iterations": 1.12,
  "average_ea_score": 0.91,
  "results": [
    {
      "benchmark_id": 1,
      "question": "...",
      "attempts": 1,
      "first_shot_correct": true,
      "eventually_correct": true,
      "ea_score": 1.0,
      "iterations_log": [...]
    }
  ]
}
```

## Testing Checklist

Once implementation is complete, test:

1. **HuggingFace model (fine-tuned)**
   ```bash
   python evaluate_ftv2_models.py \
     --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
     --model hf:taherdoust/llama31-8b-cim-q2sql \
     --mode Q2SQL \
     --model_type llama \
     --db_uri "postgresql://..." \
     --output test_hf.json
   ```

2. **Ollama model (plain) with schema**
   ```bash
   python evaluate_ftv2_models.py \
     --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
     --model ollama:qwen2.5-coder:14b \
     --mode Q2SQL \
     --model_type qwen \
     --include_schema \
     --db_uri "postgresql://..." \
     --output test_ollama_schema.json
   ```

3. **OpenRouter model with schema**
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   python evaluate_ftv2_models.py \
     --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
     --model openrouter:anthropic/claude-3.5-sonnet \
     --mode Q2SQL \
     --model_type openrouter \
     --include_schema \
     --db_uri "postgresql://..." \
     --output test_openrouter.json
   ```

4. **Agent mode (EA) evaluation**
   ```bash
   python evaluate_ftv2_models.py \
     --benchmark ../ai4db/ftv2_evaluation_benchmark_100.jsonl \
     --model hf:taherdoust/llama31-8b-cim-q2sql \
     --mode Q2SQL \
     --model_type llama \
     --agent_mode \
     --max_iterations 5 \
     --db_uri "postgresql://..." \
     --output test_ea.json
   ```

## Estimated Implementation Time

- Update prompt functions: 30 minutes
- Implement agent mode functions: 2-3 hours
- Update generation dispatch: 1 hour
- Update main() and argparse: 1 hour
- Testing and debugging: 2-3 hours

**Total: 6-8 hours**

## Priority Recommendations

1. **High Priority**: Complete agent mode (EA) - Critical for thesis evaluation
2. **High Priority**: Add schema context to all prompts - Required for fair comparison
3. **Medium Priority**: Full OpenRouter integration - Nice to have for frontier model comparison
4. **Low Priority**: Additional error handling and edge cases

## Notes for Implementation

- Schema context should be optional (flag controlled)
- Fine-tuned models trained on CIM data should NOT receive schema (redundant)
- Plain models and frontier models SHOULD receive schema (fair comparison)
- EA mode requires careful error message extraction and feedback formatting
- EA score penalty (0.15 per iteration) is configurable - consider making it a parameter
- OpenRouter rate limits may require retry logic
- Consider caching OpenRouter responses to avoid repeat API calls during debugging

## Current State

The groundwork is laid with:
- Schema file created and documented
- Model loading infrastructure supports all three sources
- OpenRouter API call function implemented
- Documentation updated in README and thesis

Next step: Complete the evaluation functions with agent mode and schema integration.

