(ai4cimdb) castangia@ipazia:~/Ali_workspace$ python3 assist_cim/evaluate_ea_models.py \
>   --benchmark /home/castangia/Ali_workspace/ai4db/ftv2_evaluation_benchmark_100_easy.jsonl \
>   --model openrouter:openai/gpt-4o-mini \
>   --mode Q2SQL \
>   --max_iterations 5 \
>   --include_schema \
>   --db_uri postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated \
>   --artifacts_dir /home/castangia/Ali_workspace/ea_artifacts
[env] Loaded OPENROUTER_API_KEY from environment
[path] Remapped Benchmark file from /home/castangia/Ali_workspace/ai4db/ftv2_evaluation_benchmark_100_easy.jsonl to /media/space/castangia/Ali_workspace/ai4db/ftv2_evaluation_benchmark_100_easy.jsonl
Loading benchmark from: /media/space/castangia/Ali_workspace/ai4db/ftv2_evaluation_benchmark_100_easy.jsonl
Loaded 150 benchmark items

Loading model: openrouter:openai/gpt-4o-mini
OpenRouter API model: openai/gpt-4o-mini
[path] Remapped Artifacts directory from /home/castangia/Ali_workspace/ea_artifacts to /media/space/castangia/Ali_workspace/ea_artifacts

Evaluating Q2SQL with EA (Eventual Accuracy)...
Max iterations: 5
Schema context: FULL DATABASE SCHEMA
Evaluating EA: 100%|██████████████████████████| 150/150 [09:05<00:00,  3.63s/it]

Saving results to: ea_results_q2sql_gpt-4o-mini.json
Results saved

======================================================================
EA EVALUATION COMPLETE - Q2SQL MODE
======================================================================

Results:
  Total samples: 150
  First-shot accuracy: 0.00%
  Eventual accuracy (EA): 0.00%
  Self-correction rate: 0.00%
  Average iterations: 1.26
  Average EA score: 0.000

Improvement: 0.00 percentage points




