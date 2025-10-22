# Chain of Thought Visualization

## New Feature: `query_agent_with_thinking()`

A new helper function that shows you **exactly how the agent thinks** and makes decisions step-by-step!

## Why This Is Useful

Understanding the agent's reasoning process helps you:
- 🔍 **Debug issues** - See where the agent goes wrong
- 📚 **Learn** - Understand how LLM agents work
- ⚡ **Optimize** - Identify slow steps or unnecessary tool calls
- 🎯 **Improve prompts** - See what instructions the agent follows (or ignores)

## Usage

### Basic Usage
```python
query_agent_with_thinking("How many buildings are in the database?")
```

### With Raw Data
```python
query_agent_with_thinking("Your question", show_raw=True)
```

### With Iteration Limit
```python
query_agent_with_thinking("Your question", max_iterations=10)
```

## Example Output

Here's what you'll see:

```
================================================================================
🤔 QUESTION: How many buildings are in the database?
================================================================================

────────────────────────────────────────────────────────────────────────────────
🧠 AGENT THINKING (Step 1, 0.5s)
────────────────────────────────────────────────────────────────────────────────

  💡 Decision: Call tool 'sql_db_schema'
     → Examining schema of table: cim_wizard_building

────────────────────────────────────────────────────────────────────────────────
🔧 TOOL RESULT (Step 2, 1.2s)
────────────────────────────────────────────────────────────────────────────────
  📊 Table schema:
     
     CREATE TABLE cim_wizard_building (
         building_id UUID DEFAULT public.uuid_generate_v4() NOT NULL, 
         lod INTEGER NOT NULL, 
         ...
     )

────────────────────────────────────────────────────────────────────────────────
🧠 AGENT THINKING (Step 3, 2.1s)
────────────────────────────────────────────────────────────────────────────────

  💡 Decision: Call tool 'sql_db_query'
     → Executing SQL query:
        SELECT COUNT(*) FROM cim_wizard_building;

────────────────────────────────────────────────────────────────────────────────
🔧 TOOL RESULT (Step 4, 2.8s)
────────────────────────────────────────────────────────────────────────────────
  ✓ Query succeeded:
     [(1234,)]

────────────────────────────────────────────────────────────────────────────────
✅ FINAL ANSWER (Step 5, 3.5s)
────────────────────────────────────────────────────────────────────────────────

There are 1,234 buildings in the database.

================================================================================
📊 SUMMARY:
   • Total steps: 5
   • Tool calls: 2
   • Time: 3.50s
================================================================================
```

## What You See

### 🧠 Agent Thinking
Shows when the agent decides to use a tool:
- **sql_db_list_tables** - Checking available tables
- **sql_db_schema** - Examining table structure
- **sql_db_query_checker** - Validating SQL query
- **sql_db_query** - Executing SQL query

### 🔧 Tool Results
Shows what each tool returns:
- ✓ Success (with formatted output)
- ❌ Error (with error message)
- Truncated for readability (first 10-20 lines)

### ✅ Final Answer
The agent's final response to your question

### 📊 Summary
- Total steps taken
- Number of tool calls
- Total time elapsed

## Comparison with Other Functions

| Function | Use Case | Output |
|----------|----------|--------|
| `query_agent()` | Normal queries | Clean answer only |
| `query_agent_with_thinking()` | **Understanding agent behavior** | **Step-by-step reasoning** ⭐ |
| `query_agent_stream()` | Deep debugging | Raw technical data |
| `execute_sql()` | Direct queries | SQL results only |

## Real-World Example

### Problem: Agent keeps failing spatial queries

**Without chain of thought:**
```python
result = query_agent("Find nearby buildings")
# Result: Error after 2 minutes
```
❌ You don't know WHY it failed

**With chain of thought:**
```python
query_agent_with_thinking("Find nearby buildings")
```

You see:
```
🧠 AGENT THINKING
  💡 Decision: Call tool 'sql_db_query'
     → Executing SQL query:
        SELECT * FROM cim_wizard_building WHERE ST_DWithin(geom, ...)

🔧 TOOL RESULT
  ❌ Query failed:
     Error: column "geom" does not exist
     
🧠 AGENT THINKING
  💡 Decision: Call tool 'sql_db_query'
     → Executing SQL query:
        SELECT * FROM cim_wizard_building WHERE ST_DWithin(geom, ...)
        
🔧 TOOL RESULT
  ❌ Query failed:
     Error: column "geom" does not exist
```

✅ **Now you know**: Agent is using wrong column name and not learning from errors!

**Solution**: Update system prompt to be more explicit about column names.

## Tips for Using Chain of Thought

### 1. Start Simple
```python
# Good first test
query_agent_with_thinking("What tables are available?")

# Then try more complex
query_agent_with_thinking("Count buildings by type")
```

### 2. Watch for Patterns
Look for:
- ❌ **Repeated errors** - Agent not learning
- ⏱️ **Slow steps** - Which tool is slow?
- 🔄 **Unnecessary calls** - Agent checking schema multiple times?
- 🎯 **Wrong tools** - Agent using wrong approach?

### 3. Compare Approaches
```python
# Try different phrasings
query_agent_with_thinking("How many buildings?")
query_agent_with_thinking("Count the buildings")
query_agent_with_thinking("SELECT COUNT(*) FROM cim_wizard_building")

# See which works best!
```

### 4. Use for Debugging
```python
# If a query fails:
query_agent_with_thinking("Your failing query", show_raw=True)

# Check:
# - What tools were called?
# - What errors occurred?
# - Did agent try to adapt?
```

## Advanced: Understanding Agent Behavior

### Good Agent Behavior ✅
```
Step 1: Check schema → Gets column names
Step 2: Write query → Uses correct columns
Step 3: Execute query → Success!
```

### Bad Agent Behavior ❌
```
Step 1: Execute query → Error (column doesn't exist)
Step 2: Execute query → Same error (didn't learn!)
Step 3: Execute query → Same error (still not learning!)
...
```

### Agent Learning ✅
```
Step 1: Execute query → Error (column doesn't exist)
Step 2: Check schema → Gets correct column names
Step 3: Execute query → Success! (adapted based on error)
```

## Troubleshooting

### Issue: Too much output
**Solution**: Output is automatically truncated
- Tables: First 10 shown
- Schema: First 20 lines
- Query results: First 10 rows

### Issue: Want to see everything
**Solution**: Use `show_raw=True`
```python
query_agent_with_thinking("Your question", show_raw=True)
```

### Issue: Agent takes too long
**Solution**: Set lower iteration limit
```python
query_agent_with_thinking("Your question", max_iterations=5)
```

### Issue: Need raw technical data
**Solution**: Use `query_agent_stream()` instead
```python
query_agent_stream("Your question")
```

## Summary

The `query_agent_with_thinking()` function is your **window into the agent's mind**. Use it to:

1. ✅ **Understand** how the agent solves problems
2. ✅ **Debug** when things go wrong
3. ✅ **Learn** about LLM agent behavior
4. ✅ **Optimize** your prompts and queries

**Pro tip**: Always use this function when developing new queries or debugging issues!

## Quick Reference

```python
# Basic usage
query_agent_with_thinking("Your question")

# With raw data
query_agent_with_thinking("Your question", show_raw=True)

# With iteration limit
query_agent_with_thinking("Your question", max_iterations=10)

# For normal queries (no thinking shown)
query_agent("Your question")

# For direct SQL (bypass agent)
execute_sql("SELECT * FROM table")
```

---

**Remember**: Understanding HOW the agent thinks is just as important as WHAT it answers!


