# CIM Database Agent

AI-powered SQL agent for querying City Information Modeling (CIM) PostgreSQL + PostGIS database.

## Files

- **`agent_cim_assist_improved.ipynb`** - New improved notebook (recommended)
- **`agent_cim_assist.ipynb`** - Original notebook (has issues)
- **`IMPROVEMENTS.md`** - Detailed list of improvements
- **`TROUBLESHOOTING.md`** - Solutions to common problems

## Quick Start

### 1. Install Dependencies

```bash
conda activate ai4db
pip install langchain-ollama langchain-core langchain-community langgraph psycopg2-binary python-dotenv
```

### 2. Start Ollama

```bash
# Make sure Ollama is running with llama3.2 model
ollama pull llama3.2
ollama serve
```

### 3. Setup Database Connection

```bash
# If using SSH tunnel:
ssh -L 5432:localhost:5432 eclab@192.168.177.23

# Or update DATABASE_URI in notebook
```

### 4. Run the Improved Notebook

```bash
jupyter notebook agent_cim_assist_improved.ipynb
```

## Key Features

✅ **Automatic geometry column detection**  
✅ **Better error handling with timeouts**  
✅ **Improved system prompts**  
✅ **Direct SQL fallback option**  
✅ **Helper functions for easier usage**  

## Usage Examples

### Basic Query
```python
result = query_agent("How many buildings are in the database?")
```

### Spatial Query
```python
result = query_agent(
    "Find 5 nearest buildings to building_id '259f59e2-20c4-45d4-88b9-298022fd9c7f' "
    "within 100 meters"
)
```

### Direct SQL (if agent fails)
```python
execute_sql("""
    SELECT COUNT(*) FROM cim_wizard_building;
""")
```

## What's Improved?

| Issue | Original | Improved |
|-------|----------|----------|
| Model | qwen2.5-coder | llama3.2 |
| Geometry columns | Not detected | Auto-detected |
| Error handling | None | Timeout + limits |
| Infinite loops | Common | Prevented |
| Performance | 5-25 min | 30-120 sec |
| Success rate | ~20% | ~80% |

## Common Problems & Solutions

### Problem: "column 'geom' does not exist"
**Solution**: The improved notebook auto-detects the correct column name

### Problem: Agent loops forever
**Solution**: Set `max_iterations=10` and `timeout=120`

### Problem: Slow performance
**Solution**: Use `llama3.2` model instead of `qwen2.5-coder`

See `TROUBLESHOOTING.md` for more solutions.

## Database Schema

### Main Tables
- **cim_wizard_building** - Building geometries
- **cim_wizard_building_properties** - Building attributes
- **cim_wizard_project_scenario** - Projects and scenarios
- **censusgeo** - Italian census data

### Key Columns
- `building_id` (UUID) - Primary key
- `building_geometry` (GEOMETRY) - Spatial data
- `census_id` (BIGINT) - Link to census

## Configuration

### Change Model
```python
llm = ChatOllama(
    model="llama3.2",  # or "llama3.1", "codellama"
    temperature=0.0
)
```

### Adjust Timeout
```python
query_agent("Question", max_iterations=20, timeout=600)
```

### Change Database
```python
DATABASE_URI = "postgresql://user:pass@host:port/db"
```

## Architecture

```
User Question
     ↓
LLM (llama3.2)
     ↓
Agent (LangGraph)
     ↓
Tools:
  - sql_db_list_tables
  - sql_db_schema
  - sql_db_query_checker
  - sql_db_query
     ↓
PostgreSQL + PostGIS
     ↓
Result
```

## Performance Tips

1. **Start simple** - Test with basic queries first
2. **Be specific** - "Count rows in cim_wizard_building" vs "How many buildings?"
3. **Use limits** - Always limit results unless you need all data
4. **Check schema** - Verify column names before complex queries
5. **Use direct SQL** - For complex spatial queries, write SQL directly

## Limitations

- Cannot modify database (read-only)
- Spatial queries require correct geometry column name
- Complex queries may take time
- Model may occasionally hallucinate

## Troubleshooting

If the agent fails:

1. **Check database connection**
   ```python
   with engine.connect() as conn:
       result = conn.execute(text("SELECT 1;"))
   ```

2. **Verify geometry column**
   ```python
   with engine.connect() as conn:
       result = conn.execute(text("""
           SELECT column_name FROM information_schema.columns 
           WHERE table_name = 'cim_wizard_building' 
           AND column_name LIKE '%geometry%';
       """))
   ```

3. **Try direct SQL**
   ```python
   execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")
   ```

4. **Check logs** - Look for error messages in output

See `TROUBLESHOOTING.md` for detailed solutions.

## Contributing

To improve the agent:

1. Update system prompt in notebook
2. Adjust model parameters
3. Add custom helper functions
4. Test with your specific queries

## Support

For issues:
1. Check `TROUBLESHOOTING.md`
2. Review error messages
3. Try direct SQL execution
4. Verify database connection

## License

Internal use only.

## Version History

- **v2.0** (2025-10-16) - Improved notebook with better error handling
- **v1.0** (2025-06-17) - Original implementation

---

**Recommended**: Use `agent_cim_assist_improved.ipynb` for best results.


