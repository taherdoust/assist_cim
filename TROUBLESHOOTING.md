# Troubleshooting Guide for CIM Agent

## Quick Diagnostics

Run these checks first to identify the problem:

### 1. Check Database Connection
```python
with engine.connect() as conn:
    # Test basic connection
    result = conn.execute(text("SELECT version();"))
    print(result.fetchone()[0])
    
    # Test PostGIS
    result = conn.execute(text("SELECT PostGIS_version();"))
    print(result.fetchone()[0])
```

### 2. Check Geometry Columns
```python
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name, column_name, udt_name
        FROM information_schema.columns
        WHERE udt_name = 'geometry'
        ORDER BY table_name;
    """))
    
    print("Geometry columns found:")
    for row in result:
        print(f"  {row[0]}.{row[1]} ({row[2]})")
```

### 3. Check LLM Connection
```python
try:
    response = llm.invoke("Say OK")
    print(f"LLM working: {response.content}")
except Exception as e:
    print(f"LLM error: {e}")
```

## Common Problems

### Problem 1: "column 'geom' does not exist"

**Symptoms**: Agent keeps trying to use `geom` or wrong column name

**Root Cause**: 
- SQLAlchemy can't reflect PostGIS geometry columns
- Agent doesn't know the correct column name

**Solution**:
```python
# Method 1: Check actual column name
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'cim_wizard_building' 
        AND column_name LIKE '%geometry%';
    """))
    geom_col = result.fetchone()[0]
    print(f"Actual geometry column: {geom_col}")

# Method 2: Update system prompt with exact name
SQL_PREFIX = f"""
CRITICAL: The geometry column in cim_wizard_building is called '{geom_col}'.
ALWAYS use '{geom_col}' not 'geom' or 'geometry'.
...
"""
```

### Problem 2: Agent Loops Forever

**Symptoms**: Same query repeated 10+ times

**Root Cause**:
- Agent not learning from errors
- No iteration limit set
- Model not capable of adapting

**Solution**:
```python
# Method 1: Set strict limits
result = query_agent(
    "Your question",
    max_iterations=10,  # Stop after 10 tries
    timeout=120         # Stop after 2 minutes
)

# Method 2: Use different model
llm = ChatOllama(
    model="llama3.2",  # Better at reasoning
    temperature=0.0
)

# Method 3: Use direct SQL instead
execute_sql("YOUR SQL QUERY HERE")
```

### Problem 3: "table 'buildings' does not exist"

**Symptoms**: Agent uses wrong table name

**Root Cause**:
- Model hallucinating table names
- System prompt not clear enough

**Solution**:
```python
# Method 1: Be very explicit in prompt
SQL_PREFIX = """
CRITICAL TABLE NAMES:
- cim_wizard_building (NOT 'buildings' or 'building')
- cim_wizard_building_properties (NOT 'properties')
- cim_wizard_project_scenario (NOT 'projects')

NEVER use shortened or guessed table names!
"""

# Method 2: List actual tables first
result = query_agent("First, list all available tables")
# Then ask your real question
```

### Problem 4: Slow Performance

**Symptoms**: Queries take 5+ minutes

**Root Cause**:
- Model too slow
- Too many iterations
- Complex reasoning

**Solution**:
```python
# Method 1: Use faster model
llm = ChatOllama(
    model="llama3.2",  # Faster than qwen2.5-coder
    num_predict=1024   # Shorter responses
)

# Method 2: Reduce iterations
query_agent("Question", max_iterations=5)

# Method 3: Use direct SQL for complex queries
execute_sql("""
    SELECT building_id, 
           ST_Distance(b1.building_geometry, b2.building_geometry) as dist
    FROM cim_wizard_building b1
    CROSS JOIN cim_wizard_building b2
    WHERE b2.building_id = 'target-id'
    LIMIT 10;
""")
```

### Problem 5: "Did not recognize type 'public.geometry'"

**Symptoms**: Warning about geometry type

**Root Cause**:
- SQLAlchemy doesn't understand PostGIS types by default

**Solution**:
```python
# Method 1: Install GeoAlchemy2
# pip install geoalchemy2

from geoalchemy2 import Geometry
# Then recreate engine

# Method 2: Ignore the warning (it's just a warning)
import warnings
warnings.filterwarnings('ignore', message='Did not recognize type')

# Method 3: Use raw SQL for geometry queries
execute_sql("SELECT ST_AsText(building_geometry) FROM cim_wizard_building LIMIT 1;")
```

### Problem 6: Connection Timeout

**Symptoms**: "Connection timeout" or "Lost connection"

**Root Cause**:
- Database server too slow
- Network issues
- Query too complex

**Solution**:
```python
# Method 1: Increase timeout
engine = create_engine(
    DATABASE_URI,
    connect_args={'connect_timeout': 60}
)

# Method 2: Use connection pooling
engine = create_engine(
    DATABASE_URI,
    pool_size=5,
    max_overflow=10
)

# Method 3: Simplify query
# Instead of: "Find all buildings with complex conditions"
# Try: "Count buildings" first, then add conditions
```

### Problem 7: Wrong Results

**Symptoms**: Agent returns incorrect data

**Root Cause**:
- Model misunderstood question
- Wrong SQL generated
- Data interpretation error

**Solution**:
```python
# Method 1: Be more specific
# Bad: "How many buildings?"
# Good: "Count the total number of rows in cim_wizard_building table"

# Method 2: Verify with direct SQL
execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")

# Method 3: Ask agent to show the SQL first
query_agent("What SQL query would you use to count buildings? Don't execute it.")
```

## Advanced Debugging

### Enable Debug Mode
```python
agent_executor = create_react_agent(
    llm, 
    tools, 
    state_modifier=system_message,
    debug=True  # Shows all tool calls
)
```

### Stream Output for Debugging
```python
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Your question")]},
    config={"recursion_limit": 10}
):
    print(f"\n--- Step ---")
    print(step)
```

### Check What Agent Sees
```python
# Check schema info
schema_info = db.get_table_info(['cim_wizard_building'])
print(schema_info)

# Check if geometry column is visible
if 'building_geometry' in schema_info:
    print("✓ Geometry column visible")
else:
    print("✗ Geometry column NOT visible")
```

### Test Individual Tools
```python
# Test schema tool
from langchain_community.tools.sql_database.tool import InfoSQLDatabaseTool

schema_tool = InfoSQLDatabaseTool(db=db)
result = schema_tool.invoke("cim_wizard_building")
print(result)

# Test query tool
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

query_tool = QuerySQLDataBaseTool(db=db)
result = query_tool.invoke("SELECT COUNT(*) FROM cim_wizard_building;")
print(result)
```

## Performance Optimization

### 1. Limit Result Size
```python
# Add LIMIT to all queries
SQL_PREFIX = """
...
IMPORTANT: Always add LIMIT clause to queries unless user asks for all results.
Default LIMIT: 10 rows
"""
```

### 2. Use Spatial Indexes
```sql
-- Check if spatial index exists
SELECT indexname 
FROM pg_indexes 
WHERE tablename = 'cim_wizard_building' 
AND indexname LIKE '%geom%';

-- Create if missing (run in psql, not in agent)
CREATE INDEX IF NOT EXISTS idx_building_geom 
ON cim_wizard_building USING GIST (building_geometry);
```

### 3. Optimize Spatial Queries
```sql
-- Bad: No spatial index usage
SELECT * FROM cim_wizard_building 
WHERE ST_Distance(building_geometry, point) < 100;

-- Good: Uses spatial index
SELECT * FROM cim_wizard_building 
WHERE ST_DWithin(building_geometry, point, 100);
```

## When to Give Up on Agent

If after trying all solutions:
- Agent still loops forever
- Wrong results consistently
- Takes >5 minutes per query

**Use direct SQL instead**:
```python
# Create a simple query function
def query_buildings_nearby(building_id, distance=100, limit=10):
    query = f"""
    SELECT b1.building_id,
           ST_Distance(b1.building_geometry, b2.building_geometry) as distance
    FROM cim_wizard_building b1
    CROSS JOIN cim_wizard_building b2
    WHERE b2.building_id = '{building_id}'
      AND b1.building_id != b2.building_id
      AND ST_DWithin(b1.building_geometry, b2.building_geometry, {distance})
    ORDER BY distance
    LIMIT {limit};
    """
    return execute_sql(query)

# Use it
result = query_buildings_nearby('259f59e2-20c4-45d4-88b9-298022fd9c7f')
```

## Getting Help

If still stuck:
1. Check the error message carefully
2. Verify database connection
3. Verify geometry column name
4. Try direct SQL to isolate the problem
5. Try a different model
6. Simplify the question

## Useful SQL Queries for Debugging

```sql
-- Check table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_name = 'cim_wizard_building'
);

-- Check column exists
SELECT EXISTS (
    SELECT FROM information_schema.columns 
    WHERE table_name = 'cim_wizard_building' 
    AND column_name = 'building_geometry'
);

-- Check PostGIS installed
SELECT PostGIS_version();

-- Check SRID of geometry column
SELECT Find_SRID('public', 'cim_wizard_building', 'building_geometry');

-- Test spatial query
SELECT building_id, ST_AsText(building_geometry) 
FROM cim_wizard_building 
LIMIT 1;
```


