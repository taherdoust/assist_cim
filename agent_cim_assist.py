#!/usr/bin/env python3
"""
CIM Database Agent - Python Script Version
==========================================

This script implements an improved SQL agent for querying the CIM Wizard database
with better PostGIS support, improved error handling, and explicit prompts.

Requirements:
- PostgreSQL database on localhost:15432
- Ollama running on localhost:11434 (or via SSH tunnel to ipazia126)
- Conda environment: aienv

Usage:
    python agent_cim_assist.py

Then use the functions:
    - query_agent("How many buildings are there?")
    - execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")
    - query_agent_with_thinking("Your question here")
"""

# ============================================================================
# IMPORTS
# ============================================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import time
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Load environment variables from .env file (if exists)
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection string
# Update this if your database is on a different host/port
DATABASE_URI = "postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated?options=-csearch_path=cim_vector,cim_census,cim_raster,cim_network"

# Ollama configuration
# If using SSH tunnel from ipazia126, this should be localhost:11434
OLLAMA_BASE_URL = "http://localhost:11434"

# Model selection
# Options: "qwen2.5-coder:32b", "qwen2.5-coder:14b", "llama3.2"
# Note: 32b is more powerful but 14b is more obedient with tools
OLLAMA_MODEL = "qwen2.5-coder:14b"  # Using 14b - better at following tool usage instructions

# Model parameters
TEMPERATURE = 0.0  # 0.0 for deterministic responses
REQUEST_TIMEOUT = 120.0  # seconds
NUM_PREDICT = 2048  # max tokens to generate

# ============================================================================
# STEP 1: INITIALIZE LLM
# ============================================================================

print("=" * 80)
print("STEP 1: Initializing LLM")
print("=" * 80)

# Create the LLM instance with Ollama
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=TEMPERATURE,
    request_timeout=REQUEST_TIMEOUT,
    num_predict=NUM_PREDICT
)

# Test LLM connection
print(f"\nTesting LLM connection to {OLLAMA_BASE_URL}...")
print(f"Model: {OLLAMA_MODEL}")
try:
    response = llm.invoke('Say OK if you can hear me.')
    print(f"✓ LLM Response: {response.content}")
except Exception as e:
    print(f"✗ LLM Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if Ollama is running: curl http://localhost:11434")
    print("  2. If using SSH tunnel: ssh -L 11434:localhost:11434 castangia@ipazia126.polito.it")
    print("  3. Check if model is pulled: ollama list")
    exit(1)

# ============================================================================
# STEP 2: CONNECT TO DATABASE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Connecting to Database")
print("=" * 80)

# Create SQLAlchemy engine
# NullPool prevents connection pool issues in interactive use
engine = create_engine(DATABASE_URI, poolclass=NullPool, echo=False)

# Test database connection
print(f"\nConnecting to database at localhost:15432...")
try:
    with engine.connect() as conn:
        # Check PostgreSQL version
        result = conn.execute(text("SELECT version();"))
        pg_version = result.fetchone()[0]
        print(f"✓ PostgreSQL: {pg_version[:50]}...")
        
        # Check PostGIS availability
        # Note: PostGIS is in public schema for Docker databases
        try:
            result = conn.execute(text("SELECT public.PostGIS_version();"))
            postgis_version = result.fetchone()[0]
            print(f"✓ PostGIS: {postgis_version}")
            POSTGIS_AVAILABLE = True
        except Exception as e:
            print(f"✗ PostGIS: NOT AVAILABLE - {str(e)[:50]}...")
            POSTGIS_AVAILABLE = False
except Exception as e:
    print(f"✗ Database Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if database is running: docker ps | grep cim-integrateddb")
    print("  2. Test connection: psql -h localhost -p 15432 -U cim_wizard_user -d cim_wizard_integrated")
    print("  3. Check port: sudo lsof -i :15432")
    exit(1)

# Create LangChain SQL database wrapper
# sample_rows_in_table_info=2 includes 2 sample rows in schema descriptions
db = SQLDatabase(engine=engine, sample_rows_in_table_info=2)

# List available tables
print(f"\n✓ Available tables:")
for table in db.get_usable_table_names():
    print(f"  - {table}")

print(f"\n✓ PostGIS Status: {'Available' if POSTGIS_AVAILABLE else 'Not Available'}")

# ============================================================================
# STEP 3: DETECT GEOMETRY COLUMNS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Detecting Geometry Columns")
print("=" * 80)

# Query database to find geometry columns
print("\nChecking geometry columns in cim_wizard_building table...")
with engine.connect() as conn:
    # Get all columns with their data types
    result = conn.execute(text("""
        SELECT table_schema, column_name, data_type, udt_name 
        FROM information_schema.columns 
        WHERE table_name = 'cim_wizard_building'
        ORDER BY table_schema, ordinal_position;
    """))
    
    print("\nColumns found:")
    all_cols = []
    for row in result:
        print(f"  - {row[0]}.{row[1]}: {row[2]} ({row[3]})")
        all_cols.append((row[0], row[1], row[3]))
    
    # Find geometry columns specifically
    # Look for columns with name containing 'geometry' or type 'geometry'
    result = conn.execute(text("""
        SELECT table_schema, column_name, udt_name
        FROM information_schema.columns 
        WHERE table_name = 'cim_wizard_building' 
        AND (column_name LIKE '%geometry%' OR udt_name = 'geometry');
    """))
    
    geometry_cols = [(row[0], row[1]) for row in result]
    print(f"\n✓ Geometry columns: {geometry_cols}")
    
    # Store the first geometry column for use in queries
    if geometry_cols:
        GEOMETRY_SCHEMA, GEOMETRY_COLUMN = geometry_cols[0]
        print(f"✓ Using geometry column: {GEOMETRY_SCHEMA}.{GEOMETRY_COLUMN}")
    else:
        GEOMETRY_SCHEMA = None
        GEOMETRY_COLUMN = None
        print("✗ WARNING: No geometry column found!")
        print("  Spatial queries will not work.")

# ============================================================================
# STEP 4: CREATE SYSTEM PROMPT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Creating System Prompt")
print("=" * 80)

# Build comprehensive system prompt for the agent
# This prompt instructs the agent on how to use tools and query the database
SQL_PREFIX = f"""
YOU ARE A TOOL-CALLING AGENT. YOU OUTPUT EITHER TOOL CALLS OR FINAL ANSWERS.

## STEP 1: If you haven't executed a query yet, output a tool call:

{{
  "name": "sql_db_query",
  "arguments": {{
    "query": "YOUR_SQL_HERE"
  }}
}}

## STEP 2: After receiving tool results, give a HUMAN-READABLE ANSWER:

DO NOT output more tool calls if you already have the answer!
Instead, respond in natural language with the result.

**EXAMPLE:**

User: "How many buildings?"

Response 1 (Tool call):
{{
  "name": "sql_db_query",
  "arguments": {{
    "query": "SELECT COUNT(*) FROM cim_vector.cim_wizard_building;"
  }}
}}

[Tool returns: [(4336,)]]

Response 2 (Final answer - NO MORE TOOL CALLS):
"There are 4,336 buildings in the database."

## CRITICAL RULES:
- First response: JSON tool call
- Second response (after getting results): Human-readable answer
- NEVER repeat the same tool call
- NEVER output tool calls after you have the answer

You are a PostgreSQL + PostGIS expert for City Information Modeling (CIM).

## CRITICAL RULES:

1. **ALWAYS use EXACT column names from sql_db_schema** - NEVER guess or hallucinate
2. **ALWAYS use schema-qualified table names** (e.g., `cim_census.censusgeo`, not just `censusgeo`)
3. **NEVER repeat the same failed query** - adapt based on errors
4. **If a column doesn't exist, CHECK THE SCHEMA AGAIN** using sql_db_schema
5. **If a task is impossible, explain why clearly** to the user
6. **Maximum 3 attempts** - if still failing, explain the problem

## DATABASE INFORMATION:

**Database Schemas:**
- `cim_vector` - Building geometries and spatial data
- `cim_census` - Census and demographic data
- `cim_raster` - Raster/elevation data
- `cim_network` - Network/infrastructure data

**Main Tables (ALWAYS use schema prefix):**
- `cim_vector.cim_wizard_building` - Building geometries and metadata
- `cim_vector.cim_wizard_building_properties` - Building attributes (height, area, type, etc.)
- `cim_vector.cim_wizard_project_scenario` - Project and scenario management
- `cim_census.censusgeo` - Italian census data

**Key Columns in cim_wizard_building:**
- building_id (UUID) - Primary key
- {f'{GEOMETRY_COLUMN} (GEOMETRY)' if GEOMETRY_COLUMN else 'NO GEOMETRY COLUMN FOUND'} - Spatial data
- lod (INTEGER) - Level of detail
- census_id (BIGINT) - Link to census data

**PostGIS Status:** {'✓ AVAILABLE' if POSTGIS_AVAILABLE else '✗ NOT AVAILABLE (spatial queries will fail)'}

## QUERY WORKFLOW:

1. **First**: Call sql_db_list_tables to see available tables
2. **Second**: Call sql_db_schema on relevant tables to see EXACT column names
3. **Third**: Write query using ONLY the column names you saw in step 2
4. **Fourth**: Use sql_db_query_checker to validate the query
5. **Fifth**: Execute with sql_db_query
6. **If error**: Analyze error message, check schema again, try different approach

## SPATIAL QUERIES (PostGIS required):

{'**PostGIS Functions Available:**' if POSTGIS_AVAILABLE else '**WARNING: PostGIS NOT available - spatial queries will fail!**'}

**CRITICAL: All PostGIS functions MUST use the public schema prefix:**
- public.ST_DWithin(geom1, geom2, distance) - find features within distance
- public.ST_Distance(geom1, geom2) - calculate distance between geometries
- public.ST_Intersects(geom1, geom2) - check if geometries intersect
- public.ST_Area(geom) - calculate area of geometry
- public.ST_Buffer(geom, distance) - create buffer around geometry
- public.ST_GeomFromText(text) - create geometry from WKT text

**Example:** Use `public.ST_Distance(...)` NOT `ST_Distance(...)`

## EXAMPLE QUERIES (ALWAYS use schema prefixes):

**Count buildings:**
```sql
SELECT COUNT(*) FROM cim_vector.cim_wizard_building;
```

**Count census zones:**
```sql
SELECT COUNT(*) FROM cim_census.censusgeo;
```

{'**Find nearby buildings (spatial):**' if GEOMETRY_COLUMN else '**Spatial queries NOT possible - no geometry column!**'}
{f'''```sql
SELECT b1.building_id, 
       public.ST_Distance(b1.{GEOMETRY_COLUMN}, b2.{GEOMETRY_COLUMN}) as distance_meters
FROM cim_vector.cim_wizard_building b1
CROSS JOIN cim_vector.cim_wizard_building b2
WHERE b2.building_id = 'target-uuid-here'
  AND b1.building_id != b2.building_id
  AND public.ST_DWithin(b1.{GEOMETRY_COLUMN}, b2.{GEOMETRY_COLUMN}, 100)
ORDER BY distance_meters
LIMIT 10;
```''' if GEOMETRY_COLUMN else ''}

## ERROR HANDLING:

**If you see "column does not exist":**
1. Call sql_db_schema again to see actual column names
2. Use the EXACT name from the schema output
3. Do NOT try the same wrong name again

**If you see "table does not exist":**
1. Call sql_db_list_tables to see actual table names
2. Use the EXACT name from the list
3. Do NOT try the same wrong name again

**If you see "function does not exist" for PostGIS functions:**
1. Make sure you used the public schema prefix: public.ST_Distance, public.ST_DWithin, etc.
2. Check if PostGIS is available (status above)
3. If not available, explain spatial queries won't work
4. Suggest non-spatial alternatives

## IMPORTANT REMINDERS:

- ✓ Check schema BEFORE writing queries
- ✓ Use exact column names from schema
- ✓ Adapt when you get errors
- ✓ Explain clearly if something is impossible
- ✗ NEVER repeat the same failed query
- ✗ NEVER guess column or table names
- ✗ NEVER use 'geom' or 'geometry' unless schema shows it

Remember: The schema is the source of truth. Always check it first!
"""

print(f"\n✓ System prompt created")
print(f"  - PostGIS: {'✓ Available' if POSTGIS_AVAILABLE else '✗ Not Available'}")
print(f"  - Geometry column: {GEOMETRY_COLUMN if GEOMETRY_COLUMN else '✗ Not Found'}")

# ============================================================================
# STEP 5: CREATE AGENT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Creating Agent")
print("=" * 80)

# Create SQL database toolkit
# This provides tools for the agent to interact with the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

print(f"\n✓ Tools available: {[tool.name for tool in tools]}")

# CRITICAL FIX: Bind tools to LLM explicitly for Ollama compatibility
# Ollama models need explicit tool binding to properly recognize tool calls
llm_with_tools = llm.bind_tools(tools)

print(f"✓ Tools bound to LLM for Ollama compatibility")

# Create prompt template with system message
# MessagesPlaceholder allows dynamic messages to be inserted
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SQL_PREFIX),  # System prompt with instructions
    MessagesPlaceholder(variable_name="messages"),  # User messages go here
])

# Create the ReAct agent
# ReAct = Reasoning + Acting (thinks about what to do, then does it)
agent_executor = create_react_agent(
    llm_with_tools,  # Language model WITH tools bound
    tools,  # Database tools
    prompt=prompt_template,  # System prompt
    debug=False  # Set to True for verbose output
)

print(f"✓ Agent created successfully!")
print(f"  - Model: {OLLAMA_MODEL}")
print(f"  - Tools: {len(tools)}")
print(f"  - System prompt: Injected via ChatPromptTemplate")

# ============================================================================
# OLLAMA JSON TOOL CALL PARSER
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Ollama Tool Call Parser")
print("=" * 80)

import json
from typing import List, Tuple, Optional

def parse_ollama_tool_call(content: str) -> Optional[Tuple[str, dict]]:
    """
    Parse Ollama's JSON-formatted tool calls from message content.
    
    Ollama outputs tool calls as JSON in the content field:
    {
      "name": "tool_name",
      "arguments": {...}
    }
    
    Args:
        content: The message content from Ollama
    
    Returns:
        Tuple of (tool_name, arguments) or None if no valid tool call found
    """
    if not content or not content.strip():
        return None
    
    content = content.strip()
    
    # Try to parse as JSON
    try:
        parsed = json.loads(content)
        
        # Check if it's a tool call format
        if isinstance(parsed, dict) and 'name' in parsed and 'arguments' in parsed:
            tool_name = parsed['name']
            tool_args = parsed['arguments']
            
            # Validate tool name is one of our SQL tools
            valid_tools = ['sql_db_query', 'sql_db_schema', 'sql_db_list_tables', 'sql_db_query_checker']
            if tool_name in valid_tools:
                return (tool_name, tool_args)
    except json.JSONDecodeError:
        pass
    
    return None


def execute_tool_manually(tool_name: str, tool_args: dict, tools: List) -> str:
    """
    Manually execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        tools: List of available tools
    
    Returns:
        Tool execution result as string
    """
    # Find the tool
    tool_obj = None
    for tool in tools:
        if tool.name == tool_name:
            tool_obj = tool
            break
    
    if not tool_obj:
        return f"Error: Tool '{tool_name}' not found"
    
    # Execute the tool
    try:
        result = tool_obj.invoke(tool_args)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent_with_ollama_fix(question: str, max_iterations: int = 15) -> dict:
    """
    Run the agent with Ollama tool call parsing.
    
    This function handles the agent loop manually because Ollama outputs
    tool calls as JSON text instead of proper tool call objects.
    
    Args:
        question: The question to ask
        max_iterations: Maximum iterations
    
    Returns:
        Dictionary with answer, queries, results, etc.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    messages = [
        SystemMessage(content=SQL_PREFIX),
        HumanMessage(content=question)
    ]
    
    sql_queries = []
    sql_results = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get LLM response
        response = llm.invoke(messages)
        
        # Check if response contains a JSON tool call
        tool_call = parse_ollama_tool_call(response.content)
        
        if tool_call:
            tool_name, tool_args = tool_call
            
            # Track SQL queries
            if tool_name == 'sql_db_query':
                sql_queries.append(tool_args.get('query', ''))
            
            # Execute the tool
            tool_result = execute_tool_manually(tool_name, tool_args, tools)
            
            # Track SQL results
            if tool_name == 'sql_db_query':
                sql_results.append(tool_result)
            
            # Add AI message and tool result to conversation
            messages.append(AIMessage(content=f"Calling {tool_name}"))
            messages.append(ToolMessage(content=tool_result, tool_call_id=f"call_{iteration}"))
            
        else:
            # No tool call - this is the final answer
            return {
                'success': True,
                'answer': response.content,
                'sql_queries': sql_queries,
                'sql_results': sql_results,
                'iterations': iteration
            }
    
    # Max iterations reached
    return {
        'success': False,
        'error': 'Max iterations reached',
        'sql_queries': sql_queries,
        'sql_results': sql_results,
        'iterations': iteration
    }

print("✓ Ollama tool call parser created")
print("✓ Manual tool execution handler created")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Helper Functions Ready")
print("=" * 80)

def query_agent(question: str, max_iterations: int = 15, timeout: int = 300) -> Dict[str, Any]:
    """
    Query the agent with Ollama-compatible tool call parsing.
    
    This function uses a custom agent loop that parses Ollama's JSON tool calls
    and manually executes them, working around Ollama's incompatibility with
    LangGraph's native tool calling.
    
    Args:
        question: The natural language question to ask
        max_iterations: Maximum number of agent reasoning steps (default: 15)
        timeout: Maximum time in seconds (default: 300)
    
    Returns:
        Dictionary with:
            - success: True if query succeeded
            - answer: The agent's response
            - sql_queries: List of SQL queries executed
            - sql_results: List of SQL query results
            - time: Elapsed time in seconds
            - iterations: Number of iterations
            OR
            - success: False if query failed
            - error: Error message
            - time: Elapsed time
    
    Example:
        >>> result = query_agent("How many buildings are there?")
        >>> print(result['answer'])
        >>> print(result['sql_queries'])
        >>> print(result['sql_results'])
    """
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    start = time.time()
    
    try:
        # Use Ollama-compatible agent loop
        result = run_agent_with_ollama_fix(question, max_iterations)
        
        elapsed = time.time() - start
        
        sql_queries = result.get('sql_queries', [])
        sql_results = result.get('sql_results', [])
        answer = result.get('answer', result.get('error', 'No answer'))
        
        # Print SQL queries executed
        if sql_queries:
            print(f"\n{'='*80}")
            print(f"SQL QUERIES EXECUTED:")
            print(f"{'='*80}")
            for i, query in enumerate(sql_queries, 1):
                print(f"\nQuery {i}:")
                print(query)
        
        # Print SQL results
        if sql_results:
            print(f"\n{'='*80}")
            print(f"DATABASE RESULTS:")
            print(f"{'='*80}")
            for i, result_text in enumerate(sql_results, 1):
                print(f"\nResult {i}:")
                if 'Error' in result_text:
                    print(f"✗ {result_text}")
                else:
                    print(f"✓ {result_text}")
        
        # Print final answer
        print(f"\n{'='*80}")
        print(f"AGENT ANSWER:")
        print(f"{'='*80}")
        print(answer)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Time: {elapsed:.2f}s | Iterations: {result.get('iterations', 0)} | SQL queries: {len(sql_queries)}")
        print(f"{'='*80}")
        
        result['time'] = elapsed
        return result
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        print(f"Time: {elapsed:.2f}s")
        return {
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def execute_sql(query: str) -> Optional[str]:
    """
    Execute SQL directly (fallback when agent fails).
    
    This bypasses the agent and runs SQL directly on the database.
    Useful when you know exactly what query you want to run.
    
    Args:
        query: SQL query string to execute
    
    Returns:
        Query result as string, or None on error
    
    Example:
        >>> result = execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")
        >>> print(result)
    """
    print(f"\n{'='*80}")
    print(f"EXECUTING SQL DIRECTLY:")
    print(f"{'='*80}")
    print(f"\n{query}\n")
    
    try:
        # Use the LangChain SQLDatabase wrapper to run the query
        result = db.run(query)
        print(f"{'='*80}")
        print(f"RESULT:")
        print(f"{'='*80}")
        print(result)
        return result
    except Exception as e:
        print(f"{'='*80}")
        print(f"ERROR:")
        print(f"{'='*80}")
        print(f"{e}")
        return None


def query_agent_with_thinking(question: str, max_iterations: int = 15, show_raw: bool = False):
    """
    Query the agent with detailed chain-of-thought visualization using Ollama-compatible approach.
    
    This function shows you exactly what the agent is thinking and doing
    at each step. Very useful for debugging and understanding agent behavior.
    
    Args:
        question: The question to ask
        max_iterations: Maximum number of reasoning steps
        show_raw: If True, also show raw message data (verbose)
    
    Example:
        >>> query_agent_with_thinking("How many buildings are there?")
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    messages = [
        SystemMessage(content=SQL_PREFIX),
        HumanMessage(content=question)
    ]
    
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        
        print(f"\n{'─'*80}")
        print(f"ITERATION {iteration} ({elapsed:.1f}s)")
        print(f"{'─'*80}")
        
        # Get LLM response
        response = llm.invoke(messages)
        
        # Check if response contains a JSON tool call
        tool_call = parse_ollama_tool_call(response.content)
        
        if tool_call:
            tool_name, tool_args = tool_call
            
            print(f"\n  → Agent Decision: Call tool '{tool_name}'")
            
            # Explain what each tool does
            if tool_name == 'sql_db_list_tables':
                print(f"     Purpose: Checking what tables are available")
            elif tool_name == 'sql_db_schema':
                table = tool_args.get('table_names', 'unknown')
                print(f"     Purpose: Examining schema of table: {table}")
            elif tool_name == 'sql_db_query_checker':
                query = tool_args.get('query', '')[:100]
                print(f"     Purpose: Validating SQL query")
                print(f"     Query: {query}...")
            elif tool_name == 'sql_db_query':
                query = tool_args.get('query', '')
                print(f"     Purpose: Executing SQL query")
                print(f"     Query: {query}")
            
            if show_raw:
                print(f"\n     [Raw args: {tool_args}]")
            
            # Execute the tool
            print(f"\n  ⚙ Executing tool...")
            tool_result = execute_tool_manually(tool_name, tool_args, tools)
            
            # Show result
            print(f"\n  ✓ Tool Result:")
            if tool_name == 'sql_db_query':
                if 'Error' in tool_result:
                    print(f"     ✗ {tool_result[:200]}")
                else:
                    lines = tool_result.split('\n')
                    for line in lines[:10]:
                        if line.strip():
                            print(f"     {line}")
                    if len(lines) > 10:
                        print(f"     ... ({len(lines)-10} more rows)")
            else:
                result_preview = tool_result[:200] if len(tool_result) > 200 else tool_result
                print(f"     {result_preview}")
            
            # Add messages to conversation
            messages.append(AIMessage(content=f"Calling {tool_name}"))
            messages.append(ToolMessage(content=tool_result, tool_call_id=f"call_{iteration}"))
            
        else:
            # No tool call - this is the final answer
            print(f"\n{'─'*80}")
            print(f"FINAL ANSWER")
            print(f"{'─'*80}")
            print(f"\n{response.content}\n")
            break
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"   - Total iterations: {iteration}")
    print(f"   - Time: {total_time:.2f}s")
    print(f"{'='*80}\n")


# Print available functions
print("\n✓ Helper functions ready:")
print("  - query_agent(question, max_iterations=15, timeout=300)")
print("  - query_agent_with_thinking(question, max_iterations=15, show_raw=False)")
print("  - execute_sql(query)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CIM DATABASE AGENT - READY")
    print("=" * 80)
    
    print("\n✓ Setup complete! You can now use the following functions:")
    print("\nBasic usage:")
    print('  result = query_agent("How many buildings are in the database?")')
    print('  print(result["answer"])')
    
    print("\nDirect SQL:")
    print('  execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")')
    
    print("\nDebug mode (see agent thinking):")
    print('  query_agent_with_thinking("How many buildings are there?")')
    
    print("\nSpatial query example:")
    if GEOMETRY_COLUMN:
        print(f'  query_agent("Find 5 nearest buildings to building_id \'259f59e2-20c4-45d4-88b9-298022fd9c7f\' within 1000 meters")')
    else:
        print("  ✗ Spatial queries not available (no geometry column found)")
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nThe agent is ready. You can now import this script and use the functions,")
    print("or run queries directly in this session.")
    print("\nPress Ctrl+D to exit, or continue with your queries below:")
    print("=" * 80 + "\n")
    
    # Example test query (comment out if you don't want it to run)
    # Uncomment the line below to test:
    query_agent("How many buildings are in the database?")
