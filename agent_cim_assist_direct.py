#!/usr/bin/env python3
"""
CIM Database Agent - Direct SQL Version
========================================

This version bypasses the agent framework issues with Ollama and uses
a hybrid approach: the LLM generates SQL, we execute it directly.

This is more reliable than tool-calling agents with Ollama.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import json
import re
import time

load_dotenv()

# Configuration
DATABASE_URI = "postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated?options=-csearch_path=cim_vector,cim_census,cim_raster,cim_network"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:14b"

print("="*80)
print("CIM DATABASE AGENT - DIRECT SQL VERSION")
print("="*80)

# Initialize database
print("\n[1/3] Connecting to database...")
engine = create_engine(DATABASE_URI, poolclass=NullPool, echo=False)
db = SQLDatabase(engine=engine, sample_rows_in_table_info=2)

with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print(f"✓ PostgreSQL: {result.fetchone()[0][:50]}...")
    
    try:
        result = conn.execute(text("SELECT public.PostGIS_version();"))
        print(f"✓ PostGIS: {result.fetchone()[0]}")
        POSTGIS_AVAILABLE = True
    except:
        print("✗ PostGIS: Not available")
        POSTGIS_AVAILABLE = False

# Initialize LLM
print("\n[2/3] Connecting to Ollama...")
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.0,
    request_timeout=120.0
)

response = llm.invoke('Say OK')
print(f"✓ LLM: {response.content}")

# Get schema information
print("\n[3/3] Loading database schema...")
tables = db.get_usable_table_names()
print(f"✓ Tables: {', '.join(tables)}")

# Build system prompt
SYSTEM_PROMPT = f"""You are a PostgreSQL + PostGIS expert for City Information Modeling (CIM).

## YOUR TASK:
Generate ONLY the SQL query to answer the user's question. 
Return ONLY valid SQL, nothing else. No explanations, no markdown, just pure SQL.

## DATABASE SCHEMA:

Tables available:
{', '.join(tables)}

Key columns in cim_wizard_building:
- building_id (UUID) - Primary key
- building_geometry (GEOMETRY) - Spatial data
- lod (INTEGER) - Level of detail
- census_id (BIGINT) - Link to census data

## POSTGIS FUNCTIONS (use public schema):
- public.ST_Distance(geom1, geom2) - distance in meters
- public.ST_DWithin(geom1, geom2, distance) - within distance
- public.ST_Intersects(geom1, geom2) - check intersection
- public.ST_Area(geom) - calculate area
- public.ST_Buffer(geom, distance) - create buffer

## EXAMPLES:

Count buildings:
SELECT COUNT(*) FROM cim_wizard_building;

Find nearby buildings:
SELECT b1.building_id, 
       public.ST_Distance(b1.building_geometry, b2.building_geometry) as distance_meters
FROM cim_wizard_building b1
CROSS JOIN cim_wizard_building b2
WHERE b2.building_id = 'target-uuid'
  AND b1.building_id != b2.building_id
  AND public.ST_DWithin(b1.building_geometry, b2.building_geometry, 1000)
ORDER BY distance_meters
LIMIT 5;

## RULES:
1. Return ONLY the SQL query
2. No markdown, no ```sql blocks
3. Use exact column names from schema
4. Use public.ST_* for PostGIS functions
5. End with semicolon
"""

print("✓ System prompt created")

def query_database(question: str, show_sql: bool = True):
    """
    Ask a question in natural language and get SQL results.
    
    Args:
        question: Natural language question
        show_sql: If True, print the generated SQL
    
    Returns:
        Query results as string
    """
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    start = time.time()
    
    # Step 1: Generate SQL
    print("\n[Generating SQL...]")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    sql_raw = response.content.strip()
    
    # Clean up SQL (remove markdown if present)
    sql = sql_raw
    if '```sql' in sql:
        sql = re.search(r'```sql\n(.*?)\n```', sql, re.DOTALL).group(1)
    elif '```' in sql:
        sql = re.search(r'```\n?(.*?)\n?```', sql, re.DOTALL).group(1)
    
    sql = sql.strip()
    
    if show_sql:
        print("\n" + "="*80)
        print("SQL GENERATED:")
        print("="*80)
        print(sql)
    
    # Step 2: Execute SQL
    print("\n[Executing SQL...]")
    try:
        result = db.run(sql)
        elapsed = time.time() - start
        
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(result)
        print("\n" + "="*80)
        print(f"✓ Success in {elapsed:.2f}s")
        print("="*80)
        
        return result
    except Exception as e:
        elapsed = time.time() - start
        print("\n" + "="*80)
        print("ERROR:")
        print("="*80)
        print(f"✗ {e}")
        print("\n" + "="*80)
        print(f"Time: {elapsed:.2f}s")
        print("="*80)
        return None

def execute_sql(query: str):
    """Execute SQL directly without LLM."""
    print("\n" + "="*80)
    print("EXECUTING SQL DIRECTLY:")
    print("="*80)
    print(f"\n{query}\n")
    
    try:
        result = db.run(query)
        print("="*80)
        print("RESULT:")
        print("="*80)
        print(result)
        return result
    except Exception as e:
        print("="*80)
        print("ERROR:")
        print("="*80)
        print(f"✗ {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("READY!")
    print("="*80)
    print("\nUsage:")
    print('  query_database("How many buildings are there?")')
    print('  execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")')
    print("\n" + "="*80 + "\n")
    
    # Test query
    query_database("How many buildings are in the database?")

