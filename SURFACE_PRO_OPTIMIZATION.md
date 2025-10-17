# Surface Pro 5 Optimization Guide

## Architecture Overview
```
Surface Pro 5 (Local)                    ipazia126 Server (Remote)
├── LangChain Agent (Lightweight)        ├── HF Model API (Heavy)
├── Dockerized PostGIS DB                ├── GPU Processing
└── Local Development Environment        └── Model Inference
```

## Surface Pro 5 Specifications
- **CPU**: Intel Core i5-7300U (2 cores, 4 threads)
- **RAM**: 8GB (typical configuration)
- **Storage**: SSD
- **No dedicated GPU**

## Computational Analysis

### What Runs Locally (Lightweight)
- **LangChain Agent**: Very lightweight, just orchestration
- **SQL Database**: PostGIS in Docker (moderate memory usage)
- **API Calls**: Simple HTTP requests to remote server
- **Local Development**: Your IDE and notebooks

### What Runs on Server (Heavy)
- **HF Model Loading**: 7B-14B parameter models
- **Model Inference**: GPU-accelerated text generation
- **Memory-intensive operations**: Model weights and activations

## Memory Usage Breakdown

### Local (Surface Pro 5)
- **Docker PostGIS**: ~500MB-1GB RAM
- **LangChain Agent**: ~50-100MB RAM
- **Python Environment**: ~200-300MB RAM
- **IDE/Notebook**: ~300-500MB RAM
- **Total Local Usage**: ~1-2GB RAM (well within 8GB limit)

### Remote (ipazia126)
- **HF Model (7B)**: ~14-16GB RAM
- **Model Inference**: Additional 2-4GB during generation
- **API Server**: ~100-200MB RAM
- **Total Remote Usage**: ~16-20GB RAM

## Performance Optimizations

### 1. Local Optimizations
```python
# Use connection pooling for database
from sqlalchemy.pool import QueuePool

# Optimize LangChain settings
llm = HFApiLLM(
    api_url="http://ipazia126.polito.it:9000",
    max_tokens=256,  # Reduce for faster responses
    temperature=0.0
)

# Use async operations where possible
import asyncio
from langchain_community.utilities import SQLDatabase
```

### 2. Database Optimizations
```bash
# Limit Docker memory usage
docker run -d \
  --name cim_wizard_integrated \
  --memory="2g" \
  --memory-swap="2g" \
  -p 5433:5432 \
  postgis/postgis:15-3.4
```

### 3. Network Optimizations
```python
# Use connection timeouts
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

## Recommended Setup

### 1. Local Environment
```bash
# Install minimal dependencies
pip install langchain langchain-community langgraph
pip install psycopg2-binary sqlalchemy
pip install requests
```

### 2. Docker Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgis:
    image: postgis/postgis:15-3.4
    container_name: cim_wizard_integrated
    ports:
      - "5433:5432"
    environment:
      POSTGRES_DB: cim_wizard_integrated
      POSTGRES_USER: cim_wizard_user
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgis_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  postgis_data:
```

### 3. Optimized LangChain Configuration
```python
# Optimized agent setup
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

# Database with connection pooling
db = SQLDatabase.from_uri(
    "postgresql://cim_wizard_user:password@localhost:5433/cim_wizard_integrated",
    engine_kwargs={
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }
)

# Lightweight toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Optimized agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier=ChatPromptTemplate.from_messages([
        ("system", "You are a helpful SQL assistant. Keep responses concise."),
        ("placeholder", "{messages}")
    ])
)
```

## Performance Monitoring

### Local Monitoring
```python
import psutil
import time

def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU: {cpu_percent}%, RAM: {memory.percent}%")

# Monitor during agent execution
monitor_resources()
result = agent.invoke({"messages": [HumanMessage(content="your query")]})
monitor_resources()
```

### Network Monitoring
```python
import requests
import time

def test_api_latency():
    start_time = time.time()
    response = requests.get("http://ipazia126.polito.it:9000/health")
    latency = time.time() - start_time
    print(f"API Latency: {latency:.2f} seconds")
    return latency < 5.0  # Should be under 5 seconds
```

## Expected Performance

### Response Times
- **Simple SQL queries**: 2-5 seconds
- **Complex queries**: 5-15 seconds
- **API calls**: 1-3 seconds
- **Database operations**: 0.1-1 seconds

### Resource Usage
- **CPU**: 20-40% during active use
- **RAM**: 1-2GB total usage
- **Network**: Minimal bandwidth usage
- **Storage**: ~1GB for Docker and dependencies

## Troubleshooting

### High Memory Usage
```bash
# Check Docker memory usage
docker stats cim_wizard_integrated

# Restart container if needed
docker restart cim_wizard_integrated
```

### Slow API Responses
```python
# Implement timeout and retry
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session
```

### Network Issues
```bash
# Test connectivity
ping ipazia126.polito.it
curl -I http://ipazia126.polito.it:9000/health

# Check if port is accessible
telnet ipazia126.polito.it 9000
```

## Conclusion

**Yes, this setup is perfectly suitable for your Surface Pro 5!**

The local agent is very lightweight and will run smoothly on your machine. The heavy computational work (model inference) happens on the server, so your local machine only needs to:
- Handle lightweight LangChain orchestration
- Manage a small PostGIS database
- Make HTTP API calls

This architecture gives you the best of both worlds: powerful AI capabilities via the server and responsive local development environment.
