# Hugging Face API Server Setup Guide

## Overview
This setup allows you to host a Hugging Face model on the ipazia server and use it via API from your local machine with LangChain.

## Architecture
```
Local Machine (Your IDE) ←→ API Calls ←→ ipazia Server (HF Model)
```

## Files Created
1. `hf_api_server.py` - FastAPI server for hosting HF models
2. `requirements_api_server.txt` - Dependencies for the server
3. `hf_api_client.py` - Simple client for testing
4. `langchain_hf_api.py` - LangChain integration
5. `agent_cim_assist_api.ipynb` - Updated notebook using API
6. `deploy_to_ipazia.sh` - Deployment script

## Step 1: Transfer Files to ipazia Server

From your local machine:
```bash
# Navigate to your project directory
cd D:\coesi\db_assist\cim_assist

# Transfer files to ipazia
scp hf_api_server.py castangia@ipazia126.polito.it:~/Ali_workspace/
scp requirements_api_server.txt castangia@ipazia126.polito.it:~/Ali_workspace/
scp deploy_to_ipazia.sh castangia@ipazia126.polito.it:~/Ali_workspace/
```

## Step 2: Deploy on ipazia Server

SSH into ipazia and run:
```bash
ssh castangia@ipazia126.polito.it
cd ~/Ali_workspace

# Make deployment script executable
chmod +x deploy_to_ipazia.sh

# Run deployment
./deploy_to_ipazia.sh
```

## Step 3: Test the API Server

On ipazia server:
```bash
# Check if service is running
sudo systemctl status hf-api

# Check logs
sudo journalctl -u hf-api -f

# Test API locally on server
curl http://localhost:8000/health
```

## Step 4: Use from Local Machine

### Option A: Use the API Client
```python
from hf_api_client import HFApiClient

client = HFApiClient("http://ipazia126.polito.it:9000")
response = client.chat("Hello! How are you?")
print(response)
```

### Option B: Use with LangChain
```python
from langchain_hf_api import HFApiLLM

llm = HFApiLLM(api_url="http://ipazia126.polito.it:9000")
response = llm("Hello! How are you?")
print(response)
```

### Option C: Use the Updated Notebook
Open `agent_cim_assist_api.ipynb` and run the cells.

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /chat` - Generate text
- `GET /docs` - API documentation (Swagger UI)

## Configuration

### Environment Variables (on ipazia server):
```bash
export HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  # Change model here
export API_HOST="0.0.0.0"                        # Listen on all interfaces
export API_PORT="9000"                           # Port number
```

### Available Models:
- `Qwen/Qwen2.5-7B-Instruct` (default)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `codellama/CodeLlama-7b-Instruct-hf`
- `meta-llama/Llama-3.1-8B-Instruct`

## Monitoring

### Check Service Status:
```bash
sudo systemctl status hf-api
```

### View Logs:
```bash
sudo journalctl -u hf-api -f
```

### Restart Service:
```bash
sudo systemctl restart hf-api
```

### Stop Service:
```bash
sudo systemctl stop hf-api
```

## Troubleshooting

### 1. Service Won't Start
```bash
# Check logs
sudo journalctl -u hf-api -n 50

# Check if port is in use
sudo netstat -tlnp | grep 8000
```

### 2. Model Loading Issues
```bash
# Check GPU availability
nvidia-smi

# Check memory usage
free -h
```

### 3. API Connection Issues
```bash
# Test from server
curl http://localhost:9000/health

# Test from local machine
curl http://ipazia126.polito.it:9000/health
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available on ipazia
2. **Memory Optimization**: Use quantization for lower memory usage
3. **Batch Requests**: The API can handle multiple concurrent requests
4. **Caching**: Consider implementing response caching for repeated queries

## Security Notes

- The API is currently open to all interfaces (0.0.0.0)
- Consider adding authentication if needed
- Use HTTPS in production environments
- Monitor resource usage to prevent abuse

## Next Steps

1. Test the basic API functionality
2. Integrate with your LangChain agents
3. Monitor performance and resource usage
4. Consider adding authentication if needed
5. Implement streaming responses for better UX
