#!/usr/bin/env python3
"""
Hugging Face API Server
Hosts a Hugging Face model as a REST API for remote access
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import uvicorn
import logging
from typing import Optional, List
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hugging Face Model API",
    description="API server for hosting Hugging Face models",
    version="1.0.0"
)

# Global variables for model and pipeline
model = None
tokenizer = None
pipe = None

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    model_name: str
    tokens_generated: int

class ModelInfo(BaseModel):
    model_name: str
    model_loaded: bool
    device: str
    memory_usage: dict

@app.on_event("startup")
async def load_model():
    """Load the Hugging Face model on startup"""
    global model, tokenizer, pipe
    
    # Model configuration - change this to your preferred model
    MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    
    # Alternative models (uncomment one to use):
    # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    # MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
    # MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    
    logger.info(f"Loading model: {MODEL_NAME}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=True,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info(f"✅ Model {MODEL_NAME} loaded successfully!")
        logger.info(f"Device: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hugging Face Model API Server",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get memory usage
    memory_usage = {}
    if torch.cuda.is_available():
        memory_usage = {
            "gpu_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "gpu_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            "gpu_max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        }
    
    return ModelInfo(
        model_name=os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        model_loaded=True,
        device=str(next(model.parameters()).device),
        memory_usage=memory_usage
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate text using the loaded model"""
    if model is None or pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating response for prompt: {request.prompt[:100]}...")
        
        # Generate response
        result = pipe(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = result[0]['generated_text']
        
        # Count tokens (approximate)
        input_tokens = len(tokenizer.encode(request.prompt))
        output_tokens = len(tokenizer.encode(response_text))
        total_tokens = input_tokens + output_tokens
        
        logger.info(f"Generated {output_tokens} tokens")
        
        return ChatResponse(
            response=response_text,
            model_name=os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            tokens_generated=output_tokens
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream text generation (for future implementation)"""
    # This could be implemented for streaming responses
    raise HTTPException(status_code=501, detail="Streaming not implemented yet")

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("API_HOST", "0.0.0.0")  # Listen on all interfaces
    PORT = int(os.getenv("API_PORT", "9000"))
    WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting API server on {HOST}:{PORT}")
    logger.info(f"Workers: {WORKERS}")
    
    # Run the server
    uvicorn.run(
        "hf_api_server:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level="info",
        reload=False  # Set to True for development
    )
