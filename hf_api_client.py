#!/usr/bin/env python3
"""
Hugging Face API Client
Local client to interact with the HF API server running on ipazia
"""

import requests
import json
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFApiClient:
    """Client for interacting with Hugging Face API server"""
    
    def __init__(self, base_url: str = "http://ipazia126.polito.it:9000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the API server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def chat(self, 
             prompt: str, 
             max_tokens: int = 512, 
             temperature: float = 0.0,
             top_p: float = 0.9,
             do_sample: bool = True) -> Dict[str, Any]:
        """Send a chat request to the API server"""
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample
        }
        
        try:
            logger.info(f"Sending request: {prompt[:100]}...")
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """Test if the API server is reachable"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = HFApiClient()
    
    # Test connection
    print("🔍 Testing connection to API server...")
    if client.test_connection():
        print("✅ Connection successful!")
        
        # Check health
        health = client.health_check()
        print(f"Health status: {health}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test chat
        test_prompt = "Hello! Can you help me with SQL queries? Please respond briefly."
        print(f"\n🧪 Testing chat with prompt: {test_prompt}")
        
        response = client.chat(test_prompt)
        if "error" not in response:
            print(f"✅ Response: {response['response']}")
            print(f"Tokens generated: {response['tokens_generated']}")
        else:
            print(f"❌ Error: {response['error']}")
    else:
        print("❌ Connection failed! Make sure the API server is running.")
