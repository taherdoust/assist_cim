#!/usr/bin/env python3
"""
LangChain Integration with Hugging Face API
Custom LLM class to use the HF API server with LangChain
"""

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import requests
import json
import logging

logger = logging.getLogger(__name__)

class HFApiLLM(LLM):
    """Custom LangChain LLM that uses the Hugging Face API server"""
    
    api_url: str = "http://ipazia126.polito.it:9000"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = requests.Session()
    
    @property
    def _llm_type(self) -> str:
        return "hf_api"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Hugging Face API server"""
        
        # Prepare payload
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "do_sample": kwargs.get("do_sample", self.do_sample)
        }
        
        try:
            logger.info(f"Calling HF API with prompt: {prompt[:100]}...")
            
            response = self.session.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return f"Error: Failed to get response from API server - {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check API server health"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.api_url}/model/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Example usage with LangChain
if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # Initialize the custom LLM
    llm = HFApiLLM()
    
    # Test health
    health = llm.health_check()
    print(f"API Health: {health}")
    
    # Test model info
    model_info = llm.get_model_info()
    print(f"Model Info: {model_info}")
    
    # Test direct call
    response = llm("Hello! Can you help me with SQL queries?")
    print(f"Direct response: {response}")
    
    # Test with LangChain chain
    prompt = ChatPromptTemplate.from_template("You are a helpful SQL assistant. {question}")
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"question": "How do I write a SELECT query?"})
    print(f"Chain response: {result}")
