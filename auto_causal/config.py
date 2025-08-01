# auto_causal/config.py
"""Central configuration for AutoCausal, including LLM client setup."""

import os
import logging
from typing import Optional

# Langchain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI # Default
from langchain_anthropic import ChatAnthropic # Example
# Add other providers if needed, e.g.:
# from langchain_community.chat_models import ChatOllama 
from dotenv import load_dotenv

# Import Together provider
from langchain_together import ChatTogether

logger = logging.getLogger(__name__)

# Load .env file when this module is loaded
load_dotenv()

def get_llm_client(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Initializes and returns the chosen LLM client based on provider.
    
    Reads provider, model, and API keys from environment variables if not passed directly.
    Defaults to OpenAI GPT-4o-mini if no provider/model specified.
    """
    # Prioritize arguments, then environment variables, then defaults
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    provider = provider.lower()
    
    # Default model depends on provider
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "together": "Qwen/Qwen2.5-72B-Instruct-Turbo"  # Default Together model
    }
    
    model_name = model_name or os.getenv("LLM_MODEL", default_models.get(provider, default_models["openai"]))
    
    api_key = None
    kwargs.setdefault("temperature", 0) # Default temperature if not provided

    logger.info(f"Initializing LLM client: Provider='{provider}', Model='{model_name}'")

    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)
        
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)
        
        elif provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in environment.")
            return ChatTogether(model=model_name, api_key=api_key, **kwargs)
            
        # Example for Ollama (ensure langchain_community is installed)
        # elif provider == "ollama":
        #     try:
        #         from langchain_community.chat_models import ChatOllama
        #         return ChatOllama(model=model_name, **kwargs)
        #     except ImportError:
        #         raise ValueError("langchain_community needed for Ollama. Run `pip install langchain-community`")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    except Exception as e:
        logger.error(f"Failed to initialize LLM (Provider: {provider}, Model: {model_name}): {e}")
        raise # Re-raise the exception 