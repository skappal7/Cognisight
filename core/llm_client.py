"""
LLM Client - Unified interface for multiple LLM providers
Supports OpenRouter (free/paid), Ollama, and LM Studio
"""

import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime


class LLMClient:
    """
    Unified client for interacting with various LLM providers
    """
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Token cost tracking (approximate costs per 1M tokens)
        self.cost_per_token = self._get_cost_per_token()
    
    def _get_cost_per_token(self) -> Dict[str, float]:
        """Get approximate token costs for different models"""
        costs = {
            # OpenRouter Free Models (approximate)
            'google/gemini-flash-1.5-8b': {'input': 0.0, 'output': 0.0},
            'meta-llama/llama-3.2-3b-instruct': {'input': 0.0, 'output': 0.0},
            'qwen/qwen-2-7b-instruct': {'input': 0.0, 'output': 0.0},
            
            # OpenRouter Paid Models (per 1M tokens)
            'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},
            'openai/gpt-4o': {'input': 2.5, 'output': 10.0},
            'google/gemini-pro-1.5': {'input': 1.25, 'output': 5.0},
            'deepseek/deepseek-chat': {'input': 0.14, 'output': 0.28},
            
            # Local models (free)
            'local': {'input': 0.0, 'output': 0.0}
        }
        
        model_cost = costs.get(self.model, {'input': 0.0, 'output': 0.0})
        
        # For local models
        if 'local' in self.provider.lower() or 'ollama' in self.provider.lower():
            model_cost = {'input': 0.0, 'output': 0.0}
        
        return model_cost
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate completion from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (optional, uses default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses default if not provided)
            
        Returns:
            Dict containing content and usage information
        """
        if 'openrouter' in self.provider.lower():
            return self._generate_openrouter(prompt, system_prompt, temperature, max_tokens)
        elif 'ollama' in self.provider.lower():
            return self._generate_ollama(prompt, system_prompt, temperature, max_tokens)
        elif 'lm studio' in self.provider.lower():
            return self._generate_lm_studio(prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Generate using OpenRouter API"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/data-analyst-agent",
            "X-Title": "Data Analyst Agent"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract usage information
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            
            # Calculate cost
            estimated_cost = (
                prompt_tokens * self.cost_per_token['input'] / 1_000_000 +
                completion_tokens * self.cost_per_token['output'] / 1_000_000
            )
            
            return {
                'content': result['choices'][0]['message']['content'],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'estimated_cost': estimated_cost
                },
                'model': result.get('model', self.model),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Generate using Ollama API"""
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature or self.temperature,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'content': result['response'],
                'usage': {
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                    'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0),
                    'estimated_cost': 0.0
                },
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def _generate_lm_studio(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Generate using LM Studio API (OpenAI-compatible)"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            usage = result.get('usage', {})
            
            return {
                'content': result['choices'][0]['message']['content'],
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'estimated_cost': 0.0
                },
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"LM Studio API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count
        Generally, 1 token ≈ 4 characters for English text
        """
        return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately fit within token limit"""
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Truncate to fit
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        return text[:max_chars] + "..."
