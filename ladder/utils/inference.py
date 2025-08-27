import os
import openai
import anthropic
import aiohttp
import json
from together import AsyncTogether
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import requests




# Load environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
deepinfra_api_key = os.getenv('DEEPINFRA_API_KEY')
#hf_client = InferenceClient(token=hf_api_token)  # Replace with your Hugging Face token

# Initialize clients
async_openai_client = AsyncOpenAI(api_key=openai_api_key)

# Add NVIDIA API key

deepseek_client = AsyncOpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

# Add DeepInfra API key

# Add DeepInfra client initialization
deepinfra_client = AsyncOpenAI(
    api_key=deepinfra_api_key,
    base_url="https://api.deepinfra.com/v1/openai"
)

async def generate_text(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 0) -> str:
    """
    Asynchronously generate text using various AI models.
    
    :param model: The name of the model to use (e.g., "gpt-3.5-turbo", "claude-2", "meta-llama/Llama-2-70b-chat-hf")
    :param prompt: The input prompt for text generation
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in generation (0.0 to 1.0)
    :return: Generated text as a string
    """
    
    # OpenAI models
    
   
    if model.startswith("gpt-") or model.startswith("o1"):
        response = await async_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
          
        )
        return response.choices[0].message.content.strip()
    
    elif model.startswith("ft:gpt") or model.startswith("o1"):
        response = await async_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            
        )
        return response.choices[0].message.content.strip()
    
    # Anthropic (Claude) models
    elif model.startswith("claude-"):
        async def run_anthropic():
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            if model.startswith("claude-3"):
                response = client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text.strip()
            else:
                response = client.completions.create(
                    model=model,
                    prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion.strip()
        
        return await run_anthropic()
    
    # Together AI models

   
    elif model.startswith("meta-llama/") or model.startswith("deepseek-ai") or model.startswith("Qwen/") or model.startswith("Meta-Llama"):
        response = await deepinfra_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    
    
    
    # NVIDIA models
    elif model.startswith("nvidia"):
        nvidia_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        response = await nvidia_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        return full_response.strip()

    # Add DeepSeek model support after the NVIDIA section
    elif model.startswith("deepseek-"):
        if model == "deepseek-reasoner":
            response = await deepseek_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Combine reasoning and content for the reasoner model
            reasoning = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
            return f"\nAnswer:\n{content}".strip()
        else:
            response = await deepseek_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

    # Hugging Face models (fallback for any other model)
    else:
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        async def query_huggingface():
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        else:
                            return str(result)
                    else:
                        raise Exception(f"Error with Hugging Face Inference API for model {model}: {await response.text()}")
        
        return await query_huggingface()
