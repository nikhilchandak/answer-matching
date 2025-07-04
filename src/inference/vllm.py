import logging
import torch
from typing import List, Dict, Any

from inference.base import BaseInference

logger = logging.getLogger(__name__)

class VLLMInference(BaseInference):
    """vLLM-based inference engine"""
    
    def __init__(self, model_path: str, max_tokens: int = 2048, temperature: float = 0.7):
        """
        Initialize vLLM inference engine
        
        Args:
            model_path: Path to the HuggingFace model
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine = None
        
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            logger.info(f"Initializing vLLM engine with model {self.model_path}")
            self.engine = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            # Store SamplingParams class for later use
            self.SamplingParams = SamplingParams
            logger.info("vLLM engine initialized successfully")
        except ImportError:
            logger.error("vLLM not installed. Run: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Error initializing vLLM engine: {e}")
            raise
            
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for prompts using vLLM
        
        Args:
            prompts: List of prompts to generate completions for
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated completions
        """
        if not self.engine:
            raise ValueError("vLLM engine not initialized")
            
        batch_size = kwargs.get('batch_size', 5)
        results = []
        
        # Create sampling parameters for vLLM
        sampling_params = self.SamplingParams(
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            top_p=kwargs.get('top_p', 0.95),
        )
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Generate completions (vLLM is synchronous)
            outputs = self.engine.generate(
                prompts=batch,
                sampling_params=sampling_params
            )
            
            # Process outputs
            for output in outputs:
                results.append(output.outputs[0].text)
        
        return results 