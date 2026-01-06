"""
Inference Module
Core Stable Diffusion + ControlNet pipeline for face stylization
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import time
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)


class StyleLensInference:
    """
    Main inference pipeline for Snap-style GenAI Lens
    Uses Stable Diffusion + ControlNet for identity-preserving stylization
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-canny",
        device: str = "auto",
        use_fp16: bool = True
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_id: HuggingFace model ID for base Stable Diffusion
            controlnet_id: ControlNet model ID
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_fp16: Use FP16 for faster inference
        """
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.use_fp16 = use_fp16
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32
        
        # Pipeline will be loaded on first use
        self.pipe = None
        self.loaded = False
        
        # Style presets
        self.style_prompts = {
            'anime': {
                'prompt': 'anime style portrait, vibrant colors, cel shaded, manga style, detailed eyes',
                'negative': 'realistic, photo, 3d render, blurry, low quality'
            },
            'cyberpunk': {
                'prompt': 'cyberpunk style portrait, neon lights, futuristic, high tech, glowing elements, dramatic lighting',
                'negative': 'natural, plain, old fashioned, low quality'
            },
            'sketch': {
                'prompt': 'pencil sketch portrait, hand drawn, black and white, detailed shading, artistic',
                'negative': 'photo, colorful, digital, low quality'
            },
            'oil_painting': {
                'prompt': 'oil painting portrait, artistic, painterly, brushstrokes, classical art style',
                'negative': 'photo, digital, low quality, blurry'
            },
            'watercolor': {
                'prompt': 'watercolor painting portrait, soft colors, artistic, flowing, delicate',
                'negative': 'photo, harsh lines, digital, low quality'
            }
        }
    
    def load_model(self):
        """Load Stable Diffusion + ControlNet pipeline"""
        if self.loaded:
            return
        
        print(f"Loading models on {self.device}...")
        start_time = time.time()
        
        try:
            # Load ControlNet
            print("Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                torch_dtype=self.dtype
            )
            
            # Load SD pipeline with ControlNet
            print("Loading Stable Diffusion...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                safety_checker=None  # Disable for faster loading
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Use faster scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Enable optimizations for GPU
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                # Try to enable xformers if available
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✓ Enabled xformers optimization")
                except:
                    print("ℹ xformers not available")
            
            load_time = time.time() - start_time
            print(f"✓ Models loaded in {load_time:.2f}s")
            
            self.loaded = True
            
        except Exception as e:
            print(f"✗ Failed to load models: {e}")
            raise
    
    def generate(
        self,
        conditioning_image: np.ndarray,
        style: str = 'anime',
        custom_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate stylized image
        
        Args:
            conditioning_image: Conditioning image (edges, landmarks, etc.)
            style: Style preset name
            custom_prompt: Optional custom prompt (overrides style)
            negative_prompt: Optional custom negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet strength
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with generated image and metadata
        """
        if not self.loaded:
            self.load_model()
        
        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
            neg_prompt = negative_prompt or self.style_prompts.get(style, {}).get('negative', '')
        else:
            style_config = self.style_prompts.get(style, self.style_prompts['anime'])
            prompt = style_config['prompt']
            neg_prompt = negative_prompt or style_config['negative']
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Convert conditioning image to PIL
        if isinstance(conditioning_image, np.ndarray):
            conditioning_pil = Image.fromarray(conditioning_image)
        else:
            conditioning_pil = conditioning_image
        
        # Resize to 512x512 for optimal performance
        conditioning_pil = conditioning_pil.resize((512, 512))
        
        # Measure inference time
        start_time = time.time()
        
        try:
            # Generate
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=conditioning_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator
                )
            
            generated_image = output.images[0]
            
            inference_time = time.time() - start_time
            
            # Convert to numpy
            generated_np = np.array(generated_image)
            
            return {
                'success': True,
                'image': generated_np,
                'pil_image': generated_image,
                'inference_time': inference_time,
                'metadata': {
                    'style': style,
                    'prompt': prompt,
                    'negative_prompt': neg_prompt,
                    'steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'controlnet_scale': controlnet_conditioning_scale,
                    'seed': seed,
                    'device': self.device,
                    'dtype': str(self.dtype)
                }
            }
            
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def generate_batch(
        self,
        conditioning_images: List[np.ndarray],
        style: str = 'anime',
        **kwargs
    ) -> List[Dict]:
        """
        Generate multiple images in batch
        
        Args:
            conditioning_images: List of conditioning images
            style: Style to apply
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generation results
        """
        results = []
        
        for img in conditioning_images:
            result = self.generate(img, style=style, **kwargs)
            results.append(result)
        
        return results
    
    def optimize_for_speed(
        self,
        num_steps: int = 15,
        use_fp16: bool = True
    ) -> Dict[str, any]:
        """
        Get optimized settings for faster inference
        
        Args:
            num_steps: Reduced number of steps
            use_fp16: Use FP16 precision
            
        Returns:
            Dictionary of optimized settings
        """
        return {
            'num_inference_steps': num_steps,
            'guidance_scale': 7.0,  # Slightly lower
            'controlnet_conditioning_scale': 0.9,
            'notes': [
                f'Reduced steps to {num_steps} (from 50)',
                'Using FP16 precision' if use_fp16 else 'Using FP32',
                'Memory-efficient attention enabled',
                'Expected 2-4x speedup with minimal quality loss'
            ]
        }
    
    def get_available_styles(self) -> List[str]:
        """Get list of available style presets"""
        return list(self.style_prompts.keys())
    
    def add_custom_style(
        self,
        name: str,
        prompt: str,
        negative_prompt: str = ''
    ):
        """
        Add custom style preset
        
        Args:
            name: Style name
            prompt: Positive prompt
            negative_prompt: Negative prompt
        """
        self.style_prompts[name] = {
            'prompt': prompt,
            'negative': negative_prompt
        }
    
    def benchmark(
        self,
        conditioning_image: np.ndarray,
        steps_list: List[int] = [10, 15, 20, 30, 50]
    ) -> List[Dict]:
        """
        Benchmark inference at different step counts
        
        Args:
            conditioning_image: Test conditioning image
            steps_list: List of step counts to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        print("Running benchmark...")
        
        for steps in steps_list:
            print(f"Testing {steps} steps...")
            
            result = self.generate(
                conditioning_image,
                style='anime',
                num_inference_steps=steps,
                seed=42  # Fixed seed for fair comparison
            )
            
            results.append({
                'steps': steps,
                'time': result['inference_time'],
                'success': result['success']
            })
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': torch.cuda.get_device_name(0)
            }
        else:
            return {'message': 'CPU mode - no GPU memory tracking'}
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.loaded = False
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            print("✓ Cleaned up resources")
