"""
Simplified Identity Enforcement for Snap GenAI Lens
Uses prompt engineering + rejection sampling instead of complex embedding blending
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional
import time
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)


class SimpleIdentityEnforcer:
    """
    Identity enforcement using rejection sampling
    More reliable than complex embedding blending
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-canny",
        use_fp16: bool = True
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.use_fp16 = use_fp16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32
        self.pipe = None
        self.loaded = False
        
        self.style_prompts = {
            'anime': {
                'prompt': 'anime style portrait, same person, preserve facial features, vibrant colors, cel shaded, manga style, detailed eyes, high quality',
                'negative': 'different person, changed face, realistic, photo, 3d render, blurry, low quality, distorted face'
            },
            'cyberpunk': {
                'prompt': 'cyberpunk style portrait, same person, preserve identity, neon lights, futuristic, high tech, glowing elements, dramatic lighting',
                'negative': 'different person, changed face, natural, plain, low quality, distorted'
            },
            'sketch': {
                'prompt': 'pencil sketch portrait, same person, preserve facial features, hand drawn, black and white, detailed shading, artistic',
                'negative': 'different person, changed face, photo, colorful, digital, low quality'
            },
            'oil_painting': {
                'prompt': 'oil painting portrait, same person, preserve identity, artistic, painterly, brushstrokes, classical art style',
                'negative': 'different person, changed face, photo, digital, low quality, blurry'
            }
        }
    
    def load_model(self):
        if self.loaded:
            return
        
        print(f"Loading ControlNet pipeline on {self.device}...")
        start_time = time.time()
        
        try:
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                torch_dtype=self.dtype
            )
            
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            
            self.pipe = self.pipe.to(self.device)
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("  ✓ xformers enabled")
                except:
                    pass
            
            load_time = time.time() - start_time
            print(f"✓ Pipeline loaded in {load_time:.2f}s")
            print(f"✓ Identity enforcement: Rejection Sampling")
            
            self.loaded = True
            
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            raise
    
    def generate_with_identity_enforcement(
        self,
        conditioning_image: np.ndarray,
        original_image: np.ndarray,
        style: str = 'anime',
        identity_threshold: float = 0.75,
        max_attempts: int = 3,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate with identity enforcement via rejection sampling
        
        Args:
            conditioning_image: ControlNet conditioning
            original_image: Original face for identity verification
            style: Style preset
            identity_threshold: Minimum acceptable similarity (0.75 recommended)
            max_attempts: Maximum generation attempts
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet strength
            seed: Random seed
            
        Returns:
            Best result meeting identity threshold
        """
        if not self.loaded:
            self.load_model()
        
        # Load identity checker
        from identity.identity_preserver import IdentityPreserver
        identity_preserver = IdentityPreserver()
        identity_preserver.load_model()
        
        # Extract target embedding once
        target_embedding = identity_preserver.extract_embedding(original_image)
        
        # Get prompts (enhanced for identity preservation)
        style_config = self.style_prompts.get(style, self.style_prompts['anime'])
        prompt = style_config['prompt']
        neg_prompt = style_config['negative']
        
        # Prepare conditioning
        if isinstance(conditioning_image, np.ndarray):
            conditioning_pil = Image.fromarray(conditioning_image).resize((512, 512))
        else:
            conditioning_pil = conditioning_image
        
        # Track best result
        best_result = None
        best_similarity = 0.0
        
        print(f"🎨 Generating with identity enforcement...")
        print(f"   Threshold: {identity_threshold:.2f}")
        print(f"   Max attempts: {max_attempts}")
        
        for attempt in range(max_attempts):
            # Use different seed each attempt if original seed provided
            current_seed = seed + attempt if seed is not None else None
            generator = torch.Generator(device=self.device).manual_seed(current_seed) if current_seed else None
            
            print(f"\n   Attempt {attempt + 1}/{max_attempts}...")
            
            # Generate
            start_time = time.time()
            
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
            generated_np = np.array(generated_image)
            inference_time = time.time() - start_time
            
            # Verify identity
            generated_embedding = identity_preserver.extract_embedding(generated_np)
            
            if generated_embedding is None:
                print(f"      ⚠️ Could not extract embedding")
                continue
            
            similarity = identity_preserver.compute_similarity(
                target_embedding,
                generated_embedding
            )
            
            print(f"      Identity similarity: {similarity:.3f}")
            
            # Track best
            if similarity > best_similarity:
                best_similarity = similarity
                best_result = {
                    'image': generated_np,
                    'pil_image': generated_image,
                    'similarity': similarity,
                    'inference_time': inference_time,
                    'attempt': attempt + 1
                }
            
            # Check if meets threshold
            if similarity >= identity_threshold:
                print(f"      ✅ Identity preserved! ({similarity:.3f} >= {identity_threshold:.2f})")
                
                return {
                    'success': True,
                    'image': generated_np,
                    'pil_image': generated_image,
                    'inference_time': inference_time,
                    'identity_enforced': True,
                    'identity_similarity': similarity,
                    'attempts_needed': attempt + 1,
                    'threshold_met': True,
                    'metadata': {
                        'style': style,
                        'prompt': prompt,
                        'negative_prompt': neg_prompt,
                        'steps': num_inference_steps,
                        'guidance_scale': guidance_scale,
                        'controlnet_scale': controlnet_conditioning_scale,
                        'seed': seed,
                        'device': self.device,
                        'method': 'Rejection Sampling (Identity Enforced)'
                    }
                }
            else:
                print(f"      ❌ Below threshold, regenerating...")
        
        # Return best attempt if threshold not met
        print(f"\n   ⚠️ Could not meet threshold after {max_attempts} attempts")
        print(f"   Best similarity achieved: {best_similarity:.3f}")
        
        if best_result:
            return {
                'success': True,
                'image': best_result['image'],
                'pil_image': best_result['pil_image'],
                'inference_time': best_result['inference_time'],
                'identity_enforced': False,
                'identity_similarity': best_similarity,
                'attempts_needed': max_attempts,
                'threshold_met': False,
                'metadata': {
                    'style': style,
                    'prompt': prompt,
                    'negative_prompt': neg_prompt,
                    'steps': num_inference_steps,
                    'device': self.device,
                    'method': f'Rejection Sampling (Best: {best_similarity:.3f})'
                }
            }
        else:
            return {
                'success': False,
                'error': 'All generation attempts failed',
                'identity_enforced': False
            }
    
    def cleanup(self):
        if self.pipe:
            del self.pipe
        if self.device == 'cuda':
            torch.cuda.empty_cache()