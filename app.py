"""
Snap GenAI Lens - Gradio Demo Application
Main interface for the face stylization system
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.face_processor import FacePreprocessor
from conditioning.condition_generator import ConditioningGenerator
from identity.identity_preserver import IdentityPreserver
from models.inference import StyleLensInference
from evaluation.evaluator import LensEvaluator


class SnapLensApp:
    """
    Main application class for Snap GenAI Lens
    """
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing Snap GenAI Lens...")
        
        # Initialize modules
        self.face_processor = FacePreprocessor(confidence_threshold=0.5)
        self.conditioning_gen = ConditioningGenerator()
        self.identity_preserver = IdentityPreserver()
        self.evaluator = LensEvaluator()
        
        # Inference pipeline (loaded on first use)
        self.inference = None
        
        print("✓ Modules initialized")
    
    def load_inference_pipeline(self):
        """Load the inference pipeline (heavy operation)"""
        if self.inference is None:
            print("Loading Stable Diffusion pipeline...")
            self.inference = StyleLensInference(
                model_id="runwayml/stable-diffusion-v1-5",
                controlnet_id="lllyasviel/sd-controlnet-canny",
                use_fp16=True
            )
            self.inference.load_model()
            self.identity_preserver.load_model()
            print("✓ All models loaded")
    
    def process_image(
        self,
        input_image: np.ndarray,
        style: str,
        num_steps: int,
        guidance_scale: float,
        controlnet_scale: float,
        seed: int,
        show_preprocessing: bool
    ):
        """
        Main processing pipeline
        
        Args:
            input_image: Input image from Gradio
            style: Selected style
            num_steps: Number of diffusion steps
            guidance_scale: CFG scale
            controlnet_scale: ControlNet conditioning scale
            seed: Random seed
            show_preprocessing: Whether to show preprocessing visualization
            
        Returns:
            Tuple of (output_image, info_text, preprocessing_viz)
        """
        try:
            # Validate input
            if input_image is None:
                return None, "❌ Please upload an image", None
            
            # Convert to RGB if needed
            if len(input_image.shape) == 2:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            elif input_image.shape[2] == 4:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
            
            # 1. Face Preprocessing
            print("Step 1: Face preprocessing...")
            preprocessing_results = self.face_processor.process_image(input_image)
            
            if not preprocessing_results['success']:
                error_msg = preprocessing_results.get('error', 'Unknown error')
                return None, f"❌ Face detection failed: {error_msg}", None
            
            # Create preprocessing visualization
            preprocessing_viz = None
            if show_preprocessing:
                preprocessing_viz = self.face_processor.visualize_preprocessing(
                    input_image, preprocessing_results
                )
            
            # 2. Generate Conditioning
            print("Step 2: Generating conditioning signals...")
            conditions = self.conditioning_gen.generate_all_conditions(
                input_image,
                preprocessing_results,
                use_canny=True,
                use_landmarks=True,
                use_depth=False,  # Disabled for speed
                use_mask=True
            )
            
            # Combine conditions (weighted)
            weights = {
                'canny': 0.6,
                'landmarks': 0.4,
                'mask': 0.3
            }
            
            combined_conditioning = self.conditioning_gen.combine_conditions(
                conditions, weights
            )
            
            # 3. Load models if needed
            self.load_inference_pipeline()
            
            # 4. Generate styled image
            print("Step 3: Generating styled image...")
            generation_result = self.inference.generate(
                conditioning_image=combined_conditioning,
                style=style,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                seed=seed if seed > 0 else None
            )
            
            if not generation_result['success']:
                error_msg = generation_result.get('error', 'Unknown error')
                return None, f"❌ Generation failed: {error_msg}", preprocessing_viz
            
            generated_image = generation_result['image']
            inference_time = generation_result['inference_time']
            
            # 5. Evaluate identity preservation
            print("Step 4: Evaluating results...")
            identity_metrics = self.identity_preserver.measure_identity_preservation(
                input_image,
                generated_image,
                preprocessing_results['face_info']['bbox']
            )
            
            # 6. Evaluate overall quality
            eval_metrics = self.evaluator.evaluate_generation(
                original_image=input_image,
                generated_image=generated_image,
                conditioning_image=combined_conditioning,
                style_prompt=generation_result['metadata']['prompt'],
                identity_metrics=identity_metrics,
                inference_time=inference_time
            )
            
            # 7. Create info text
            info_text = self._create_info_text(
                eval_metrics,
                generation_result['metadata'],
                preprocessing_results
            )
            
            print("✓ Processing complete!")
            
            return generated_image, info_text, preprocessing_viz
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg, None
    
    def _create_info_text(
        self,
        eval_metrics: dict,
        generation_metadata: dict,
        preprocessing_results: dict
    ) -> str:
        """Create formatted info text"""
        lines = []
        
        lines.append("=" * 50)
        lines.append("✅ GENERATION SUCCESSFUL")
        lines.append("=" * 50)
        lines.append("")
        
        # Performance
        lines.append("⏱️ PERFORMANCE")
        lines.append(f"  Inference Time: {eval_metrics.get('inference_time', 0):.2f}s")
        lines.append(f"  FPS: {eval_metrics.get('fps', 0):.2f}")
        lines.append("")
        
        # Quality Metrics
        lines.append("📊 QUALITY METRICS")
        lines.append(f"  Identity Similarity: {eval_metrics.get('identity_similarity', 0):.3f}")
        lines.append(f"  Identity Status: {eval_metrics.get('confidence', 'unknown')}")
        
        if 'clip_similarity' in eval_metrics:
            lines.append(f"  CLIP Similarity: {eval_metrics.get('clip_similarity', 0):.3f}")
        
        if 'overall_quality' in eval_metrics:
            lines.append(f"  Overall Quality: {eval_metrics.get('overall_quality', 0):.3f}")
        
        lines.append("")
        
        # Generation Settings
        lines.append("⚙️ GENERATION SETTINGS")
        lines.append(f"  Style: {generation_metadata.get('style', 'unknown')}")
        lines.append(f"  Steps: {generation_metadata.get('steps', 0)}")
        lines.append(f"  Guidance Scale: {generation_metadata.get('guidance_scale', 0):.1f}")
        lines.append(f"  Device: {generation_metadata.get('device', 'unknown')}")
        lines.append("")
        
        # Face Detection
        lines.append("👤 FACE DETECTION")
        face_conf = preprocessing_results['face_info']['confidence']
        lines.append(f"  Confidence: {face_conf:.3f}")
        lines.append(f"  Landmarks: {len(preprocessing_results['landmarks'])} points")
        lines.append("")
        
        return "\n".join(lines)
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Define interface
        with gr.Blocks(title="Snap GenAI Lens", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🎨 Snap GenAI Lens Demo
            
            **Identity-Preserving Face Stylization using Stable Diffusion + ControlNet**
            
            Upload a selfie and transform it into different artistic styles while preserving your identity!
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input
                    gr.Markdown("### 📸 Input")
                    input_image = gr.Image(
                        label="Upload Selfie",
                        type="numpy",
                        height=400
                    )
                    
                    # Style selection
                    style_dropdown = gr.Dropdown(
                        choices=['anime', 'cyberpunk', 'sketch', 'oil_painting', 'watercolor'],
                        value='anime',
                        label="Style"
                    )
                    
                    # Advanced settings
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        num_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Inference Steps (lower = faster)"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=5.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                        
                        controlnet_scale = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.1,
                            label="ControlNet Strength"
                        )
                        
                        seed = gr.Number(
                            value=42,
                            label="Seed (0 for random)",
                            precision=0
                        )
                        
                        show_preprocessing = gr.Checkbox(
                            value=False,
                            label="Show preprocessing visualization"
                        )
                    
                    # Generate button
                    generate_btn = gr.Button(
                        "🎨 Generate",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output
                    gr.Markdown("### ✨ Generated Output")
                    output_image = gr.Image(
                        label="Stylized Result",
                        type="numpy",
                        height=400
                    )
                    
                    # Info text
                    info_text = gr.Textbox(
                        label="Generation Info",
                        lines=15,
                        max_lines=20
                    )
            
            # Preprocessing visualization (hidden by default)
            with gr.Row(visible=True):
                preprocessing_viz = gr.Image(
                    label="Preprocessing Visualization",
                    type="numpy",
                    visible=True
                )
            
            # Examples
            gr.Markdown("### 📚 Tips")
            gr.Markdown("""
            - Use clear, well-lit photos for best results
            - Face should be clearly visible and frontal
            - Higher steps = better quality but slower
            - Lower ControlNet strength for more creative freedom
            """)
            
            # Connect processing function
            generate_btn.click(
                fn=self.process_image,
                inputs=[
                    input_image,
                    style_dropdown,
                    num_steps,
                    guidance_scale,
                    controlnet_scale,
                    seed,
                    show_preprocessing
                ],
                outputs=[output_image, info_text, preprocessing_viz]
            )
        
        return demo


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("SNAP GENAI LENS - Starting Demo")
    print("="*60 + "\n")
    
    # Create app
    app = SnapLensApp()
    
    # Create interface
    demo = app.create_interface()
    
    # Launch
    print("\n🚀 Launching Gradio interface...")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=True
    )


if __name__ == "__main__":
    main()
