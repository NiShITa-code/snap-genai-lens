"""
Example Usage Script
Demonstrates how to use the Snap GenAI Lens modules programmatically
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from preprocessing.face_processor import FacePreprocessor
from conditioning.condition_generator import ConditioningGenerator
from identity.identity_preserver import IdentityPreserver
from models.inference import StyleLensInference
from evaluation.evaluator import LensEvaluator


def example_full_pipeline(image_path: str, output_path: str = "output.png"):
    """
    Complete example of the full pipeline
    
    Args:
        image_path: Path to input image
        output_path: Path to save output
    """
    print("=" * 60)
    print("Snap GenAI Lens - Example Pipeline")
    print("=" * 60)
    
    # Load image
    print("\n1. Loading image...")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {image_rgb.shape}")
    
    # Initialize modules
    print("\n2. Initializing modules...")
    face_processor = FacePreprocessor()
    cond_gen = ConditioningGenerator()
    identity_preserver = IdentityPreserver()
    evaluator = LensEvaluator()
    
    # Process face
    print("\n3. Detecting face...")
    preprocessing_results = face_processor.process_image(image_rgb)
    
    if not preprocessing_results['success']:
        print(f"   ❌ Error: {preprocessing_results['error']}")
        return
    
    print(f"   ✓ Face detected with confidence: {preprocessing_results['face_info']['confidence']:.3f}")
    print(f"   ✓ Extracted {len(preprocessing_results['landmarks'])} landmarks")
    
    # Generate conditioning
    print("\n4. Generating conditioning signals...")
    conditions = cond_gen.generate_all_conditions(
        image_rgb,
        preprocessing_results,
        use_canny=True,
        use_landmarks=True,
        use_depth=False,
        use_mask=True
    )
    print(f"   ✓ Generated {len(conditions)} conditioning signals")
    
    # Combine conditions
    weights = {'canny': 0.6, 'landmarks': 0.4, 'mask': 0.3}
    combined_cond = cond_gen.combine_conditions(conditions, weights)
    print(f"   ✓ Combined conditioning: {combined_cond.shape}")
    
    # Load inference pipeline
    print("\n5. Loading Stable Diffusion + ControlNet...")
    inference = StyleLensInference(use_fp16=True)
    inference.load_model()
    
    # Generate
    print("\n6. Generating styled image...")
    result = inference.generate(
        conditioning_image=combined_cond,
        style='anime',
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=42
    )
    
    if not result['success']:
        print(f"   ❌ Generation failed: {result.get('error')}")
        return
    
    print(f"   ✓ Generated in {result['inference_time']:.2f}s")
    print(f"   ✓ FPS: {1/result['inference_time']:.2f}")
    
    # Evaluate identity
    print("\n7. Evaluating identity preservation...")
    identity_preserver.load_model()
    identity_metrics = identity_preserver.measure_identity_preservation(
        image_rgb,
        result['image'],
        preprocessing_results['face_info']['bbox']
    )
    print(f"   ✓ Identity similarity: {identity_metrics['identity_similarity']:.3f}")
    print(f"   ✓ Identity preserved: {identity_metrics['identity_preserved']}")
    
    # Complete evaluation
    print("\n8. Computing quality metrics...")
    eval_results = evaluator.evaluate_generation(
        original_image=image_rgb,
        generated_image=result['image'],
        conditioning_image=combined_cond,
        style_prompt=result['metadata']['prompt'],
        identity_metrics=identity_metrics,
        inference_time=result['inference_time']
    )
    
    if 'clip_similarity' in eval_results:
        print(f"   ✓ CLIP similarity: {eval_results['clip_similarity']:.3f}")
    if 'overall_quality' in eval_results:
        print(f"   ✓ Overall quality: {eval_results['overall_quality']:.3f}")
    
    # Save output
    print(f"\n9. Saving output to {output_path}...")
    output_pil = Image.fromarray(result['image'])
    output_pil.save(output_path)
    print("   ✓ Saved!")
    
    # Create visualization
    print("\n10. Creating comparison visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(combined_cond)
    axes[1].set_title('Conditioning')
    axes[1].axis('off')
    
    axes[2].imshow(result['image'])
    axes[2].set_title(f"Generated ({result['metadata']['style']})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved comparison to comparison.png")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)
    
    return result, eval_results


def example_batch_generation(image_path: str, styles: list = None):
    """
    Example of generating multiple styles
    
    Args:
        image_path: Path to input image
        styles: List of styles to generate
    """
    if styles is None:
        styles = ['anime', 'cyberpunk', 'sketch']
    
    print(f"\nGenerating {len(styles)} styles...")
    
    # Load and preprocess once
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_processor = FacePreprocessor()
    cond_gen = ConditioningGenerator()
    
    preprocessing_results = face_processor.process_image(image_rgb)
    
    if not preprocessing_results['success']:
        print(f"Error: {preprocessing_results['error']}")
        return
    
    conditions = cond_gen.generate_all_conditions(
        image_rgb, preprocessing_results,
        use_canny=True, use_landmarks=True
    )
    
    combined_cond = cond_gen.combine_conditions(
        conditions,
        {'canny': 0.6, 'landmarks': 0.4}
    )
    
    # Load model once
    inference = StyleLensInference()
    inference.load_model()
    
    # Generate all styles
    results = {}
    for style in styles:
        print(f"Generating {style}...")
        result = inference.generate(
            combined_cond,
            style=style,
            num_inference_steps=20,
            seed=42
        )
        
        if result['success']:
            results[style] = result['image']
            print(f"  ✓ {result['inference_time']:.2f}s")
    
    # Visualize
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, (style, img) in enumerate(results.items(), start=1):
        axes[idx].imshow(img)
        axes[idx].set_title(style.title())
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved batch results to batch_results.png")
    
    return results


def example_benchmark():
    """
    Example of benchmarking performance
    """
    print("\nRunning performance benchmark...")
    
    # Create dummy conditioning
    dummy_cond = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Load model
    inference = StyleLensInference()
    inference.load_model()
    
    # Benchmark
    results = inference.benchmark(
        dummy_cond,
        steps_list=[10, 15, 20, 30, 50]
    )
    
    # Display
    print("\n📊 Benchmark Results:")
    print("-" * 40)
    print(f"{'Steps':<10} {'Time (s)':<15} {'FPS':<10}")
    print("-" * 40)
    
    for r in results:
        fps = 1 / r['time'] if r['time'] > 0 else 0
        print(f"{r['steps']:<10} {r['time']:<15.2f} {fps:<10.2f}")
    
    return results


if __name__ == "__main__":
    print("\n🎨 Snap GenAI Lens - Example Usage\n")
    
    # You need to provide a test image
    # Example: python example_usage.py
    
    # Uncomment the example you want to run:
    
    # Example 1: Full pipeline
    # result, metrics = example_full_pipeline("path/to/your/image.jpg")
    
    # Example 2: Batch generation
    # results = example_batch_generation("path/to/your/image.jpg")
    
    # Example 3: Benchmark
    # benchmark_results = example_benchmark()
    
    print("\n💡 Tip: Edit this file to uncomment the example you want to run")
    print("   or import these functions in your own script")
