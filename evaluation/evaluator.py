"""
Evaluation Module
Measure quality and performance metrics for generated images
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import time
from PIL import Image


class LensEvaluator:
    """
    Comprehensive evaluation for GenAI Lens
    Measures both technical quality and user-facing metrics
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.clip_model = None
        self.clip_processor = None
    
    def load_clip(self):
        """Load CLIP model for similarity measurement"""
        if self.clip_model is not None:
            return
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = self.clip_model.to(device)
            self.clip_model.eval()
            
            print(f"✓ CLIP loaded on {device}")
            
        except ImportError:
            print("⚠ CLIP not available. Install transformers for CLIP metrics")
            self.clip_model = None
    
    def compute_clip_similarity(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> float:
        """
        Compute CLIP similarity between two images
        
        Args:
            image1: First image (RGB)
            image2: Second image (RGB)
            
        Returns:
            Similarity score [0, 1]
        """
        if self.clip_model is None:
            self.load_clip()
        
        if self.clip_model is None:
            # Fallback to simple structural similarity
            return self._compute_structural_similarity(image1, image2)
        
        try:
            import torch
            
            # Convert to PIL
            pil1 = Image.fromarray(image1)
            pil2 = Image.fromarray(image2)
            
            # Process images
            inputs = self.clip_processor(
                images=[pil1, pil2],
                return_tensors="pt",
                padding=True
            )
            
            device = next(self.clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = torch.nn.functional.cosine_similarity(
                image_features[0:1],
                image_features[1:2]
            )
            
            return float(similarity.cpu().item())
            
        except Exception as e:
            print(f"⚠ CLIP similarity failed: {e}")
            return self._compute_structural_similarity(image1, image2)
    
    def _compute_structural_similarity(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> float:
        """
        Fallback structural similarity using SSIM
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Similarity score [0, 1]
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Resize to same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            # Compute SSIM
            score = ssim(gray1, gray2)
            
            # Convert to [0, 1] range
            score = (score + 1.0) / 2.0
            
            return float(score)
            
        except ImportError:
            # Ultimate fallback: simple pixel correlation
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            correlation = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]
            return float((correlation + 1.0) / 2.0)
    
    def compute_style_alignment(
        self,
        image: np.ndarray,
        style_prompt: str
    ) -> float:
        """
        Measure how well image aligns with style description
        
        Args:
            image: Generated image
            style_prompt: Style description
            
        Returns:
            Alignment score [0, 1]
        """
        if self.clip_model is None:
            self.load_clip()
        
        if self.clip_model is None:
            return 0.5  # Neutral score if CLIP unavailable
        
        try:
            import torch
            
            # Convert to PIL
            pil_image = Image.fromarray(image)
            
            # Process
            inputs = self.clip_processor(
                text=[style_prompt],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            )
            
            device = next(self.clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get features
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            
            # Compute similarity
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            return float(probs[0][0].cpu().item())
            
        except Exception as e:
            print(f"⚠ Style alignment computation failed: {e}")
            return 0.5
    
    def measure_inference_performance(
        self,
        inference_times: List[float]
    ) -> Dict[str, float]:
        """
        Compute performance statistics
        
        Args:
            inference_times: List of inference times in seconds
            
        Returns:
            Performance metrics
        """
        if not inference_times:
            return {}
        
        times_array = np.array(inference_times)
        
        return {
            'mean_time': float(np.mean(times_array)),
            'median_time': float(np.median(times_array)),
            'std_time': float(np.std(times_array)),
            'min_time': float(np.min(times_array)),
            'max_time': float(np.max(times_array)),
            'fps': 1.0 / np.mean(times_array) if np.mean(times_array) > 0 else 0,
            'num_samples': len(inference_times)
        }
    
    def evaluate_generation(
        self,
        original_image: np.ndarray,
        generated_image: np.ndarray,
        conditioning_image: np.ndarray,
        style_prompt: str,
        identity_metrics: Optional[Dict] = None,
        inference_time: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive evaluation of a single generation
        
        Args:
            original_image: Original input image
            generated_image: Generated output
            conditioning_image: Conditioning used
            style_prompt: Style description
            identity_metrics: Optional identity preservation metrics
            inference_time: Optional inference time
            
        Returns:
            Complete evaluation metrics
        """
        metrics = {}
        
        # 1. Image Quality Metrics
        metrics['clip_similarity'] = self.compute_clip_similarity(
            original_image, generated_image
        )
        
        metrics['style_alignment'] = self.compute_style_alignment(
            generated_image, style_prompt
        )
        
        # 2. Identity Preservation
        if identity_metrics:
            metrics.update(identity_metrics)
        
        # 3. Performance
        if inference_time:
            metrics['inference_time'] = inference_time
            metrics['fps'] = 1.0 / inference_time if inference_time > 0 else 0
        
        # 4. Image Statistics
        metrics['image_stats'] = self._compute_image_stats(generated_image)
        
        # 5. Overall Quality Score
        quality_components = []
        
        if 'clip_similarity' in metrics:
            quality_components.append(metrics['clip_similarity'])
        
        if 'style_alignment' in metrics:
            quality_components.append(metrics['style_alignment'])
        
        if identity_metrics and 'identity_similarity' in identity_metrics:
            quality_components.append(identity_metrics['identity_similarity'])
        
        if quality_components:
            metrics['overall_quality'] = np.mean(quality_components)
        
        return metrics
    
    def _compute_image_stats(self, image: np.ndarray) -> Dict:
        """
        Compute basic image statistics
        
        Args:
            image: Input image
            
        Returns:
            Statistics dictionary
        """
        return {
            'mean_brightness': float(np.mean(image)),
            'std_brightness': float(np.std(image)),
            'mean_rgb': [float(np.mean(image[:,:,i])) for i in range(3)],
            'size': image.shape[:2]
        }
    
    def create_comparison_grid(
        self,
        original: np.ndarray,
        conditioning: np.ndarray,
        generated: np.ndarray,
        metrics: Dict,
        title: str = "Generation Results"
    ) -> np.ndarray:
        """
        Create visual comparison grid
        
        Args:
            original: Original image
            conditioning: Conditioning image
            generated: Generated image
            metrics: Evaluation metrics
            title: Grid title
            
        Returns:
            Comparison grid image
        """
        # Resize all to same height
        h = 512
        
        def resize_keep_aspect(img, target_h):
            aspect = img.shape[1] / img.shape[0]
            target_w = int(target_h * aspect)
            return cv2.resize(img, (target_w, target_h))
        
        orig_resized = resize_keep_aspect(original, h)
        cond_resized = resize_keep_aspect(conditioning, h)
        gen_resized = resize_keep_aspect(generated, h)
        
        # Create grid
        grid = np.hstack([orig_resized, cond_resized, gen_resized])
        
        # Add labels
        labels = ["Original", "Conditioning", "Generated"]
        x_positions = [10, orig_resized.shape[1] + 10, 
                      orig_resized.shape[1] + cond_resized.shape[1] + 10]
        
        for label, x in zip(labels, x_positions):
            cv2.putText(grid, label, (x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add metrics at bottom
        y_pos = h - 100
        metrics_text = [
            f"CLIP Sim: {metrics.get('clip_similarity', 0):.3f}",
            f"Identity: {metrics.get('identity_similarity', 0):.3f}",
            f"Time: {metrics.get('inference_time', 0):.2f}s"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(grid, text, (10, y_pos + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return grid
    
    def generate_evaluation_report(
        self,
        evaluations: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate text report from evaluations
        
        Args:
            evaluations: List of evaluation results
            save_path: Optional path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("SNAP GENAI LENS - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Aggregate metrics
        clip_scores = [e.get('clip_similarity', 0) for e in evaluations]
        identity_scores = [e.get('identity_similarity', 0) for e in evaluations]
        times = [e.get('inference_time', 0) for e in evaluations if e.get('inference_time')]
        
        report.append(f"Total Evaluations: {len(evaluations)}")
        report.append("")
        
        # Quality Metrics
        report.append("QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"CLIP Similarity:")
        report.append(f"  Mean:   {np.mean(clip_scores):.3f}")
        report.append(f"  Median: {np.median(clip_scores):.3f}")
        report.append(f"  Std:    {np.std(clip_scores):.3f}")
        report.append("")
        
        report.append(f"Identity Preservation:")
        report.append(f"  Mean:   {np.mean(identity_scores):.3f}")
        report.append(f"  Median: {np.median(identity_scores):.3f}")
        report.append(f"  Std:    {np.std(identity_scores):.3f}")
        report.append("")
        
        # Performance Metrics
        if times:
            perf = self.measure_inference_performance(times)
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"Mean Inference Time: {perf['mean_time']:.3f}s")
            report.append(f"Median Time:         {perf['median_time']:.3f}s")
            report.append(f"FPS:                 {perf['fps']:.2f}")
            report.append(f"Min Time:            {perf['min_time']:.3f}s")
            report.append(f"Max Time:            {perf['max_time']:.3f}s")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        avg_clip = np.mean(clip_scores)
        avg_identity = np.mean(identity_scores)
        
        if avg_clip < 0.6:
            report.append("⚠ Low CLIP similarity - consider adjusting prompts")
        if avg_identity < 0.6:
            report.append("⚠ Low identity preservation - strengthen identity guidance")
        if times and np.mean(times) > 2.0:
            report.append("⚠ High inference time - consider optimization")
        
        if avg_clip >= 0.7 and avg_identity >= 0.7:
            report.append("✓ Excellent quality metrics!")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Report saved to {save_path}")
        
        return report_text
