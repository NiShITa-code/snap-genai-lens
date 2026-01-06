"""
Identity Preservation Module
Maintains face identity during stylization using face embeddings
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import torch


class IdentityPreserver:
    """
    Preserve face identity during generation
    Uses face embeddings to measure identity consistency
    """
    
    def __init__(self, model_name: str = 'arcface'):
        """
        Initialize identity preserver
        
        Args:
            model_name: Face recognition model ('arcface' or 'insightface')
        """
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load face recognition model"""
        try:
            # Try to import insightface
            import insightface
            from insightface.app import FaceAnalysis
            
            # Initialize face analysis
            self.model = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            
            print(f"✓ Loaded InsightFace model on {self.device}")
            
        except ImportError:
            print("⚠ InsightFace not available. Using fallback identity measurement")
            self.model = None
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding
        
        Args:
            image: RGB image with face
            
        Returns:
            Face embedding vector or None
        """
        if self.model is None:
            # Fallback: use simple histogram-based "embedding"
            return self._extract_simple_embedding(image)
        
        try:
            # Detect and get embedding
            faces = self.model.get(image)
            
            if not faces:
                return None
            
            # Get first face embedding
            embedding = faces[0].embedding
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"⚠ Embedding extraction failed: {e}")
            return self._extract_simple_embedding(image)
    
    def _extract_simple_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Simple fallback embedding using color/texture features
        
        Args:
            image: RGB image
            
        Returns:
            Simple feature vector
        """
        # Resize to standard size
        face_img = cv2.resize(image, (112, 112))
        
        # Extract features
        features = []
        
        # Color histogram
        for i in range(3):
            hist = cv2.calcHist([face_img], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            features.extend(hist)
        
        # Texture features (LBP-like)
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        gray_flat = gray.flatten()
        features.extend(gray_flat[::10])  # Subsample
        
        embedding = np.array(features)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
        
        return embedding
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score [0, 1]
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Convert to [0, 1] range
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def extract_face_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Extract and crop face region with padding
        
        Args:
            image: Full image
            bbox: (x, y, width, height)
            padding: Padding ratio around face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop
    
    def measure_identity_preservation(
        self,
        original_image: np.ndarray,
        generated_image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, float]:
        """
        Measure how well identity is preserved
        
        Args:
            original_image: Original input image
            generated_image: Generated output image
            face_bbox: Optional face bounding box
            
        Returns:
            Dictionary with identity metrics
        """
        # Extract face regions if bbox provided
        if face_bbox is not None:
            orig_face = self.extract_face_region(original_image, face_bbox)
            gen_face = self.extract_face_region(generated_image, face_bbox)
        else:
            orig_face = original_image
            gen_face = generated_image
        
        # Get embeddings
        orig_embedding = self.extract_embedding(orig_face)
        gen_embedding = self.extract_embedding(gen_face)
        
        # Compute similarity
        identity_score = self.compute_similarity(orig_embedding, gen_embedding)
        
        # Additional metrics
        metrics = {
            'identity_similarity': identity_score,
            'identity_preserved': identity_score > 0.6,  # Threshold
            'confidence': 'high' if identity_score > 0.7 else 'medium' if identity_score > 0.5 else 'low'
        }
        
        return metrics
    
    def get_identity_conditioning(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Dict:
        """
        Get identity information for conditioning
        
        Args:
            image: Input face image
            target_size: Target image size
            
        Returns:
            Dictionary with identity conditioning info
        """
        # Extract embedding
        embedding = self.extract_embedding(image)
        
        if embedding is None:
            return {
                'success': False,
                'embedding': None
            }
        
        return {
            'success': True,
            'embedding': embedding,
            'embedding_dim': len(embedding)
        }
    
    def apply_identity_guidance(
        self,
        generated_image: np.ndarray,
        target_embedding: np.ndarray,
        strength: float = 0.8
    ) -> np.ndarray:
        """
        Apply post-processing identity guidance
        (Placeholder for advanced techniques)
        
        Args:
            generated_image: Generated image
            target_embedding: Target identity embedding
            strength: Guidance strength [0, 1]
            
        Returns:
            Adjusted image
        """
        # This is a placeholder
        # In production, this would use techniques like:
        # - IP-Adapter
        # - InstantID
        # - Face swapping with blending
        
        # For now, just return the image
        return generated_image
    
    def visualize_identity_comparison(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        metrics: Dict[str, float]
    ) -> np.ndarray:
        """
        Create visualization comparing original and generated
        
        Args:
            original: Original image
            generated: Generated image
            metrics: Identity metrics
            
        Returns:
            Comparison visualization
        """
        # Resize to same size
        h, w = original.shape[:2]
        generated_resized = cv2.resize(generated, (w, h))
        
        # Create side-by-side comparison
        comparison = np.hstack([original, generated_resized])
        
        # Add metrics text
        identity_score = metrics.get('identity_similarity', 0.0)
        confidence = metrics.get('confidence', 'unknown')
        
        text = f"Identity: {identity_score:.2f} ({confidence})"
        cv2.putText(comparison, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add labels
        cv2.putText(comparison, "Original", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Generated", (w + 10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return comparison
