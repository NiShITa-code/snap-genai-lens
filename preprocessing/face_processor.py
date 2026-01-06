"""
Face Preprocessing Module
Handles face detection, landmark extraction, and mask generation
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict
from PIL import Image


class FacePreprocessor:
    """Face detection and preprocessing for GenAI Lens"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize face preprocessor
        
        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence_threshold
        )
        
        # Initialize MediaPipe Face Detection for bounding box
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=confidence_threshold
        )
    
    def validate_input(self, image: np.ndarray) -> Dict[str, any]:
        """
        Validate input image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check image size
        h, w = image.shape[:2]
        if h < 256 or w < 256:
            validation['errors'].append(f"Image too small: {w}x{h}. Minimum 256x256")
            validation['valid'] = False
        
        if h > 2048 or w > 2048:
            validation['warnings'].append(f"Image large: {w}x{h}. Will be resized")
        
        # Check brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 30:
            validation['warnings'].append("Image very dark - may affect quality")
        elif mean_brightness > 225:
            validation['warnings'].append("Image very bright - may affect quality")
        
        return validation
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face and extract bounding box
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary with detection info or None
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # Detect face
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        # Get first detection
        detection = results.detections[0]
        confidence = detection.score[0]
        
        if confidence < self.confidence_threshold:
            return None
        
        # Get bounding box
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Ensure bbox is within image bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return {
            'bbox': (x, y, width, height),
            'confidence': float(confidence),
            'center': (x + width // 2, y + height // 2)
        }
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 468 facial landmarks
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Landmarks array (468, 2) or None
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = image.shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return np.array(landmarks)
    
    def create_face_mask(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Create binary face mask from landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks (468, 2)
            
        Returns:
            Binary mask (H, W)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use face contour landmarks (indices from MediaPipe)
        # Face oval indices
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Get face contour points
        contour_points = landmarks[face_oval_indices]
        
        # Fill polygon
        cv2.fillPoly(mask, [contour_points], 255)
        
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        return mask
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (RGB or BGR)
            
        Returns:
            Dictionary with all preprocessing results
        """
        # Validate input
        validation = self.validate_input(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect face
        face_info = self.detect_face(image_rgb)
        
        if face_info is None:
            return {
                'success': False,
                'error': 'No face detected',
                'validation': validation
            }
        
        # Extract landmarks
        landmarks = self.extract_landmarks(image_rgb)
        
        if landmarks is None:
            return {
                'success': False,
                'error': 'Failed to extract landmarks',
                'validation': validation,
                'face_detected': True,
                'face_info': face_info
            }
        
        # Create face mask
        face_mask = self.create_face_mask(image_rgb, landmarks)
        
        return {
            'success': True,
            'validation': validation,
            'face_info': face_info,
            'landmarks': landmarks,
            'face_mask': face_mask,
            'image_shape': image_rgb.shape
        }
    
    def visualize_preprocessing(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Visualize preprocessing results
        
        Args:
            image: Original image
            results: Preprocessing results
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        if not results['success']:
            # Add error text
            cv2.putText(vis_image, results['error'], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return vis_image
        
        # Draw bounding box
        bbox = results['face_info']['bbox']
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw landmarks
        landmarks = results['landmarks']
        for lm in landmarks:
            cv2.circle(vis_image, tuple(lm), 1, (255, 0, 0), -1)
        
        # Add confidence text
        conf = results['face_info']['confidence']
        cv2.putText(vis_image, f"Conf: {conf:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
