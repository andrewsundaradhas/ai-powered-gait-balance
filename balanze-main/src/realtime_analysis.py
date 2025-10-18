"""
Real-time Gait and Balance Analysis

This module provides functionality for real-time analysis of gait and balance
using a webcam or video feed.
"""

import cv2
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from enum import Enum
import mediapipe as mp
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisMode(Enum):
    """Available analysis modes."""
    GAIT = "gait"
    BALANCE = "balance"
    TANDEM = "tandem"
    TUG = "tug"  # Timed Up and Go

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    frame: np.ndarray
    metrics: Dict[str, float]
    landmarks: Dict[str, Any]
    timestamp: float
    frame_number: int
    fps: float

class RealTimeAnalyzer:
    """Real-time gait and balance analysis using MediaPipe."""
    
    def __init__(self, 
                 mode: AnalysisMode = AnalysisMode.GAIT,
                 output_dir: str = 'data/realtime',
                 show_feed: bool = True,
                 save_output: bool = False,
                 model_complexity: int = 2):
        """
        Initialize the real-time analyzer.
        
        Args:
            mode: Analysis mode (gait, balance, etc.)
            output_dir: Directory to save output files
            show_feed: Whether to display the video feed
            save_output: Whether to save the analysis results
            model_complexity: MediaPipe model complexity (0-2)
        """
        self.mode = mode
        self.show_feed = show_feed
        self.save_output = save_output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Analysis state
        self.frame_count = 0
        self.start_time = time.time()
        self.results = []
        self.metrics_history = []
        
        # Create output files if saving
        if self.save_output:
            self._init_output_files()
    
    def _init_output_files(self):
        """Initialize output files for saving results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_video_path = self.output_dir / f"{self.mode.value}_analysis_{timestamp}.mp4"
        self.output_csv_path = self.output_dir / f"{self.mode.value}_metrics_{timestamp}.csv"
        self.landmarks_path = self.output_dir / f"{self.mode.value}_landmarks_{timestamp}.jsonl"
        
        # Will be initialized when we know the frame size
        self.video_writer = None
    
    def analyze_video(self, video_source: int = 0, max_frames: Optional[int] = None):
        """
        Analyze video from a source (webcam or file).
        
        Args:
            video_source: Path to video file or camera index
            max_frames: Maximum number of frames to process (None for no limit)
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {video_source}")
        
        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if saving output
        if self.save_output and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                self.fps if self.fps > 0 else 30.0,
                (self.frame_width, self.frame_height)
            )
        
        logger.info(f"Starting {self.mode.value} analysis...")
        
        try:
            frame_count = 0
            pbar = tqdm(desc="Processing frames", unit="frames")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Display frame if enabled
                if self.show_feed:
                    self._display_frame(result)
                
                # Save results
                if self.save_output:
                    self._save_frame_result(result)
                
                frame_count += 1
                pbar.update(1)
                
                # Check for exit key (press 'q' to quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.close()
            
        finally:
            # Release resources
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Save metrics to CSV
            if self.save_output and self.metrics_history:
                pd.DataFrame(self.metrics_history).to_csv(self.output_csv_path, index=False)
                logger.info(f"Saved metrics to: {self.output_csv_path}")
    
    def process_frame(self, frame: np.ndarray) -> AnalysisResult:
        """
        Process a single frame and return analysis results.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            AnalysisResult containing processed frame and metrics
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Calculate metrics based on detected landmarks
        metrics = self._calculate_metrics(pose_results, hand_results, face_results)
        
        # Draw landmarks and annotations on frame
        annotated_frame = self._annotate_frame(
            frame.copy(), pose_results, hand_results, face_results, metrics
        )
        
        # Update frame counter and timing
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add FPS to metrics
        metrics['fps'] = fps
        
        # Create result object
        result = AnalysisResult(
            frame=annotated_frame,
            metrics=metrics,
            landmarks=self._extract_landmarks(pose_results, hand_results, face_results),
            timestamp=current_time,
            frame_number=self.frame_count,
            fps=fps
        )
        
        return result
    
    def _calculate_metrics(self, pose_results, hand_results, face_results) -> Dict[str, float]:
        """Calculate analysis metrics from detection results."""
        metrics = {}
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate joint angles
            angles = self._calculate_joint_angles(landmarks)
            metrics.update(angles)
            
            # Calculate balance metrics
            if self.mode in [AnalysisMode.BALANCE, AnalysisMode.TANDEM]:
                balance_metrics = self._calculate_balance_metrics(landmarks)
                metrics.update(balance_metrics)
            
            # Calculate gait metrics
            if self.mode == AnalysisMode.GAIT:
                gait_metrics = self._calculate_gait_metrics(landmarks)
                metrics.update(gait_metrics)
            
            # Calculate TUG metrics
            if self.mode == AnalysisMode.TUG:
                tug_metrics = self._calculate_tug_metrics(landmarks)
                metrics.update(tug_metrics)
        
        return metrics
    
    def _calculate_joint_angles(self, landmarks) -> Dict[str, float]:
        """Calculate angles between key joints."""
        angles = {}
        
        # Define joint connections for angle calculation
        joint_triplets = [
            # Left arm
            (11, 13, 15, 'left_shoulder_angle'),  # Shoulder-Elbow-Wrist
            # Right arm
            (12, 14, 16, 'right_shoulder_angle'),  # Shoulder-Elbow-Wrist
            # Left leg
            (23, 25, 27, 'left_hip_angle'),  # Hip-Knee-Ankle
            # Right leg
            (24, 26, 28, 'right_hip_angle'),  # Hip-Knee-Ankle
            # Torso
            (11, 23, 24, 'torso_angle_left'),  # Left-shoulder-hip-hip
            (12, 24, 23, 'torso_angle_right'),  # Right-shoulder-hip-hip
        ]
        
        for j1, j2, j3, name in joint_triplets:
            if (j1 < len(landmarks) and j2 < len(landmarks) and j3 < len(landmarks)):
                angle = self._get_angle(
                    (landmarks[j1].x, landmarks[j1].y),
                    (landmarks[j2].x, landmarks[j2].y),
                    (landmarks[j3].x, landmarks[j3].y)
                )
                angles[name] = angle
        
        return angles
    
    def _calculate_balance_metrics(self, landmarks) -> Dict[str, float]:
        """Calculate balance-related metrics."""
        metrics = {}
        
        # Calculate center of mass (simplified as midpoint between hips)
        if len(landmarks) > 24:  # Check if we have enough landmarks
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            com_x = (left_hip.x + right_hip.x) / 2
            com_y = (left_hip.y + right_hip.y) / 2
            
            metrics['com_x'] = com_x
            metrics['com_y'] = com_y
            
            # Calculate sway (movement of center of mass)
            if hasattr(self, 'prev_com_x'):
                sway_x = abs(com_x - self.prev_com_x)
                sway_y = abs(com_y - self.prev_com_y)
                metrics['sway_x'] = sway_x
                metrics['sway_y'] = sway_y
                metrics['sway_magnitude'] = np.sqrt(sway_x**2 + sway_y**2)
            
            self.prev_com_x = com_x
            self.prev_com_y = com_y
        
        return metrics
    
    def _calculate_gait_metrics(self, landmarks) -> Dict[str, float]:
        """Calculate gait-related metrics."""
        metrics = {}
        
        if len(landmarks) > 28:  # Check if we have enough landmarks
            # Calculate step length (distance between ankles)
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            step_length = np.sqrt(
                (left_ankle.x - right_ankle.x)**2 + 
                (left_ankle.y - right_ankle.y)**2
            )
            metrics['step_length'] = step_length
            
            # Calculate cadence (steps per minute) - simplified
            current_time = time.time()
            if hasattr(self, 'last_step_time'):
                time_since_last_step = current_time - self.last_step_time
                if step_length > 0.1:  # Threshold for step detection
                    cadence = 60 / time_since_last_step  # Steps per minute
                    metrics['cadence'] = cadence
                    self.last_step_time = current_time
            else:
                self.last_step_time = current_time
        
        return metrics
    
    def _calculate_tug_metrics(self, landmarks) -> Dict[str, float]:
        """Calculate Timed Up and Go test metrics."""
        metrics = {}
        current_time = time.time()
        
        # Initialize TUG test state if needed
        if not hasattr(self, 'tug_start_time'):
            self.tug_start_time = current_time
            self.tug_phase = 'initial'
            self.tug_phases = {}
        
        # Calculate time elapsed since test start
        elapsed_time = current_time - self.tug_start_time
        metrics['tug_elapsed_time'] = elapsed_time
        
        # Detect TUG phases (simplified)
        if len(landmarks) > 24:  # Check if we have enough landmarks
            # Get vertical position of nose (simplified head position)
            nose = landmarks[0]
            
            # Phase detection logic (simplified)
            if self.tug_phase == 'initial' and nose.y < 0.7:  # Arbitrary threshold
                self.tug_phase = 'standing'
                self.tug_phases['stand_time'] = elapsed_time
                logger.info("TUG: Subject stood up")
            
            # Add more phase detection logic here...
        
        metrics['tug_phase'] = self.tug_phase
        
        return metrics
    
    def _extract_landmarks(self, pose_results, hand_results, face_results) -> Dict[str, Any]:
        """Extract and format landmarks from detection results."""
        landmarks = {}
        
        if pose_results.pose_landmarks:
            landmarks['pose'] = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z if hasattr(lm, 'z') else 0.0, 'visibility': lm.visibility}
                for lm in pose_results.pose_landmarks.landmark
            ]
        
        if hand_results.multi_hand_landmarks:
            landmarks['hands'] = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks['hands'].append([
                    {'x': lm.x, 'y': lm.y, 'z': lm.z if hasattr(lm, 'z') else 0.0}
                    for lm in hand_landmarks.landmark
                ])
        
        if face_results.multi_face_landmarks:
            landmarks['face'] = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z if hasattr(lm, 'z') else 0.0}
                for lm in face_results.multi_face_landmarks[0].landmark
            ]
        
        return landmarks
    
    def _annotate_frame(self, frame, pose_results, hand_results, face_results, metrics):
        """Draw landmarks and annotations on the frame."""
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Draw face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        
        # Draw metrics
        self._draw_metrics(frame, metrics)
        
        return frame
    
    def _draw_metrics(self, frame, metrics):
        """Draw metrics on the frame."""
        # Display FPS and frame count
        cv2.putText(frame, f"FPS: {metrics.get('fps', 0):.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display mode-specific metrics
        y_offset = 60
        for name, value in metrics.items():
            if name != 'fps':
                text = f"{name}: {value:.2f}" if isinstance(value, (int, float)) else f"{name}: {value}"
                cv2.putText(frame, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
    
    def _display_frame(self, result: AnalysisResult):
        """Display the annotated frame."""
        cv2.imshow(f"Gait Analysis - {self.mode.value.capitalize()}", result.frame)
    
    def _save_frame_result(self, result: AnalysisResult):
        """Save frame results to output files."""
        # Save video frame
        if self.video_writer is not None:
            self.video_writer.write(result.frame)
        
        # Save metrics
        self.metrics_history.append({
            'timestamp': result.timestamp,
            'frame_number': result.frame_number,
            'fps': result.fps,
            **result.metrics
        })
        
        # Save landmarks to JSONL file
        if hasattr(self, 'landmarks_file'):
            json.dump({
                'frame_number': result.frame_number,
                'timestamp': result.timestamp,
                'landmarks': result.landmarks
            }, self.landmarks_file)
            self.landmarks_file.write('\n')
    
    @staticmethod
    def _get_angle(a: Tuple[float, float], 
                  b: Tuple[float, float], 
                  c: Tuple[float, float]) -> float:
        """Calculate the angle between three points in degrees."""
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)


def main():
    """Run real-time analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Gait and Balance Analysis')
    parser.add_argument('--mode', type=str, default='gait',
                       choices=['gait', 'balance', 'tandem', 'tug'],
                       help='Analysis mode')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index or video file path)')
    parser.add_argument('--output-dir', type=str, default='data/realtime',
                       help='Directory to save output files')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--save', action='store_true',
                       help='Save analysis results')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Initialize and run the analyzer
    analyzer = RealTimeAnalyzer(
        mode=AnalysisMode(args.mode),
        output_dir=args.output_dir,
        show_feed=not args.no_display,
        save_output=args.save
    )
    
    analyzer.analyze_video(
        video_source=source,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
