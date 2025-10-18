"""
Data Processing Pipeline for Gait and Balance Analysis

This module provides functionality to collect, preprocess, and prepare
gait and balance data for machine learning models.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import cv2
import mediapipe as mp
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GaitDataCollector:
    """Class for collecting gait and balance data from video or live feed."""
    
    def __init__(self, output_dir: str = 'data/raw', 
                 frame_rate: int = 30,
                 image_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            frame_rate: Frame rate for video capture
            image_size: Tuple of (width, height) for video capture
        """
        self.output_dir = Path(output_dir)
        self.frame_rate = frame_rate
        self.image_size = image_size
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create output directories
        self.raw_data_dir = self.output_dir / 'raw_data'
        self.processed_data_dir = self.output_dir / 'processed_data'
        self.video_dir = self.output_dir / 'videos'
        
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.video_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def collect_from_video(self, video_path: str, subject_id: str, 
                         test_type: str = 'gait', save_video: bool = False) -> pd.DataFrame:
        """
        Collect pose and movement data from a video file.
        
        Args:
            video_path: Path to the input video file
            subject_id: Unique identifier for the subject
            test_type: Type of test being performed ('gait', 'balance', etc.)
            save_video: Whether to save the annotated video
            
        Returns:
            DataFrame containing the collected data
        """
        logger.info(f"Processing video: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if saving output
        if save_video:
            output_path = self.video_dir / f"{subject_id}_{test_type}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Initialize data storage
        frames_data = []
        frame_number = 0
        
        # Process video frames
        with tqdm(total=frame_count, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                pose_results = self.pose.process(rgb_frame)
                face_results = self.face_mesh.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                # Extract and store pose data
                frame_data = self._extract_pose_data(pose_results, frame_number, fps)
                frame_data.update(self._extract_face_data(face_results, frame_number))
                frame_data.update(self._extract_hand_data(hand_results, frame_number))
                
                # Add metadata
                frame_data.update({
                    'frame_number': frame_number,
                    'timestamp': frame_number / fps,
                    'subject_id': subject_id,
                    'test_type': test_type,
                    'frame_width': width,
                    'frame_height': height
                })
                
                frames_data.append(frame_data)
                
                # Draw annotations if saving video
                if save_video:
                    annotated_frame = self._draw_landmarks(frame.copy(), pose_results, face_results, hand_results)
                    out.write(annotated_frame)
                
                frame_number += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if save_video:
            out.release()
            logger.info(f"Saved annotated video to: {output_path}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(frames_data)
        output_file = self.raw_data_dir / f"{subject_id}_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved collected data to: {output_file}")
        
        return df
    
    def _extract_pose_data(self, results, frame_number: int, fps: float) -> Dict[str, float]:
        """Extract pose landmarks and calculate features."""
        if not results.pose_landmarks:
            return {}
        
        landmarks = results.pose_landmarks.landmark
        data = {}
        
        # Extract individual landmark coordinates
        for i, landmark in enumerate(landmarks):
            data.update({
                f'pose_{i}_x': landmark.x,
                f'pose_{i}_y': landmark.y,
                f'pose_{i}_z': landmark.z if landmark.HasField('z') else 0.0,
                f'pose_{i}_visibility': landmark.visibility
            })
        
        # Calculate additional features
        if len(landmarks) > 25:  # Full body pose
            # Calculate angles between key joints
            data.update(self._calculate_joint_angles(landmarks))
            
            # Calculate distances between key points
            data.update(self._calculate_distances(landmarks))
            
            # Calculate velocities and accelerations (if previous frame data is available)
            if hasattr(self, 'prev_pose_data'):
                data.update(self._calculate_kinematics(data, self.prev_pose_data, 1/fps))
            
            self.prev_pose_data = data
        
        return data
    
    def _extract_face_data(self, results, frame_number: int) -> Dict[str, float]:
        """Extract face mesh landmarks and calculate features."""
        if not results.multi_face_landmarks:
            return {}
        
        data = {}
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract key facial landmarks
        for i, landmark in enumerate(face_landmarks):
            data.update({
                f'face_{i}_x': landmark.x,
                f'face_{i}_y': landmark.y,
                f'face_{i}_z': landmark.z if hasattr(landmark, 'z') else 0.0
            })
        
        return data
    
    def _extract_hand_data(self, results, frame_number: int) -> Dict[str, float]:
        """Extract hand landmarks and calculate features."""
        if not results.multi_hand_landmarks:
            return {}
        
        data = {}
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = 'left' if results.multi_handedness[hand_idx].classification[0].label == 'Left' else 'right'
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                data.update({
                    f'{hand_label}_hand_{i}_x': landmark.x,
                    f'{hand_label}_hand_{i}_y': landmark.y,
                    f'{hand_label}_hand_{i}_z': landmark.z if hasattr(landmark, 'z') else 0.0
                })
        
        return data
    
    def _calculate_joint_angles(self, landmarks) -> Dict[str, float]:
        """Calculate angles between key joints."""
        angles = {}
        
        # Define joint connections for angle calculation (simplified)
        joint_pairs = [
            # Left arm
            (11, 13, 15),  # Shoulder-Elbow-Wrist
            # Right arm
            (12, 14, 16),  # Shoulder-Elbow-Wrist
            # Left leg
            (23, 25, 27),  # Hip-Knee-Ankle
            # Right leg
            (24, 26, 28),  # Hip-Knee-Ankle
            # Torso
            (11, 23, 24),  # Left-hip-right-hip angle
            (12, 24, 23)   # Right-hip-left-hip angle
        ]
        
        for i, (j1, j2, j3) in enumerate(joint_pairs):
            if (j1 < len(landmarks) and j2 < len(landmarks) and j3 < len(landmarks)):
                angle = self._calculate_angle(
                    (landmarks[j1].x, landmarks[j1].y),
                    (landmarks[j2].x, landmarks[j2].y),
                    (landmarks[j3].x, landmarks[j3].y)
                )
                angles[f'angle_{j1}_{j2}_{j3}'] = angle
        
        return angles
    
    def _calculate_distances(self, landmarks) -> Dict[str, float]:
        """Calculate distances between key points."""
        distances = {}
        
        # Define point pairs for distance calculation
        point_pairs = [
            # Shoulder width
            (11, 12, 'shoulder_width'),
            # Hip width
            (23, 24, 'hip_width'),
            # Torso length (shoulder to hip)
            (11, 23, 'left_torso_length'),
            (12, 24, 'right_torso_length'),
            # Limb lengths
            (11, 13, 'left_upper_arm_length'),
            (13, 15, 'left_lower_arm_length'),
            (12, 14, 'right_upper_arm_length'),
            (14, 16, 'right_lower_arm_length'),
            (23, 25, 'left_thigh_length'),
            (25, 27, 'left_shin_length'),
            (24, 26, 'right_thigh_length'),
            (26, 28, 'right_shin_length')
        ]
        
        for i, j, name in point_pairs:
            if i < len(landmarks) and j < len(landmarks):
                dist = self._calculate_distance(
                    (landmarks[i].x, landmarks[i].y),
                    (landmarks[j].x, landmarks[j].y)
                )
                distances[name] = dist
        
        return distances
    
    def _calculate_kinematics(self, current_data: Dict[str, float], 
                            prev_data: Dict[str, float], 
                            dt: float) -> Dict[str, float]:
        """Calculate velocities and accelerations."""
        kinematics = {}
        
        # Calculate velocities and accelerations for key points
        for i in range(25):  # Assuming 25 keypoints (MediaPipe Pose)
            x_key = f'pose_{i}_x'
            y_key = f'pose_{i}_y'
            
            if x_key in current_data and x_key in prev_data:
                # Velocity
                vx = (current_data[x_key] - prev_data.get(x_key, 0)) / dt if dt > 0 else 0
                vy = (current_data[y_key] - prev_data.get(y_key, 0)) / dt if dt > 0 else 0
                
                kinematics[f'velocity_{i}_x'] = vx
                kinematics[f'velocity_{i}_y'] = vy
                kinematics[f'speed_{i}'] = np.sqrt(vx**2 + vy**2)
                
                # Acceleration (if previous velocity is available)
                if hasattr(self, 'prev_velocities'):
                    ax = (vx - self.prev_velocities.get(f'velocity_{i}_x', 0)) / dt if dt > 0 else 0
                    ay = (vy - self.prev_velocities.get(f'velocity_{i}_y', 0)) / dt if dt > 0 else 0
                    
                    kinematics[f'acceleration_{i}_x'] = ax
                    kinematics[f'acceleration_{i}_y'] = ay
                    kinematics[f'acceleration_magnitude_{i}'] = np.sqrt(ax**2 + ay**2)
        
        # Store current velocities for next frame
        self.prev_velocities = {
            k: v for k, v in kinematics.items() 
            if k.startswith('velocity_')
        }
        
        return kinematics
    
    @staticmethod
    def _calculate_angle(a: Tuple[float, float], 
                        b: Tuple[float, float], 
                        c: Tuple[float, float]) -> float:
        """Calculate the angle between three points."""
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    @staticmethod
    def _calculate_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def _draw_landmarks(self, image, pose_results, face_results, hand_results):
        """Draw detected landmarks on the image."""
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        
        return image


class GaitDataPreprocessor:
    """Class for preprocessing and feature engineering on gait data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the preprocessor with raw data.
        
        Args:
            data: Raw gait data as a pandas DataFrame
        """
        self.data = data.copy()
        self.features = None
        self.scaler = None
    
    def preprocess(self) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Returns:
            Preprocessed DataFrame with engineered features
        """
        logger.info("Starting data preprocessing...")
        
        # 1. Handle missing values
        self._handle_missing_values()
        
        # 2. Extract temporal features
        self._extract_temporal_features()
        
        # 3. Calculate spatial features
        self._calculate_spatial_features()
        
        # 4. Calculate kinematic features
        self._calculate_kinematic_features()
        
        # 5. Extract frequency domain features
        self._extract_frequency_features()
        
        # 6. Normalize features
        self._normalize_features()
        
        logger.info("Data preprocessing completed.")
        return self.data
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Forward fill for temporal data
        self.data = self.data.fillna(method='ffill')
        
        # Fill any remaining NaNs with column means
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
    
    def _extract_temporal_features(self):
        """Extract temporal features from timestamps."""
        if 'timestamp' in self.data.columns:
            # Time since start of recording
            self.data['time_elapsed'] = self.data['timestamp'] - self.data['timestamp'].min()
            
            # Frame rate variation
            self.data['frame_interval'] = self.data['timestamp'].diff().fillna(0)
            
            # Rolling statistics for temporal smoothing
            window_size = min(5, len(self.data) // 10)  # 10% of data or 5 frames, whichever is smaller
            if window_size > 1:
                for col in self._get_numeric_columns():
                    if 'velocity' in col or 'acceleration' in col:
                        self.data[f'{col}_smooth'] = self.data[col].rolling(window=window_size, center=True).mean()
    
    def _calculate_spatial_features(self):
        """Calculate spatial relationships between body parts."""
        # Calculate center of mass (simplified)
        if all(f'pose_{i}_x' in self.data.columns for i in range(24)):  # 24 keypoints
            x_cols = [f'pose_{i}_x' for i in range(24)]
            y_cols = [f'pose_{i}_y' for i in range(24)]
            
            self.data['com_x'] = self.data[x_cols].mean(axis=1)
            self.data['com_y'] = self.data[y_cols].mean(axis=1)
            
            # Calculate sway (movement of center of mass)
            self.data['com_sway'] = np.sqrt(
                self.data['com_x'].diff()**2 + 
                self.data['com_y'].diff()**2
            ).fillna(0)
    
    def _calculate_kinematic_features(self):
        """Calculate kinematic features like velocities and accelerations."""
        # These are already calculated in the data collection phase
        # This is a placeholder for additional kinematic features
        pass
    
    def _extract_frequency_features(self, signal_columns=None, sample_rate=30):
        """
        Extract frequency domain features using FFT.
        
        Args:
            signal_columns: List of columns to analyze. If None, use all numeric columns.
            sample_rate: Sampling rate in Hz
        """
        if signal_columns is None:
            signal_columns = self._get_numeric_columns()
        
        for col in signal_columns:
            if col in self.data.columns:
                signal = self.data[col].values
                
                # Apply FFT
                fft_vals = np.fft.fft(signal)
                fft_freq = np.fft.fftfreq(len(signal), 1/sample_rate)
                
                # Get magnitude spectrum (positive frequencies only)
                idx = np.where(fft_freq >= 0)
                freqs = fft_freq[idx]
                mag = np.abs(fft_vals[idx])
                
                # Calculate features
                if len(freqs) > 0:
                    # Dominant frequency
                    dom_freq = freqs[np.argmax(mag[1:]) + 1]  # Skip DC component
                    self.data[f'{col}_dom_freq'] = dom_freq
                    
                    # Power in different frequency bands (Hz)
                    bands = {
                        'delta': (0.5, 4),
                        'theta': (4, 8),
                        'alpha': (8, 13),
                        'beta': (13, 30),
                        'gamma': (30, 100)
                    }
                    
                    for band, (f_low, f_high) in bands.items():
                        band_mask = (freqs >= f_low) & (freqs <= f_high)
                        if np.any(band_mask):
                            power = np.sum(mag[band_mask] ** 2)
                            self.data[f'{col}_power_{band}'] = power
    
    def _normalize_features(self):
        """Normalize features to zero mean and unit variance."""
        numeric_cols = self._get_numeric_columns()
        
        # Skip columns that shouldn't be normalized
        skip_cols = ['frame_number', 'timestamp', 'subject_id', 'test_type', 
                    'frame_width', 'frame_height', 'time_elapsed']
        numeric_cols = [col for col in numeric_cols if col not in skip_cols]
        
        if numeric_cols:
            from sklearn.preprocessing import StandardScaler
            
            self.scaler = StandardScaler()
            self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save the preprocessed data to a file.
        
        Args:
            output_path: Path to save the processed data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            self.data.to_csv(output_path, index=False)
        elif output_path.suffix == '.parquet':
            self.data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved processed data to: {output_path}")


def process_gait_data(input_path: str, output_path: str) -> None:
    """
    Process gait data from input file and save to output file.
    
    Args:
        input_path: Path to input data file (CSV, JSON, or directory)
        output_path: Path to save processed data (CSV or Parquet)
    """
    logger.info(f"Processing data from: {input_path}")
    
    # Load data
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix == '.json':
            df = pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    elif input_path.is_dir():
        # Load and combine multiple files
        data_files = list(input_path.glob('*.csv')) + list(input_path.glob('*.json'))
        dfs = []
        for file in data_files:
            if file.suffix == '.csv':
                dfs.append(pd.read_csv(file))
            else:
                dfs.append(pd.read_json(file))
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Preprocess data
    preprocessor = GaitDataPreprocessor(df)
    processed_data = preprocessor.preprocess()
    
    # Save processed data
    preprocessor.save_processed_data(output_path)
    
    return processed_data


if __name__ == "__main__":
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process gait and balance data')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from video')
    collect_parser.add_argument('video_path', type=str, help='Path to input video file')
    collect_parser.add_argument('--subject_id', type=str, required=True, help='Subject ID')
    collect_parser.add_argument('--test_type', type=str, default='gait', 
                               choices=['gait', 'balance', 'tandem', 'tug'],
                               help='Type of test being performed')
    collect_parser.add_argument('--output_dir', type=str, default='data/raw',
                               help='Directory to save collected data')
    collect_parser.add_argument('--save_video', action='store_true',
                               help='Save annotated video')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process collected data')
    process_parser.add_argument('input_path', type=str, 
                               help='Path to input data file or directory')
    process_parser.add_argument('--output_path', type=str, default='data/processed/processed_data.parquet',
                               help='Path to save processed data')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'collect':
        # Initialize data collector
        collector = GaitDataCollector(output_dir=args.output_dir)
        
        # Collect data from video
        collector.collect_from_video(
            video_path=args.video_path,
            subject_id=args.subject_id,
            test_type=args.test_type,
            save_video=args.save_video
        )
    elif args.command == 'process':
        # Process data
        process_gait_data(args.input_path, args.output_path)
    else:
        parser.print_help()
