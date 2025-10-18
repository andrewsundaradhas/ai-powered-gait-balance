"""
Test script for the Gait and Balance Analysis data processing pipeline.

This script demonstrates how to use the data collection and processing
components of the pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import GaitDataCollector, process_gait_data

def test_data_collection(video_path: str, output_dir: str):
    """Test the data collection from a video file."""
    print(f"Testing data collection from: {video_path}")
    
    # Initialize data collector
    collector = GaitDataCollector(output_dir=output_dir)
    
    # Collect data from video
    df = collector.collect_from_video(
        video_path=video_path,
        subject_id="test_subject",
        test_type="gait",
        save_video=True
    )
    
    print(f"Collected {len(df)} frames of data")
    print("Columns in collected data:", df.columns.tolist())
    
    return df

def test_data_processing(input_path: str, output_path: str):
    """Test the data processing pipeline."""
    print(f"Testing data processing for: {input_path}")
    
    # Process the data
    processed_df = process_gait_data(input_path, output_path)
    
    print(f"Processed data shape: {processed_df.shape}")
    print("Columns in processed data:", processed_df.columns.tolist()[:20], "...")
    
    return processed_df

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Test the data processing pipeline')
    parser.add_argument('--video_path', type=str, 
                       help='Path to input video file for testing')
    parser.add_argument('--data_path', type=str,
                       help='Path to pre-collected data for processing test')
    parser.add_argument('--output_dir', type=str, default='data/test_output',
                       help='Directory to save test outputs')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests based on provided arguments
    if args.video_path:
        print("=== Testing Data Collection ===")
        test_data_collection(args.video_path, str(output_dir / "collected_data"))
    
    if args.data_path or (args.video_path and not args.data_path):
        input_path = args.data_path or str(output_dir / "collected_data")
        output_path = str(output_dir / "processed_data.parquet")
        
        print("\n=== Testing Data Processing ===")
        test_data_processing(input_path, output_path)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
