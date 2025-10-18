"""
Real-time Gait and Balance Analysis Demo

This script demonstrates how to use the RealTimeAnalyzer class to perform
real-time analysis of gait and balance using a webcam or video file.
"""

import argparse
import cv2
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.realtime_analysis import RealTimeAnalyzer, AnalysisMode

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Real-time Gait and Balance Analysis Demo')
    parser.add_argument('--mode', type=str, default='gait',
                       choices=['gait', 'balance', 'tandem', 'tug'],
                       help='Analysis mode')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index or video file path)')
    parser.add_argument('--output-dir', type=str, default='data/realtime_demo',
                       help='Directory to save output files')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--save', action='store_true',
                       help='Save analysis results')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum number of frames to process (default: 300)')
    
    args = parser.parse_args()
    
    print("""
    ===========================================
    Real-time Gait and Balance Analysis Demo
    ===========================================
    Controls:
    - Press 'q' to quit
    - Press 'p' to pause/resume
    - Press 's' to save current frame
    """)
    
    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
        source_type = f"camera {source}"
    except ValueError:
        source = args.source
        source_type = f"file: {source}"
    
    print(f"Starting {args.mode} analysis from {source_type}")
    if args.save:
        print(f"Results will be saved to: {args.output_dir}")
    
    # Initialize the analyzer
    analyzer = RealTimeAnalyzer(
        mode=AnalysisMode(args.mode),
        output_dir=args.output_dir,
        show_feed=not args.no_display,
        save_output=args.save
    )
    
    # Start the analysis
    try:
        analyzer.analyze_video(
            video_source=source,
            max_frames=args.max_frames
        )
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nDemo completed. Cleaning up...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
