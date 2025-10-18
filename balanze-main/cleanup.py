"""
Cleanup script to remove unnecessary files before pushing to GitHub.
Run this script before committing to clean up the repository.
"""
import os
import shutil
from pathlib import Path

def remove_dirs(dirs_to_remove):
    """Remove directories if they exist."""
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)

def remove_files(files_to_remove):
    """Remove files if they exist."""
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)

def main():
    # Directories to remove
    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        "venv",
        ".ipynb_checkpoints"
    ]
    
    # Files to remove
    files_to_remove = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "*.log"
    ]
    
    # Remove directories
    remove_dirs(dirs_to_remove)
    
    # Remove files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.endswith(ext) for ext in ['.pyc', '.pyo', '.pyd', '.log']):
                file_path = os.path.join(root, file)
                remove_files([file_path])
    
    print("\nCleanup complete! The repository is now ready for commit.")

if __name__ == "__main__":
    main()
