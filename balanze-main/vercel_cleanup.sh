#!/bin/bash

# Clean up any temporary files or caches
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name ".pytest_cache" -exec rm -r {} +

# Clean up any other temporary files
rm -rf .mypy_cache
rm -rf .pytest_cache
rm -rf .coverage
rm -f .coverage.*

echo "Cleanup completed successfully!"
