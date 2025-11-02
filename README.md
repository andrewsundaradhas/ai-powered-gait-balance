#  AI-Powered Gait and Balance Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive AI-powered system for analyzing human gait and balance patterns to assess mobility and detect potential disorders.

##  Features

- **Real-time Gait Analysis**: Capture and analyze walking patterns in real-time
- **Balance Assessment**: Evaluate postural stability and balance metrics
- **Disorder Detection**: AI models to detect potential mobility disorders
- **Interactive Visualizations**: Comprehensive plots and 3D visualizations
- **RESTful API**: Easy integration with other systems
- **Web Dashboard**: User-friendly interface for analysis and monitoring
- **Data Validation**: Robust data validation and versioning
- **Model Management**: Version control and management of ML models

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-gait-analysis.git
   cd ai-gait-analysis
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=sqlite:///./gait_analysis.db
   MODEL_DIR=models
   ```

##  Quick Start

### 1. Generate Synthetic Data (if needed)
```bash
python data_preparation/generate_synthetic_data.py
```

### 2. Train Models
```bash
python scripts/train_models.py
```

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```

### 4. Access the Web Interface
Open your browser and navigate to:
- API Documentation: http://localhost:8000/docs
- Web Dashboard: http://localhost:8501

##  Project Structure

```
ai-gait-analysis/
├── app/                    # FastAPI application
│   ├── main.py             # Main FastAPI application
│   ├── models.py           # Database models
│   ├── schemas.py          # Pydantic models
│   └── config.py           # Configuration settings
├── data/                   # Data storage
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration
├── reports/                # Generated reports and figures
├── scripts/                # Utility scripts
│   ├── train_models.py     # Model training script
│   └── data_processing.py  # Data preprocessing utilities
├── src/                    # Source code
│   ├── api/                # API routes and endpoints
│   ├── data/               # Data handling and validation
│   ├── features/           # Feature engineering
│   ├── models/             # Model definitions
│   └── visualization/      # Visualization utilities
├── tests/                  # Test files
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

##  Model Architecture

The system uses a combination of traditional machine learning models:

1. **Gait Classifier**: Random Forest model for classifying different gait patterns
2. **Balance Classifier**: Gradient Boosting model for assessing balance stability
3. **Disorder Detector**: Ensemble model for detecting potential mobility disorders

##  Data Flow

1. **Data Collection**: Capture motion data from sensors or video
2. **Preprocessing**: Clean and normalize the raw data
3. **Feature Extraction**: Extract relevant biomechanical features
4. **Model Inference**: Pass features through trained models
5. **Visualization**: Generate interactive plots and reports
6. **Storage**: Save results to database for future reference

##  API Endpoints

- `POST /api/analyze/gait`: Analyze gait from video or sensor data
- `GET /api/analysis/{analysis_id}`: Get analysis results
- `GET /api/models`: List available models
- `POST /api/models/upload`: Upload a new model (admin only)

##  Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8 .
```

##  Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Stop Containers
```bash
docker-compose down
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Key Features

- **Multi-Modal Analysis**
  -  Gait analysis (temporal-spatial parameters)
  -  Static and dynamic balance assessment
  -  Clinical test automation (TUG, BBS, 6MWT)
  -  Joint kinematics and kinetics

- **Advanced AI/ML Capabilities**
  -  Deep learning-powered pose estimation
  -  Automated feature extraction
  -  Time-series analysis of movement patterns
  -  Anomaly detection for movement disorders

- **Clinical Integration**
  -  Standardized clinical reports
  -  Web-based dashboard
  -  EHR/EMR integration ready
  -  HIPAA-compliant data handling

- **Visualization & Reporting**
  -  Interactive 3D motion visualization
  -  Time-series analytics dashboard
  -  Responsive web interface
  -  PDF/CSV export capabilities

##  Installation

### Prerequisites
- Python 3.9+
- CUDA 11.2+ (for GPU acceleration)
- FFmpeg (for video processing)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/balanze.git
   cd balanze
   ```

2. **Set up the environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   .\venv\Scripts\activate  # Windows

   # Install core dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

3. **Download pre-trained models**
   ```bash
   python scripts/download_models.py
   ```

##  Quick Start

### 1. Web Application (Recommended)

```bash
# Start the web server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000` in your browser.

### 2. Command Line Interface

#### Real-time Analysis
```bash
# Run with default webcam
python src/cli/analyze.py --source 0 --task gait

# Analyze a video file
python src/cli/analyze.py --source input/video.mp4 --task balance

# Batch process a directory of videos
python src/cli/batch_process.py --input-dir data/raw --output-dir results/
```

#### Available Analysis Tasks
- `gait`: Full gait cycle analysis
- `balance`: Static and dynamic balance assessment
- `tug`: Timed Up and Go test
- `sit2stand`: Sit-to-stand transfer analysis
- `custom`: Custom analysis pipeline

### 3. Python API

```python
from balanze import GaitAnalyzer, BalanceAssessor

# Initialize analyzers
gait_analyzer = GaitAnalyzer(model_path='models/gait_model.pth')
balance_assessor = BalanceAssessor()

# Analyze video
results = gait_analyzer.analyze('path/to/video.mp4')
balance_scores = balance_assessor.assess('path/to/balance_test.mp4')

# Export results
results.export('report.pdf')
```

## 🗂 Project Structure

```
balanze/
├── app/                      # Web application
│   ├── api/                  # API endpoints
│   ├── core/                 # Core application logic
│   ├── models/               # Database models
│   ├── static/               # Static files
│   └── templates/            # HTML templates
│
├── src/                      # Core Python package
│   ├── analysis/             # Analysis modules
│   │   ├── gait.py           # Gait analysis
│   │   ├── balance.py        # Balance assessment
│   │   └── clinical_tests.py # Clinical test analysis
│   │
│   ├── data/                 # Data handling
│   │   ├── preprocessing.py  # Data preprocessing
│   │   ├── augmentation.py   # Data augmentation
│   │   └── dataloader.py     # Data loading utilities
│   │
│   ├── models/               # Model architectures
│   │   ├── pose_estimation/  # Pose estimation models
│   │   ├── classifiers/      # Classification models
│   │   └── regressors/       # Regression models
│   │
│   ├── utils/                # Utility functions
│   └── visualization/        # Visualization tools
│
├── config/                   # Configuration files
│   ├── default.yaml          # Default configuration
│   └── production.yaml       # Production settings
│
├── data/                     # Data directory
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
│   └── results/              # Analysis results
│
├── models/                   # Trained models
│   ├── gait/                 # Gait analysis models
│   └── balance/              # Balance assessment models
│
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── docker/                   # Docker configurations
├── docs/                     # Documentation
├── .github/                  # GitHub configurations
├── .env.example              # Environment variables template
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                 # This file
```

##  Development

### Setting Up Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

2. **Run tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=src tests/
   
   # Run specific test
   pytest tests/test_gait_analysis.py -v
   ```

3. **Code Quality**
   ```bash
   # Run linter
   flake8 src/
   
   # Run type checking
   mypy src/
   
   # Format code
   black src/
   isort src/
   ```

##  Deployment

### Docker Deployment

```bash
# Build the Docker image
docker build -t balanze .

# Run the container
docker run -d -p 8000:8000 --name balanze_app balanze
```

### Kubernetes Deployment

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -n balanze
```

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


