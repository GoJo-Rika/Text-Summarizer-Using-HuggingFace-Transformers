# Text Summarizer Using HuggingFace Transformers

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/google/pegasus-cnn_dailymail)

A complete end-to-end machine learning project for text summarization using the HuggingFace Pegasus model. This project demonstrates a production-ready ML pipeline with proper modular architecture, configuration management, , API deployment, and containerization.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training Pipeline](#model-training-pipeline)
- [Deployment](#deployment)
- [External Tools and Dependencies](#external-tools-and-dependencies)
- [Best Practices Implemented](#best-practices-implemented)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## <a id="overview"></a>🔍 Overview
<!-- ## 🔍 Overview -->

This project provides a complete solution for summarizing conversational text. It fine-tunes Google's Pegasus model on the SAMSum dataset to generate concise summaries from dialogue. The entire system follows MLOps best practices, featuring a modular architectural design, comprehensive logging, a repeatable training pipeline, configuration management, and a deployed API for easy inference.

The pipeline processes conversational data and generates concise summaries, making it ideal for chat summarization, meeting notes, and dialogue analysis applications. This repository is designed to be both a functional application and a learning guide for building robust, production-level ML systems.

## <a id="project-architecture"></a>🔍 🏗️ Project Architecture
<!-- ## 🏗️ Project Architecture -->

The project follows a modular, pipeline-based architecture that separates concerns, making it scalable, maintainable, and easy to debug.

*   **Why this architecture?** A modular design allows individual components (like data ingestion or model training) to be developed, tested, and updated independently without affecting the rest of the system. This is crucial for collaborative projects and long-term maintenance.

### Core Components
- **Data Ingestion**: Downloads and extracts the SAMSum dataset
- **Data Transformation**: Tokenizes and preprocesses text data for model training
- **Model Training**: Fine-tunes the Pegasus model on the prepared dataset
- **Model Evaluation**: Assesses model performance using ROUGE metrics
- **Prediction Pipeline**: Serves the trained model for real-time inference

### Design Patterns
- **Modular Architecture**: Each component can be developed, tested, and executed independently, allowing for iterative development and easier debugging
- **Pipeline Pattern**: Sequential processing stages with clear boundaries enable stage-by-stage development and validation
- **Configuration Management**: Centralized YAML-based configuration supports environment-specific deployments
- **Entity-Component Pattern**: Clear data structures and component interfaces promote code reusability and maintainability
- **Dependency Injection**: Configurable components provide flexibility for different execution environments

## <a id="features"></a>✨ Features
<!-- ## ✨ Features -->

- **Production-Ready Pipeline**: Complete ML workflow from data ingestion to deployment
- **Flexible Configuration**: YAML-based configuration for easy management of paths and parameter/hyperparameters tuning
- **Multi-Device Support**: Automatic detection and utilization of CPU, CUDA (NVIDIA GPU), and Apple Silicon (MPS)
- **RESTful API**: FastAPI-based web service for serving the summarization model
- **Comprehensive Logging**: Detailed logging throughout the pipeline for easy monitoring and debugging
- **Model Evaluation**: `ROUGE score` calculation for performance assessment
- **Dockerized Deployment**: Easy containerization and deployment using `Dockerfile`

## <a id="getting-started"></a>🚀 Getting Started
<!-- ## 🚀 Getting Started -->

Follow these steps to set up and run the project locally.

### Prerequisites
Before starting this project, ensure you have the following installed:
- Python 3.8+
- Git
- Hardware Requirements:
  - Minimum 8GB RAM (16GB recommended for training)
  - GPU support (optional but recommended for faster training)
  - 5GB free disk space for datasets and models

### Step 1: Clone the Repository
First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/GoJo-Rika/Text-Summarizer-Using-HuggingFace-Transformers.git
cd Text-Summarizer-Using-HuggingFace-Transformers
```

### Step 2: Set Up The Environment and Install Dependencies
We recommend using **uv**, a fast, next-generation Python package manager.

#### Recommended Approach (using `uv`)
1.  **Install `uv`** on your system if you haven't already.
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Create a virtual environment and install dependencies** with a single command:
    ```bash
    uv sync
    ```
    This command automatically creates a `.venv` folder and installs all required packages from `requirements.txt`.

    > **Note**: For a comprehensive guide on `uv`, check out this detailed tutorial: [uv-tutorial-guide](https://github.com/GoJo-Rika/uv-tutorial-guide).

#### Alternative Approach (using `venv` and `pip`)
If you prefer to use the standard `venv` and `pip`:
1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt # Using uv: uv add -r requirements.txt
    ```

### Step 3: Initialize Project Structure
Create the necessary directory structure and empty files for the project using the following command: 
```bash
python template.py # Using uv: uv run template.py
```

## <a id="project-structure"></a>📁 Project Structure
<!-- ## 📁 Project Structure -->

```
Text-Summarizer/
├── artifacts/                   # Stores outputs:  data, models, and metrics
├── config/
│   └── config.yaml              # Static configuration (paths, model names)
├── logs/                        # Application logs
├── research/                    # Jupyter notebooks for experimentation
├── src/
│   └── text_summarizer/
│       ├── components/          # Core ML components logic for each pipeline stage
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/              # Configuration manager logic
│       │   └── configuration.py
│       ├── entity/              # Custom data structures (dataclasses) and entities
│       │   └── __init__.py
│       ├── pipeline/            Orchestrates the ML workflow stages
│       │   ├── stage_1_data_ingestion_pipeline.py
│       │   ├── stage_2_data_transformation_pipeline.py
│       │   ├── stage_3_model_trainer_pipeline.py
│       │   ├── stage_4_model_evaluation_pipeline.py
│       │   └── prediction_pipeline.py
│       └── utils/               # Utility functions# Helper functions (e.g., reading YAML)
│           └── common.py
├── app.py                       # FastAPI web application for prediction
├── main.py                      # Main script to run the training pipeline
├── params.yaml                  # Tunable hyperparameters for training
├── requirements.txt             # Python dependencies
└── Dockerfile                   # Docker containerization configuration for deployment
```

## <a id="configuration"></a>⚙️ Configuration
<!-- ## ⚙️ Configuration -->

The project uses two separate YAML files for configuration, a common best practice.

### `config/config.yaml`
Holds static configuration like file paths, artifact directories, and pre-trained model names. These rarely change.
```yaml
data_ingestion:
  source_URL: "https://github.com/GoJo-Rika/datasets/raw/refs/heads/main/summarizer-data.zip"
  
model_trainer:
  model_ckpt: "google/pegasus-cnn_dailymail"
```

### `params.yaml`
Contains hyperparameters for model training (e.g., learning rate, batch size, epochs). This allows for easy tuning and experimentation without modifying the core application code.
```yaml
TrainingArguments:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

## <a id="usage"></a>🎯 Usage
<!-- ## 🎯 Usage -->

### Training the Model

To run the complete training pipeline from scratch, execute `main.py`:
```bash
python main.py # uv run main.py
```

This command executes all four pipeline stages sequentially:
1.  **Data Ingestion**: Downloads and extracts data (ingestion and extraction).
2.  **Data Transformation**: Prepares data for the model (preprocessing and tokenization).
3.  **Model Training**: Fine-tunes the model (training).
4.  **Model Evaluation**: Calculates performance metrics (evaluation and metric calculation).

**Note on Model Training:** The model training stage in `main.py` is commented out by default to prevent accidental, resource-intensive retraining. To run a full training session, uncomment the relevant lines in `main.py`.

### Running Individual Stages

You can also run individual components for testing or debugging:

```python
from src.text_summarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline

# Run only data ingestion
pipeline = DataIngestionTrainingPipeline()
pipeline.initiate_data_ingestion()
```

### Starting the API Server

To start the web service for inference and serve the trained model via a REST API, run the `app.py` file::

```bash
python app.py # uv run app.py
```

The server will start on `http://localhost:8080` with automatic API documentation available at `http://localhost:8080/docs`.

## <a id="api-endpoints"></a>🌐 API Endpoints
<!-- ## 🌐 API Endpoints -->

### `GET /`
Redirects to the interactive API documentation.

### `GET /train`
Triggers the complete training pipeline. Useful for retraining the model via an API call.
```bash
curl -X GET "http://localhost:8080/train"
```

### `POST /predict`
Generates a summary for the provided text.
```bash
curl -X POST "http://localhost:8080/predict?text=Your%20text%20to%20summarize%20here"
```
**Example:**
```bash
curl -X POST "http://localhost:8080/predict?text=Alice%3A%20Hey%2C%20I%20can't%20make%20it%20to%20the%20meeting%20this%20afternoon.%20Bob%3A%20No%20problem!%20I'll%20send%20you%20the%20notes."
```

## <a id="model-training-pipeline"></a>🔄 Model Training Pipeline
<!-- ## 🔄 Model Training Pipeline -->

The training pipeline consists of four distinct, reuseable stages:

### Stage 1: Data Ingestion
- **Description**
  - Downloads the SAMSum dataset from the configured URL
  - Extracts the ZIP file to the artifacts directory
  - Validates data integrity and structure
- **Code**: `src/text_summarizer/pipeline/stage_1_data_ingestion_pipeline.py`

### Stage 2: Data Transformation
- **Description**
  - Loads the raw dataset using HuggingFace datasets
  - Tokenizes dialogue and summary pairs using the Pegasus tokenizer
  - Applies appropriate truncation and padding strategies
  - Saves the processed dataset for training
- **Code**: `src/text_summarizer/pipeline/stage_2_data_transformation_pipeline.py`

### Stage 3: Model Training
- **Description**
  - Loads the pre-trained Pegasus model
  - Configures training arguments from params.yaml
  - Implements data collation for sequence-to-sequence tasks
  - Fine-tunes the model on the SAMSum dataset
  - Saves the trained model and tokenizer
- **Code**: `src/text_summarizer/pipeline/stage_3_model_trainer_pipeline.py`

**Training Environment**: The model training was performed on Google Colab's free tier using T4 GPU, achieving significant performance improvements over local CPU training. The complete training process took approximately 10 minutes per epoch, with the full pipeline validation taking around 40 minutes including model downloading and file transfers. The modular architecture proved particularly valuable during development, allowing individual pipeline stages to be tested locally before moving to GPU-accelerated training in the cloud environment.

### Stage 4: Model Evaluation
- **Description**
  - Evaluates the model using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
  - Generates performance reports and saves metrics to CSV
  - Provides quantitative assessment of summarization quality
- **Code**: `src/text_summarizer/pipeline/stage_4_model_evaluation_pipeline.py`

## <a id="deployment"></a>🐳 Deployment
<!-- ## 🐳 Deployment -->

### Docker Deployment

Build and run the application in a Docker container:

```bash
# Build the Docker image
docker build -t text-summarizer .

# Run the container
docker run -p 8080:8080 text-summarizer
```

### Production Considerations

For production deployment, consider:
- **Environment Variables**: Use environment variables for sensitive configurations
- **Model Versioning**: Implement model versioning and rollback capabilities
- **Monitoring**: Add application and model performance monitoring
- **Scaling**: Use container orchestration platforms like Kubernetes
- **Security**: Implement authentication and input validation

## <a id="external-tools-and-dependencies"></a>🛠️ External Tools and Dependencies
<!-- ## 🛠️ External Tools and Dependencies -->

### Core ML Libraries
- **Transformers**: HuggingFace library for transformer models
- **Datasets**: HuggingFace datasets library for data loading
- **Torch**: PyTorch for deep learning operations
- **Evaluate**: HuggingFace evaluate library for metrics calculation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NLTK**: Natural language processing utilities
- **py7zr**: Archive extraction support

### Web Framework
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications

### Configuration and Utilities
- **PyYAML**: YAML parsing and configuration management
- **python-box**: Enhanced dictionary access for configurations
- **ensure**: Type checking and validation decorators

### Evaluation Metrics
- **ROUGE Score**: Text summarization evaluation metrics
- **sacrebleu**: BLEU score calculation for text generation

### External Monitoring and Logging
- **Weights & Biases (wandb)**: Experiment tracking and artifact storage for model training metrics and logs
- **Cloud Integration**: Google Colab integration for GPU-accelerated training on free tier resources

## <a id="best-practices-implemented"></a>🏆 Best Practices Implemented
<!-- ## 🏆 Best Practices Implemented -->

### Code Organization
- **Modular Architecture**: Clear separation of concerns with dedicated components for pipelines, and utilities
- **Configuration Management**: Centralized YAML configuration files for easy parameter tuning
- **Logging Strategy**: Comprehensive logging throughout the pipeline for traceability and debugging
- **Error Handling**: Proper exception handling and error reporting

### ML Engineering
- **Pipeline Pattern**: Sequential processing stages with clear interfaces
- **Data Validation**: Input validation and data integrity checks
- **Model Versioning**: Organized model saving and loading procedures
- **Evaluation Framework**: Systematic model evaluation using standard metrics

### Development Practices
- **Type Hints**: Enforced type hints for improved code quality and readability
- **Documentation**: Comprehensive docstrings and comments
- **Dependency Management**: Clear `requirements.txt` specification file for reproducible environments
- **Environment Isolation**: Clear instructions for using virtual environments (`uv` or `venv`)

## <a id="troubleshooting"></a>🔧 Troubleshooting
<!-- ## 🔧 Troubleshooting -->

### Common Issues and Solutions

#### Memory Issues During Training
If you encounter out-of-memory errors:
- Reduce `per_device_train_batch_size` in params.yaml
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Consider using gradient checkpointing for memory optimization

#### Model Loading Errors
If the model fails to load:
- Verify internet connectivity for downloading pre-trained models
- Check HuggingFace model hub availability
- Ensure sufficient disk space for model files

#### API Connection Issues
If the API server fails to start:
- Check if port 8080 is available and is not already in use by another application
- Verify all dependencies are installed correctly
- Review application logs for specific error messages

#### Device Compatibility
For device-specific issues:
- **Apple Silicon**: Ensure MPS support is available in your PyTorch installation
- **CUDA**: Verify CUDA drivers and PyTorch GPU support
- **CPU**: The system falls back to CPU automatically if GPU is unavailable

## <a id="contributing"></a>🤝 Contributing
<!-- ## 🤝 Contributing -->

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Note**: This project is designed for educational and research purposes. For production use, consider additional security measures, monitoring, and scalability optimizations based on your specific requirements.

For questions or issues, please refer to the project's issue tracker or contact the maintainers.