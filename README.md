
# MLOps Sentiment Analysis Pipeline

This project implements an end-to-end MLOps pipeline for Sentiment Analysis using the IMDB dataset. It demonstrates workflow orchestration, experiment tracking, containerization, and model deployment.

## Architecture

*   **Orchestration**: [Mage AI](https://www.mage.ai/) manages the data pipeline (ETL).
*   **Experiment Tracking**: [MLflow](https://mlflow.org/) tracks model parameters, metrics, and artifacts.
*   **Containerization**: [Docker](https://www.docker.com/) & Docker Compose encapsulate the entire environment.
*   **Serving**: MLflow Models handles the deployment of the trained model as a REST API.

## Project Structure

*   `sentiment_analysis_pipeline/`: The Mage project containing data loaders, transformers, and exporters.
    *   `data_loaders/load_local_data.py`: Loads the IMDB dataset from a local mount (or Kaggle Hub fallback).
    *   `transformers/preprocess_text_data.py`: cleans and processes text data (TF-IDF preparation).
    *   `data_exporters/train_and_log_mlflow.py`: Trains a Keras model, logs it to MLflow, and creates a custom PyFunc model including the Vectorizer.
*   `docker-compose.yml`: Orchestrates the `mage` (pipeline) and `mlflow-serve` (inference) services.
*   `Dockerfile`: Defines the Python environment for both Mage and MLflow serving.
*   `requirement.txt`: Python dependencies.

## Prerequisites

*   Docker && Docker Compose
*   Git

## Setup & Inspection

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/2coolkalamkaar/mlops
    cd <repo-folder>
    ```

2.  **Start the MLOps Stack:**
    ```bash
    sudo docker compose up -d --build
    ```
    This will start two containers:
    *   `mage-sentiment-analysis`: The pipeline orchestrator (Port 6789)
    *   `mlflow-serve-sentiment`: The model serving API (Port 5000)

3.  **Access Mage UI:**
    Open `http://localhost:6789` in your browser.
    *   Navigate to Pipelines -> `sentiment_analysis_workflow`.
    *   Run the pipeline manually to train a new model.

## workflow Execution

To run the pipeline via CLI (useful for CI/CD):
```bash
sudo docker exec mage-sentiment-analysis mage run sentiment_analysis_pipeline sentiment_analysis_workflow
```

This process will:
1.  Load the IMDB dataset.
2.  Preprocess the text (clean, tokenize, stem).
3.  Train a Keras Neural Network.
4.  Log the model and TfidfVectorizer to the local `mlflow.db` and `mlruns/` directory.

## Model Deployment & Testing

The `mlflow-serve` service automatically serves the registered model `SentimentAnalysis-Mage` (Version 2).

**Test Health:**
```bash
curl -X GET http://localhost:5000/health
```

**Get Predictions:**
You can send raw text reviews to the endpoint. The custom model wrapper handles tokenization internally.

```bash
curl -X POST http://localhost:5000/invocations \
     -H "Content-Type: application/json" \
     -d '{"inputs": ["This movie was fantastic! I really enjoyed it.", "Terrible acting and boring plot."]}'
```

**Expected Output:**
```json
{"predictions": [[0.85], [0.02]]}
```
*   Values close to 1.0 are Positive.
*   Values close to 0.0 are Negative.

## Development Notes

*   **Data Persistence**: `mlflow.db` and `mlruns/` are mounted volumes. Deleting them resets the experiment history.
*   **Dependency Management**: Add new packages to `requirement.txt` and rebuild the containers.
