# prefect/main_flow.py
from prefect import flow, task
import os
import sys
import subprocess
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrefectFlow")

@task(name="Ingest Data")
def ingest_data():
    logger.info("📡 Ingesting Data...")
    if not os.path.exists("ml-32m-split/movies.csv"):
        raise Exception("Source data missing!")
    return "Data Ready"

@task(name="Process Data")
def process_data():
    logger.info("🧹 Processing & Enriching Data...")
    # Simulate processing or call actual enhancement scripts if they exist as modules
    # For now, we assume the data is static locally, but in real world we'd run clean steps.
    return "Data Processed"

@task(name="Train Classifier")
def train_classifier():
    logger.info("🤖 Training Classifier...")
    # Call the training script
    result = subprocess.run(["python", "models/python_movie_classifier.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Classifier Training Failed: {result.stderr}")
        raise Exception("Classifier Training Failed")
    logger.info("Classifier Trained Successfully")

@task(name="Train Regressor")
def train_regressor():
    logger.info("📉 Training Regressor...")
    result = subprocess.run(["python", "models/rating_regressor.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Regressor Training Failed: {result.stderr}")
        raise Exception("Regressor Training Failed")
    logger.info("Regressor Trained Successfully")

@task(name="Train Association Rules")
def train_associations():
    logger.info("🔗 Mining Association Rules...")
    result = subprocess.run(["python", "models/association_rules.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Association Mining Failed: {result.stderr}")
        raise Exception("Association Mining Failed")
    logger.info("Association Rules Generated")

@task(name="Run Validation")
def validate_models():
    logger.info("🧪 Running DeepChecks Validation...")
    result = subprocess.run(["python", "tests/model_validation.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Validation Failed: {result.stderr}")
        # We might not want to break the flow depending on severity, 
        # but strictly speaking validation failure should stop deployment.
        raise Exception("Model Validation Failed")
    logger.info("Validation Passed")

@flow(name="Movie ML Pipeline")
def main_flow():
    logger.info("🚀 Starting MLOps Pipeline")
    
    # 1. Data Prep
    ingest_data()
    process_data()
    
    # 2. Training (Parallel)
    # Prefect runs tasks sequentially by default unless using task runners, 
    # but for simplicity we list them here.
    train_classifier()
    train_regressor()
    train_associations()
    
    # 3. Validation
    validate_models()
    
    logger.info("✅ Pipeline Complete")

if __name__ == "__main__":
    main_flow()
