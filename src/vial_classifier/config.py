"""Central configuration for the vial classifier pipeline.
 
All paths are relative to the project root, so run every entry point from there.
"""
 
import os
from pathlib import Path
 
PACKAGE_DIR = Path(__file__).resolve().parent
 
# --- Training ---------------------------------------------------------------
# VIAL_NUM_EPOCHS lets CI run a short smoke training without code changes.
NUM_EPOCHS = int(os.environ.get("VIAL_NUM_EPOCHS", "5"))
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42
 
# --- Images -----------------------------------------------------------------
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "good", 1: "defective"}
 
# --- Paths (relative to the project root) -----------------------------------
ARTIFACT_DIR = "artifacts"
MLFLOW_STATE_DIR = "mlflow-state"
IMAGE_DATA_DIR = "./data/images/raw"
METADATA_PATH = "./data/images/metadata.csv"
TEST_DATA_DIR = "./data/test"
TEST_METADATA_PATH = "./data/test/metadata.csv"
PYTORCH_MODEL_PATH = ARTIFACT_DIR + "/vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR + "/model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR + "/drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR + "/class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR + "/image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR + "/inference_log.json"
 
# --- Object store (public Hetzner bucket holding the images) ----------------
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"
 
# --- MLflow: local sqlite db, so no tracking server has to be running -------
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
ARTIFACT_PATH = "model_pyfunc"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = "./mlflow-state/artifacts"
 
# --- Export and serving -----------------------------------------------------
EXPORT_DIR = "models/model"
SERVE_PORT = 5001
UPSTREAM_PORT = 5002
UPSTREAM_HOST = "127.0.0.1"
PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"
 
# --- Drift detection --------------------------------------------------------
DRIFT_DEFAULT_DAYS = 7
DRIFT_KS_PVALUE_THRESHOLD = 0.05
DRIFT_MEAN_THRESHOLD = 0.1