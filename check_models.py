
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Get the latest version of the model
model_name = "SentimentAnalysis-Mage"
latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])

for version in latest_versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}, Source: {version.source}")
