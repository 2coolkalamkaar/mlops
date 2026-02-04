
import requests
import json
import pandas as pd

# The MLflow serve endpoint usually expects a JSON with "inputs" or "dataframe_split"
# Since we used autolog and it's a keras model, the input schema should match what the model expects.
# Our model expects shape (features,), but MLflow usually handles the TF-IDF vectorization inside the model IF we logged the whole pipeline.
# CAUTION: We only logged the Keras model, NOT the vectorizer. 
# The preprocessing info (vectorizer) is NOT inside the logged model artifact in the current implementation.
# The current model expects ALREADY VECTORIZED input. 
# This is a common mistake. Deployment will fail or give garbage if we send raw text.

# However, for testing, we can try to send a dummy vector of the correct shape.
# We logged "features" param: 28834 in the previous run.

# A better approach for real deployment:
# We should have logged a custom Python Model (MLflow pyfunc) that includes both the Vectorizer and Keras model.
# OR we should send raw text and have the serving layer handle it? 
# The current served model is raw Keras. It expects numbers.

# Let's try to verify what the input shape is.
# Assuming 28834 features.

features_count = 28834 # Estimated from logs, usually around 20k-30k for this dataset subset.
# Let's check the logs or use a small script to find exact shape if needed.

# But wait, sending a 28k vector via curl/python is painful manually.
# Let's verify if the user wants us to fix the deployment content (include vectorizer) or just test connectivity.
# I will try to verify connectivity first.

print("Checking connectivity...")
try:
    response = requests.get("http://localhost:5000/health")
    print(f"Health check status: {response.status_code}")
except Exception as e:
    print(f"Connection failed: {e}")

# If we want to really predict "This movie is great", we need the vectorizer.
# Since it was trained in the pipeline and not saved as a separate artifact we can easily load,
# we strictly can't reproduce the exact vector without the fitted vectorizer object.
# The vectorizer was instantiated inside the Mage block and lost after execution (unless pickle/saved).
# A key MLOps lesson: Always save your preprocessors!

print("Done.")
