
import requests
import json
import pandas as pd

# Now we test the REAL deployment with raw text interaction.
# The endpoint expects a JSON with "inputs" which is a list.

data = {
    "inputs": [
        "This movie was absolutely amazing! I loved every moment of it.",
        "Worst film I have ever seen. Total waste of time.",
        "It was okay, not great but not terrible."
    ]
}

print("Suspending belief and asking the oracle...")
try:
    response = requests.post(
        "http://localhost:5000/invocations",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    if response.status_code == 200:
        print("\nPredictions received:")
        predictions = response.json()['predictions']
        for text, pred in zip(data['inputs'], predictions):
            print(f"review: '{text}' -> Sentiment Score: {pred[0]:.4f} ({'Positive' if pred[0] > 0.5 else 'Negative'})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Request failed: {e}")
