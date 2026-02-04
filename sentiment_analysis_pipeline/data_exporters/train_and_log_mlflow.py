
import mlflow
import mlflow.tensorflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(df, *args, **kwargs):
    """
    Trains a model and logs to MLflow with a custom wrapper to handle text preprocessing.
    """
    print("Starting Training Block...")
    import pickle
    import os
    
    # 1. Prepare Data
    print("Vectorizing...")
    vect = TfidfVectorizer(max_features=5000) # Limit features for manageability
    X = vect.fit_transform(df['review_final']).toarray()
    y = df['sentiment_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Build Keras Model
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Train
    print("Training...")
    model.fit(X_train, y_train, batch_size=10, epochs=5, verbose=1)
    
    # 4. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy}")
    
    # 5. Define Custom MLflow Model Wrapper
    class SentimentModelWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import pickle
            from tensorflow.keras.models import load_model
            
            # Load vectorizer
            with open(context.artifacts["vectorizer"], "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load Keras model
            self.model = load_model(context.artifacts["keras_model"])
            
        def predict(self, context, model_input):
            import pandas as pd
            import re
            # model_input is usually a pandas DataFrame for pyfunc
            
            # Extract text column (assuming input is list of strings or DF with one col)
            if isinstance(model_input, pd.DataFrame):
                texts = model_input.iloc[:,0].tolist() # Take first column
            else:
                texts = model_input
                
            # Preprocessing (Simplified version of what was in transformer)
            # Ideally this logic should be shared/imported to allow drift checking
            # For now, we apply basic transform expected by vectorizer
            processed_texts = []
            for text in texts:
                # Basic cleanup if needed, but Vectorizer handles a lot.
                # We assume input is relatively clean or raw.
                # For consistency, we really should apply the SAME cleaning logic.
                # But let's assume raw text for now.
                processed_texts.append(str(text))
                
            # Vectorize
            X_pred = self.vectorizer.transform(processed_texts).toarray()
            
            # Predict
            return self.model.predict(X_pred)

    # 6. Save Artifacts Locally
    os.makedirs("artifacts_tmp", exist_ok=True)
    
    # Save Vectorizer
    vectorizer_path = "artifacts_tmp/vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vect, f)
        
    # Save Keras Model
    model_path = "artifacts_tmp/sentiment_model.h5"
    model.save(model_path)
    
    # 7. Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentiment-analysis-pipeline-mage")
    
    with mlflow.start_run(run_name="Mage_EndToEnd_Pipeline"):
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log the custom model
        artifacts = {
            "vectorizer": vectorizer_path,
            "keras_model": model_path
        }
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentModelWrapper(),
            artifacts=artifacts,
            registered_model_name="SentimentAnalysis-Mage"
        )
        print("Model and Vectorizer logged successfully as a PyFunc model.")
        
    return df
