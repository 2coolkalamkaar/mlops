
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc/g++ needed for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirement.txt .

# Install any needed packages specified in requirement.txt
# Using --no-cache-dir to keep image small and avoiding some cache issues
RUN pip install --no-cache-dir -r requirement.txt

# Copy the Mage project and other source code
COPY sentiment_analysis_pipeline/ /app/sentiment_analysis_pipeline/
COPY sentiment-analysis-mlflow/ /app/sentiment-analysis-mlflow/

# Expose the port the app runs on (Mage runs on 6789 by default)
EXPOSE 6789

# Define environment variables
# Tell Mage where the project is
ENV MAGE_CODE_PATH=/app/sentiment_analysis_pipeline
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Command to run the application
# We start the Mage server pointing to our project
CMD ["mage", "start", "sentiment_analysis_pipeline"]
