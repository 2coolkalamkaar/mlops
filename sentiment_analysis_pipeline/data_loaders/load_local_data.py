
import io
import pandas as pd
import requests
import os
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    csv_path = '/app/data/IMDB Dataset.csv'
    
    print(f"Checking for file at: {csv_path}")
    print(f"Directory listing of /app/data: {os.listdir('/app/data')}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    print(f"Loading data from mounted volume: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Use subset for demo speed
    df_subset = df[:5000].copy()
    
    print(f"Loaded dataset with {len(df_subset)} rows")
    return df_subset

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) > 0, 'The output has no rows'
    assert 'review' in output.columns, 'Missing review column'
    assert 'sentiment' in output.columns, 'Missing sentiment column'
