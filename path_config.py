import os
from pathlib import Path

def get_dataset_path():
    """Get path to Reuters dataset relative to project directory"""
    project_dir = Path(__file__).parent
    dataset_dir = "reuters21578"
    dataset_path = project_dir / dataset_dir
    return str(dataset_path)