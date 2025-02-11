import os
from werkzeug.utils import secure_filename
import pandas as pd

def process_file(file, upload_folder):
    """Save uploaded file and return the path."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath

def load_csv(filepath):
    """Load CSV into Pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
