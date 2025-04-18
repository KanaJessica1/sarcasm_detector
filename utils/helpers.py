import json
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        with open(self.filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    
    def prepare_data(self, test_size=0.2):
        df = self.load_data()
        X = df['headline']
        y = df['is_sarcastic']
        return train_test_split(X, y, test_size=test_size, random_state=42)
