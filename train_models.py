from utils.helpers import DataLoader
from models.model_training import ModelTrainer

# Load and prepare data
loader = DataLoader('data/Sarcasm_Headlines_Dataset.json')
X_train, X_test, y_train, y_test = loader.prepare_data()

# Train models
trainer = ModelTrainer()
print("Training Logistic Regression...")
lr_model = trainer.train_logistic_regression(X_train, y_train)
print("\nTraining LSTM...")
lstm_model = trainer.train_lstm(X_train, y_train)
