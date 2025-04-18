import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ModelTrainer:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=10000)
    
    def train_logistic_regression(self, X_train, y_train):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        params = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2']
        }
        
        gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        
        joblib.dump(gs.best_estimator_, 'models/trained_models/lr_model.pkl')
        return gs.best_estimator_

def train_lstm(self, X_train, y_train):
        # Tokenization
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        
        # Model building function for KerasClassifier
        def create_model(optimizer='adam', units=64):
            model = Sequential([
                Embedding(10000, 128, input_length=100),
                LSTM(units),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=optimizer,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            return model
        
        # Wrap Keras model
        model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=64, verbose=0)
        
        # Hyperparameters
        params = {
            'optimizer': ['adam', 'rmsprop'],
            'units': [32, 64, 128]
        }
        
        gs = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=1, verbose=1)
        gs.fit(X_train_pad, y_train)
        
        # Save best model
        gs.best_estimator_.model.save('models/trained_models/lstm_model.h5')
        return gs.best_estimator_
