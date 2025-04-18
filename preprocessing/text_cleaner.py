import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        with open('preprocessing/slang_emoji_mapping.json') as f:
            self.slang_map = json.load(f)
    
    def _replace_slang_emojis(self, text):
        # Replace slang words
        for slang, replacement in self.slang_map['slang'].items():
            text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
        
        # Replace emojis
        for emoji, meaning in self.slang_map['emojis'].items():
            text = text.replace(emoji, f' {meaning} ')
        return text
    
    def clean_text(self, text):
        # Step 1: Handle slang and emojis
        text = self._replace_slang_emojis(text)
        
        # Step 2: Standard cleaning
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
        text = re.sub(r'<.*?>+', '', text)  # HTML
        text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation (keep as space)
        text = re.sub(r'\d+', '', text)  # Numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Whitespace
        
        # Step 3: Lemmatization
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) 
                for word in words 
                if word not in self.stop_words]
        
        return ' '.join(words)
