import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import re
from typing import Tuple, List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mode':
            for col in numeric_cols:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0, inplace=True)
        
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < 3]
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_encoded = df.copy()
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        df_scaled = df.copy()
        
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
            else:
                df_scaled[columns] = self.scaler.transform(df_scaled[columns])
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
            else:
                df_scaled[columns] = self.scaler.transform(df_scaled[columns])
        
        return df_scaled
    
    def create_derived_features(self, df: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        df_features = df.copy()
        
        for feature_name, config in feature_config.items():
            if config['type'] == 'ratio':
                df_features[feature_name] = df_features[config['numerator']] / (df_features[config['denominator']] + 1e-10)
            elif config['type'] == 'difference':
                df_features[feature_name] = df_features[config['col1']] - df_features[config['col2']]
            elif config['type'] == 'product':
                df_features[feature_name] = df_features[config['col1']] * df_features[config['col2']]
            elif config['type'] == 'temporal':
                if 'date_col' in config:
                    df_features[config['date_col']] = pd.to_datetime(df_features[config['date_col']])
                    df_features[f"{config['date_col']}_year"] = df_features[config['date_col']].dt.year
                    df_features[f"{config['date_col']}_month"] = df_features[config['date_col']].dt.month
                    df_features[f"{config['date_col']}_dayofweek"] = df_features[config['date_col']].dt.dayofweek
                    df_features[f"{config['date_col']}_quarter"] = df_features[config['date_col']].dt.quarter
        
        return df_features


class TextPreprocessor:
    
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def preprocess_text_column(self, df: pd.DataFrame, text_column: str, 
                               remove_stops: bool = True) -> pd.DataFrame:
        df_processed = df.copy()
        
        df_processed[f'{text_column}_clean'] = df_processed[text_column].apply(self.clean_text)
        
        if remove_stops:
            df_processed[f'{text_column}_clean'] = df_processed[f'{text_column}_clean'].apply(self.remove_stopwords)
        
        df_processed[f'{text_column}_length'] = df_processed[f'{text_column}_clean'].apply(lambda x: len(x.split()))
        
        return df_processed
    
    def extract_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df_features = df.copy()
        
        df_features['char_count'] = df_features[text_column].apply(lambda x: len(str(x)))
        df_features['word_count'] = df_features[text_column].apply(lambda x: len(str(x).split()))
        df_features['avg_word_length'] = df_features.apply(
            lambda row: row['char_count'] / (row['word_count'] + 1e-10), axis=1
        )
        df_features['sentence_count'] = df_features[text_column].apply(
            lambda x: len(re.split(r'[.!?]+', str(x)))
        )
        
        return df_features


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, val_size: float = 0.1, 
               random_state: int = 42) -> Tuple:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' or len(y.unique()) < 20 else None
    )
    
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp if y_temp.dtype == 'object' or len(y_temp.unique()) < 20 else None
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_temp, X_test, y_temp, y_test

