"""
Data preprocessing for Heart Disease Risk Assessment
Dataset: UCI Cleveland Heart Disease Dataset
Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
Target: 0 = No disease, 1 = Disease
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise-Induced Angina (1=Yes)',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of Peak Exercise ST Segment',
    'ca': 'Major Vessels Colored by Fluoroscopy (0-3)',
    'thal': 'Thalassemia (1=Normal, 2=Fixed, 3=Reversible)'
}


def load_data(filepath=None):
    """
    Load UCI Heart Disease dataset.
    If filepath is None, downloads from UCI repository URL or creates synthetic data.
    """
    if filepath is not None:
        df = pd.read_csv(filepath)
    else:
        # Generate representative synthetic data based on UCI dataset statistics
        np.random.seed(42)
        n = 303

        data = {
            'age': np.random.normal(54.4, 9.0, n).clip(29, 77).astype(int),
            'sex': np.random.binomial(1, 0.68, n),
            'cp': np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08]),
            'trestbps': np.random.normal(131.7, 17.6, n).clip(94, 200).astype(int),
            'chol': np.random.normal(246.7, 51.8, n).clip(126, 564).astype(int),
            'fbs': np.random.binomial(1, 0.15, n),
            'restecg': np.random.choice([0, 1, 2], n, p=[0.48, 0.48, 0.04]),
            'thalach': np.random.normal(149.6, 22.9, n).clip(71, 202).astype(int),
            'exang': np.random.binomial(1, 0.33, n),
            'oldpeak': np.abs(np.random.normal(1.04, 1.16, n)).clip(0, 6.2).round(1),
            'slope': np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47]),
            'ca': np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]),
            'thal': np.random.choice([1, 2, 3], n, p=[0.18, 0.40, 0.42]),
        }
        df = pd.DataFrame(data)
        risk_score = (
            (df['cp'] == 0).astype(int) * 2 +
            (df['thalach'] < 140).astype(int) * 1.5 +
            (df['exang'] == 1).astype(int) * 1.5 +
            df['oldpeak'] * 0.5 +
            (df['ca'] > 0).astype(int) * 2 +
            (df['thal'] == 3).astype(int) * 1.5 +
            (df['sex'] == 1).astype(int) * 0.5 +
            (df['age'] > 55).astype(int) * 0.5
        )
        prob = 1 / (1 + np.exp(-(risk_score - 4)))
        df['target'] = np.random.binomial(1, prob)
    return df
def preprocess(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    df = df.copy()
   
    if df['target'].max() > 1:
        df['target'] = (df['target'] > 0).astype(int)

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    return (
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        scaler, FEATURE_NAMES
    )