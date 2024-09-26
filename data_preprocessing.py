import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('heart.csv')
    
    # Replace '?' with NaN and convert to numeric
    data.replace('?', pd.NA, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with column mean
    data.fillna(data.mean(), inplace=True)

    # Encoding categorical features
    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex'])
    data['cp'] = le.fit_transform(data['cp'])
    data['thal'] = le.fit_transform(data['thal'])

    # Scaling features
    scaler = StandardScaler()
    features_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    return data
