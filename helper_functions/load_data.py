import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_rent(budget):
    # Load data into dataframe
    streeteasy = pd.read_csv('data/rent.csv')
    df = pd.DataFrame(streeteasy)

    # Features
    df['within_budget'] = df.rent.apply(lambda x: 0 if x >= budget else 1)
    labels = df['within_budget']

    # Labels
    df.bedrooms = df.bedrooms.apply(lambda x: x if x.is_integer() else x//1+1).apply(lambda x: 1 if x==0 else x)
    features = df[['size_sqft', 'min_to_subway', 'bedrooms', 'has_gym', 'has_patio']]
    feature_names_list = [feature for feature in features.columns]

    # Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)

    # Normalise the feature data (mean = 0, std = 1 using Z-score method), only fit StandardScalar to train data.
    normalise = StandardScaler()
    X_train_norm = normalise.fit_transform(X_train)
    X_test_norm = normalise.transform(X_test)
    X_train.reset_index(inplace=True, drop=True)    
    y_train.reset_index(inplace=True, drop=True)    

    
    return feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise
