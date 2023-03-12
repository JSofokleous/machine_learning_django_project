import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_rent(budget):
    # Load data into dataframe
    streeteasy = pd.read_csv('data/rent.csv')
    df = pd.DataFrame(streeteasy)

    # Features
    df['max_rent'] = df.rent.apply(lambda x: 0 if x >= budget else 1)
    labels = df['max_rent']

    # Labels
    # df['one_bed'] = df.bedrooms.apply(lambda x: 1 if x <= 1 else 0)
    # df['two_beds'] = df.bedrooms.apply(lambda x: 1 if x <= 2 and x>1 else 0)
    # df['three_beds'] = df.bedrooms.apply(lambda x: 1 if x <= 3 and x>=2 else 0)
    # df['four_beds'] = df.bedrooms.apply(lambda x: 1 if x <= 5 and x>3 else 0)
    # df['five_beds'] = df.bedrooms.apply(lambda x: 1 if x == 5 else 0)
    # features = df[['size_sqft', 'min_to_subway', 'one_bed', 'two_beds', 'three_beds','four_beds','five_beds','has_patio', 'has_gym']]
    # bool_data_type = [0, 0, 1, 1, 1, 1, 1, 1, 1]

    df.bedrooms = df.bedrooms.apply(lambda x: x if x.is_integer() else x//1+1)
    df.bedrooms = df.bedrooms.apply(lambda x: 1 if x==0 else x)
    features = df[['size_sqft', 'min_to_subway', 'bedrooms', 'has_gym', 'has_patio']]
    # bool_data_type = [0, 0, 0, 1, 1]

    # DELETE??? Used only in knn.
    # feature_names = {}
    # for i in range(len(features.columns)):
    #     feature_names[features.columns[i]] = bool_data_type[i]
    feature_names_list = [feature for feature in features.columns]

    # Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)

    # Normalise the feature data (mean = 0, std = 1 using Z-score method), only fit StandardScalar to train data.
    normalise = StandardScaler()
    X_train_norm = normalise.fit_transform(X_train)
    X_test_norm = normalise.transform(X_test)
    X_train.reset_index(inplace=True, drop=True)    
    
    return feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise
