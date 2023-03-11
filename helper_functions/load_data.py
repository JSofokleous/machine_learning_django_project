import pandas as pd

def list_feature_names(features, binary):
    feature_names = {}
    for i in range(len(features.columns)):
        feature_names[features.columns[i]] = binary[i]
    print("Feature names: ", feature_names)
    feature_names_list = [name for name in feature_names]

    return feature_names, feature_names_list

def load_rent(budget):
    # Load data into dataframe
    streeteasy = pd.read_csv('data/rent.csv')
    df = pd.DataFrame(streeteasy)

    # Features
    df['max_rent'] = df.rent.apply(lambda x: 0 if x >= budget else 1)
    labels = df['max_rent']

    # Labels
    df['one_bed'] = df.bedrooms.apply(lambda x: 1 if x == 1 else 0)
    df['two_or_more_bed'] = df.bedrooms.apply(lambda x: 1 if x >= 2 else 0)
    features = df[['size_sqft', 'min_to_subway', 'one_bed', 'two_or_more_bed', 'has_patio', 'has_gym']]
    bool_data_type = [0, 0, 1, 1, 1, 1]

    print('\nAVG SIZE: ', df.size_sqft.mean())
    print('AVG TIME TO SUBWAY', df.min_to_subway.mean())

    return features, bool_data_type, labels 
