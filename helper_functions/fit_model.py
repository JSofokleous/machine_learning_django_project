from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from io import BytesIO
import base64 #get_plot()

## CLASSIFIER
def fit_score_model(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)

    # Score the model 
    model.score(X_train, y_train)
    model.score(X_test, y_test)

    # Analyse coefficients by printing:
    #### AttributeError: coef_ is only available when using a linear kernel
    # list(zip(['Sex','Age','FirstClass','SecondClass', 'Master'],model.coef_[0]))

    # Predict labels using test data
    y_pred = model.predict(X_test)

    # Determine accuracy and F1 score, Round to 1.d.p and convert to percentage 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(100*accuracy, 1)
    f1 = f1_score(y_test, y_pred)
    f1 = round(100*f1, 1)

    return accuracy, f1

def plot_to_html(buffer):
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png= buffer.getvalue()
    plot = base64.b64encode(image_png)
    plot = plot.decode('utf-8')
    buffer.close()    
    return plot


from helper_functions.load_data import load_rent, list_feature_names

# LOG MODEL
def log_model():
    # LOAD DATA
    features, bool_data_type, labels = load_rent(8000)
    # Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)
    # Normalise the feature data (mean = 0, std = 1 using Z-score method), only fit StandardScalar to train data.
    normalise = StandardScaler()
    X_train_norm = normalise.fit_transform(X_train)
    X_test_norm = normalise.transform(X_test)
    X_train.reset_index(inplace=True, drop=True)
    # List current selected features
    feature_names, feature_names_list = list_feature_names(features, bool_data_type)

    # Return user input sample for selected features, and normalise
    # sample_features = get_sample(feature_names) 
    sample_features = [200, 5, 1, 0, 0, 0]

    sample_features_norm = normalise.transform([sample_features])

    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data. [[failure, survival]]
    prediction_prob = classifier.predict_proba(sample_features_norm)
    prediction_prob = round(100*prediction_prob[0][1], 2)
    if prediction_prob > 50: prediction = 1
    else: prediction = 0

    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train.size_sqft, X_train.min_to_subway, X_train.one_bed, c=y_train, cmap='RdYlBu', alpha=0.25)  

    plt.xlabel('size_sqft')
    plt.ylabel('min_to_subway')
    plt.tight_layout()
    buffer = BytesIO()
    plot = plot_to_html(buffer)

    return plot, prediction, prediction_prob, accuracy, f1


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def get_best_k(X_train, y_train, X_test, y_test):
    k_list = range(1, 101)
    scores = []
    best_score, best_k = 0, 0
    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)
        if score > best_score: 
            best_score = score
            best_k = k
    # plt.plot(k_list, scores)
    # plt.show()
    return best_k

def k_distance(data_point, sample_features, feature_names_list):
    squared_difference = 0
    # Datapoint: [1, 2, 3, 4]
    # Samplepoint: [[1.3, -1.5, 1.8, -0.5, 4.9]]
    for i in range(len(data_point)):
        squared_difference += (data_point[feature_names_list[i]].item() - sample_features[i]) ** 2
        final_distance = squared_difference ** 0.5
        return final_distance

def k_classify(sample_features_norm, X_train_norm, y_train, k, feature_names_list, sample_features, X_train):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[feature_names_list[0]], X_train[feature_names_list[1]], X_train[feature_names_list[2]], c=y_train, cmap='RdYlBu', alpha=0.15)
    ax.scatter(sample_features[0], sample_features[1], sample_features[2], c='k', marker='o', s=300)
    ax.set_xlabel(feature_names_list[0])
    ax.set_ylabel(feature_names_list[1])
    ax.set_zlabel(feature_names_list[2])

    ## DETERMINE AND PLOT CLOSEST NEIGHBOURS
    # Loop through all points in the dataset X_train
    distances = []
    for row_index in range(len(X_train)):
        data_point = X_train.loc[[row_index]]
        distance_to_point = k_distance(data_point, sample_features, feature_names_list)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, row_index])

    # Taking only the k closest points
    distances.sort()
    neighbors = distances[0:k]

    # Classify point based on majority of neighbours (If equal, return label of FIRST neighbour)
    success, fail = 0, 0
    for neighbor in neighbors:
        row_index = neighbor[1]
        # Add neighbors to scatter
        row = X_train.loc[[row_index]]
        ax.scatter(row[feature_names_list[0]].item(), row[feature_names_list[1]].item(), row[feature_names_list[2]].item(), c='dimgrey', marker='1', s=500)

        if y_train.iloc[row_index] == 0: 
            fail += 1
        elif y_train.iloc[row_index] == 1:
            success += 1 


    plt.show()
       
    if success > fail: return 1
    elif fail > success: return 0
    else: 
        print('Equal number of neighbours!')
        return y_train.iloc[neighbors[0][1]]
