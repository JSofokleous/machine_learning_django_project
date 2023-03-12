
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64 #get_plot()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from helper_functions.load_data import load_rent


## CLASSIFIER
def fit_score_model(model, X_train, y_train, X_test, y_test):
    

    # Fit and score the model 
    model.fit(X_train, y_train)
    model.score(X_train, y_train)  ##### DELETE
    model.score(X_test, y_test)    ##### DELETE

    # Predict labels using test data
    y_pred = model.predict(X_test)

    # Determine accuracy and F1 score, Round to 1.d.p and convert to percentage 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(100*accuracy, 1)     ##### SIMPLIFY TO ONE LINE
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


# LOG MODEL
def log_model(budget, sample_features): 

    # Load data
    feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise = load_rent(budget)
    sample_features_norm = normalise.transform([sample_features])

    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data
    prediction_prob = classifier.predict_proba(sample_features_norm)
    prediction_prob = round(100*prediction_prob[0][1], 2)       #####SIMPLIFY
    if prediction_prob > 50: prediction = 1
    else: prediction = 0

    # Plot
    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train.size_sqft, X_train.min_to_subway, X_train.one_bed, c=y_train, cmap='RdYlBu', alpha=0.25)  

    plt.xlabel('size_sqft')
    plt.ylabel('min_to_subway') #### USE SAME METHOD AS KNN TO LABEL Z AXIS TOO
    plt.tight_layout()
    buffer = BytesIO()
    plot = plot_to_html(buffer)

    return plot, prediction, prediction_prob, accuracy, f1




def knn_model(budget, sample_features):

    # Load data
    feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise = load_rent(budget)
    sample_features_norm = normalise.transform([sample_features]) 

    # Determine best value of k
    # k = get_best_k(X_train_norm, y_train, X_test_norm, y_test)
    k = 10

    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample using own KNN model
    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_subplot(111, projection='3d')

    ##### CHANGE TO X_train.size_sqft
    ax.scatter(X_train[feature_names_list[0]], X_train[feature_names_list[1]], X_train[feature_names_list[2]], c=y_train, cmap='RdYlBu', alpha=0.15)
    ax.scatter(sample_features[0], sample_features[1], sample_features[2], c='k', marker='o', s=300)
    ax.set_xlabel(feature_names_list[0])
    ax.set_ylabel(feature_names_list[1])
    ax.set_zlabel(feature_names_list[2])

    ## DETERMINE AND PLOT CLOSEST NEIGHBOURS TO PREDICT POINT
    # Loop through all points in the dataset X_train
    distances = []
    for row_index in range(len(X_train)):
        data_point = X_train.loc[[row_index]]

        # Calculate distance to point in row_index, for each feature i
        squared_difference = 0
        for i in range(len(data_point)):
            squared_difference += (data_point[feature_names_list[i]].item() - sample_features[i]) ** 2
            distance_to_point = squared_difference ** 0.5

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

    prediction = 0
    prediction_prob = None
    if success > fail: prediction = 1
    elif fail == success: 
        prediction = y_train.iloc[neighbors[0][1]]

    plt.tight_layout() ###### POTENTIAL WEAK POINT, NOT PLT. but AX.
    buffer = BytesIO()
    plot = plot_to_html(buffer)
 
    return plot, prediction, prediction_prob, accuracy, f1


# def get_best_k(X_train, y_train, X_test, y_test):
#     k_list = range(1, 101)
#     scores = []
#     best_score, best_k = 0, 0
#     for k in k_list:
#         classifier = KNeighborsClassifier(n_neighbors=k)
#         classifier.fit(X_train, y_train)
#         score = classifier.score(X_test, y_test)
#         scores.append(score)
#         if score > best_score: 
#             best_score = score
#             best_k = k
#     return best_k



 