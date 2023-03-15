
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from io import BytesIO
import base64 #get_plot()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from helper_functions.load_data import load_rent



# LOGISTIC REGRESSION MODEL
def log_model(budget, sample_features): 
    # Load data
    feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise = load_rent(budget)
    sample_features_norm = normalise.transform([sample_features])

    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data
    prediction = classifier.predict_proba(sample_features_norm)
    prediction = round(100*prediction[0][1], 2)

    # Plot
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[feature_names_list[0]], X_train[feature_names_list[1]], X_train[feature_names_list[2]], c=y_train, cmap='RdYlBu', alpha=0.15, label='Training data')
    ax.scatter(sample_features[0], sample_features[1], sample_features[2], c='k', marker='o', s=200, label='Desired property')
    ax.set_xlabel('Size of property (sqft)')
    ax.set_ylabel('Mins to underground station')
    ax.set_zlabel('Bedrooms')
    ax.set_zticks(range(6))
    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plot = plot_to_html(buffer)
    plt.close()

    return plot, prediction, accuracy, f1


# K-NEAREST NEIGHBOURS
def knn_model(budget, sample_features):
    # Load data
    feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise = load_rent(budget)
    
    sample_features_norm = normalise.transform([sample_features]) 
    k = 10
    # k = get_best_k(X_train, y_train, X_test, y_test)

    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample using own KNN model
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[feature_names_list[0]], X_train[feature_names_list[1]], X_train[feature_names_list[2]], c=y_train, cmap='RdYlBu', alpha=0.15, label='Training data')
    ax.scatter(sample_features[0], sample_features[1], sample_features[2], c='k', marker='o', s=200, label='Desired property')
    ax.set_xlabel('Size of property (sqft)')
    ax.set_ylabel('Mins to underground station')
    ax.set_zlabel('Bedrooms')
    ax.set_zticks(range(6))

    ## DETERMINE AND PLOT CLOSEST NEIGHBOURS TO PREDICT POINT
    # Loop through all points in the dataset X_train
    distances = []

    # Filter realistic ranges to improve runtime
    if sample_features[0] <= 1200:
        subset_i = list(X_train[(X_train.size_sqft.between(0.9*sample_features[0],1.1*sample_features[0]))].index.values)
    else:
        subset_i = list(X_train[X_train.size_sqft >= 0.8*sample_features[0]].index.values)
    X_train_subset = X_train.iloc[subset_i].reset_index(drop=True)
    y_train_subset = y_train.iloc[subset_i].reset_index(drop=True)

    for row_index in range(len(X_train_subset)):
        data_point = X_train_subset.loc[[row_index]]

        # Calculate distance to point in row_index, for each feature i
        squared_difference = 0
        for i in range(len(data_point)):
            squared_difference += (data_point[feature_names_list[i]].item() - sample_features[i]) ** 2
        
        # Adding the distance and point associated with that distance
        distances.append([squared_difference ** 0.5, row_index])

    # Taking only the k closest points
    distances.sort()
    neighbors = distances[0:k]

    # Classify point based on majority of neighbours (If equal, return label of FIRST neighbour)
    success, fail = 0, 0
    for neighbor in neighbors:
        row_index = neighbor[1]
        # Add neighbors to scatter
        row = X_train_subset.loc[[row_index]]
    
        if success == 0 and fail == 0:
            ax.scatter(row[feature_names_list[0]].item(), row[feature_names_list[1]].item(), row[feature_names_list[2]].item(), c='dimgrey', marker='1', s=500, label='Nearest neighbours')
        else:
            ax.scatter(row[feature_names_list[0]].item(), row[feature_names_list[1]].item(), row[feature_names_list[2]].item(), c='dimgrey', marker='1', s=500)

        if y_train_subset.iloc[row_index] == 0: 
            fail += 1

        elif y_train_subset.iloc[row_index] == 1:
            success += 1 

    prediction = 0
    prediction_prob = None
    if success > fail: prediction = 1
    elif fail == success: 
        prediction = y_train_subset.iloc[neighbors[0][1]]

    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plot = plot_to_html(buffer)
    plt.close()

    return plot, prediction, accuracy, f1


def svm_model(budget, sample_features):
    feature_names_list, X_train, X_train_norm, X_test, X_test_norm, y_train, y_test, normalise = load_rent(budget)
    sample_features_norm = normalise.transform([sample_features])
    sample_features_norm = [sample_features_norm[0][:2]]
   
    # Only allow 2 features to compare (automatically set to first 2 columns)
    X_train_norm = np.vstack([X_train_norm[:,0], X_train_norm[:,1]]).T
    X_test_norm = np.vstack([X_test_norm[:,0], X_test_norm[:,1]]).T
    X_train = np.vstack([X_train.size_sqft, X_train.min_to_subway]).T
    X_test = np.vstack([X_test.size_sqft, X_test.min_to_subway]).T
    r = np.exp(-(X_train_norm ** 2).sum(1))

    # Fit data to model and determine accuracy 
    # classifier = SVC(kernel='linear', C = 0.01)
    # classifier = SVC(kernel='rbf', gamma = 0.7, C = 0.1)
    # classifier = SVC(kernel='poly', degree = 5)
    classifier = SVC(kernel='rbf', gamma = 0.006, C = 0.7)
    accuracy, f1 = fit_score_model(classifier, X_train, y_train, X_test, y_test)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(X_train[:,0], X_train[:,1], r, c=y_train, s=50, cmap='RdYlBu', alpha=0.15, label='Training data')
    # ax.scatter3D(X_train_norm[:,0], X_train_norm[:,1], r, c=y_train, s=50, cmap='RdYlBu', alpha=0.15)
    ax.scatter(sample_features[0], sample_features[1], 0, c='k', marker='o', s=200, label='Desired property')
    ax.set_xlabel('Size of property (sqft)')
    ax.set_ylabel('Mins to underground station')
    ax.set_zlabel('kernal')

    # Split the range into 20 equal parts and return in a 1D list
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx = np.linspace(xlim[0], xlim[1], 10)
    yy = np.linspace(ylim[0], ylim[1], 10)

    # Return two 2D lists combining the coordinates in xx and yy
    YY, XX = np.meshgrid(yy, xx)

    # Ravel flattens each array into 1D and Vstack joins the two 1D arrays into a 2D array, which is transposed 
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy)
    Z = Z.reshape(XX.shape) 

    # Show decision boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-2, 0, 2], alpha=0.5, linestyles=['--', '-', '--'], label='Decision boundary')
    # ax.contour(XX, YY, Z ,colors='k', levels=[-2, 0, 2],alpha=0.2, extend3d=True) 

    # ax.set_zlim(zmin=0, zmax=4)
    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plot = plot_to_html(buffer)
    plt.close()


    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample
    prediction = classifier.predict(sample_features_norm)
    prediction_prob = None


    return plot, prediction, accuracy, f1


    


# Helper functions:
# FIND OPTIMAL k VALUE
def get_best_k(X_train, y_train, X_test, y_test, lower_boundary=5, upper_boundary=20):
    scores = []
    best_score, best_k = 0, 0
    for k in range(lower_boundary, upper_boundary):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)
        if score > best_score: 
            best_score = score
            best_k = k
    return best_k


# CLASSIFIER
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

# PLOT TO HTML
def plot_to_html(buffer):
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png= buffer.getvalue()
    plot = base64.b64encode(image_png)
    plot = plot.decode('utf-8')
    buffer.close()    
    return plot