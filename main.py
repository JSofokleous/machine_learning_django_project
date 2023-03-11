import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 1
from helper_functions.load_data import load_titanic, load_house, list_feature_names
# 3
from helper_functions.input_sample import get_sample
# 4
from helper_functions.fit_model import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


## 1: CHOOSE MODEL
# Introduction message: List of ML models is printed to the user
print("\n" + "#"*75)
print("\nYou were aboard the Titanic when it struck an iceberg! This machine learning algorithm will determine if you will survive.")
print("\nPlease pick which machine learning model you would like to use to determine your chances of survival. \n\n\n~~~CHOICES~~~\n")
models = {'knn':'K-Nearest Neighbours', 'log':'Logistic Regression', 'svm':'Support Vector Machine', 'tree':'Decision Tree'}
for i in models:
    print("For the {} model, please write \"{}\"".format(models[i], i))
print("\n~~~~~~~~~~~~~")

# User prompted to choose a ML model, which is only accepted if a preset string
while True:
    model_name = input("\nEnter here: ")
    if model_name in models: 
        print("You have picked a {} model!".format(models[model_name]))
        break
    print("Please pick a valid model")



## 2: LOAD DATA
# Load cleaned data into a dataframe
budget = 0
while budget < 1000 or budget > 20000:
    budget = int(input("\nBudget in usd/month (avg = 5000): "))
features, binary, labels = load_house(budget)

# Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)

# Normalise the feature data (mean = 0, std = 1 using Z-score method), only fit StandardScalar to train data.
normalise = StandardScaler()
X_train_norm = normalise.fit_transform(X_train)
X_test_norm = normalise.transform(X_test)
X_train.reset_index(inplace=True, drop=True)

# List current selected features
feature_names, feature_names_list = list_feature_names(features, binary)


## 3: LOAD SAMPLE AND CREATE/USE MODEL 

# sample_features = [1 for col in features.columns]

# Logistic Regression Model (works best with binary LABELS? +can predict proba)
if model_name == 'log': 
    # Return user input sample for selected features, and normalise
    sample_features = get_sample(feature_names) 
    sample_features_norm = normalise.transform([sample_features])

    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data. [[failure, survival]]
    prediction_prob = classifier.predict_proba(sample_features_norm)
    prediction_prob = round(100*prediction_prob[0][1], 2)
    print("\nThe probability of finding a property with these features within your budget is {0}%! The accuracy of this model is {1}% and the f1 score is {2}%\n".format(prediction_prob, accuracy, f1))


# K-Nearest Neighbors Model (Can only display 3 features on a graph, same as log. -no proba)
elif model_name == 'knn': 
    # Return user input sample for selected features, and normalise
    sample_features = get_sample(feature_names) #1D ARRAY
    sample_features_norm = normalise.transform([sample_features]) #2D ARRAY

    # Determine best value of k
    # k = get_best_k(X_train_norm, y_train, X_test_norm, y_test)
    # print("~~~K = {}~~~".format(k))
    k = 10


    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample using own KNN model
    if k_classify(sample_features_norm, X_train_norm, y_train, k, feature_names_list, sample_features, X_train) == 1: 
        print("\nA property with these features is within your budget. The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("\nA property with these features is not within your budget. The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))


# Support Vector Model (-Can only display 2 features (on 2D or 3D), as transform radially to split data by a plane. -no proba)
elif model_name == 'svm': 

    # Return user input sample for selected features, and normalise
    # sample_features = get_sample(feature_names)     # UNCOMMENT TO SELECT CUSTOM FEATURES
    sample_features = [1 for col in features.columns]     # UNCOMMENT FOR DEFAULT FEATURES: 1st class female, 1yo  
    sample_features_norm = normalise.transform([sample_features])

    # Only allow 2 features to compare (automatically set to first 2 columns)
    print("Picking first 2 features: {0} and {1}".format(feature_names[:2][0], feature_names[:2][1]))
    X_train_norm = np.vstack([X_train_norm[:,0], X_train_norm[:,1]]).T
    X_test_norm = np.vstack([X_test_norm[:,0], X_test_norm[:,1]]).T
    sample_features_norm = [sample_features_norm[0][:2]]

    # Fit data to model and determine accuracy 
    classifier = SVC(kernel='linear', C = 0.01)
    # classifier = SVC(kernel='rbf', gamma = 0.05, C = 1000)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)
    three_d_svm(classifier, X_train_norm, y_train)
    two_d_svm(classifier, X_train_norm, y_train)

    # Predict label of sample
    if classifier.predict(sample_features_norm) == 1:
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))


# Tree
elif model_name == 'tree': 
    print("This feature is coming soon!")


# Error
else: print("Error loading model")