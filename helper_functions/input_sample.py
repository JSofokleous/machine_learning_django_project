
def get_sample(feature_names):
    # Dictionary of all features and their values initally set to -1
    dict = {}
    list = []

    for feature in feature_names:
        # Set initial value to -1, so can re-query until correct input 
        dict[feature] = -1

        # CONTINUOUS
        if feature_names[feature] == 0:
            while dict[feature] < 0:
                dict[feature] = int(input("\n{}: ".format(feature)))
        
        # BINARY
        elif feature_names[feature] == 1:
            while dict[feature] != 0 and  dict[feature] != 1:
                dict[feature] = int(input("\n{}: ".format(feature)))
        
        list.append(dict[feature])

    return list