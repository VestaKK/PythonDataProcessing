import pandas as pd
import numpy as np
import csv
import math
import json

from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, f1_score

"""
    Takes an encoded dataframe and outputs its numpy array
    equivalent
"""
def DataEncoder(df):
    output_list = []
    for index, row in df.iterrows():
        output_list.append(row.tolist())
    return np.asarray(output_list)

"""
    Goes through a numpy array, imputes mode of columns
    on nan entries and returns the altered array
"""
def impute_mode(numpy):
    
    mode = stats.mode(numpy, axis=0, keepdims=True)
    mode = list(mode[0][0])
    NUM_COLUMNS = len(numpy[0])
    NUM_ROWS = len(numpy)

    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            if math.isnan(numpy[i, j]):
                numpy[i, j] = mode[j]
                
    return numpy


"""
    Determines the most effective supervised ML model for the data set using
    encoded data, and evaluates this
"""
def data_analysis():

    bin_features = ['binned_damage_objectives', 'binned_damage_building',
       'binned_damage_taken', 'binned_damage_total', 'binned_gold_earned',
       'binned_kda', 'binned_level', 'binned_kills', 'binned_deaths', 
       'binned_assists', 'binned_time_cc', 'binned_vision_score']

    # Read in reformated dataset 
    binned_data = pd.read_csv('CSVFILES/BINNED_DATA.csv')

    # Sets target variable
    class_label = 'summoner_spells'

    # gets all variables in the data set
    features = list(binned_data.columns)

    # remove class label and non-binned features
    features.remove(class_label)
    for feature in bin_features:
        features.remove(feature[7:])

    # Generate 80/20 train-test split using binned_data
    X = binned_data[features]
    y = binned_data[class_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    train = pd.concat([X_train, y_train.to_frame()], axis=1)
    test = pd.concat([X_test, y_test.to_frame()], axis=1)

    """
        sets training variable to none for the sake of testing
    """
    def set_XY_None():
        XY_variables = [X, y, X_train, X_test, y_train, y_test]
        for variable in XY_variables:
            variable = None

    set_XY_None()


    # write final output to a .txt file
    f = open('TXT_FILES/final_model_evaluation.txt', 'w')

    # Calculates the mutual information score of each value
    feature_NMI = {}
    print(f'Calculating NMI between our target variable and features of the data set:')
    f.write('Calculating NMI between our target variable and features of the data set:\n')
    for feature in features:
        # Use train to avoid data leakage
        feature_NMI[feature] = normalized_mutual_info_score(train.dropna()[class_label], 
                                train.dropna()[feature], 
                                average_method='min')
        print(f'\tThe NMI score between {class_label} and {feature} is {feature_NMI[feature]}')
        f.write(f'\tThe NMI score between {class_label} and {feature} is {feature_NMI[feature]}\n')

    # K-Fold for data evaluation
    print("\nEvaluating best model for data set using K-Fold:")
    f.write("\nEvaluating best model for data set using K-Fold:\n")
    # Defines K-Fold function
    kf_CV = KFold(n_splits=5, shuffle=True, random_state=42)

    # Different num_features that could be used in model                
    possible_num_features = [1, 2, 3, 4, 5]

    # Sorts features by highest NMI score
    features.sort(key = lambda feature: feature_NMI[feature], reverse=True) 

    # Define function models
    k_options = [3, 4, 5, 6, 7]
    criterion_options = ['entropy', 'gini']

    # Defines dictionary to contain the best model attributes
    best_model = {}
    best_model['accuracy'] = 0
    best_model['model'] = None
    best_model['num_features'] = None
    best_model['option'] = None

    # Evaluating Decision Tree using K-Fold
    # -----------------------------------------------------------------------------
    # Iterates through the possible criterion for measuring the quality of a split
    for chosen_criterion in criterion_options:
        dt = DecisionTreeClassifier(criterion=chosen_criterion)
        print(f'\tK-Fold using a(n) {chosen_criterion} based decision tree:') 
        f.write(f'\tK-Fold using a(n) {chosen_criterion} based decision tree:\n')
        # Iterates through the number of possible features
        for num_features in possible_num_features:
            # Stores results of K-Fold
            results = []

            # Calculates filtered_features with num_features features
            filtered_features = features[:num_features]
            # Gets X and y variables from training_set
            X = DataEncoder(train[filtered_features])
            y = DataEncoder(train[[class_label]])

            for train_idx, test_idx in kf_CV.split(X):
                # train-test split
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Pre-processing     
                X_train = impute_mode(X_train)
                X_test = impute_mode(X_test)
                
                # Training
                dt.fit(X_train, y_train)    
                
                # Predictions
                results.append(dt.score(X_test, y_test))

            accuracy = np.mean(results)
            print(f"\t\tAverage Accuracy with {num_features} feature(s): {accuracy}")
            f.write(f"\t\tAverage Accuracy with {num_features} feature(s): {accuracy}\n")
            if accuracy > best_model['accuracy']:
                best_model['accuracy'] = accuracy
                best_model['model'] = 'Decision Tree'
                best_model['num_features'] = num_features
                best_model['option'] = chosen_criterion

            set_XY_None()

        print()
        f.write('\n')

    # Evaluating KNN using K-Fold 

    # Iterates through possible values of k for knn
    for k in k_options:
        knn = KNN(n_neighbors=k)
        print(f'\tK-Fold using {k}NN:')
        f.write(f'\tK-Fold using {k}NN:\n')
        # Iterates through the number of possible features
        for num_features in possible_num_features:
            # Stores results of K-Fold
            results = []

            # Calculates filtered_features with num_features features
            filtered_features = features[:num_features]
            # Gets X and y variables from training_set
            X = DataEncoder(train[filtered_features])
            y = DataEncoder(train[[class_label]])

            for train_idx, test_idx in kf_CV.split(X):
                # train-test split
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Pre-processing     
                X_train = impute_mode(X_train)
                X_test = impute_mode(X_test) 
                # Standardises the data
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                # Training
                knn.fit(X_train, y_train.ravel())

                # Predictions
                y_pred = knn.predict(X_test)
                results.append(accuracy_score(y_test, y_pred))

            accuracy = np.mean(results)
            print(f"\t\tAverage Accuracy with {num_features} feature(s): {accuracy}\n")
            f.write(f"\t\tAverage Accuracy with {num_features} feature(s): {accuracy}\n\n")

            if accuracy > best_model['accuracy']:
                best_model['accuracy'] = accuracy
                best_model['model'] = 'KNN'
                best_model['num_features'] = num_features
                best_model['option'] = k_options

            set_XY_None()
        
        print()
        f.write('\n')
    # -----------------------------------------------------------------------------


    # Prints the best model by accuracy
    print(f'The best model by accuracy is a(n) {best_model["option"]} {best_model["model"]} with {best_model["num_features"]} feature(s)')
    f.write(f'The best model by accuracy is a(n) {best_model["option"]} {best_model["model"]} with {best_model["num_features"]} feature(s)\n')

    print(f'It gives an accuracy of {best_model["accuracy"]}\n')
    f.write(f'It gives an accuracy of {best_model["accuracy"]}\n\n')

    # Calculates filtered_features with optimal NUM_FEATURES features from K-Fold
    NUM_FEATURES = best_model['num_features']

    features.sort(key = lambda feature: feature_NMI[feature], reverse=True)
    filtered_features = features[:NUM_FEATURES]

    print(f'The {NUM_FEATURES} best features for predicting {class_label} by NMI are:')
    f.write(f'The {NUM_FEATURES} best features for predicting {class_label} by NMI are:\n')

    for feature in filtered_features:
        print(f'\t{feature} with a NMI score of {feature_NMI[feature]}')
        f.write(f'\t{feature} with a NMI score of {feature_NMI[feature]}\n')

    with open('CSVFILES/BINNED_DATA_ENCODES.json', 'r') as fp:
        summoner_spell_encode = json.load(fp)['summoner_spell_encode']

    # Gets full set of train data and processes it
    print('\nTraining Decision Tree Model...')
    f.write('\nTraining Decision Tree Model...\n')

    X = impute_mode(DataEncoder(train[filtered_features]))
    y = DataEncoder(train[[class_label]])

    # Fit decision tree using optimal criterion from K-Fold
    dt = DecisionTreeClassifier(criterion=best_model['option'])
    dt.fit(X, y)

    # --------------------------------
    GENERATE_DECISION_TREE_PNG = False
    # --------------------------------

    # Outputs decision_tree.png if allowed
    if GENERATE_DECISION_TREE_PNG:
        print('Generating decision_tree.png...\n')
        f.write('Generating decision_tree.png...\n\n')
        class_labels = list(summoner_spell_encode.keys())
        plt.figure(figsize=(10, 15))
        plot_tree(dt, # the DT classifier
                feature_names=filtered_features, # feature names
                class_names=class_labels, # class labels
                filled=True, # fill in the rectangles
                fontsize = 6
                )

        plt.title("Decision Tree Classifier")
        plt.savefig('Graphs/decision_tree.png', bbox_inches = "tight")
    else:
        print('\n\t============ PNG generation for decision tree has been disabled ============')
        print('\t                            IMPORTANT MESSAGE:                              \n')
        print('\tGeneration of "decision_tree.png" tends to break the python kernel on Ed')
        print('\tWe suspect this is an issue with the website and not with the code itself.')
        print('\tIf you would like to generate "decision_tree.png", you can go into')
        print('\tdata_analysis.py and redefine GENERATE_DECISION_TREE_PNG as True.\n')
        print('\tMore information in README.\n')
        print('\t============================================================================\n')

    
    # Evaluate model on test set
    print('Evaluating Model...\n')
    f.write('Evaluating Model...\n\n')

    X_test = test[filtered_features]
    y_test = test[[class_label]]

    X_test = impute_mode(DataEncoder(X_test))
    y_test = DataEncoder(y_test)

    y_pred = dt.predict(X_test)

    class_labels = list(summoner_spell_encode.values())
    class_labels_display = [i for i in summoner_spell_encode.items()]
    class_labels_display = [i[0] for i in sorted(class_labels_display, key=lambda x: x[1])]

    # Confusion matrix
    print('Generating Confusion Matrix...')
    f.write('Generating Confusion Matrix...\n')
    cm = confusion_matrix(y_test, y_pred, labels=sorted(class_labels))

    # key=lambda x: x[1])
    fig = plt.figure(figsize=(100, 140))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels_display).plot()
    plt.title('Confusion matrix for imbalanced dataset')
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    print('Generating confusion_matrix.png...')
    f.write('Generating confusion_matrix.png...\n')
    plt.savefig('Graphs/confusion_matrix.png', bbox_inches = "tight")
    

    print('\nEvaluation statistics:')
    f.write('Evaluation statistics:\n')

    for method in ['weighted', None]:
        if method == None:
            print(f'\tSubset Accuracy: \n\t{list(recall_score(y_test, y_pred, average = None))}\n')
            f.write(f'\tSubset Accuracy: \n\t{list(recall_score(y_test, y_pred, average = None))}\n\n')

            print(f'\tSubset Precision: \n\t{list(precision_score(y_test, y_pred, average = None))}\n')
            f.write(f'\tSubset Precision: \n\t{list(precision_score(y_test, y_pred, average = None))}\n\n')

            print(f'\tSubset F1: \n\t{list(f1_score(y_test, y_pred, average = None))}\n')
            f.write(f'\tSubset F1: \n\t{list(f1_score(y_test, y_pred, average = None))}\n')

        else:
            print(f'\n\tAccuracy ({method}): {recall_score(y_test, y_pred, average = method)}')
            f.write(f'\n\tAccuracy ({method}): {recall_score(y_test, y_pred, average = method)}\n')

            print(f'\tPrecision ({method}): {precision_score(y_test, y_pred, average = method)}')
            f.write(f'\tPrecision ({method}): {precision_score(y_test, y_pred, average = method)}\n')

            print(f'\tF1 ({method}): {f1_score(y_test, y_pred, average = method)}\n')
            f.write(f'\tF1 ({method}): {f1_score(y_test, y_pred, average = method)}\n\n')

    f.close()
    return






