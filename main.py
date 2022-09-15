# Imports
import sys

import pandas as pd

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

from calculate_embeddings import get_dataframe_with_embeddings

# Calculate the total number of articles screened so far (train / train + test)
def calcPercentScreened(initial, remaining):
    return (initial.index[-1] + 1) / ((remaining.index[-1] + 1) + (initial.index[-1] + 1))

# Calculate the number of articles needed to screen out of entire dataset to find all included articles
# (train + last_index / train + test)
def calcPercentNeedToScreen(initial, remaining, last_index):
    return ((initial.index[-1] + 1) + (last_index + 1)) / ((initial.index[-1] + 1) + (remaining.index[-1] + 1))

# Print stats (return false when all included articles found)
def printStats(initial, remaining):
    # Find index of last included article
    include_indicies = remaining[remaining.code == 1].index
    # Calculate total percentage of articles that need to be screened
    if (len(include_indicies) == 0):
        print("All included articles found after screening: %d (%.2f percent)" % (len(initial), calcPercentScreened(initial, remaining) * 100))
        return False
    else:
        print("Screened: %.2f (%d) Need to screen: %.2f (%d)" % (
            calcPercentScreened(initial, remaining) * 100,
            len(initial),
            calcPercentNeedToScreen(initial, remaining, include_indicies[-1]) * 100,
            len(remaining)
         ))
        return True

from math import ceil

# Count number of included articles
def countIncludes(df):
    return len(df[df.code == 1].index)

# Create training and testing set
# Use 20% of included articles and equal number of excludes
def createTrainTest(df, train_proportion=0.2):
    # Split include and exclude data
    includes = df[df.code == 1]
    excludes = df[df.code == 0]
    # Take 20% of the total includes
    train_size = ceil(len(includes) * train_proportion)
    # Get training data
    train_includes = includes.sample(train_size)
    train_excludes = excludes.sample(train_size)
    # Remove training data from testing data
    includes.drop(train_includes.index)
    excludes.drop(train_excludes.index)
    # Return train and test
    return pd.concat([train_includes, train_excludes]), pd.concat([includes, excludes])

# Function to calculate probabilities of each remaining article
def calcProb(model, initial, remaining):
    # Get initial training data and labels
    initial_data = initial['embedding'].tolist()
    initial_labels = initial['code'].tolist()

    # Fit model to initial training data
    model.fit(initial_data, initial_labels)

    # Get remaining data for testing
    remaining_data = remaining['embedding'].tolist()

    # Predict probability [exclusion, inclusion] on remaining articles
    pred = model.predict_proba(remaining_data)

    # Calculate score (x[1] = probability of inclusion)
    pred = list(map(lambda x: x[1], pred))
    # Add probability to dataframe
    remaining['prob'] = pred
    # Sort by probability
    remaining = remaining.sort_values(by=['prob'], ascending=False).reset_index(drop=True)

    return remaining

# Simulate screening process and return effort and accuracy metrics
def simulateScreening(df, model, randomOrder = False):
    # Intialize metrics
    effort_list = []
    accuracy_list = []
    # Create training/testing data
    labelled, unlabelled = createTrainTest(df)

    # Shuffle data
    labelled = labelled.sample(frac=1).reset_index(drop=True)
    unlabelled = unlabelled.sample(frac=1).reset_index(drop=True)

    # Find total number of includes
    total_includes = countIncludes(df)

    while(unlabelled.index[-1] > 0):
        # Calculate number of included articles found so far
        includes_found = countIncludes(labelled)

        # Calculate the effort of the model so far
        effort = len(labelled) / len(df)
        # Calculate the accuracy of the model at this point
        accuracy = includes_found/total_includes
        # Append stats to lists
        effort_list.append(effort)
        accuracy_list.append(accuracy)

        # Early termination
        if(includes_found == total_includes):
            # Have found all articles
            break

        # Calculate and sort unlabelled data (to get documents rankings)
        if not randomOrder:
            unlabelled = calcProb(model, labelled, unlabelled)

        # Take highest ranking remaining article and add it to labelled data
        # Drop it from unlabbeled data
        # (This simuates querying/screening the highest ranked document)
        labelled = pd.concat([labelled, unlabelled.iloc[[0]]], ignore_index=True)
        unlabelled.drop(0, inplace=True)
        unlabelled.reset_index(drop=True, inplace=True)

    # Return effort and accuracy data for screening simulation
    return effort_list, accuracy_list

def load_classifier(model_name):
    if "svc" in model_name:
        model = SVC(kernel='rbf', class_weight='balanced', probability=True)
    elif "svclinear" in model_name:
        model = SVC(kernel='linear', class_weight='balanced', probability=True)
    elif "mlp" in model_name:
        model = MLPClassifier()
    elif "lr" in model_name:
        model = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000)
    else:
        print("Falling back to default logistic regression model")
        model = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000)
    return model

# Main function
if __name__ == "__main__":
    # Datasets to use # UP TO BOW-mlp overdiagnosis
    dataset_names = ["bacteriuria", "telehealth"]

    # Models to use
    # Simple
    # model_names = ["bow-lr", "bow-svc", "bow-mlp", "tfidf-lr", "tfidf-svc", "tfidf-mlp", "doc2vec-lr", "doc2vec-svc", "doc2vec-mlp"]
    # Scibert
    model_names = ["scibert", "scibert-average", "scibert-concat"]
    # Blooom
    # model_names = ["bloom-350m", "bloom-1b7"]

    for model_name in model_names:
        randomOrder = False
        if model_name == "control":
            randomOrder = True
        model = load_classifier(model_name)

        for dataset_name in dataset_names:
            stats = []
            df =  get_dataframe_with_embeddings(dataset_name, model_name)
            # Simulate screening 10 times
            for i in range(10):
                print(model_name, dataset_name)
                print(i+1)
                stats.append(simulateScreening(df, model, randomOrder))

            stats_df = pd.DataFrame(stats, columns=["effort", "accuracy"])
            # Save to csv
            stats_df.to_csv("./stats/stats-" + dataset_name + "-" + model_name + ".csv")
