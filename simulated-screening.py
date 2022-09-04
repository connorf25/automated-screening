# Imports
import sys

import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xml.dom import minidom

import numpy as np

from IPython.display import clear_output

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
    initial_data = initial['embeddings'].tolist()
    initial_labels = initial['code'].tolist()

    # Fit model to initial training data
    model.fit(initial_data, initial_labels)

    # Get remaining data for testing
    remaining_data = remaining['embeddings'].tolist()

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
def simulateScreening(df, randomOrder = False):
    # Intialize metrics
    effort_list = []
    accuracy_list = []
    # Create training/testing data
    labelled, unlabelled = createTrainTest(df)

    # Shuffle data
    labelled = labelled.sample(frac=1).reset_index(drop=True)
    unlabelled = unlabelled.sample(frac=1).reset_index(drop=True)

    # Load model
    model = LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000)

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

def read_embeddings(name, method, layers_to_use):
    # If running the control test just load the tf-idf embeddings (smallest file)
    if method == "control":
        df = pd.read_pickle("./" + name + "/" + name + "-embeddings-simple.pkl")
    else:
        df = pd.read_pickle("./" + name + "/" + name + "-embeddings-" + method + ".pkl")
    if "bloom" in method or "scibert" in method:
        if layers_to_use == "average":
            # Average all the last 5 hidden layers
            df["embeddings"] = df["embeddings"].map(lambda layers: average_layers(layers))
        elif layers_to_use == "concat":
            # TODO: take concatenation of all word embeddings
            print("TODO")
        else:
            # Take just last layer of model
            df["embeddings"] = df["embeddings"].map(lambda layers: layers[-1])
    return df

# Take the average of multiple hidden layers
def average_layers(layers):
    data = np.array(layers)
    return np.average(data, axis=0)

def calculate_embeddings_bow(name):
    df = loadXmlToDataframe(name)
    countVectorizer = CountVectorizer(stop_words="english", max_features=768)
    df["embeddings"] = countVectorizer.fit_transform(df["abstract"]).todense().tolist()
    return df

def calculate_embeddings_tfidf(name):
    df = loadXmlToDataframe(name)
    tfidfVectorizer = TfidfVectorizer(stop_words="english", max_features=768)
    df["embeddings"] = tfidfVectorizer.fit_transform(df["abstract"]).todense().tolist()
    return df

def calculate_embeddings_word2vec(name):
    df = loadXmlToDataframe(name)
    # TODO: Return word2vec embeddings
    return df

def loadXmlToDataframe(name):
    abstractsInclude, tagsInclude = parseXML(name + '/' + name + 'Include.xml', 1)
    abstractsExclude, tagsExclude = parseXML(name + '/' + name + 'Exclude.xml', 0)
    df = pd.DataFrame(list(zip(tagsInclude + tagsExclude, abstractsInclude + abstractsExclude)), columns =['code', 'abstract'])
    return df

#Function to parse xml
def parseXML(filename, isInclude):
    abstracts = []
    tags = []
    xmldoc = minidom.parse(filename)
    for node in xmldoc.getElementsByTagName('abstract'):
        abstract = node.getElementsByTagName('style')[0].firstChild.nodeValue
        abstracts.append(abstract)
        tags.append(isInclude)
    return abstracts, tags

# Main function
if __name__ == "__main__":
    method = sys.argv[1]
    layers = sys.argv[2] if len(sys.argv) > 2 else False
    randomOrder = False
    if method == "control":
        randomOrder = True

    names = ["cellulitis", "copper", "search", "uti", "overdiagnosis"]

    for name in names:
        stats = []
        if method == "bow":
            df = calculate_embeddings_bow(name)
        elif method == "tfidf":
            df = calculate_embeddings_tfidf(name)
        elif method == "word2vec":
            df = calculate_embeddings_word2vec(name)
        else:
            df = read_embeddings(name, method, layers)
        # Simulate screening 10 times
        for i in range(10):
            clear_output(wait=True)
            print(name)
            print(i+1)
            stats.append(simulateScreening(df, randomOrder))

        stats_df = pd.DataFrame(stats, columns=["effort", "accuracy"])
        if layers:
            stats_df.to_csv("./stats-" + name + "-" + method + "-" + layers + ".csv")
        else:
            stats_df.to_csv("./stats-" + name + "-" + method + ".csv")