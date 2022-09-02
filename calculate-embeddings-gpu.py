

# Imports
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xml.dom import minidom

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer, BloomTokenizerFast, BloomModel
import torch
import numpy as np

#Timing
from IPython.display import clear_output
import timeit

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

# Function to return last 5 hidden layers for bloom and scibert
def get_embeddings(abstracts, model_name):
    # Load model and tokenizer
    if "bloom" in model_name:
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/" + model_name)
        model = BloomModel.from_pretrained("bigscience/" + model_name, output_hidden_states=True).to(device)
    elif "scibert" in model_name:
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True).to(device)
    else:
        print("Invalid model name")

    print('tokenizer is type:', type(tokenizer))
    print('model is type:', type(model))

    embeddings = []
    length = len(abstracts.tolist())
    index = 0

    start = timeit.default_timer()
    for sentence in abstracts.tolist():
        clear_output(wait=True)
        index += 1
        sen_emb = get_embedding(model, tokenizer, sentence)
        embeddings.append(sen_emb)

        if (index/length*100) < 1:
            expected_time = "Calculating..."

        else:
            time_perc = timeit.default_timer()
            expected_time = np.round( (time_perc-start) /(index/length) /60,2)

        print(index, length)
        print(expected_time)
    return embeddings

def get_embedding(model, tokenizer, text):

    # Encode with special tokens ([CLS] and [SEP], returning pytorch tensors
    encoded_dict = tokenizer.encode_plus(
                        text,
                        truncation=True,
                        max_length=512,
                        add_special_tokens = True,
                        return_tensors = 'pt'
                )
    input_ids = encoded_dict['input_ids']
    # Convert input to cuda
    input_ids = input_ids.to(device)
    # Set model to evaluation mode
    model.eval()

    # Run through model
    with torch.no_grad():
        outputs = model(input_ids)
        # Extract hidden states
        hidden_states = outputs.hidden_states


    # Select the word embeddings on the last 5 layers
    last_layers = hidden_states[-5:]
    print("Number layers:", len(last_layers))
    document_embedding_layers = []
    for layer in last_layers:
        # Take first item from batch (only one document per batch anyway)
        layer = layer[0]
        # Calculate average of all word embeddings to get document embedding
        document_embedding_layers.append(torch.mean(layer, dim=0).cpu().detach().numpy())
    # Convert to np array
    document_embedding_layers = np.asarray(document_embedding_layers)
    print("Document embedding shape:", document_embedding_layers.shape)

    return document_embedding_layers

def calculate_embeddings(name, method):
    # Parse XML
    abstractsInclude, tagsInclude = parseXML(name + '/' + name + 'Include.xml', 1)
    abstractsExclude, tagsExclude = parseXML(name + '/' + name + 'Exclude.xml', 0)
    df = pd.DataFrame(list(zip(tagsInclude + tagsExclude, abstractsInclude + abstractsExclude)), columns =['code', 'abstract'])

    df['embeddings'] = get_embeddings(df['abstract'], method)

    # Save dataframe to prevent recalculation
    df.to_pickle("./" + name + "/" + name + "-embeddings-" + method + ".pkl")



# Main function
if __name__ == "__main__":
    method = sys.argv[1]
    names = ["cellulitis", "copper", "search", "uti", "overdiagnosis"]

    # Initialize cuda
    print("Is CUDA avaliable:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for name in names:
        calculate_embeddings(name, method)

