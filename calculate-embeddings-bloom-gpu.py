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
    itemlist = xmldoc.getElementsByTagName('abstract')
    for node in xmldoc.getElementsByTagName('abstract'):
        abstract = node.getElementsByTagName('style')[0].firstChild.nodeValue
        abstracts.append(abstract)
        tags.append(isInclude)
    return abstracts, tags

# Find bloom embeddings
def get_bloom_embeddings(abstracts):
    # Load scibert
    bloom_tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
    bloom_model = BloomModel.from_pretrained("bigscience/bloom-350m", output_hidden_states=True).to(device)

    print('bloom_tokenizer is type:', type(bloom_tokenizer))
    print('bloom_model is type:', type(bloom_model))

    embeddings = []
    length = len(abstracts.tolist())
    index = 0

    start = timeit.default_timer()
    for sentence in abstracts.tolist():
        clear_output(wait=True)
        index += 1
        sen_emb = get_bloom_embedding(bloom_model, bloom_tokenizer, sentence)
        embeddings.append(sen_emb)

        stop = timeit.default_timer()

        if (index/length*100) < 1:
            expected_time = "Calculating..."

        else:
            time_perc = timeit.default_timer()
            expected_time = np.round( (time_perc-start) /(index/length) /60,2)

        print(index, length)
        print(expected_time)
    return embeddings

def get_bloom_embedding(model, tokenizer, text):

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

    # Run through Bloom
    with torch.no_grad():
        outputs = model(input_ids)
        # Extract hidden states
        hidden_states = outputs.hidden_states


    # Select the word embeddings on the last 5 layers
    token_vecs = hidden_states[-5:][0]
    print("Token vecs:", token_vecs.shape)
    # Calculate average of token vectors/word embeddings
    document_embedding = torch.mean(token_vecs, dim=0)
    # Convert to np array
    document_embedding = document_embedding.cpu().detach().numpy()
    print("Document embedding shape:", document_embedding.shape)

    return document_embedding

def calculate_embeddings(name, method):
    # Parse XML
    abstractsInclude, tagsInclude = parseXML(name + '/' + name + 'Include.xml', 1)
    abstractsExclude, tagsExclude = parseXML(name + '/' + name + 'Exclude.xml', 0)
    df = pd.DataFrame(list(zip(tagsInclude + tagsExclude, abstractsInclude + abstractsExclude)), columns =['code', 'abstract'])

    if method == "bloom-350m":
        df['embeddings'] = get_bloom_embeddings(df['abstract'])

    # Save dataframe to prevent recalculation
    # df.to_pickle("./" + name + "/" + name + "-embeddings-" + method + ".pkl")

# Main function
if __name__ == "__main__":
    method = sys.argv[1]
    # names = ["cellulitis", "copper", "search", "uti", "overdiagnosis"]
    names = ["copper"]

    # Initialize cuda
    print("Is CUDA avaliable:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for name in names:
        calculate_embeddings(name, method)

