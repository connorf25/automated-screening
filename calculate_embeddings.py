import pandas as pd
import numpy as np
from xml.dom import minidom

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Doc2Vec
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from transformers import BertModel, BertTokenizer, BloomTokenizerFast, BloomModel
import torch

#Timing
import timeit

# MAIN function used to load xml and return embeddingsP
def get_dataframe_with_embeddings(dataset_name, model_name):
    # Initialize cuda
    print("Is CUDA avaliable:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Parse XML and build dataset
    abstractsInclude, tagsInclude = parseXML("datasets/" + dataset_name + '/' + dataset_name + 'Include.xml', 1)
    abstractsExclude, tagsExclude = parseXML("datasets/" + dataset_name + '/' + dataset_name + 'Exclude.xml', 0)
    df = pd.DataFrame(list(zip(tagsInclude + tagsExclude, abstractsInclude + abstractsExclude)), columns =['code', 'abstract'])

    # Get embeddings
    df['embedding'] = get_embeddings(df['abstract'], model_name, device)

    # Return dataframe
    return df

# Function to decide what embeddings to return
def get_embeddings(abstracts, model_name, device):
    if "bloom" in model_name:
        if "560m" in model_name:
            tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
            model = BloomModel.from_pretrained("bigscience/bloom-560m", output_hidden_states=True).to(device)
        elif "1b7" in model_name:
            tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
            model = BloomModel.from_pretrained("bigscience/bloom-1b7", output_hidden_states=True).to(device)
        else:
            tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
            model = BloomModel.from_pretrained("bigscience/bloom", output_hidden_states=True).to(device)
        return calculate_embeddings_huggingface(model_name, abstracts, tokenizer, model, device)

    elif "scibert" in model_name:
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True).to(device)
        return  calculate_embeddings_huggingface(model_name, abstracts, tokenizer, model, device)

    elif "bow" in model_name:
        return calculate_embeddings_bow(abstracts)

    elif "tfidf" in model_name:
        return calculate_embeddings_tfidf(abstracts)

    elif "doc2vec" in model_name:
        return calculate_embeddings_doc2vec(abstracts)

    else:
        print("Invalid model name")
        return

#Function to parse endnote xml file
def parseXML(filename, isInclude):
    abstracts = []
    tags = []
    xmldoc = minidom.parse(filename)
    for node in xmldoc.getElementsByTagName('abstract'):
        abstract = node.getElementsByTagName('style')[0].firstChild.nodeValue
        abstracts.append(abstract)
        tags.append(isInclude)
    return abstracts, tags

# Function to return embeddings for huggingface models (bloom and scibert)
def calculate_embeddings_huggingface(model_name, abstracts, tokenizer, model, device):
    print('tokenizer is type:', type(tokenizer))
    print('model is type:', type(model))

    embeddings = []
    length = len(abstracts.tolist())
    index = 0

    start = timeit.default_timer()
    for sentence in abstracts.tolist():
        index += 1
        doc_emb = get_embedding_huggingface(model_name, model, tokenizer, sentence, device)
        print("Document embedding shape:", doc_emb.shape)
        embeddings.append(doc_emb)

        if (index/length*100) < 1:
            expected_time = "Calculating..."
        else:
            time_perc = timeit.default_timer()
            expected_time = np.round( (time_perc-start) /(index/length) /60,2)

        print(index, length)
        print(expected_time)
    return embeddings

# Get huggingface embedding as np array
def get_embedding_huggingface(model_name, model, tokenizer, text, device):
    # Encode with special tokens ([CLS] and [SEP], returning pytorch tensors
    padding = False
    # 1024 set to prevent cuda memory error
    max_length = 1024
    # Add padding for concat so all inputs are same length
    if "concat" in model_name:
        max_length = 512
        padding = "max_length"
    # Limit scibert max length to prevent error
    if "scibert" in model_name:
        max_length = 512
    # Tokenize text
    encoded_dict = tokenizer.encode_plus(
        text,
        truncation=True,
        max_length=max_length,
        padding=padding,
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

    # Return embeddings in correct format
    if "average" in model_name:
        last_layers = hidden_states[-5:]
        document_embedding_layers = []
        for layer in last_layers:
            # Take first item from batch (only one document per batch anyway)
            layer = layer[0]
            # Calculate average of all word embeddings to get document embedding
            document_embedding_layers.append(torch.mean(layer, dim=0).cpu().detach().numpy())
        # Take average of last 5 layers
        document_embedding_layers = np.asarray(document_embedding_layers)
        return np.average(document_embedding_layers, axis=0)

    elif "concat" in model_name:
        last_layer = hidden_states[-1]
        last_layer_of_first_document = last_layer[0]
        return torch.flatten(last_layer_of_first_document).cpu().detach().numpy()

    else: # Take avg of last layer
        last_layer = hidden_states[-1]
        last_layer_of_first_document = last_layer[0]
        return torch.mean(last_layer_of_first_document, dim=0).cpu().detach().numpy()

def calculate_embeddings_bow(abstracts):
    countVectorizer = CountVectorizer(stop_words="english", max_features=768)
    embeddings = countVectorizer.fit_transform(abstracts).todense().tolist()
    return embeddings

def calculate_embeddings_tfidf(abstracts):
    tfidfVectorizer = TfidfVectorizer(stop_words="english", max_features=768)
    embeddings = tfidfVectorizer.fit_transform(abstracts).todense().tolist()
    return embeddings

def calculate_embeddings_doc2vec(abstracts):
    embeddings = []
    processed_abstracts=[]
    for abstract in abstracts:
        # Cleaning the text
        processed_abstract = abstract.lower()
        processed_abstract = re.sub('[^a-zA-Z]', ' ', processed_abstract )
        processed_abstract = re.sub(r'\s+', ' ', processed_abstract)

        # Preparing the dataset
        tokenized_abstract = nltk.word_tokenize(processed_abstract)

        # Removing Stop Words
        tokenized_abstract = [w for w in tokenized_abstract if w not in nltk.corpus.stopwords.words('english')]
        processed_abstracts.append(tokenized_abstract)

    # Convert to TaggedDocument for efficiency
    tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_abstracts)]
    model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=5)

    for processed_abstract in processed_abstracts:
        embeddings.append(model.infer_vector(processed_abstract))

    return embeddings
