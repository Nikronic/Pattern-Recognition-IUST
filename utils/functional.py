# %% import libraries
import pandas as pd
import numpy as np
import operator

from sentiment_analysis.utils import ios as IO

from typing import Dict, List, Tuple

import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import torch
from torch.nn import functional
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer

# %% constants and inits
PUNCTUATION = r""""#$%&()'*+-./<=>@[\]^_`{|}~"""


# %% functions
def generate_vocabulary(df: pd.DataFrame, save: bool = False, path: str = None,
                        names: Tuple[str] = ('text', 'label')) -> Dict:
    """
    Gets a pandas `DataFrame` and extracts all word occurrence in the entries of the `df` and build a dictionary
    that each word's number of repetition in whole texts.

    :param df: a pandas DataFrame with a column called 'text' to be analyzed.
    :param save: Whether or not saving dictionary as a file in disc
    :param path: Only applicable if `save` is True. The path which contains name of file to save vocab.
    :param names: A tuple of strings containing key values of "feature" and "label" in the dataframes
    :return: A dictionary of words and their counts
    """
    vocabulary = {}
    for reviews in df[names[0]]:
        for word in reviews:
            if word not in vocabulary.keys():
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1
    if save:
        with open(path, 'w+') as file:
            for k, v in zip(vocabulary.keys(), vocabulary.values()):
                file.write(str(k) + ' ' + str(v) + '\n')
    return vocabulary


def read_vocabulary(path: str) -> Dict:
    """
    Reads saved vocabulary at the path as a dictionary

    :param path: path to file of vocabs
    :return: A dictionary of words and their frequencies
    """

    vocab = {}
    with open(path) as file:
        line = file.readline()
        while line:
            key, value = line.split(' ')
            vocab[key] = int(value)
            line = file.readline()
    return vocab


def filter_vocabulary(vocab: Dict, threshold: float, bidirectional: bool):
    """
    Applies a filter regarding given arguments and return a subset of given vocab dictionary.

    Methods:
    1. threshold = -1: no cutting
    2. 0 < threshold <1 : cuts percentage=`threshold` of data
    3. 2 < threshold < len(vocab): cuts `threshold` number of words

    Note: in all cases output is sorted dictionary.

    :param vocab: A `Dict` of vocabularies where keys are words and values are int of repetition.
    :param threshold: a float or int number in specified format.
    :param bidirectional: Cutting process will be applied on most frequent and less frequent if `True`, else
            only less frequent.
    :return: A sorted `Dict` based on the values
    """
    # convert percentage to count
    if threshold <= 1:
        threshold = int(threshold) * 100
        threshold *= len(vocab)

    # sort vocab
    sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=operator.itemgetter(1))}

    if threshold < 0:
        return sorted_vocab

    # filter
    if bidirectional:
        sorted_vocab = dict(list(sorted_vocab.items())[threshold:-threshold])
        return sorted_vocab
    else:
        sorted_vocab = dict(list(sorted_vocab.items())[threshold:])
        return sorted_vocab


def build_bow(words: List[str], vocabulary: Dict):
    """
    Build Bag of Words model for given `List` of words regarding given `vocabulary` dictionary.

    :param words: A list of words to be converted to BOW
    :param vocabulary: A list of vocabulary as reference
    :return: A `List` of int
    """
    return np.array([1 if wv in words else 0 for wv in vocabulary])


def tokenizer(text: str, omit_stopwords: bool = True, root: str = 'lem') -> List[str]:
    """
    Tokenizes a review into sentences then words using regex

    :param text: a string to be tokenized
    :param omit_stopwords: whether remove stopwords or not using NLTK
    :param root: using `stemming`, `lemmatization` or none of them to get to the root of words ['lem', 'stem', 'none']
    :return: a `List` of string
    """

    # inits
    words = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    strings = re.sub(r'((<[ ]*br[ ]*[/]?[ ]*>)[ ]*)+', '\n', text).lower().split('\n')
    for line in strings:
        # remove punctuations
        line = line.translate(str.maketrans('', '', string.punctuation))
        words_of_line = re.sub(r'[\d+.]+', '', line.strip()).split(' ')  # remove numbers
        for w in words_of_line:
            if omit_stopwords:
                if w in stopwords.words('english'):
                    continue

            if root == 'none':  # no stemming or lemmatization
                pass
            elif root == 'lem':  # lemmatization using WordNet (DOWNLOAD NEEDED)
                w = lemmatizer.lemmatize(w)
            elif root == 'stem':  # stemming using Porter's algorithm
                w = stemmer.stem(w)
            else:
                raise Exception('root mode not defined. Use "lem", "stem" or "none"')
            if len(w) > 0:
                words.append(w)
    return words


def bert_tokenizer(text: str, pretrained: str = 'bert-large-uncased'):
    """
    Preprocesses the input in BERT convention and use BERT tokenizer to tokenize and convert words to
    IDs. In the end, we obtain hidden layers' values using PyTorch and pad all of them to have same
    sized tensors to apply average at the end of process for each review.

    example:

    input: "Hi Nik. How you doing?"
    1. preprocessed: "[CLS] hi nik . [SEP] how you doing ? [SEP]"
    2. tokenization
    3. convert to IDs
    4. model values
    5. padding
    6. average row-wise (each row corresponds to a sentence of a review)

    :param text: a string to be tokenized
    :param pretrained: Which pretrained model to use
    :return: A string in BERT  convention
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(pretrained)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = []
    segments_ids = []
    max_len = 0  # used for padding to take average over same sized tensors

    text = text.lower()
    strings = re.sub(r'((<[ ]*br[ ]*[/]?[ ]*>)[ ]*)+', ' ', text).split(r'. ')
    for idx, s in enumerate(strings):
        s = s.translate(str.maketrans('', '', PUNCTUATION))
        s = f'[CLS] {s}. [SEP]'
        words = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
        segments_ids.append([0] * len(words))
        tokens_tensor = torch.tensor([words])
        segments_tensors = torch.tensor([segments_ids[idx]])
        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            encoded_seq = outputs[0]  # 1*len(words)*hidden_size
            if len(encoded_seq[0]) > max_len:
                max_len = len(encoded_seq[0])
            encoded.append(encoded_seq[0])
    for idx, v in enumerate(encoded):
        encoded[idx] = functional.pad(input=v, pad=(0, 0, 0, max_len - v.shape[0]), mode='constant', value=0)
    output = encoded[0].unsqueeze(0)
    for idx, v in enumerate(encoded[1:]):
        output = torch.cat((output, v.unsqueeze(0)), dim=0)
    return output.mean(0)


def build_tf_idf(df: pd.DataFrame, custom_tokenizer: callable = tokenizer, names: Tuple[str] = ('text', 'label'),
                 **kwargs):
    """
    Builds TF-IDF embeddings for given corpus in a DataFrame

    :param df: a pandas DataFrame
    :param custom_tokenizer: A custom tokenizer function to replace default one
    :param names: A tuple of strings containing key values of "feature" and "label" in the dataframes
    :param kwargs: Optional parameters of object
    :return: A numpy ndarray object containing the features of given corpus
    """
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, **kwargs)
    x = vectorizer.fit_transform(df[names[0]].to_numpy())
    return x.toarray().astype(np.float16)


# %% test
if __name__ == '__main__':
    root = 'sentiment_analysis/data/sample/train/'
    df = IO.read_all_files(root, True)
    df['text'] = df['text'].apply(tokenizer)
    vocabs = generate_vocabulary(df, True, 'vocab.vocab')
    filtered_vocab = filter_vocabulary(vocabs, 50, False)
    df['text'] = df['text'].apply(build_bow, args=(filtered_vocab,))

    path = 'sentiment_analysis/data/sample/train/neg/3_4.txt'
    df_single = IO.read_single_file(path)
    df_single['text'] = df_single['text'].apply(tokenizer)

    dic = read_vocabulary('vocab.vocab')
    x = build_tf_idf(df=df, custom_tokenizer=tokenizer, analyzer='word', max_features=60000,
                     lowercase=False)
