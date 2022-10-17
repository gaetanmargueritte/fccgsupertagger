from copy import deepcopy
import torch
from torch import Tensor
from typing import Any, Dict, List, Tuple
from gensim.models import KeyedVectors
from collections import Counter
import random
import numpy as np

## useful tags added to data
START_TAG = "<START>"
END_TAG = "<END>"
PAD_TAG = "<PAD>"
UNK_TAG = "<UNK>"

# useful type alias
Dataset = List[List[Tuple[str, str, str, str, str]]]

# loads all words from word2vec into a dictionary of numpy arrays
# embed file is expected to be text file from Jean-Philippe Fauconnier, who kindly proposed usefeul ressources on his github page
# https://fauconnier.github.io/
def load_words_embedding_file(embed_file: str) -> Dict[str, np.ndarray]:
    print("Loading embed model (binary file) using Gensim...")
    model = KeyedVectors.load_word2vec_format(embed_file, binary=True, unicode_errors='strict')
    dict = {}
    for key in model.index_to_key:
        dict[key.lower()] = np.copy(model.word_vec(key))
    print("Done!")
    return dict

# loads FrenchCCGBank's data. This is a french ccg corpus created by Le Ngoc Luyen on a .txt format, generated using his code available on github.
# https://github.com/lengocluyen/FrenchCCGBank
#### dataset_file content
## corpus = [
##           [sentence01:[
##                       word1: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
##                       word2: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
##                       .....]
##           ]
##           [sentence02:[
##                       word1: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
##                       word2: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
##                       .....]
##           ]
##           ......
##         ]
def load_dataset(dataset_file: str) -> Dataset:
    print("Loading dataset (txt file)...")
    dataset: Dataset = []
    # 0:id, 1:words, 2:lemma, 3:postag, 4:deprel, 5:ccgtag, 6:deprel_id
    with open(dataset_file, "r") as f:
        sentence = []
        for line in f:
            if len(line) > 1:
                l = line.rstrip().split('\t')
                # word, lemma, xpostag, deprel, ccgtag
                word_tuple = (l[1].strip(), l[2].strip(), l[4].strip(), l[6].strip(), l[8].strip())
                sentence.append(word_tuple)
            elif len(sentence) > 0: # end of bloc separated by empty line (only \n)
                dataset.append(sentence)
                sentence = []
    print("Done!")
    return dataset

# sorts by frequency a dictionary
def sort_dict(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    # lambda key: sorts by frequency then by alphanumerical order
    sorted_items = sorted(dict.items(), key=lambda x: (-x[1], x[0]))
    item2id = {item[0]: id for id, item in enumerate(sorted_items)}
    id2item = {id: item[0] for id, item in enumerate(sorted_items)}
    return item2id, id2item   

# creates item2id and id2item, sorted by frequency
def map_dictionary(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    # add pad and unk so that they are the 2 first values
    dict[PAD_TAG] = 100001
    dict[UNK_TAG] = 100000
    return sort_dict(dict)
        

# maps tags in the same fashion as map_dictionary, but adds start and end tags for computation
def map_tags(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    dict[START_TAG] = -1
    dict[END_TAG] = -2
    dict[PAD_TAG] = -3
    return sort_dict(dict)

# build vocabulary for each feature (word, lemma, postag, deprel) and ccg outputs from dataset
def build_vocab(dataset: List[List[Tuple[str, str, str, str, str,]]]) -> List[Tuple[Dict[str, int], Dict[int, str]]]:
    print("Building vocabulary...")
    words, lemmas, postags, deprels, ccgs = [], [], [], [], []
    for sentence in dataset:
        for word, lemma, postag, deprel, ccg in sentence:
            words.append(word.lower())
            lemmas.append(lemma)
            postags.append(postag)
            deprels.append(deprel)
            ccgs.append(ccg)
    # dict(Counter(x)) will create a dictionary of each feature with the number of times it appears in the dataset
    words = dict(Counter(words))
    lemmas = dict(Counter(lemmas))
    postags = dict(Counter(postags))
    deprels = dict(Counter(deprels))
    ccgs = dict(Counter(ccgs))
    words2id, id2words = map_dictionary(words)
    lemmas2id, id2lemmas = map_dictionary(lemmas)
    postags2id, id2postags = map_dictionary(postags)
    deprels2id, id2deprels = map_dictionary(deprels)
    ## we need START and END tags for ccg tags
    ccgs2id, id2ccgs = map_tags(ccgs)
    print("Done!")
    return [(words2id, id2words), (lemmas2id, id2lemmas), (postags2id, id2postags), (deprels2id, id2deprels), (ccgs2id, id2ccgs)]

# adds words from pre_words_embedding to vocabulary
def enhance_vocabulary(words2id: Dict[str, int], id2words: Dict[int, str], pre_words_embedding: Dict[str, Tensor]) -> Tuple[Dict[str, int], Dict[int, str]]:
    print("Enhancing vocabulary using pre_trained embedding...")
    for w in pre_words_embedding.keys():
        word = w.lower()
        if word not in words2id.keys():
            id = len(words2id)
            words2id[word] = id
            id2words[id] = word
    print("Done!")
    return (words2id, id2words)

# splits randomly into 3 subsets (train, test and validation) a given dataset
def shuffle_and_split(dataset: Dataset, vratio: float, tratio: float, seed: int) -> Dataset:
    random.seed(seed)
    random.shuffle(dataset)
    test_index = int(len(dataset) * tratio)
    test_dataset = dataset[:test_index]
    train_dataset = dataset[test_index:]
    random.shuffle(train_dataset)
    validation_index = int(len(train_dataset) * vratio)
    validation_dataset = train_dataset[:validation_index]
    train_dataset = train_dataset[validation_index:]
    return (train_dataset, validation_dataset, test_dataset)

# creates proper data from a given dataset, using the vocabulary.
# transforms each feature into its proper id in order to feed it later in the neural model
# returns a list (dataset) of dictionaries (sentences) of lists (words)
def get_data_from_dataset(dataset: Dataset, words2id: Dict[str, int], lemmas2id: Dict[str, int], postags2id: Dict[str, int], deprels2id: Dict[str, int], ccgs2id: Dict[str, int]) -> List[Dict[str, List[str]]]:
    print("Creating data from dataset...")
    data = []
    for sentence in dataset:
        sentence_text = []
        words_id = []
        lemmas_id = []
        postags_id = []
        deprels_id = []
        ccgs_id = []
        for word, lemma, postag, deprel, ccg in sentence:
            sentence_text.append(word)
            wordl = word.lower()
            # we can expect typo or error in words and lemma
            words_id.append(words2id[wordl] if wordl in words2id else UNK_TAG)
            lemmas_id.append(lemmas2id[lemma] if lemma in lemmas2id else UNK_TAG)
            postags_id.append(postags2id[postag])
            deprels_id.append(deprels2id[deprel])
            ccgs_id.append(ccgs2id[ccg])
        data.append({
            "sentence_text": sentence_text,
            "words_id": words_id,
            "lemmas_id": lemmas_id,
            "postags_id": postags_id,
            "deprels_id": deprels_id,
            "ccgs_id": ccgs_id
        })
    print("Done!")
    return data

# pads a given sequence in parameter until it reaches sequence_length. 
def pad_sequence(sequence: List[int], sequence_length: int, pad_value: int) -> List[int]:
    padded_sequence = [pad_value for _ in range(sequence_length)]
    padded_sequence[:len(sequence)] = sequence[:]
    return padded_sequence
