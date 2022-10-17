# Callable file reading a discourse provided in french via input file
# outputs a file with the same discourse supertagged with CCG
# Tags are generated by a trained model which is either trained or to-be-trained

import pickle
import sys
import os
import argparse
import spacy
import subprocess
import time
import torch
from typing import Dict, List, Any, Tuple
from nltk.parse import malt
from spacy_lefff import POSTagger, LefffLemmatizer
from spacy.language import Language
from torch import Tensor
from model import Tagger
from utils import PAD_TAG, UNK_TAG, pad_sequence, load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUTS_FOLDER = "./outputs/"

if not os.path.exists(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)

DICT_COARSE_POSTAG = {
    "ADJ": "A",
    "ADJWH": "A",
    "ADV": "ADV",
    "ADVWH": "ADV",
    "CC": "C",
    "CLO": "CL",
    "CLR": "CL",
    "CLS": "CL",
    "CS": "C",
    "DET": "D",
    "DETWH": "D",
    "ET": "ET",
    "I": "I",
    "NC": "N",
    "NPP": "N",
    "P": "P",
    "P+D": "P",
    "P+PRO": "P",
    "PONCT": "PONCT",
    "PRO": "PRO",
    "PROREL": "PRO",
    "PROWH": "PRO",
    "V": "V", 
    "VIMP": "V", 
    "VINF": "V",
    "VPP": "V",
    "VPR": "V",
    "VS": "V"
}

@Language.factory('melt_tagger')  
def create_melt_tagger(nlp, name):
    return POSTagger()

@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer(after_melt=True, default=True)

# takes a list of nltk Doc and generates a temporary file under conll format
# used in order to execute french maltparser and retreive sentence depedencies
def to_conll(sentences: List[Any]) -> None:
    with open(OUTPUTS_FOLDER + "inputs.conll", "w") as f:
        for s in sentences:
            i = 1
            for doc in sentences[s]:
                f.write(f"{i}\t{doc.text}\t{doc._.lefff_lemma}\t{DICT_COARSE_POSTAG[doc._.melt_tagger]}\t{doc._.melt_tagger}\t_\n")
                i += 1
            f.write("\n")

def get_id(dictionary: Dict, obj: str) -> int:
    return dictionary[obj] if obj in dictionary else dictionary[UNK_TAG]

def fetch_and_format_data(input_file: str, model_data: Dict[str, Dict], max_sequence_length: int) -> Tuple[Tensor]:
    print("Analysing input sentences.")
    print("Language - [French]")
    print("POS tags - [MElt]")
    pos_lem_tagger = spacy.load('fr_core_news_sm')
    pos_lem_tagger.add_pipe('melt_tagger', after="parser")
    # Lefff French lemmatizer. Tests are required in order to prove whether it is more accurate or not than spacy-native French lemmatizer.
    pos_lem_tagger.add_pipe('french_lemmatizer', after="melt_tagger")
    inputs: Dict[str, List[Any]] = {}
    with open(input_file, "r") as f:
        lines = f.readlines()
        for x in lines:
            inputs[x.strip()] = pos_lem_tagger(x.strip())
    to_conll(inputs)
    in_filename = OUTPUTS_FOLDER + "inputs.conll"
    out_filename = OUTPUTS_FOLDER + "inputs_with_depedency.conll"
    print(f"Saving input sentences analysis at location {in_filename}.")
    print("Adding depedency analysis")
    print("Parser - [MaltParser]")
    mp = malt.MaltParser('maltparser-1.7', "fremalt-1.7.mco")
    subprocess.run(mp.generate_malt_command(os.path.abspath(in_filename), 
                                            outputfilename=os.path.abspath(out_filename), mode="parse"), 
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.STDOUT)
    
    words2id = model_data["words2id"]
    lemmas2id = model_data["lemmas2id"]
    postags2id = model_data["postags2id"]
    deprels2id = model_data["deprels2id"]
    sentences: List[Dict] = []
    # french maltparser line is organised as such:
    # id, word, lemma, coarse_postag, postag, _, head, deprel, _, _
    sentences_text, sentences_masks, sentences_words, sentences_lemmas, sentences_postags, sentences_deprels = [], [], [], [], [], []
    with open(out_filename, "r") as f:
        words, words_id, lemmas, postags, deprels = [], [], [], [], []
        for line in f:
            if len(line) > 1:
                l = line.rstrip().split('\t')
                words.append(l[1])
                words_id.append(get_id(words2id, l[1].lower()))
                lemmas.append(get_id(lemmas2id, l[2]))
                postags.append(get_id(postags2id, l[4]))
                deprels.append(get_id(deprels2id, l[7].replace('_', '.')))
            elif len(line) > 0: #\n
                sentences_text.append(' '.join(words))
                sentences_masks.append(pad_sequence([1 for _ in range(len(words))], max_sequence_length, 0))
                sentences_words.append(pad_sequence(words_id, max_sequence_length, words2id[PAD_TAG]))
                sentences_lemmas.append(pad_sequence(lemmas, max_sequence_length, lemmas2id[PAD_TAG]))
                sentences_postags.append(pad_sequence(postags, max_sequence_length, postags2id[PAD_TAG]))
                sentences_deprels.append(pad_sequence(deprels, max_sequence_length, deprels2id[PAD_TAG]))
                words, words_id, lemmas, postags, deprels = [], [], [], [], []
    masks_batch = torch.tensor(sentences_masks, device=device, dtype=torch.long)
    words_batch = torch.tensor(sentences_words, device=device, dtype=torch.long)
    lemmas_batch = torch.tensor(sentences_lemmas, device=device, dtype=torch.long)
    postags_batch = torch.tensor(sentences_postags, device=device, dtype=torch.long)
    deprels_batch = torch.tensor(sentences_deprels, device=device, dtype=torch.long)
    return (sentences_text, words_batch, lemmas_batch, postags_batch, deprels_batch, masks_batch)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="French CCG supertagger [main].")
    parser.add_argument("-m", "--model", default="", type=str, help="Trained model weights dictionary file")
    parser.add_argument("-d", "--data", default="", type=str, help="Trained model data file (.pickle file)")
    parser.add_argument("-i", "--input", default="", type=str, help="Discourse formulated in French natural language.")
    parser.add_argument("-o", "--output", default="tagged_discourse.txt", type=str, help="tagged discourse file (default: tagged_discourse.txt")
    parser.add_argument("--max-sequence-length", default=120, type=int, help="Max sequence length (default: 120)")
    parameters = parser.parse_args()
    model_file: str = parameters.model
    data_file: str = parameters.data
    input_file: str = parameters.input
    output_file: str = parameters.output
    max_sequence_length: int = parameters.max_sequence_length


    if not os.path.exists(model_file) or not os.path.isfile(model_file):
        print("Model weights dictionary file invalid. It can be generated by calling the file train.py.")
        print("Exiting program.")
        sys.exit(1)
    if not os.path.exists(data_file) or not os.path.isfile(data_file):
        print("Model data file invalid. It can be generated by calling the file train.py.")
        print("Exiting program.")
        sys.exit(1)
    if not os.path.exists(input_file) or not os.path.isfile(input_file):
        print("Input file invalid.")
        print("Exiting program.")
        sys.exit(1)
    
    start = time.time()
    print("**** Model loading... ****")
    with open(data_file, 'rb') as f:
        model_data = pickle.load(f)
    model = Tagger(None, model_data["lemma_embed_size"], model_data["postag_embed_size"], model_data["deprel_embed_size"], model_data["nb_output_class"],
                    model_data["ccgs2id"], model_data["vocab_size"], hidden_size=model_data["hidden_size"], max_sequence_length=model_data["max_sequence_length"])
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    print("**** Done. ****")

    with torch.no_grad():
        print("**** Starting data preprocessing... ****")
        batch_text, batch_words, batch_lemmas, batch_postags, batch_deprels, batch_masks = fetch_and_format_data(input_file, model_data, max_sequence_length)
        print("**** Done. ****")
        predictions = model(batch_words, batch_lemmas, batch_postags, batch_deprels, batch_masks)
        end = time.time()
        print(f"Predicted {len(batch_text)} sentences in {end-start}s.")
        id2ccgs = model_data["id2ccgs"]
        for i in range(len(batch_text)):
            for t, p in zip(batch_text[i].split(), predictions[i]):
                print(f"{t} - {id2ccgs[p]}")
            print("------------")
