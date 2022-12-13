# file used to parse and format different possible input data.


import subprocess
import torch
import os
import spacy
import subprocess
from typing import Dict, List, Any, Tuple
from nltk.parse import malt
from spacy_lefff import POSTagger, LefffLemmatizer
from spacy.language import Language, Doc
from torch import Tensor
from utils import PAD_TAG, UNK_TAG, pad_sequence, Dataset

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
    "PREF": "PREF",
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
    "VS": "V",
}



class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
           spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


@Language.factory("melt_tagger")
def create_melt_tagger(nlp, name):
    return POSTagger()


@Language.factory("french_lemmatizer")
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer(after_melt=True, default=True)


def to_conll(dataset: Dataset, filename: str) -> None:
    with open(filename, "w") as f:
        for words, lemmas, postags, deprels, _ in dataset:
            i = 1
            for w, l, p, d in zip(words, lemmas, postags, deprels):
                f.write(
                    f"{i}\t{w}\t{l}\t{DICT_COARSE_POSTAG[p]}\t{p}\t{d}\n"
                )
            f.write("\n")



def get_id(dictionary: Dict, obj: str) -> int:
    return dictionary[obj] if obj in dictionary else dictionary[UNK_TAG]

def load_tlgbank_dataset_camembert(dataset_file: str) -> Any:
    print(f"Loading dataset from {dataset_file} for CamemBERTTager architecture...")
    # only words and tags
    data = []
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            elements = l.split(' ')  
            words_text = []
            words_pos = []
            words_category = []
            for words in elements:
                w = words.split('|')
                words_text.append(w[0].strip())
                words_pos.append(w[1].split('-')[0].split("+")[0].strip())
                words_category.append(w[2].strip())
            data.append((words_text, words_pos, words_category))
    print("Done!")
    return data

def load_tlgbank_dataset(dataset_file: str) -> Dataset:
    print(f"Loading dataset from {dataset_file}...")
    data = []
    pos_lem_tagger = spacy.load("fr_core_news_sm")
    # input data is already tokenized.
    # cases like "aujourd'hui" might be separated. We don't want that.
    pos_lem_tagger.tokenizer = WhitespaceTokenizer(pos_lem_tagger.vocab)
    pos_lem_tagger.add_pipe('melt_tagger', after='parser')
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            elements = l.split(' ')
            words_text = []
            words_pos = []
            words_category = []
            for words in elements:
                w = words.split('|')
                words_text.append(w[0].strip())
                words_pos.append(w[1].split('-')[0].split("+")[0].strip())
                words_category.append(w[2].strip())
            text = ' '.join(words_text)
            doc = pos_lem_tagger(text)
            words_lemma = [d.lemma_ for d in doc]
            data.append((words_text, words_lemma, words_pos, ['_' for _ in range(len(words_text))], words_category))
    conll_filename = OUTPUTS_FOLDER + "tlgbank_unprocessed.conll"
    dataset_filename = OUTPUTS_FOLDER + "tlgbank.conll"
    try:
        to_conll(data, conll_filename)
        mp = malt.MaltParser("maltparser-1.7", "fremalt-1.7.mco")
        subprocess.run(
            mp.generate_malt_command(
                os.path.abspath(conll_filename),
                outputfilename=os.path.abspath(dataset_filename),
                mode="parse"
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
    finally:
        os.remove(conll_filename)

    dataset: Dataset = []
    with open(dataset_filename, 'r') as f:
        lines = f.readlines()
        words, lemmas, postags, _, categories = data.pop(0)

        sentence = []
        i = 0
        for l in lines:
            if len(l) > 1:
                assert l.split('\t')[1] in words
                sentence.append((words[i], lemmas[i], postags[i], l.split('\t')[7], categories[i]))
                i += 1
            elif len(sentence) > 0:
                if len(sentence) < 120:
                    dataset.append(sentence)
                    #for w, l, p, d, c in sentence:
                    #    print(f"{w}\t{l}\t{p}\t{d}\t{c}")
                # Il faut vraiment changer ça. Tout le processing de "data" au dessus doit se faire sous le même format que dataset
                # todo, wip, tout ce que tu veux
                # refactoring to be done
                if len(data) > 0:
                    words, lemmas, postags, _, categories = data.pop(0)
                sentence = []
                i = 0
    print("Done!")
    return dataset
    

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
def load_fccgbank_dataset(dataset_file: str) -> Dataset:
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


def fetch_data_camembert(
    input_file: str
) -> Tensor:
    print("Utilizing CamemBERTTagger. Applying whitespace tokenization.")
    sentences = []
    with open(input_file, "r") as f:
        lines = f.readlines()
        for x in lines:
            sentences.append(x.split())
    return sentences

def fetch_and_format_data(
    input_file: str, model_data: Dict[str, Dict], max_sequence_length: int
) -> Tuple[Tensor]:
    print("Analysing input sentences.")
    print("Language - [French]")
    print("POS tags - [MElt]")
    pos_lem_tagger = spacy.load("fr_core_news_sm")
    pos_lem_tagger.add_pipe("melt_tagger", after="parser")
    # Lefff French lemmatizer. Tests are required in order to prove whether it is more accurate or not than spacy-native French lemmatizer.
    #pos_lem_tagger.add_pipe("french_lemmatizer", after="melt_tagger")
    inputs: List[Any] = {}
    with open(input_file, "r") as f:
        lines = f.readlines()
        for x in lines:
            inputs.append(pos_lem_tagger(x.strip()))
    to_conll(inputs)
    in_filename = OUTPUTS_FOLDER + "inputs_unprocessed.conll"
    out_filename = OUTPUTS_FOLDER + "inputs.conll"
    print(f"Saving input sentences analysis at location {in_filename}.")
    print("Adding depedency analysis")
    print("Parser - [MaltParser]")
    mp = malt.MaltParser("maltparser-1.7", "fremalt-1.7.mco")
    subprocess.run(
        mp.generate_malt_command(
            os.path.abspath(in_filename),
            outputfilename=os.path.abspath(out_filename),
            mode="parse",
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    words2id = model_data["words2id"]
    lemmas2id = model_data["lemmas2id"]
    postags2id = model_data["postags2id"]
    deprels2id = model_data["deprels2id"]
    sentences: List[Dict] = []
    # french maltparser line is organised as such:
    # id, word, lemma, coarse_postag, postag, _, head, deprel, _, _
    (
        sentences_text,
        sentences_masks,
        sentences_words,
        sentences_lemmas,
        sentences_postags,
        sentences_deprels,
    ) = ([], [], [], [], [], [])
    with open(out_filename, "r") as f:
        words, words_id, lemmas, postags, deprels = [], [], [], [], []
        for line in f:
            if len(line) > 1:
                l = line.rstrip().split("\t")
                words.append(l[1])
                words_id.append(get_id(words2id, l[1].lower()))
                lemmas.append(get_id(lemmas2id, l[2]))
                postags.append(get_id(postags2id, l[4]))
                deprels.append(get_id(deprels2id, l[7].replace("_", ".")))
            elif len(line) > 0:  # \n
                sentences_text.append(" ".join(words))
                sentences_masks.append(
                    pad_sequence([1 for _ in range(len(words))], max_sequence_length, 0)
                )
                sentences_words.append(
                    pad_sequence(words_id, max_sequence_length, words2id[PAD_TAG])
                )
                sentences_lemmas.append(
                    pad_sequence(lemmas, max_sequence_length, lemmas2id[PAD_TAG])
                )
                sentences_postags.append(
                    pad_sequence(postags, max_sequence_length, postags2id[PAD_TAG])
                )
                sentences_deprels.append(
                    pad_sequence(deprels, max_sequence_length, deprels2id[PAD_TAG])
                )
                words, words_id, lemmas, postags, deprels = [], [], [], [], []
    masks_batch = torch.tensor(sentences_masks, dtype=torch.long)
    words_batch = torch.tensor(sentences_words, dtype=torch.long)
    lemmas_batch = torch.tensor(sentences_lemmas, dtype=torch.long)
    postags_batch = torch.tensor(sentences_postags, dtype=torch.long)
    deprels_batch = torch.tensor(sentences_deprels, dtype=torch.long)
    return (
        sentences_text,
        words_batch,
        lemmas_batch,
        postags_batch,
        deprels_batch,
        masks_batch,
    )

if __name__ == "__main__":
    input_file = "./data/tlgbankRaw.txt"
    load_tlgbank_dataset(input_file)