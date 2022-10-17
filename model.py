import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
from torchcrf import CRF
from typing import Tuple, Any, Dict, List


def init_lstm(lstm: nn.Module) -> None:
    for param in lstm.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)

def init_correl(feature: List[int], hidden_size: int, dropout: float, num_layers: int, output_dim: int) -> List[nn.Module]:
    lstm, tdd = [], []
    for i in range(len(feature)):
        dim1 = feature[i]
        for j in range(i + 1, len(feature)):

            dim2 = feature[j]

            feature_lstm = nn.LSTM(
            input_size = dim1 + dim2,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = True,
            batch_first = True
            )
            init_lstm(feature_lstm)
            
            # mimics time distributed dense layer of keras
            feature_tdd = nn.Linear(
                in_features=hidden_size*2,
                out_features=output_dim
            )

            init.xavier_normal_(feature_tdd.weight.data)
            init.normal_(feature_tdd.bias.data)
            
            lstm.append(feature_lstm)
            tdd.append(feature_tdd)
    
    return lstm, tdd

## we consider 4 features for each word: word, lemma, postage, deprel
class Tagger(nn.Module):
    def __init__(
        self, 
        pre_words_embedding: Any,
        lemma_embed_size: int,
        postag_embed_size: int,
        deprel_embed_size: int,
        nb_output_class: int,
        dict2id: Dict[str, int],
        vocab_size: int,
        dropout: float = 0.1,
        hidden_size: int = 128,
        max_sequence_length: int = 120, 
        max_embedding_dim: int = 200,
        test_split_ratio: float = 0.2,
        batch_size: int = 32,
        max_word_length: int = 30,
        num_layers: int = 2,
        use_gpu: bool = True) -> None:
        super().__init__()

        self._lemma_embed_size = lemma_embed_size
        self._postag_embed_size = postag_embed_size
        self._deprel_embed_size = deprel_embed_size
        self._nb_output_class = nb_output_class
        self._dict2id = dict2id
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._vocab_size = vocab_size
        self._max_embedding_dim = max_embedding_dim
        self._test_split_ratio = test_split_ratio
        self._batch_size = batch_size
        self._max_word_length = max_word_length
        self._num_layers = num_layers
        self._device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        print("Using device:", self._device)

        ### creating layers
        ## 1 embedding layer per feature
        # words
        self._words_embed = nn.Embedding(self._vocab_size, self._max_embedding_dim)
        if pre_words_embedding is not None:
            self._words_embed.weight = nn.Parameter(torch.FloatTensor(pre_words_embedding))
            self._words_embed.weight.requires_grad = False # do not re-learn pre_words_embedding
        else:
            torch.nn.init.xavier_uniform_(self._words_embed.weight)
        # lemma
        self._lemma_embed = nn.Embedding(self._lemma_embed_size, self._max_embedding_dim)
        torch.nn.init.xavier_uniform_(self._lemma_embed.weight)
        # postag
        self._postag_embed = nn.Embedding(self._postag_embed_size, self._postag_embed_size)
        torch.nn.init.xavier_uniform_(self._postag_embed.weight)
        # deprel
        self._deprel_embed = nn.Embedding(self._deprel_embed_size, self._deprel_embed_size)
        torch.nn.init.xavier_uniform_(self._deprel_embed.weight)
        # usual dropout layer
        self._dropout_layer = nn.Dropout(self._dropout)

        ## lstm layers learning correlation between 2 features.
        # stocking each feature dim to prepare lstm input
        self._features_dim = [self._max_embedding_dim, self._max_embedding_dim, self._postag_embed_size, self._deprel_embed_size]
        self._correl_lstm, self._correl_tdd = init_correl(self._features_dim, self._hidden_size, self._dropout, self._num_layers, self._nb_output_class)
        ## first bilstm layer after concat of all embeds of a sequence and the output of correl lstms
        size_correl_output = self._nb_output_class * len(self._correl_lstm)
        self._lstm = nn.LSTM(
            input_size = max_embedding_dim * 2 + self._deprel_embed_size + self._postag_embed_size + size_correl_output,
            hidden_size = self._hidden_size,
            num_layers = self._num_layers,
            dropout = self._dropout,
            bidirectional = True,
            batch_first = True
        ) 

        init_lstm(self._lstm)

        # final layer and crf
        self._hidden2tags = nn.Linear(self._hidden_size * 2, self._nb_output_class)

        init.xavier_normal_(self._hidden2tags.weight.data)
        init.normal_(self._hidden2tags.bias.data)

        # relu activation to trim out negative probabilities
        self._activation = nn.ReLU()

        self._crf = CRF(self._nb_output_class, batch_first=True)

    
    #negative loss likelihood computation
    def nll(self, sentence: Tensor, lemma: Tensor, postag: Tensor, deprel: Tensor, real_tags: Tensor, masks: Tensor) -> Tensor:
        predictions = self.get_sentence_prediction_emission(sentence, lemma, postag, deprel)
        nll = -self._crf(predictions, real_tags, masks.bool())
        return nll

    # outputs the correlation between each pair of embedded features and merges them in a tuple with said features
    def _merge_feature_correlations(self, embeds_list: List[Tensor], sentence_length: int) -> Tuple[Tensor]:
        correlations = embeds_list[:]
        correl_ptr = 0
        for i in range(len(embeds_list)):
            for j in range(i+1, len(embeds_list)):
                correl_tensor = torch.cat((embeds_list[i], embeds_list[j]), 2)
                correl_lstm = self._correl_lstm[correl_ptr].to(self._device)
                correl_tdd = self._correl_tdd[correl_ptr].to(self._device)
                
                correl_ptr += 1

                feature_correl, _ = correl_lstm(correl_tensor)
                feature_correl = self._activation(feature_correl)
                feature_correl = correl_tdd(feature_correl)
                correlations.append(feature_correl)
        
        return tuple(correlations)

    def get_sentence_prediction_emission(self, sentence: Tensor, lemma: Tensor, postag: Tensor, deprel: Tensor) -> Tensor:
        # embedding inputs
        sentence_embeded = self._words_embed(sentence)
        lemma_embeded = self._lemma_embed(lemma)
        postag_embeded = self._postag_embed(postag)
        deprel_embeded = self._deprel_embed(deprel)
        # doing correlation of pairs of features
        # order is important, as embed size was used to init correl_lstm
        embeds_list = [sentence_embeded, lemma_embeded, postag_embeded, deprel_embeded]
        full_merge = self._merge_feature_correlations(embeds_list, len(sentence))
        features = torch.cat(full_merge, 2)
        #embeds = self._dropout_layer(embeds)
        prediction_lstm, _ = self._lstm(features)
        #prediction1 = self._activation(prediction1)
        #prediction1 = self._dropout_layer(prediction1)
        
        prediction_lstm = self._activation(prediction_lstm)
        
        emissions = self._hidden2tags(prediction_lstm)
        
        return emissions


    def forward(self, sentence: Tensor, lemma: Tensor, postag: Tensor, deprel: Tensor, mask: Tensor) -> Tuple[List[float], List[int]]:
        predictions = self.get_sentence_prediction_emission(sentence, lemma, postag, deprel)
        paths = self._crf.decode(predictions, mask.bool())
        return paths


    