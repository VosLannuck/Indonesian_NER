#%%

from Models import BiLSTM as BILSTMModule 
from Utils import Corpus as CorpusModule
import NER as NERModule
import torch
import importlib
from torch.optim import Adam
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from spacy.lang.id import Indonesian
from typing import List
importlib.reload(BILSTMModule)
importlib.reload(CorpusModule)

corpus : CorpusModule.Corpus = CorpusModule.Corpus()

INPUT_DIMENSION : int =  len(corpus.word_field.vocab) # Because its sentence
EMBEDDING_DIMENSION : int = 300  # Embedding dims within each words ( For semantic)
HIDDEN_DIMENSION : int = 64
OUTPUT_DIMENSION : int = len(corpus.tag_field.vocab) # Tag ( NER )
TOTAL_LSTM_LAYERS : int = 2
EMBEDDING_LAYER_DROPOUT_Percentage : float = 0.5
LSTM_LAYER_DROPOUT_Percentage : float = 0.1
FC_LAYER_DROPOUT_Percentage : float = 0.25
WORD_PAD_IDX : int = corpus.word_pad_idx

bilstm = BILSTMModule.BILstm(
    inputDimension=INPUT_DIMENSION,
    embeddingDimension=EMBEDDING_DIMENSION,
    hiddenLayerDimension=HIDDEN_DIMENSION,
    outputLayerDimension=OUTPUT_DIMENSION,
    numLstmLayers=TOTAL_LSTM_LAYERS,
    embeddingDropout=EMBEDDING_LAYER_DROPOUT_Percentage,
    lstmDropout=LSTM_LAYER_DROPOUT_Percentage,
    fullyConnectedDropout=FC_LAYER_DROPOUT_Percentage,
    wordPaddingIndex=WORD_PAD_IDX
)

EPOCH : int = 30
ner : NERModule.NER = NERModule.NER(
    bilstm,
    corpus,
    Adam,
    CrossEntropyLoss
)
ner.Train(1)
#%%
# Ner 1 Epoch Train
sentence = "Sementara itu, Kepala Pelaksana BPBD Luwu Utara Muslim Muchtar mengatakan, terdapat 15.000 jiwa mengungsi akibat banjir bandang."
tags = ["O", "O", "O", "O", "O", "B-ORGANIZATION", "I-ORGANIZATION", "L-ORGANIZATION", "B-PERSON", "L-PERSON", "O", "O", "O", "U-QUANTITY", "O", "O", "O", "O", "O", "O"]
words, infer_tags, unknown_tokens = ner.Inference(sentence=sentence, true_tags=tags)


#%% Ner 30 Epoch
ner.Train(30)
#%%
sentence = "Sementara itu, Kepala Pelaksana BPBD Luwu Utara Muslim Muchtar mengatakan, terdapat 15.000 jiwa mengungsi akibat banjir bandang."
tags = ["O", "O", "O", "O", "O", "B-ORGANIZATION", "I-ORGANIZATION", "L-ORGANIZATION", "B-PERSON", "L-PERSON", "O", "O", "O", "U-QUANTITY", "O", "O", "O", "O", "O", "O"]
words, infer_tags, unknown_tokens = ner.Inference(sentence=sentence, true_tags=tags)


