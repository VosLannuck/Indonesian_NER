#%%
from Utils import Corpus as CorpusModule
import importlib
import torch
from torch import nn
from torch.optim import Adam

from typing import List, Tuple, Dict

importlib.reload(CorpusModule)

class BILstm(nn.Module):
    """ 
        BIDirectional LSTM ( Fancy Name yk )
    """
    def __init__(self, inputDimension : int, embeddingDimension : int,
                 hiddenLayerDimension : int, outputLayerDimension : int,
                 numLstmLayers : int , embeddingDropout : float,
                 lstmDropout : float, fullyConnectedDropout : float,
                 wordPaddingIndex : int  
                 ):
        super().__init__()
        
        self.embeddingDimension = embeddingDimension
        
        ## Layer - Layer 
        self.embeddingLayer : nn.Embedding = nn.Embedding(
            num_embeddings = inputDimension,
            embedding_dim = embeddingDimension,
            padding_idx = wordPaddingIndex
        )
        
        self.embeddingDropoutLayer : nn.Module = nn.Dropout(embeddingDropout)
        
        self.BILstmLayer : nn.Module = nn.LSTM(
             input_size = embeddingDimension,
             hidden_size = hiddenLayerDimension,
             num_layers=numLstmLayers,
             bidirectional=True,
             dropout= lstmDropout if numLstmLayers > 1 else 0
        )
        
        self.fullyConnectedDropout : nn.Module = nn.Dropout(fullyConnectedDropout)
        self.fcLayer : nn.Module = nn.Linear(hiddenLayerDimension * 2, outputLayerDimension)
        
    def forward(self, sentence):
        ## Sentence.shape --> [sentence length, batch size]
        ## embeddingOutput.shape --> [sentence length, batch size, embedding dim]
        ## lstmOutput.shape --> [sentence length, batch size, hidden dim * 2 ]
        embeddingOutput : torch.Tensor = self.embeddingDropoutLayer(self.embeddingLayer(sentence))
        lstmOutput , _ = self.BILstmLayer(embeddingOutput)
        NEROutput : torch.Tensor = self.fullyConnectedDropout(self.fcLayer(lstmOutput))
        
        return NEROutput
    
    def initWeights(self):
        
        for _, params in self.named_parameters():
            nn.init.normal_(tensor=params.data, mean=0, std=1)
    
    def initEmbedding(self, word_pad_idx):
        self.embeddingLayer.weight.data[word_pad_idx] = torch.zeros(self.embeddingDimension)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#%%
