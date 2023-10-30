#%%
from Models import BiLSTM as BILSTMModule 
from Utils import Corpus as CorpusModule
import torch
import importlib
from torch.optim import Adam
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from spacy.lang.id import Indonesian
from typing import List
importlib.reload(BILSTMModule)
importlib.reload(CorpusModule)
#%%
#%%

class NER(object):
    
    def __init__(self, model : Module,
                 corpus : CorpusModule.Corpus,
                 optimizer_cls : Adam,
                 loss_fn_cls :  CrossEntropyLoss,
                 lr : float = 0.01
                 ):
        self.model : Module = model
        self.corpus : CorpusModule.Corpus = corpus
        self.optimizer : Adam = optimizer_cls(model.parameters(), lr=lr) 
        self.loss_fn_cls : CrossEntropyLoss = loss_fn_cls(ignore_index=self.corpus.tag_pad_idx)
    def Accuracy(self, preds: torch.Tensor, y : torch.Tensor):
        maxPredictions : torch.Tensor = preds.argmax(dim=1, keepdim=True) # For each Sentence
        nonPadElement : torch.Tensor = (y != self.corpus.tag_pad_idx).nonzero()
        correctPredictions : torch.Tensor = maxPredictions[nonPadElement].squeeze(dim=1).eq(y[nonPadElement])
        lenPred : torch.Tensor = torch.Tensor([y[nonPadElement].shape[0]])
        return correctPredictions.sum() / lenPred
        
    def Train(self, nEpoch : int):
        trainLosses : torch.Tensor = 0.0
        trainAccuracy : torch.Tensor = 0.0
        for epoch in range(nEpoch):
            epochLoss : torch.Tensor = 0.0
            epochAccuracy : torch.Tensor = 0.0
            self.model.train()
            iterator = self.corpus.trainIterator
            for batch in iterator:
                # text = [sentence Length, batch_size ] 
                # tags = [sentence Length, batch_size ]
                texts : torch.Tensor = batch.word
                tags  : torch.Tensor = batch.tag
                
                self.optimizer.zero_grad()
                predTags : torch.Tensor = self.model(texts) # [sen Length, batch, outputDim]
                
                predTags = predTags.view(-1, predTags.shape[-1]) # [senLength * batch, outputDim]
                tags = tags.view(-1)# [senLength * batchSize]
                
                batchLoss : torch.Tensor = self.loss_fn_cls(predTags, tags)
                batchAccuracy : torch.Tensor = self.Accuracy(predTags, tags)
                
                batchLoss.backward()
                self.optimizer.step()
                
                epochLoss += batchLoss.item()
                epochAccuracy += batchAccuracy.item()
            trainLosses += epochLoss / len(iterator)
            trainAccuracy += epochAccuracy / len(iterator)
            print(f"Epoch {epoch + 1} : ")
            print(f"Train Lossess : {trainLosses}")
            print(f"Train Accuracy : {trainAccuracy}")
            
            print("Validation Result : ")
            _,_ = self.Evaluate(self.corpus.valIterator)
            print("Test Result : ")
            _,_ = self.Evaluate(self.corpus.testIterator)
            print("\n")

    
    def Evaluate(self, iterator):
        
        totalLoss : torch.Tensor = 0.0
        totalAccuracy : torch.Tensor = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                texts : torch.Tensor = batch.word
                tags : torch.Tensor =  batch.tag
                
                predTags : torch.Tensor = self.model(texts)
                predTags = predTags.view(-1, predTags.shape[-1])
                
                tags = tags.view(-1)
                
                batchLoss : torch.Tensor = self.loss_fn_cls(predTags, tags)
                batchAccuracy : torch.Tensor = self.Accuracy(predTags, tags)
                
                totalLoss += batchLoss.item()
                totalAccuracy += batchAccuracy.item()
        print(f"Loss : {totalLoss / len(iterator)} ")
        print(f"Acc : {totalAccuracy / len(iterator)}")
        
        return totalLoss / len(iterator ), totalAccuracy / len(iterator)
        

    def Inference(self, sentence : torch.Tensor, true_tags = None):
        self.model.eval()
        
        indonesian : Indonesian = Indonesian()
        sentenceTokens : List[str] = [token.text.lower() for token in indonesian(sentence)]
        
        sentenceNumRepresentation : List[int] = [self.corpus.word_field.vocab.stoi[token] for token in sentenceTokens]
        unknownIndx : List[int] = self.corpus.word_field.vocab.stoi[self.corpus.word_field.unk_token]
        
        unknownTokens : List[int] =  [t for t, n in zip(sentenceTokens, sentenceNumRepresentation) if n == unknownIndx]

        sentenceTensor : torch.Tensor = torch.LongTensor(sentenceNumRepresentation)# Flatten [sen length]
        prediction : torch.Tensor = self.model(sentenceTensor.unsqueeze(-1)) #[outputDim]
        bestPrediction : torch.Tensor = prediction.squeeze(-1).argmax(-1)
        predictionTag : List[int] = [self.corpus.tag_field.vocab.itos[t.item()] for t in bestPrediction]
        
        maxLenToken = max([len(token) for token in sentenceTokens ] + [len("word")])
        maxLenTag = max([len(tag) for tag in predictionTag] + [len("pred")])
        
        print(
        f"{'word'.ljust(maxLenToken)}\t{'unk'.ljust(maxLenToken)}\t{'pred tag'.ljust(maxLenTag)}" 
        + ("\ttrue tag" if true_tags else "")
        )
        for i, token in enumerate(sentenceTokens):
            is_unk = "✓" if token in unknownTokens else ""
            print(
                f"{token.ljust(maxLenToken)}\t{is_unk.ljust(maxLenToken)}\t{predictionTag[i].ljust(maxLenTag)}" 
                + (f"\t{true_tags[i]}" if true_tags else "")
                )
        return sentenceTokens, predictionTag, unknownTokens
        