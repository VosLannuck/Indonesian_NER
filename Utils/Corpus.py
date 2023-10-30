#%%
import os
from torchtext.data import Field, BucketIterator

from torchtext.datasets import SequenceTaggingDataset 
from torchtext.data.dataset import Dataset
from torchtext.data import Iterator

""" 
    Using torchtext==0.6.0

"""
#%%
""" 
    Constaznce

"""
INPUT_FOLDER : str = os.getcwd() +"/Dataset/"
MINIMUM_WORD_FREQ : int = 3
BATCH_SIZE : int = 64

class Corpus(object):
    
    """ 
        Corpus for Indonesian NER
        
        Using .py because i dont want go into github just for reading the
        code 
    """
    
    def __init__(self, inputFolder : str = INPUT_FOLDER,
                 minimumWordFreq : int = MINIMUM_WORD_FREQ, 
                 batchSize : int = BATCH_SIZE,
                 trainFilename="train.tsv",
                 validationFilename = "val.tsv",
                 testingFilename = "test.tsv"
                 ):
        
        self.word_field : Field = Field(lower=False, tokenizer_language="id", )
        self.tag_field : Field = Field(unk_token=None, tokenizer_language="id", )
        
        self.batchSize : int = batchSize
        self.minimumWordFreq = minimumWordFreq
        
        self.trainFilename : str= trainFilename
        self.validationFilename : str = validationFilename
        self.testingFilename : str = testingFilename
        
        self.trainDataset : Dataset 
        self.validationDataset : Dataset 
        self.testDataset : Dataset 
        
        self.trainIterator : Iterator
        self.valIterator : Iterator
        self.testIterator : Iterator
        
        self.trainDataset, self.validationDataset, self.testDataset = SequenceTaggingDataset.splits(
            path = inputFolder,
            train = self.trainFilename,
            validation = self.validationFilename,
            test = self.validationFilename,
            fields=(("word", self.word_field), ("tag", self.tag_field))
        )
        
        self.trainIterator, self.valIterator, self.testIterator = BucketIterator.splits(
            datasets=(self.trainDataset, self.validationDataset, self.testDataset),
            batch_size=self.batchSize)
        
        self.word_field.build_vocab(self.trainDataset, min_freq=self.minimumWordFreq)
        self.tag_field.build_vocab(self.trainDataset.tag)
        
        # this just for padding INDX
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token] 
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]      
        
    
# %%
#%%
""" 

if __name__ == "__main__":
    corpus = Corpus(
        inputFolder= INPUT_FOLDER,
        minimumWordFreq=MINIMUM_WORD_FREQ,
        batchSize=BATCH_SIZE,
        )
print(f"Train set: {len(corpus.trainDataset)} sentences")
print(f"Val set: {len(corpus.validationDataset)} sentences")
print(f"Test set: {len(corpus.testDataset)} sentences")

print(corpus.word_field.vocab.stoi) # Checkhing out the stoi

"""
