import torch
import torch.nn as nn
from torch.nn import utils
import torch.nn.functional as F

import pandas as pd

import sceptr

MyAminoAcidTokenIndex = {
    "NULL" : 0,
    "MASK" : 1,
    "CLS" : 2,
    "A" : 3,
    "C" : 4,
    "D" : 5,
    "E" : 6,
    "F" : 7,
    "G" : 8,
    "H" : 9,
    "I" : 10,
    "K" : 11,
    "L" : 12,
    "M" : 13,
    "N" : 14,
    "P" : 15,
    "Q" : 16,
    "R" : 17,
    "S" : 18,
    "T" : 19,
    "V" : 20,
    "W" : 21,
    "Y" : 22
}

MyCdrCompartmentIndex = {
    "CDR1A": 1,
    "CDR2A": 2,
    "CDR3A": 3,
    "CDR1B": 4,
    "CDR2B": 5,
    "CDR3B": 6
}


def tokenise_each_tuple(entry: tuple):
    '''
    input:
        each row of a DataFrame as a tuple with named entries ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
        where each entry in the tuple contains just the amino acid sequences, NOT gene names!

    return:
        tokenised single vector representing the entire TCR
    '''
    initial_cls_vector = (MyAminoAcidTokenIndex["CLS"], 0, 0, 0)


    tokenised = []
    tokenised.append(initial_cls_vector)

    for k, v in MyCdrCompartmentIndex.items():
        tokenised.extend(
            list(zip(
                    [MyAminoAcidTokenIndex[aa] for aa in getattr(entry, k)],       # token_indices 
                    [idx for idx, _ in enumerate(getattr(entry, k), start=1)],   # token_positions
                    [len(getattr(entry, k)) for _ in getattr(entry, k)],                  # cdr_length
                    [v for _ in getattr(entry, k)]                               # compartment_index 
            ))
        )

    return tokenised


def cdr_tokenise(df: pd.DataFrame):
    '''
    input:
        entire DataFrame containing amino acid sequences of all CDRs of both chains of TCRs
        where column has names ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    
    output:
        padded tokenised vectors, where each vector represent a single TCR
    '''
    tokenised = []
    for entry in df.itertuples():
        tokenised.append(torch.tensor(tokenise_each_tuple(entry)))

    padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised,
                batch_first=True,
                padding_value=0,
            ) 
    return padded_batch



class SceptrFineTuneModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim_1=128, hidden_dim_2=64):
        '''
        input_dim=64 is the output dimension of the default Sceptr model
        '''
        super().__init__()
        sceptr_model = sceptr.variant.default()
        self.bert = sceptr_model._bert
        self.device = sceptr_model._device

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        x = self.bert.get_vector_representations_of(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x