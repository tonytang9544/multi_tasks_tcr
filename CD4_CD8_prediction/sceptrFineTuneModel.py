import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from torchinfo import summary

import sceptr

from dataset_utils import generate_all_three_cdrs
from sceptr_tokeniser import sceptr_tokenise



class SceptrFineTuneModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim_1=128, device=None):
        '''
        input_dim=64 is the output dimension of the default Sceptr model
        device=None will move the model to the default device initialised with the Sceptr model:
            normally if CUDA available, move to CUDA, otherwise stay at CPU
        '''
        super().__init__()
        sceptr_model = sceptr.variant.default()
        
        if device is None:
            self.device = sceptr_model._device
        else:
            self.device = device

        self.bert = sceptr_model._bert.to(self.device)

        self.fc1 = nn.Linear(input_dim, hidden_dim_1).to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)
        self.fc2 = nn.Linear(hidden_dim_1, 1).to(self.device)


    def forward(self, x):
        x = self.bert.get_vector_representations_of(x)
        x = self.dropout(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
if __name__ == "__main__":
    tcrs = pd.DataFrame(
    data = {
            "TRAV": ["TRAV38-1*01", "TRAV3*01", "TRAV13-2*01", "TRAV38-2/DV8*01"],
            "CDR3A": ["CAHRSAGGGTSYGKLTF", "CAVDNARLMF", "CAERIRKGQVLTGGGNKLTF", "CAYRSAGGGTSYGKLTF"],
            "TRBV": ["TRBV2*01", "TRBV25-1*01", "TRBV9*01", "TRBV2*01"],
            "CDR3B": ["CASSEFQGDNEQFF", "CASSDGSFNEQFF", "CASSVGDLLTGELFF", "CASSPGTGGNEQYF"],
        },
        index = [0,1,2,3]
    )

    aa_sequences = generate_all_three_cdrs(tcrs)

    model = SceptrFineTuneModel()
    print(summary(model))
    print(model(sceptr_tokenise(aa_sequences).to(model.device)))