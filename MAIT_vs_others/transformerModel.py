import torch.nn.functional as F
import torch.nn as nn
import torch
from dataset_utils import generate_all_three_cdrs


class TransformerTCRModel(nn.Module):
    def __init__(self, transformer_model_dim=64, embedding_dim=4, embedding_bias=True, hidden_dim=128, nhead=8, num_layer=1, dim_feedforward=2048):
        '''
        transformer_model_dim=64 is the dimension of the transformer embedding
        hidden_dim is the dimension of the linear classification layer
        embedding_dim is the dimension of the amino acid embedding space
        '''
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_model_dim, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward),
            num_layers=num_layer
        )
        self.amino_acid_projection = nn.Linear(embedding_dim, transformer_model_dim, bias=embedding_bias)
        self.fc1 = nn.Linear(transformer_model_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = self.amino_acid_projection(x)
        x = self.transformer(x)[:, 0, :]    # only get the CLS token, only works with batch_first=True!
        x = self.dropout(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    

if __name__ == "__main__":

    from torchinfo import summary
    import pandas as pd
    from sceptr_tokeniser import sceptr_tokenise, tokenise_each_tuple, one_hot_vector_from_tokenised_list


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
    print(sceptr_tokenise(aa_sequences))
    tokenise_method = lambda x: one_hot_vector_from_tokenised_list(tokenise_each_tuple(x))
    print(sceptr_tokenise(aa_sequences, tokenise_method=tokenise_method))

    model = TransformerTCRModel()
    print(summary(model))
    print(model(sceptr_tokenise(aa_sequences).to(torch.float)))