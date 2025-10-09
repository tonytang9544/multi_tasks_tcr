import torch.nn.functional as F
import torch.nn as nn
import torch
from dataset_utils import generate_all_three_cdrs


class TransformerTCRModel(nn.Module):
    def __init__(self, 
                 transformer_model_dim=128, 
                 embedding_dim=4, 
                 embedding_bias=False, 
                 hidden_dim=128, 
                 nhead=32, 
                 num_layer=3, 
                 dim_feedforward=512,
                 num_output_labels:int=1,
                 dropout_chance: float=0.2
        ):
        '''
        transformer_model_dim: dimension of the transformer representation
        hidden_dim: dimension of the linear classification layer
        embedding_dim: dimension of the amino acid embedding
        num_output_labels: number of (orthogonal) labels to predict
        '''
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_model_dim, 
                nhead=nhead, 
                batch_first=True, 
                dim_feedforward=dim_feedforward,
                dropout=dropout_chance,
            ),
            num_layers=num_layer
        )
        self.amino_acid_projection = nn.Linear(embedding_dim, transformer_model_dim, bias=embedding_bias)
        self.fc1 = nn.Linear(transformer_model_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_chance)
        self.fc2 = nn.Linear(hidden_dim, num_output_labels)


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