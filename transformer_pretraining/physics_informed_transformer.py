import torch.nn.functional as F
import torch.nn as nn
import torch
from dataset_utils import generate_all_three_cdrs


class PhysCDREncoder(nn.Module):
    def __init__(self, transformer_model_dim=64, embedding_dim=4, embedding_bias=True, hidden_dim=128, nhead=8, num_layer=1, dim_feedforward=256):
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
        self.amino_acid_projection = nn.Linear(embedding_dim, transformer_model_dim-2, bias=embedding_bias)
        self.register_buffer(
            "physics_aa_projection",
            torch.tensor([
                # [0, 0], # "NULL"
                [0, 0], # "MASK"
                [0, 0], # "CLS"
                [0, 1], # "A"
                [0, 0], # "C"
                [-1, 0], # "D"
                [-1, 0], # "E"
                [0, 1], # "F"
                [0, 1], # "G"
                [1, 0], # "H"
                [0, 1], # "I"
                [1, 0], # "K"
                [0, 1], # "L"
                [0, 1], # "M"
                [0, 0], # "N"
                [0, 1], # "P"
                [0, 0], # "Q"
                [1, 0], # "R"
                [0, 0], # "S"
                [0, 0], # "T"
                [0, 1], # "V"
                [0, 1], # "W"
                [0, 1], # "Y"
            ],
            dtype=torch.float)
        )


    def forward(self, x):
        aa_prj = self.amino_acid_projection(x)
        phy_prj = torch.matmul(x[:, :, 1:23], self.physics_aa_projection)
        x = torch.concat([aa_prj, phy_prj], axis=-1)
        return self.transformer(x)


class PhysTransformerTCRModel(nn.Module):
    def __init__(self, transformer_model_dim=64, embedding_dim=4, embedding_bias=True, hidden_dim=128, nhead=8, num_layer=1, dim_feedforward=256):
        '''
        transformer_model_dim=64 is the dimension of the transformer embedding
        hidden_dim is the dimension of the linear classification layer
        embedding_dim is the dimension of the amino acid embedding space
        '''
        super().__init__()

        self.cdr_encoder = PhysCDREncoder(
            transformer_model_dim=transformer_model_dim,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layer=num_layer,
            dim_feedforward=dim_feedforward
        )
        self.fc1 = nn.Linear(transformer_model_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = self.cdr_encoder(x)[:, 0, :]    # only get the CLS token, only works with batch_first=True!
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

    model = PhysTransformerTCRModel(
        embedding_dim=29
    )
    print(summary(model))
    print(model(sceptr_tokenise(aa_sequences, tokenise_method=tokenise_method).to(torch.float)))
