import transformers
from torchinfo import summary
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

'''
From pytorch documentations
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        '''
        Note this assumes the tensor of dimension (batch, seq, feature)
        '''
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :,  0::2] = torch.sin(position * div_term)
        pe[0, :,  1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TCR_peptide_model(nn.Module):
    def __init__(
            self, 
            tokenisation_dim: int = 29, 
            transformer_model_dim: int = 64, 
            feedforward_dim: int = 256, 
            nhead: int = 8, 
            num_layer: int = 3, 
            dropout: float = 0.1, 
            batch_first: bool = True,
            max_seq_len: int = 200
        ):
        '''
        transformer_model_dim=64 is the dimension of the transformer embedding
        feedforward_dim is the dimension of the feedforward network of the transformer encoder layer, usually set to 4 * transformer_model_dim
        '''
        super().__init__()

        # store a matrix as a loop up table for the tokenised one-hot vector
        self.embedder = nn.Linear(tokenisation_dim, transformer_model_dim, bias=False)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_model_dim,
                nhead=nhead,
                batch_first=batch_first,
                dim_feedforward=feedforward_dim,
                dropout=dropout
            ),
            num_layers=num_layer
        )

    def forward(self, x):
        embedded = self.embedder(x)
        token_type_embedded = self.token_type_embedder(x["token_type_ids"])
        src_key_padding_mask = x["attention_mask"] == 0
        embedded = pos_embedded + token_type_embedded
        return self.encoder(
                embedded,
                src_key_padding_mask=src_key_padding_mask
            )


if __name__ == "__main__": 
    # tokenizer = transformers.BertTokenizerFast(
    #         "aa_vocab.txt",
    #         do_lower_case=False,
    #         do_basic_tokenize=True,
    #         tokenize_chinese_chars=False,
    #         padding_side="right",
    #     )  
    # model = TCR_peptide_model(tokenizer.vocab_size)
    # print(summary(model))
    # print(tokenizer(["E E A P U, A F G", "E E A P U, A F G", "E E A P U, A F G E F A"], ["G F P", "A A A A A", "C F G"], return_tensors="pt", padding=True))
    # print(model(tokenizer(["E E A P U, A F G", "E E A P U, A F G", "E E A P U, A F G E F A"], ["G F P", "A A A A A", "C F G"], return_tensors="pt", padding=True)).shape)
    # print(tokenizer.mask_token_id, tokenizer.cls_token)

    import matplotlib.pyplot as plt
    pos_enc_vec = PositionalEncoding(64).pe.squeeze()

    print(pos_enc_vec)
    print(pos_enc_vec.shape)

    for i in range(6):
        plt.plot(pos_enc_vec[:20, i], label=str(i))
    plt.legend()
    plt.savefig("position_encoding_vector.png")
    plt.cla()
    plt.close()

    # pretrain_model = TCRtransformerPretrainer()
    # print(summary(pretrain_model))