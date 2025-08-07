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
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TCRtransformer(nn.Module):
    def __init__(
            self, 
            vocab_size=26, 
            transformer_model_dim=64, 
            feedforward_dim=256, 
            nhead=8, 
            num_layer=3, 
            dropout=0.1, 
            batch_first=True,
            max_seq_len=200
        ):
        '''
        transformer_model_dim=64 is the dimension of the transformer embedding
        feedforward_dim is the dimension of the feedforward network of the transformer encoder layer, usually set to 4 * transformer_model_dim
        '''
        super().__init__()

        self.aa_embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=transformer_model_dim
        )

        self.pos_embedder = PositionalEncoding(
            d_model=transformer_model_dim,
            max_len=max_seq_len
        )

        # self.cdr_embedder = nn.Embedding(
        #     num_embeddings=6,
        #     embedding_dim=transformer_model_dim
        # )

        self.token_type_embedder = nn.Embedding(
            num_embeddings=2,
            embedding_dim=transformer_model_dim
        )

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
        aa_embedded = self.aa_embedder(x["input_ids"])
        pos_embedded = self.pos_embedder(aa_embedded)
        token_type_embedded = self.token_type_embedder(x["token_type_ids"])
        src_key_padding_mask = x["attention_mask"] == 0
        embedded = pos_embedded + token_type_embedded
        return self.encoder(
                embedded,
                src_key_padding_mask=src_key_padding_mask
            )


if __name__ == "__main__": 
    tokenizer = transformers.BertTokenizerFast(
            "aa_vocab.txt",
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            padding_side="right",
        )  
    model = TCRtransformer(tokenizer.vocab_size)
    print(summary(model))
    print(tokenizer(["E E A P U, A F G", "E E A P U, A F G", "E E A P U, A F G E F A"], ["G F P", "A A A A A", "C F G"], return_tensors="pt", padding=True))
    print(model(tokenizer(["E E A P U, A F G", "E E A P U, A F G", "E E A P U, A F G E F A"], ["G F P", "A A A A A", "C F G"], return_tensors="pt", padding=True)).shape)
    print(tokenizer.mask_token_id, tokenizer.cls_token)

    # pretrain_model = TCRtransformerPretrainer()
    # print(summary(pretrain_model))