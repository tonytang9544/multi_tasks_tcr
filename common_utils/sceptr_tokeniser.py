import torch
from torch.nn import utils

import pandas as pd
import numpy as np


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


def one_hot_vector_from_tokenised_list(tokenised_list: list, one_hot_dim=29):
    pos_embed_idx = 0
    cdr_embed_start_idx = 22
    one_hot_embed = np.zeros((len(tokenised_list), one_hot_dim))
    for i in range(len(tokenised_list)):
        if tokenised_list[i][0] != 0:
            one_hot_embed[i][tokenised_list[i][0]] = 1
        if tokenised_list[i][2] != 0:
            one_hot_embed[i][pos_embed_idx] = tokenised_list[i][1] / tokenised_list[i][2]
        if tokenised_list[i][3] != 0:
            one_hot_embed[i][cdr_embed_start_idx + tokenised_list[i][3]] = 1
    return one_hot_embed


def sceptr_tokenise(df: pd.DataFrame, tokenise_method=tokenise_each_tuple):
    '''
    input:
        entire DataFrame containing amino acid sequences of all CDRs of both chains of TCRs
        where column has names ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    
    output:
        padded tokenised vectors, where each vector represent a single TCR
    '''
    tokenised = []
    for entry in df.itertuples():
        tokenised.append(torch.tensor(tokenise_method(entry)))

    padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised,
                batch_first=True,
                padding_value=0,
            ) 
    return padded_batch


def nested_tensor_tokenise(df: pd.DataFrame, tokenise_method=tokenise_each_tuple):
    '''
    input:
        entire DataFrame containing amino acid sequences of all CDRs of both chains of TCRs
        where column has names ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    
    output:
        nested tensors with layout=torch.jagged, where each element is  represent a single TCR
    '''
    tokenised = []
    for entry in df.itertuples():
        tokenised.append(tokenise_method(entry))
 
    return torch.nested.nested_tensor(tokenised, layout=torch.jagged)


if __name__ == "__main__":

    from dataset_utils import generate_all_three_cdrs
    from transformerModel import TransformerTCRModel

    tcrs = pd.DataFrame(
    data = {
            "TRAV": ["TRAV38-1*01", "TRAV3*01", "TRAV13-2*01", "TRAV38-2/DV8*01"],
            "CDR3A": ["CAHRSAGGGTSYGKLTF", "CAVDNARLMF", "CAERIRKGQVLTGGGNKLTF", "CAYRSAGGGTSYGKLTF"],
            "TRBV": ["TRBV2*01", "TRBV25-1*01", "TRBV9*01", "TRBV2*01"],
            "CDR3B": ["CASSEFQGDNEQFF", "CASSDGSFNEQFF", "CASSVGDLLTGELFF", "CASSPGTGGNEQYF"],
        },
        index = [0,1,2,3]
    )

    aa_seq = generate_all_three_cdrs(tcrs)

    print(sceptr_tokenise(aa_seq))

    nested_tokenised = nested_tensor_tokenise(aa_seq)
    print([x for x in nested_tokenised])

    # # normal nn.MultiHeadedAttention does not work with nested tensor?
    # print(TransformerTCRModel()(nested_tokenised.to(torch.float)))




