import sceptr
from torchinfo import summary
import pandas as pd
from torch.nn import utils
import torch

from libtcrlm import schema
from libtcrlm.tokeniser import CdrTokeniser
from libtcrlm.tokeniser.token_indices import AminoAcidTokenIndex



def generate_all_three_cdrs(dataset: pd.DataFrame):
    CDR1A = []
    CDR2A = []
    CDR1B = []
    CDR2B = []

    aa_seq_df = pd.DataFrame()

    for idx, entry in dataset.iterrows():
        tcr = schema.make_tcr_from_components(
            trav_symbol=entry["TRAV"],
            junction_a_sequence=entry["CDR3A"],
            trbv_symbol=entry["TRBV"],
            junction_b_sequence=entry["CDR3B"]
        )

        CDR1A.append(tcr.cdr1a_sequence)
        CDR2A.append(tcr.cdr2a_sequence)
        CDR1B.append(tcr.cdr1b_sequence)
        CDR2B.append(tcr.cdr2b_sequence)

    aa_seq_df["CDR1A"] = pd.Series(CDR1A)
    aa_seq_df["CDR2A"] = pd.Series(CDR2A)
    aa_seq_df["CDR1B"] = pd.Series(CDR1B)
    aa_seq_df["CDR2B"] = pd.Series(CDR2B)
    aa_seq_df["CDR3A"] = dataset["CDR3A"].copy()
    aa_seq_df["CDR3B"] = dataset["CDR3B"].copy()

    return aa_seq_df


MyCdrCompartmentIndex = {
    "CDR1A": 1,
    "CDR2A": 2,
    "CDR3A": 3,
    "CDR1B": 4,
    "CDR2B": 5,
    "CDR3B": 6
}


def tokenise_each_entry(entry: pd.Series):
    '''
    input:
        each row of a DataFrame with columns ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
        where each column contains just the amino acid sequences, NOT gene names!

    return:
        tokenised single vector representing the entire TCR
    '''
    initial_cls_vector = (AminoAcidTokenIndex.CLS, 0, 0, 0)

    tokenised = []
    tokenised.append(initial_cls_vector)

    for k, v in MyCdrCompartmentIndex.items():
        tokenised.extend(
            list(zip(
                (
                    [AminoAcidTokenIndex[aa] for aa in entry[k]],       # token_indices 
                    [idx for idx, _ in enumerate(entry[k], start=1)],   # token_positions
                    [len(entry[k]) for _ in entry[k]],                  # cdr_length
                    [v for _ in entry[k]]                               # compartment_index
                )
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
    for i, entry in df.iterrows():
        tokenised.append(tokenise_each_entry(entry))
    
    padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_tcrs,
                batch_first=True,
                padding_value=0,
            ) 
    return padded_batch


model = sceptr.variant.default()

print(summary(model._bert))

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


# print(model._calc_torch_representations(tcrs))

# model._bert.set_fine_tuning_mode(True) # this does not work
tcr_series = schema.generate_tcr_series(tcrs)
# print(tcr_series)

# all these three below are equivalent
tokenised_tcrs_1 = [model._tokeniser.tokenise(tcr) for tcr in tcr_series]
tokenised_tcrs_2 = [CdrTokeniser().tokenise(tcr) for tcr in tcr_series]
tokenised_tcrs = tcr_series.apply(lambda tcr: CdrTokeniser().tokenise(tcr)).tolist()
# print([tokenised_tcrs[i] - tokenised_tcrs_2[i] for i in range(4)])
# # proves equivalency

tokenised_tcrs_4 = cdr_tokenise(aa_sequences)
print(tokenised_tcrs_4)

# print(tokenised_tcrs)
# padded_batch = utils.rnn.pad_sequence(
#                 sequences=tokenised_tcrs,
#                 batch_first=True,
#                 padding_value=0,
# #             )#.to(model._device)
# print(padded_batch - tokenised_tcrs_4)
output = model._bert.get_vector_representations_of(tokenised_tcrs_4.to(model._device))
print(output)
print(output.shape)


