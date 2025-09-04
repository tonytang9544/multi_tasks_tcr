import pandas as pd
from libtcrlm import schema

def generate_all_three_cdrs(dataset: pd.DataFrame):
    CDR1A = []
    CDR2A = []
    CDR1B = []
    CDR2B = []

    aa_seq_df = dataset.copy()

    for entry in dataset.itertuples():
        tcr = schema.make_tcr_from_components(
            trav_symbol=entry.TRAV,
            junction_a_sequence=entry.CDR3A,
            trbv_symbol=entry.TRBV,
            junction_b_sequence=entry.CDR3B
        )

        CDR1A.append(tcr.cdr1a_sequence)
        CDR2A.append(tcr.cdr2a_sequence)
        CDR1B.append(tcr.cdr1b_sequence)
        CDR2B.append(tcr.cdr2b_sequence)

    aa_seq_df["CDR1A"] = pd.Series(CDR1A)
    aa_seq_df["CDR2A"] = pd.Series(CDR2A)
    aa_seq_df["CDR1B"] = pd.Series(CDR1B)
    aa_seq_df["CDR2B"] = pd.Series(CDR2B)

    return aa_seq_df