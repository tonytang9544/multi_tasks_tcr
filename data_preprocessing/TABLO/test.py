import pandas as pd

tcrs = pd.DataFrame(

    data = {

            "TRAV": ["TRAV38-1*01", "TRAV3*01", "TRAV13-2*01", "TRAV38-2/DV8*01", "TRAJ58"],

            "CDR3A": ["CAHRSAGGGTSYGKLTF", "CAVDNARLMF", "CAERIRKGQVLTGGGNKLTF", "CAYRSAGGGTSYGKLTF", "CAVDNARLMF"],

            "TRBV": ["TRBV2*01", "TRBV25-1*01", "TRBV9*01", "TRBV2*01", "TRBV6-7"],

            "CDR3B": ["CASSEFQGDNEQFF", "CASSDGSFNEQFF", "CASSVGDLLTGELFF", "CASSPGTGGNEQYF", "CASSVGDLLTGELFF"],

    },

    index = [0,1,2,3,4]

)



print(tcrs)

import sceptr, tidytcells

# print(sceptr.calculate_vector_representations(tcrs.iloc[0, 1, 2, 3]))

new_df = pd.DataFrame()

new_df["TRAV"] = tcrs["TRAV"].apply(lambda x: tidytcells.tr.standardise(x, enforce_functional=True))

print(new_df)

new_df = new_df.dropna()

print(new_df)