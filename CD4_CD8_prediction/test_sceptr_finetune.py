import sceptr
from torchinfo import summary
import pandas as pd
from torch.nn import utils
import torch

from libtcrlm import schema
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex

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


print(model._calc_torch_representations(tcrs))

# model._bert.set_fine_tuning_mode(True)
tokenised_tcrs = [model._tokeniser.tokenise(tcr) for tcr in schema.generate_tcr_series(tcrs)]
padded_batch = utils.rnn.pad_sequence(
                sequences=tokenised_tcrs,
                batch_first=True,
                padding_value=DefaultTokenIndex.NULL,
            ).to(model._device)
print(model._bert.get_vector_representations_of(padded_batch))
