import torch
import transformers

import pandas as pd


def random_aa_masking(
        input_ids, 
        tokeniser,
        change_proportion=0.15, 
        masking_chance=0.8, 
        randomise_chance=0.1, 
        no_change_chance=0.1
    ):
    rand_num = torch.rand_like(input_ids, dtype=torch.float)
    final_masking_chance = change_proportion * masking_chance
    final_randomise_chance = change_proportion * randomise_chance + final_masking_chance
    final_no_change_chance = change_proportion * no_change_chance + final_randomise_chance
    special_tokens_mask = (input_ids == tokenizer.mask_token_id) | (input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.pad_token_id) | (input_ids == tokenizer.unk_token_id) | (input_ids == tokenizer.sep_token_id) 

    masked_input = input_ids.clone()

    masked_input[(rand_num <= final_masking_chance) & ~special_tokens_mask] = tokeniser.mask_token_id
    masked_input[(final_masking_chance < rand_num) & (rand_num <= final_randomise_chance) & ~special_tokens_mask] = torch.randint_like(
        masked_input[(final_masking_chance < rand_num) & (rand_num <= final_randomise_chance) & ~special_tokens_mask], 
        low=1, 
        high=20)
    
    masked_labels = input_ids.clone()
    masked_labels[final_no_change_chance < rand_num] = -100 # for pytorch loss function to ignore

    return masked_input, masked_labels


def generate_full_tcr_sample_peptide_and_generate_labels(df: pd.DataFrame, batch_size=128):
    dataset = df.sample(batch_size)
    # cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    all_CDRs = dataset["CDR1A"] + "," + dataset["CDR2A"] + dataset["CDR3A"] + dataset["CDR1B"] + dataset["CDR2B"] + dataset["CDR3B"]
    white_spaced_TCRs = all_CDRs.apply(lambda x: " ".join(x)).to_list()
    target_epitope = dataset["epitope"].apply(lambda x: " ".join(x))
    random_epitope = df.sample(batch_size)["epitope"].apply(lambda x: " ".join(x))
    random_mask = torch.rand(batch_size) > 0.5
    mixed_epitope = target_epitope.where(random_mask, random_epitope)
    labels = torch.ones(batch_size, dtype=torch.int8)
    labels[random_mask] = 0
    return white_spaced_TCRs, mixed_epitope, labels


if __name__ == "__main__":
    a = torch.randint(26, (3, 10))
    print(a)
    tokenizer = transformers.BertTokenizerFast(
            "aa_vocab.txt",
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            padding_side="right",
        )
    print(tokenizer.mask_token_id)
    print(random_aa_masking(a, tokenizer))