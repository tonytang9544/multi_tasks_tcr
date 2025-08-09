import torch
import transformers

import pandas as pd


def random_aa_masking(
        input_ids, 
        tokenizer,
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

    masked_input[(rand_num <= final_masking_chance) & ~special_tokens_mask] = tokenizer.mask_token_id
    masked_input[(final_masking_chance < rand_num) & (rand_num <= final_randomise_chance) & ~special_tokens_mask] = torch.randint_like(
        masked_input[(final_masking_chance < rand_num) & (rand_num <= final_randomise_chance) & ~special_tokens_mask], 
        low=1, 
        high=20)

    full_mask = (rand_num <= final_no_change_chance) & ~special_tokens_mask
    
    masked_labels = input_ids.clone()
    masked_labels[final_no_change_chance < rand_num] = -100 # for pytorch loss function to ignore

    return masked_input, masked_labels, full_mask


def generate_full_tcr_sample_peptide_and_generate_labels(df: pd.DataFrame, batch_size=128):
    dataset = df.sample(batch_size, replace=True)
    # cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    all_CDRs = dataset["CDR1A"] + "," + dataset["CDR2A"] + "," +  dataset["CDR3A"] + "," + dataset["CDR1B"] + "," + dataset["CDR2B"] + "," + dataset["CDR3B"]
    white_spaced_TCRs = all_CDRs.apply(lambda x: " ".join(str(x))).to_list()
    target_epitope = dataset["epitope.id"].apply(lambda x: " ".join(x))
    random_epitope = df.sample(batch_size, replace=True)["epitope.id"].apply(lambda x: " ".join(x))
    random_mask = torch.rand(batch_size) > 0.5
    target_epitope.iloc[random_mask].replace(to_replace=random_epitope)
    target_epitope = target_epitope.to_list()
    labels = torch.ones(batch_size, dtype=torch.float)
    labels[random_mask] = 0
    return white_spaced_TCRs, target_epitope, labels


if __name__ == "__main__":
    a = torch.randint(26, (3, 10))
    print(a)
    tokenizer = transformers.BertTokenizerFast(
            "/home/minzhetang/Documents/github/multi_tasks_tcr/transformer_pretraining/aa_vocab.txt",
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            padding_side="right",
        )

    print(tokenizer.mask_token_id)
    print(random_aa_masking(a, tokenizer))

    import pandas as pd
    df = pd.read_csv("~/Documents/results/data_preprocessing/vdjdb/VDJDB_sceptr_nr_cdr.csv").dropna().reset_index(drop=True)
    print(generate_full_tcr_sample_peptide_and_generate_labels(df))
    generate_full_tcr_sample_peptide_and_generate_labels(df)