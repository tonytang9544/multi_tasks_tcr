import torch
import transformers

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
    masked_labels[(final_no_change_chance < rand_num) & ~special_tokens_mask] = -100 # for pytorch loss function to ignore

    return masked_input, masked_labels

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