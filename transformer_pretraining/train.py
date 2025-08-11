import pandas as pd
import numpy as np
from torchinfo import summary

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import transformers

from tqdm import tqdm


from models import TCR_peptide_model
from dataset_utils import generate_full_tcr_sample_peptide_and_generate_labels, random_aa_masking

train_config_dict = {
    "lr": 1e-4,
    "num_epoch": 30,
    "num_train_batches": int(2e4),
    "transformer_model_d": 64,
    "has_scheduler": False,
    "num_warmup_proportion": 0.1,
    "batch_size": 128,
    "dataset_path": "~/Documents/results/data_preprocessing/vdjdb/VDJDB_sceptr_nr_cdr.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_file": "/home/minzhetang/Documents/github/multi_tasks_tcr/transformer_pretraining/aa_vocab.txt"
}

tokenizer = transformers.BertTokenizerFast(
            train_config_dict["vocab_file"],
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            padding_side="right",
        )

model = TCR_peptide_model(
    tokenizer.vocab_size,
    transformer_model_dim=train_config_dict["transformer_model_d"]
).to(train_config_dict["device"])
classifier = nn.Linear(train_config_dict["transformer_model_d"], 1).to(train_config_dict["device"])

print("training parameters:")
print(train_config_dict)

# CD_label_col = "CD4_or_CD8"

tcr_data_path = train_config_dict["dataset_path"]

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)
# print(tc_df.head())

train, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.2, random_state=42)

joint_params = list(model.parameters()) + list(classifier.parameters()) # for joint training




num_epochs = train_config_dict["num_epoch"]
batch_size = train_config_dict["batch_size"]
# num_train_batches = int(train.shape[0] / batch_size)
num_train_batches = train_config_dict["num_train_batches"]
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)

if train_config_dict["has_scheduler"]:
    optimizer = optim.AdamW(joint_params, lr=train_config_dict["lr"])
    total_steps = num_epochs*num_train_batches
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(train_config_dict["num_warmup_proportion"]*total_steps), 
        num_training_steps=total_steps
    )
else:
    optimizer = optim.Adam(joint_params, lr=train_config_dict["lr"])

best_val_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    classifier.train()
    running_loss = 0.0
    for i in range(num_train_batches):
        # Clear the gradients

        optimizer.zero_grad()

        # Forward pass
        white_spaced_TCRs, mixed_epitope, binding_labels = generate_full_tcr_sample_peptide_and_generate_labels(train, train_config_dict["batch_size"])
        
        tokenised = tokenizer(
            white_spaced_TCRs,
            mixed_epitope,
            return_tensors="pt", 
            padding=True
        )

        masked_aa_tokens, masked_aa_labels, aa_mask = random_aa_masking(tokenised["input_ids"], tokenizer=tokenizer)
        aa_masked_full_input = {
            "input_ids": masked_aa_tokens.to(train_config_dict["device"]),
            "token_type_ids" : tokenised["token_type_ids"].to(train_config_dict["device"]),
            "attention_mask": tokenised["attention_mask"].to(train_config_dict["device"])
        }

        outputs = model(aa_masked_full_input).squeeze()
        pred_binding_labels = F.sigmoid(classifier(outputs[:, 0, :]))
        pred_masked_aa_representations = outputs[aa_mask, :]
        cos_sim = torch.matmul(pred_masked_aa_representations, model.aa_embedder.weight.T)
        pred_masked_aa_tokens = F.softmax(cos_sim, dim=-1)

        loss = F.binary_cross_entropy(
            pred_binding_labels.squeeze(),
            binding_labels.to(train_config_dict["device"])) + F.cross_entropy(
                pred_masked_aa_tokens, 
                tokenised["input_ids"][aa_mask].to(train_config_dict["device"]))
        
        # Backward pass and optimization
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()
        if train_config_dict["has_scheduler"]:
            scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/num_train_batches:.4f}')

    model.eval()
    classifier.eval()
    with torch.no_grad():
        running_loss = 0.0

        for i in range(num_val_batches):
            # batch = val.iloc[i*batch_size: (i+1)*batch_size]
            
            # Forward pass
            white_spaced_TCRs, mixed_epitope, binding_labels = generate_full_tcr_sample_peptide_and_generate_labels(val, train_config_dict["batch_size"])
            
            tokenised = tokenizer(
                white_spaced_TCRs,
                mixed_epitope,
                return_tensors="pt", 
                padding=True
            )

            masked_aa_tokens, masked_aa_labels, aa_mask = random_aa_masking(tokenised["input_ids"], tokenizer=tokenizer)
            aa_masked_full_input = {
                "input_ids": masked_aa_tokens.to(train_config_dict["device"]),
                "token_type_ids" : tokenised["token_type_ids"].to(train_config_dict["device"]),
                "attention_mask": tokenised["attention_mask"].to(train_config_dict["device"])
            }

            outputs = model(aa_masked_full_input).squeeze()
            pred_binding_labels = F.sigmoid(classifier(outputs[:, 0, :]))
            pred_masked_aa_representations = outputs[aa_mask, :]
            cos_sim = torch.matmul(pred_masked_aa_representations, model.aa_embedder.weight.T)
            pred_masked_aa_tokens = F.softmax(cos_sim, dim=-1)

            loss = F.binary_cross_entropy(
                pred_binding_labels.squeeze(),
                binding_labels.to(train_config_dict["device"])) + F.cross_entropy(
                    pred_masked_aa_tokens, 
                    tokenised["input_ids"][aa_mask].to(train_config_dict["device"]))
        
            running_loss += loss.item()
        curr_val_loss = running_loss/num_val_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {curr_val_loss:.4f}')
        if curr_val_loss < best_val_loss:
            torch.save(model, "model.pt")
            best_val_loss = curr_val_loss

print('Training complete')

model = torch.load("model.pt", weights_only=False).eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(num_test_batches):
        white_spaced_TCRs, mixed_epitope, binding_labels = generate_full_tcr_sample_peptide_and_generate_labels(test, train_config_dict["batch_size"])
            
        tokenised = tokenizer(
            white_spaced_TCRs,
            mixed_epitope,
            return_tensors="pt", 
            padding=True
        )

        masked_aa_tokens, masked_aa_labels, aa_mask = random_aa_masking(tokenised["input_ids"], tokenizer=tokenizer)
        aa_masked_full_input = {
            "input_ids": masked_aa_tokens.to(train_config_dict["device"]),
            "token_type_ids" : tokenised["token_type_ids"].to(train_config_dict["device"]),
            "attention_mask": tokenised["attention_mask"].to(train_config_dict["device"])
        }

        outputs = model(aa_masked_full_input).squeeze()
        pred_binding_labels = F.sigmoid(classifier(outputs[:, 0, :]))
        pred_masked_aa_representations = outputs[aa_mask, :]
        cos_sim = torch.matmul(pred_masked_aa_representations, model.aa_embedder.weight.T)
        pred_masked_aa_tokens = F.softmax(cos_sim, dim=-1)

        # loss = F.binary_cross_entropy(
        #     pred_binding_labels.squeeze(),
        #     binding_labels.to(train_config_dict["device"])) + F.cross_entropy(
        #         pred_masked_aa_tokens, 
        #         tokenised["input_ids"][aa_mask].to(train_config_dict["device"]))
    
        # running_loss += loss.item()
        all_preds.extend(pred_binding_labels.cpu().numpy())
        all_labels.extend(binding_labels.numpy())

auc = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("AUC plot")
plt.clf()
plt.close()