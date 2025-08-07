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


from model import TCR_peptide_model


train_config_dict = {
    "lr": 2e-4,
    "num_epoch": 30,
    "transformer_model_d": 64,
    "has_scheduler": False,
    "batch_size": 1024*4,
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

tokenizer = transformers.BertTokenizerFast(
            "aa_vocab.txt",
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            padding_side="right",
        )

model = TCR_peptide_model(
    tokenizer.vocab_size,
    transformer_model_dim=train_config_dict["transformer_model_d"]
).to(train_config_dict["device"])
classifier = nn.Linear(train_config_dict["transformer_model_d"], 1)

print("training parameters:")
print(train_config_dict)

# CD_label_col = "CD4_or_CD8"

tcr_data_path = train_config_dict["dataset_path"]

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)
# print(tc_df.head())




# aa_sequences = generate_all_three_cdrs(tc_df)
# print(model(sceptr_tokenise(aa_sequences).to("cuda")).shape)

# tc_df["label"] = LabelEncoder().fit_transform(tc_df["CD4_or_CD8"])

train, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.2, random_state=42)

criterion = nn.CrossEntropyLoss()



















optimizer = optim.Adam(model.parameters(), lr=train_config_dict["lr"])
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

num_epochs = train_config_dict["num_epoch"]
batch_size = train_config_dict["batch_size"]
num_train_batches = int(train.shape[0] / batch_size)
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)

best_val_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(num_train_batches):
        # Clear the gradients
        optimizer.zero_grad()
        batch = train.iloc[i*batch_size: (i+1)*batch_size]
        
        # Forward pass
        outputs = model(sceptr_tokenise(batch).to(model.device)).squeeze()
        loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(model.device))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    # scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/num_train_batches:.4f}')

    model.eval()
    with torch.no_grad():
        running_loss = 0.0

        for i in range(num_val_batches):
            batch = val.iloc[i*batch_size: (i+1)*batch_size]
            
            # Forward pass
            outputs = model(sceptr_tokenise(batch).to(model.device)).squeeze()
            loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(model.device))
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
        batch = test.iloc[i*batch_size: (i+1)*batch_size]
        outputs = model(sceptr_tokenise(batch).to(model.device)).squeeze()
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(batch["label"].to_numpy())

auc = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("AUC plot")
plt.clf()
plt.close()