import pandas as pd
import numpy as np
from torchinfo import summary

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import transformers

from tqdm import tqdm
import pickle


from sceptrFineTuneModel import SceptrFineTuneModel
from sceptr_tokeniser import sceptr_tokenise


train_config_dict = {
    "lr": 3e-4,
    "num_epoch": 50,
    "classifier_hid_dim": 128,
    "has_scheduler": True,
    "batch_size": 1024,
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "sceptr_model_variant": "default"
}

print("training parameters:")
print(train_config_dict)

label_col = "MAIT_or_NOT"

tcr_data_path = train_config_dict["dataset_path"]

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)#.iloc[:1000]
tc_df[label_col] = tc_df["annotation_L3"] == "MAIT"

print(tc_df.head())
print(tc_df[label_col].unique())
print(tc_df[label_col].value_counts())

tc_df["label"] = LabelEncoder().fit_transform(tc_df[label_col])

train, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.2, random_state=42)
  
    
model = SceptrFineTuneModel(
    hidden_dim_1=train_config_dict["classifier_hid_dim"],
    model_variant=train_config_dict["sceptr_model_variant"]
)
summary(model)

num_epochs = train_config_dict["num_epoch"]
batch_size = train_config_dict["batch_size"]
num_train_batches = int(train.shape[0] / batch_size)
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)



criterion = nn.BCELoss()

if train_config_dict["has_scheduler"]:
    optimizer = optim.AdamW(model.parameters(), lr=train_config_dict["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
    # total_steps = num_epochs*num_train_batches
    # scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=int(train_config_dict["num_warmup_proportion"]*total_steps), 
    #     num_training_steps=total_steps
    # )
else:
    optimizer = optim.Adam(model.parameters(), lr=train_config_dict["lr"])



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
    if train_config_dict["has_scheduler"]:
        print(f"scheduler learn rate: {scheduler.get_last_lr()}")
        scheduler.step()
    
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

with open("true_pred_dict.pkl", "wb") as f:
    pickle.dump(
        {
            "y_true": all_labels,
            "y_pred": all_preds
        },
        f
    )


print(f'ROC_AUC: {roc_auc_score(all_labels, all_preds)}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("ROC_AUC plot")
plt.clf()
plt.close()
print(f"Average precision score: {average_precision_score(all_labels, all_preds)}")
PrecisionRecallDisplay.from_predictions(all_labels, all_preds)
plt.savefig("Precision Recall curve plot")
plt.clf()
plt.close()