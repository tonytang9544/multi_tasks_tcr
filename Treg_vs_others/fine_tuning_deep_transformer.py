import pandas as pd
import numpy as np
from torchinfo import summary
import pickle

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


from transformerModel import TransformerTCRModel
from sceptr_tokeniser import sceptr_tokenise, tokenise_each_tuple, one_hot_vector_from_tokenised_list


train_config_dict = {
    "lr": 3e-4,
    "num_epoch": 50,
    "one_hot_feature_embedding": True,
    "classifier_hid_dim": 256,
    "transformer_model_dim": 128,
    "encoder_feedforward_dim": 512,
    "num_encoder_layers": 3,
    "has_scheduler": True,
    "scheduler_gamma": 0.5,
    "scheduler_step_size": 6,
    "batch_size": 1024,
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "num_warmup_proportion": 0.02,
    "max_grad_norm": None
}

device = "cuda" if torch.cuda.is_available() else "cpu"

print("training parameters:")
print(train_config_dict)

label_col = "Phenotype_Label"

tcr_data_path = train_config_dict["dataset_path"]

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)#.iloc[:1000]
tc_df[label_col] = tc_df["annotation_L3"] == "Tregs"
# print(tc_df.head())
# print(tc_df[label_col].unique())
# print(tc_df[label_col].value_counts())
# input("press any key to continue")

if train_config_dict["one_hot_feature_embedding"]:
    model = TransformerTCRModel(
        hidden_dim=train_config_dict["classifier_hid_dim"],
        dim_feedforward=train_config_dict["encoder_feedforward_dim"],
        num_layer=train_config_dict["num_encoder_layers"],
        transformer_model_dim=train_config_dict["transformer_model_dim"],
        embedding_bias=False,
        embedding_dim=29
    )
    tokenise_method = lambda x: one_hot_vector_from_tokenised_list(tokenise_each_tuple(x))
else:
    model = TransformerTCRModel(
        hidden_dim=train_config_dict["classifier_hid_dim"],
        dim_feedforward=train_config_dict["encoder_feedforward_dim"],
        num_layer=train_config_dict["num_encoder_layers"],
        transformer_model_dim=train_config_dict["transformer_model_dim"]
    )
    tokenise_method = lambda x: tokenise_each_tuple(x)

model = model.to(device)
summary(model)


# aa_sequences = generate_all_three_cdrs(tc_df)
# print(model(sceptr_tokenise(aa_sequences).to("cuda")).shape)

tc_df["label"] = LabelEncoder().fit_transform(tc_df[label_col])

train, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.2, random_state=42)



num_epochs = train_config_dict["num_epoch"]
batch_size = train_config_dict["batch_size"]
num_train_batches = int(train.shape[0] / batch_size)
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)

criterion = nn.BCELoss()
if train_config_dict["has_scheduler"]:
    optimizer = optim.AdamW(model.parameters(), lr=train_config_dict["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=train_config_dict["scheduler_step_size"], 
        gamma=train_config_dict["scheduler_gamma"]
    )
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
        outputs = model(sceptr_tokenise(batch, tokenise_method=tokenise_method).to(device=device, dtype=torch.float)).squeeze()
        loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(device))
        
        # Backward pass and optimization
        loss.backward()
        if train_config_dict["max_grad_norm"] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), train_config_dict["max_grad_norm"])
        optimizer.step()
        
        running_loss += loss.item()
        
    if train_config_dict["has_scheduler"]:
        print(f"Learn rate: {scheduler.get_last_lr()}")
        scheduler.step()    
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/num_train_batches:.4f}')
    

    model.eval()
    with torch.no_grad():
        running_loss = 0.0

        for i in range(num_val_batches):
            batch = val.iloc[i*batch_size: (i+1)*batch_size]
            
            # Forward pass
            outputs = model(sceptr_tokenise(batch, tokenise_method=tokenise_method).to(device=device, dtype=torch.float)).squeeze()
            loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(device))
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
        outputs = model(sceptr_tokenise(batch, tokenise_method=tokenise_method).to(device=device, dtype=torch.float)).squeeze()
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