from transformers import BertTokenizer, BertModel

# Load tokenizer from local directory
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model from local directory
model = BertModel.from_pretrained("bert-base-uncased")

# Example usage
inputs = tokenizer("M A A E B", return_tensors="pt")
# outputs = model(**inputs)

# print(outputs.last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)

print(inputs)
