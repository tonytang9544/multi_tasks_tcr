import transformers
from torchinfo import summary

bert = transformers.BertModel.from_pretrained("bert-base-uncased")

print(summary(bert))