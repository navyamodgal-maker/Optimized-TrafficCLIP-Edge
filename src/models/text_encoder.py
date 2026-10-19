import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", output_dim=1024):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.projection = nn.Linear(768, output_dim)

    def forward(self, text_list):
        device = next(self.parameters()).device
        tokens = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.bert(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)

if __name__ == "__main__":
    encoder = TextEncoder()
    texts = ["A network traffic photo of Skype", "A network traffic photo of Zoom"]
    out = encoder(texts)
    print(out.shape)  # expect torch.Size([2, 1024])