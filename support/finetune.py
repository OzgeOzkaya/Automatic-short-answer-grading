import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv("ds_finetune_90.csv", encoding="ANSI", sep=";")

model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length=384):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = self.data.iloc[index]["context"]
        question = self.data.iloc[index]["question"]
        answer = self.data.iloc[index]["answer"]

        encoding = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation="only_second",
            padding="max_length"
        )

        # Answer'ı context içindeki başlangıç ve bitiş pozisyonlarına dönüştürün
        start_positions = torch.tensor([context.find(answer)], dtype=torch.long)
        end_positions = torch.tensor([context.find(answer) + len(answer)], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "start_positions": start_positions,
            "end_positions": end_positions
        }


custom_dataset = CustomDataset(df, tokenizer)

train_dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

# Optimizer ve scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Fine-tuning işlemi
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    print(epoch)
    model.train()
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(batch)

    scheduler.step()

model.save_pretrained("fine_tuned_bert_model")
tokenizer.save_pretrained("fine_tuned_bert_model")
