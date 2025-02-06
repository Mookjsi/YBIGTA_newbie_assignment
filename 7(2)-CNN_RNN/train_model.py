import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *

if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    # train
    for epoch in tqdm(range(num_epochs)):
        loss_sum = 0.0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()

            input_ids = tokenizer(
                data["verse_text"], 
                padding=True, 
                return_tensors="pt"
            ).input_ids.to(device)

            labels_tensor = data["label"].to(device)
            if labels_tensor.dim() == 2:  # (batch_size, 1)인 경우
                labels_tensor = labels_tensor.squeeze(-1)  # (batch_size,)

            all_logits = model(input_ids)

            logits = all_logits[:, -1, :]

            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # validation
        preds = []
        gold_labels = []
        model.eval()
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(
                    data["verse_text"], 
                    padding=True, 
                    return_tensors="pt"
                ).input_ids.to(device)

                val_labels_tensor = data["label"]
                if val_labels_tensor.dim() == 2:  
                    val_labels_tensor = val_labels_tensor.squeeze(-1)
                gold_labels.extend(val_labels_tensor.tolist())

                val_logits = model(input_ids) 
                val_logits = val_logits[:, -1, :]  

                predicted = val_logits.argmax(-1).cpu().tolist()
                preds.extend(predicted)

        macro = f1_score(gold_labels, preds, average='macro')
        micro = f1_score(gold_labels, preds, average='micro')
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"loss: {loss_sum/len(train_loader):.6f} | "
              f"macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")
