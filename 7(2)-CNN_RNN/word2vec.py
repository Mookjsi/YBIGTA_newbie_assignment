import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal
from typing import List
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        self.embeddings.weight.data.normal_(0, 0.1)

        self.to(device)
        pass  

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach().cpu()

    def fit(
        self,
        corpus: List[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        
        logging.info("Starting Word2Vec training on device: %s", device)
        start_time = time.time()
        

        tokenized_corpus: List[List[int]] = []
        for line in corpus:
            tokens = tokenizer.encode(line, add_special_tokens=False)

            if tokenizer.pad_token_id is not None:
                tokens = [t for t in tokens if t != tokenizer.pad_token_id]
            if tokens:
                tokenized_corpus.append(tokens)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for tokens in tqdm(tokenized_corpus, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
                for i, center_token in enumerate(tokens):
                    start = max(i - self.window_size, 0)
                    end = min(i + self.window_size + 1, len(tokens))
                    context_tokens = [tokens[j] for j in range(start, end) if j != i]
                    if not context_tokens:
                        continue

                    if self.method == "skipgram":
                        loss_tensor = self._train_skipgram(
                            center_token,
                            context_tokens,
                            criterion,
                            optimizer
                        )
                    else:
                        loss_tensor = self._train_cbow(
                            center_token,
                            context_tokens,
                            criterion,
                            optimizer
                        )
                    total_loss += loss_tensor.item()

            logging.info(f"Epoch {epoch+1}/{num_epochs} - total_loss: {total_loss:.4f}")

        elapsed = time.time() - start_time
        logging.info(f"Training finished! Elapsed: {elapsed:.2f} seconds")
        pass

    def _train_cbow(
        self,
        center_token: int,
        context_tokens: List[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> Tensor:
        self.zero_grad()
        context_ids = torch.LongTensor(context_tokens).to(device)  # (C,)
        context_embed = self.embeddings(context_ids)                # (C, d_model)
        context_vec = torch.mean(context_embed, dim=0, keepdim=True)  # (1, d_model)

        logits = self.weight(context_vec)  # (1, vocab_size)
        center_label = torch.LongTensor([center_token]).to(device)  # (1,)

        loss = criterion(logits, center_label)
        loss.backward()
        optimizer.step()
        return loss

    def _train_skipgram(
        self,
        center_token: int,
        context_tokens: List[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> Tensor:
        self.zero_grad()

        center_id = torch.LongTensor([center_token]).to(device)   # (1,)
        center_embed = self.embeddings(center_id)                 # (1, d_model)

        ctx_ids = torch.LongTensor(context_tokens).to(device)     # (C,)
        center_embed_expanded = center_embed.expand(len(context_tokens), -1)  # (C, d_model)

        logits = self.weight(center_embed_expanded)               # (C, vocab_size)
        loss = criterion(logits, ctx_ids)
        loss.backward()
        optimizer.step()
        return loss

    pass
