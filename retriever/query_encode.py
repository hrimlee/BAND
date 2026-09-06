import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def query_encode(input_list, model_path=None):
    model_path = model_path or "ncbi/MedCPT-Query-Encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    for text in tqdm(input_list, desc="query encoding"):
        with torch.no_grad():
            encoded = tokenizer(
                [text],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = model(**encoded).last_hidden_state[:, 0, :]
            embeddings.extend(emb.detach().cpu().numpy())

    return np.vstack(embeddings)
