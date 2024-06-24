from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import requests
import os
from tqdm import tqdm
import collections
import torch.nn as nn
from transformers import AutoModel
import json

app = Flask(__name__)

# GitHub credentials
GITHUB_USERNAME = 'Macaulay001'
GITHUB_TOKEN = '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'


# LFS Pointer Contents
LFS_POINTERS = {
    "model": {
        "oid": "354ca3ab6a18a5356f6e3930054e9346d454e238945042593cc31c07fe597559",
        "size": 443496479
    },
    "gene_embeddings": {
        "oid": "8e3b9c43c406c7bb9fc57b6d49608a0447d05cb5eb86bd602eb6174055e7c918",
        "size": 125817032
    },
    "disease_embeddings": {
        "oid": "df80dae77669ff5489e2fcb72c129dad14aad79644904d8a6000a669a122805d",
        "size": 84821597
    }
}

# Function to get LFS file download URL
def get_lfs_file_download_url(repo, auth, sha256, size):
    session = requests.Session()
    url = f"https://github.com/{repo}/info/lfs/objects/batch"
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json"
    }
    data = {
        "operation": "download",
        "transfer": ["basic"],
        "objects": [
            {"oid": sha256, "size": size}
        ]
    }
    response = session.post(url, headers=headers, json=data, auth=auth)
    if response.status_code != 200:
        raise Exception(f"Failed to get LFS download URL: {response.text}")
    download_action = response.json()['objects'][0]['actions']['download']
    return download_action['href']

# Function to download file
def download_file(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.text}")
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Example usage
repo = "Macaulay001/GENELLM_WEBAPP"
auth = (GITHUB_USERNAME, GITHUB_TOKEN)

# Download the model file
model_lfs = LFS_POINTERS["model"]
model_path = "state_dict_0.pth"
download_url = get_lfs_file_download_url(repo, auth, model_lfs["oid"], model_lfs["size"])
download_file(download_url, model_path)

# Download the gene embeddings file
gene_lfs = LFS_POINTERS["gene_embeddings"]
gene_embeddings_path = "gene_embeddings.csv"
download_url = get_lfs_file_download_url(repo, auth, gene_lfs["oid"], gene_lfs["size"])
download_file(download_url, gene_embeddings_path)

# Download the disease embeddings file
disease_lfs = LFS_POINTERS["disease_embeddings"]
disease_embeddings_path = "disease_embeddings.csv"
download_url = get_lfs_file_download_url(repo, auth, disease_lfs["oid"], disease_lfs["size"])
download_file(download_url, disease_embeddings_path)

# Load embeddings
try:
    gene_embeddings = pd.read_csv(gene_embeddings_path)
    disease_embeddings = pd.read_csv(disease_embeddings_path)
    print("Embeddings loaded successfully.")
except Exception as e:
    print(f"Error loading embeddings: {e}")

# Define the model class
class FineTunedBERT(nn.Module):
    def __init__(self, pool="mean", model_name="bert-base-cased", device="cuda"):
        super(FineTunedBERT, self).__init__()
        self.model_name = model_name
        self.pool = pool
        self.device = device
        
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.bert_hidden = self.bert.config.hidden_size
        self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))

    def forward(self, input_ids_, attention_mask_):
        hiddenState, ClsPooled = self.bert(input_ids=input_ids_, attention_mask=attention_mask_).values()

        # Perform pooling on the hidden state embeddings
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask_)
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask_)
        else:
            raise ValueError('Invalid pooling method.')

        return embeddings, hiddenState, self.pipeline(embeddings)

    def max_pooling(self, hidden_state, attention_mask):
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_embeddings

# Function to load model and filter state_dict
def load_model_with_filtered_state_dict(model_class, state_dict_path, device):
    print(f"Loading state dictionary from {state_dict_path}...")
    model = model_class(pool="mean", model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=device)
    try:
        state_dict = torch.load(state_dict_path, map_location=device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if "gene2vecFusion" not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
    return model

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_with_filtered_state_dict(FineTunedBERT, model_path, device)
model.eval()

def getEmbeddings(text, model, max_length=512, batch_size=1000, pool="mean"):
    tokenizer = BertTokenizerFast.from_pretrained(model.model_name)
    tokens = tokenizer.batch_encode_plus(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenization Done.")
    print("Get Embeddings ...")
    
    embeddings = []
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(model.device), batch_attention_mask.to(model.device))
            embeddings.append(pooled_embeddings)
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    print(concat_embeddings.size())
    return concat_embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    embedding = getEmbeddings([text], model).detach().cpu().numpy()

    if gene_embeddings.empty or disease_embeddings.empty:
        return render_template('result.html', top_genes=[], top_diseases=[], error="Embeddings data is not loaded correctly.")

    gene_similarities = cosine_similarity(embedding, gene_embeddings.iloc[:, 1:].values)
    disease_similarities = cosine_similarity(embedding, disease_embeddings.iloc[:, 1:].values)

    top_genes = gene_embeddings.iloc[np.argsort(-gene_similarities[0])[:10]]['Gene name'].tolist()
    top_diseases = disease_embeddings.iloc[np.argsort(-disease_similarities[0])[:10]]['Disease'].tolist()

    return render_template('result.html', top_genes=top_genes, top_diseases=top_diseases)

if __name__ == '__main__':
    app.run(debug=True)
