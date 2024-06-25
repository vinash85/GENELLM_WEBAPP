from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from GENELLM_WEBAPP.GENELLM_WEBAPP.util import FineTunedBERT, getEmbeddings
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset

# app = Flask(__name__)







import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import collections
import pandas as pd
from tqdm import tqdm
from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import collections
from tqdm import tqdm

app = Flask(__name__)

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
        token_embeddings[input_mask_expanded == 0] = -1e9
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_embeddings

def getEmbeddings(text, model=None, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", max_length=512, batch_size=1000, gene2vec_flag=False, gene2vec_hidden=200, pool="mean"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, FineTunedBERT):
        model_name = model.model_name            
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif isinstance(model, collections.OrderedDict):
        state_dict = model.copy() 
        model = FineTunedBERT(pool=pool, model_name=model_name, device=device).to(device)
        model.load_state_dict(state_dict)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    else:
        model = FineTunedBERT(pool=pool, model_name=model_name, device=device).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    model = nn.DataParallel(model)
    tokens = tokenizer.batch_encode_plus(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device), batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)
    
    concat_embeddings = torch.cat(embeddings, dim=0)
    return concat_embeddings

def load_model_with_filtered_state_dict(model_class, state_dict_path, device):
    model = model_class(pool="mean", model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=device)
    state_dict = torch.load(state_dict_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "gene2vecFusion" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gene_embeddings = pd.read_csv('/home/tailab/data/gene_embeddings.csv')
disease_embeddings = pd.read_csv('/home/tailab/data/disease_embeddings.csv')

gene_names = gene_embeddings['Gene name'].tolist()
disease_names = disease_embeddings['Disease'].tolist()

model = load_model_with_filtered_state_dict(FineTunedBERT, '/home/tailab/data/state_dict_0.pth', device)
model.eval()

def compute_embeddings(text, model, max_length=512, batch_size=16):
    tokenizer = BertTokenizerFast.from_pretrained(model.model_name)
    tokens = tokenizer.batch_encode_plus(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    model.eval()
    for batch_input_ids, batch_attention_mask in dataloader:
        with torch.no_grad():
            pooled_embeddings, _, _ = model(batch_input_ids.to(device), batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)

    concat_embeddings = torch.cat(embeddings, dim=0)
    return concat_embeddings

@app.route('/')
def index():
    return render_template('index.html', gene_names=gene_names, disease_names=disease_names)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()
    selected_type = request.form['type']
    selected_name = request.form['name']

    if text:
        # Use text input for prediction
        embedding = compute_embeddings([text], model).detach().cpu().numpy()
        
        gene_similarities = cosine_similarity(embedding, gene_embeddings.iloc[:, 1:].values)
        disease_similarities = cosine_similarity(embedding, disease_embeddings.iloc[:, 1:].values)
        
        top_genes_indices = np.argsort(-gene_similarities[0])[:10]
        top_diseases_indices = np.argsort(-disease_similarities[0])[:10]
        
        top_genes_indices = top_genes_indices[top_genes_indices < len(gene_embeddings)]
        top_diseases_indices = top_diseases_indices[top_diseases_indices < len(disease_embeddings)]
        
        top_genes = gene_embeddings.iloc[top_genes_indices]['Gene name'].tolist()
        top_diseases = disease_embeddings.iloc[top_diseases_indices]['Disease'].tolist()
    else:
        # Use dropdown selection for prediction
        if selected_type == 'gene':
            selected_embedding = gene_embeddings[gene_embeddings['Gene name'] == selected_name].iloc[:, 1:].values
            similarities = cosine_similarity(selected_embedding, gene_embeddings.iloc[:, 1:].values)
            top_genes_indices = np.argsort(-similarities[0])[:10]
            top_genes_indices = top_genes_indices[top_genes_indices < len(gene_embeddings)]
            
            similarities = cosine_similarity(selected_embedding, disease_embeddings.iloc[:, 1:].values)
            top_diseases_indices = np.argsort(-similarities[0])[:10]
            top_diseases_indices = top_diseases_indices[top_diseases_indices < len(disease_embeddings)]
            
            top_genes = gene_embeddings.iloc[top_genes_indices]['Gene name'].tolist()
            top_diseases = disease_embeddings.iloc[top_diseases_indices]['Disease'].tolist()
        else:
            selected_embedding = disease_embeddings[disease_embeddings['Disease'] == selected_name].iloc[:, 1:].values
            similarities = cosine_similarity(selected_embedding, disease_embeddings.iloc[:, 1:].values)
            top_diseases_indices = np.argsort(-similarities[0])[:10]
            top_diseases_indices = top_diseases_indices[top_diseases_indices < len(disease_embeddings)]
            
            similarities = cosine_similarity(selected_embedding, gene_embeddings.iloc[:, 1:].values)
            top_genes_indices = np.argsort(-similarities[0])[:10]
            top_genes_indices = top_genes_indices[top_genes_indices < len(gene_embeddings)]
            
            top_genes = gene_embeddings.iloc[top_genes_indices]['Gene name'].tolist()
            top_diseases = disease_embeddings.iloc[top_diseases_indices]['Disease'].tolist()

    return render_template('result.html', top_genes=top_genes, top_diseases=top_diseases)




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5000)
