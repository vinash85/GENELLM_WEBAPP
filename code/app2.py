import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
import requests
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import os
from markupsafe import Markup
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import certifi

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
special_tokens = set(['[CLS]', '[SEP]'])

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '3425'

# SMTP Configuration
SMTP_SERVER = 'smtp.gmail.com'  # For example, Gmail SMTP server
SMTP_PORT = 465
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_FROM = SMTP_USERNAME
EMAIL_TO = 'asahu@salud.unm.edu'
# EMAIL_TO = 'macaulayoladimeji15@gmail.com'
EMAIL_SUBJECT_FEEDBACK = 'New Feedback from LitGENE'
EMAIL_SUBJECT_CONTACT = 'New Contact Message from LitGENE'

# Define Fine-Tuned BERT Model
class FineTunedBERT(nn.Module):
    def __init__(self, pool="mean", model_name="bert-base-cased", device="cuda"):
        super(FineTunedBERT, self).__init__()
        self.model_name = model_name
        self.pool = pool
        self.device = device
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.bert_hidden = self.bert.config.hidden_size
        self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))

    def forward(self, input_ids, attention_mask):
        hiddenState, ClsPooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).values()
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask)
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask)
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

def load_model_with_filtered_state_dict(model_class, state_dict_path, device):
    model = model_class(pool="mean", model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=device)
    state_dict = torch.load(state_dict_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "gene2vecFusion" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embeddings
gene_embeddings = pd.read_csv('/home/tailab/data/gene_embeddings_merged.csv')
disease_embeddings = pd.read_csv('/home/tailab/data/disease_embeddings_merged.csv')
drug_embeddings = pd.read_csv('/home/tailab/data/drug_embeddings_merged.csv')

gene_names = gene_embeddings['Gene name'].tolist()
disease_names = disease_embeddings['Disease'].tolist()
drug_names = drug_embeddings['Drug name'].tolist()

# Load model
model = load_model_with_filtered_state_dict(FineTunedBERT, '/home/tailab/data/state_dict_0.pth', device)
model = nn.DataParallel(model)
model.eval()

def compute_embeddings(text, model, max_length=512, batch_size=16):
    tokenizer = BertTokenizerFast.from_pretrained(model.module.model_name)
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
    with open('counter.txt', 'r') as file:
        visit_count = int(file.read())
    visit_count += 1
    with open('counter.txt', 'w') as file:
        file.write(str(visit_count))
    return render_template('index.html', visit_count=visit_count)

@app.route('/about')
def about():
    return render_template('about.html')

def send_email(subject, body, receiver_email=EMAIL_TO):
    message = MIMEMultipart()
    message["From"] = EMAIL_FROM
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, receiver_email, message.as_string())
        return True
    except smtplib.SMTPException as e:
        print(f"SMTP error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        feedback_type = request.form.get('feedback_type')
        message = request.form.get('message')
        if not message:
            flash('Feedback message is required', 'error')
            return redirect(url_for('feedback'))

        # Prepare email content
        body = f"Name: {name}\nEmail: {email}\nFeedback Type: {feedback_type}\nMessage:\n{message}"
        subject = EMAIL_SUBJECT_FEEDBACK

        if send_email(subject, body):
            flash('Thank you for your feedback. We will get back to you as soon as possible if you provided your email address.', 'success')
        else:
            flash('An error occurred while sending your feedback. Please try again later.', 'error')
        
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        if not name or not email or not subject or not message:
            flash('Name, email, subject, and message are required', 'error')
            return redirect(url_for('contact'))

        # Prepare email content
        body = f"Name: {name}\nEmail: {email}\nMessage:\n{message}"

        if send_email(subject, body):
            flash('Thank you for reaching out to us. We will get back to you shortly.', 'success')
        else:
            flash('An error occurred while sending your message. Please try again later.', 'error')

        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/choose_type', methods=['GET', 'POST'])
def choose_type():
    if request.method == 'POST':
        selected_type = request.form.get('type')
        if selected_type == 'custom':
            return render_template('edit_prompt.html', type='custom', member='', prompt='')
        if selected_type == 'gene':
            item_list = gene_names
        elif selected_type == 'disease':
            item_list = disease_names
        elif selected_type == 'drug':
            item_list = drug_names
        else:
            flash('Invalid type selected', 'error')
            return redirect(url_for('choose_type'))
        return render_template('choose_member.html', type=selected_type, item_list=item_list)
    return render_template('choose_type.html')

@app.route('/edit_prompt', methods=['POST'])
def edit_prompt():
    selected_type = request.form.get('type')
    member = request.form.get('member')
    
    if not member:
        # If no selection was made, proceed with an empty prompt
        prompt = ""
    else:
        if selected_type == 'gene' and member not in gene_names:
            flash(f"The gene '{member}' was not found.", 'error')
            return redirect(url_for('choose_type'))
        elif selected_type == 'disease' and member not in disease_names:
            flash(f"The disease '{member}' was not found.", 'error')
            return redirect(url_for('choose_type'))
        elif selected_type == 'drug' and member not in drug_names:
            flash(f"The drug '{member}' was not found.", 'error')
            return redirect(url_for('choose_type'))
        
        if selected_type == 'gene':
            prompt = gene_embeddings.set_index('Gene name').at[member, 'Summary']
        elif selected_type == 'disease':
            prompt = disease_embeddings.set_index('Disease').at[member, 'Summary']
        elif selected_type == 'drug':
            prompt = drug_embeddings.set_index('Drug name').at[member, 'Summary']
        else:
            prompt = ""
    
    return render_template('edit_prompt.html', type=selected_type, member=member, prompt=prompt)

@app.route('/submit_prompt', methods=['POST'])
def submit_prompt():
    text = request.form.get('text').strip()
    if len(text.split()) > 3:
        try:
            embedding = compute_embeddings([text], model).detach().cpu().numpy()
            gene_similarities = cosine_similarity(embedding, gene_embeddings.iloc[:, 2:].values)
            disease_similarities = cosine_similarity(embedding, disease_embeddings.iloc[:, 2:].values)
            drug_similarities = cosine_similarity(embedding, drug_embeddings.iloc[:, 2:].values)
            top_genes_indices = np.argsort(-gene_similarities[0])[:10]
            top_diseases_indices = np.argsort(-disease_similarities[0])[:10]
            top_drugs_indices = np.argsort(-drug_similarities[0])[:10]
            top_genes = gene_embeddings.iloc[top_genes_indices]['Gene name'].tolist()
            gene_scores = gene_similarities[0][top_genes_indices].tolist()
            top_diseases = disease_embeddings.iloc[top_diseases_indices]['Disease'].tolist()
            disease_scores = disease_similarities[0][top_diseases_indices].tolist()
            top_drugs = drug_embeddings.iloc[top_drugs_indices]['Drug name'].tolist()
            drug_scores = drug_similarities[0][top_drugs_indices].tolist()
            top_genes_data = list(zip(top_genes, gene_scores))
            top_diseases_data = list(zip(top_diseases, disease_scores))
            top_drugs_data = list(zip(top_drugs, drug_scores))
            enrichment_results = enrich_genes(top_genes)
            if enrichment_results:
                kegg_pathways = [entry for entry in enrichment_results if entry['source'] == 'KEGG' and entry['significant']]
                if kegg_pathways:
                    kegg_pathways.sort(key=lambda x: x['p_value'])
                    top_kegg_pathway = kegg_pathways[0]
                else:
                    top_kegg_pathway = None
            else:
                top_kegg_pathway = None

            global pmc_embeddings_df, gene_articles, disease_articles, drug_articles, global_list
            if not global_list:
                pmc_embeddings_df = pd.read_csv('/home/tailab/data/high_new_article_embeddings.csv')
                gene_articles = pd.read_csv('/home/tailab/data/high_gene_pubchem_similarity.csv')
                disease_articles = pd.read_csv('/home/tailab/data/high_disease_pubchem_similarity.csv')
                drug_articles = pd.read_csv('/home/tailab/data/high_drug_pubchem_similarity.csv')
                global_list = True
            return render_template('result.html', top_genes_data=top_genes_data, top_diseases_data=top_diseases_data, top_drugs_data=top_drugs_data, top_kegg_pathway=top_kegg_pathway, text=text)
        except Exception as e:
            return render_template('error.html', error_message=f"An error occurred: {e}")
    else:
        return render_template('error.html', error_message="Please enter a longer prompt.")

@app.route('/analyze', methods=['POST'])
def analyze():
    target_name = request.form['target_name']
    target_type = request.form['target_type']
    try:
        # Fetch the summary based on the target type
        if target_type == 'gene':
            target_row = gene_embeddings[gene_embeddings['Gene name'] == target_name]
        elif target_type == 'disease':
            target_row = disease_embeddings[disease_embeddings['Disease'] == target_name]
        elif target_type == 'drug':
            target_row = drug_embeddings[drug_embeddings['Drug name'] == target_name]

        if not target_row.empty:
            text = target_row.iloc[0]['Summary']
        else:
            return render_template('error.html', error_message=f"{target_type.capitalize()} '{target_name}' not found.")

        embedding = compute_embeddings([text], model).detach().cpu().numpy()
        gene_similarities = cosine_similarity(embedding, gene_embeddings.iloc[:, 2:].values)
        disease_similarities = cosine_similarity(embedding, disease_embeddings.iloc[:, 2:].values)
        drug_similarities = cosine_similarity(embedding, drug_embeddings.iloc[:, 2:].values)
        top_genes_indices = np.argsort(-gene_similarities[0])[:10]
        top_diseases_indices = np.argsort(-disease_similarities[0])[:10]
        top_drugs_indices = np.argsort(-drug_similarities[0])[:10]
        top_genes = gene_embeddings.iloc[top_genes_indices]['Gene name'].tolist()
        gene_scores = gene_similarities[0][top_genes_indices].tolist()
        top_diseases = disease_embeddings.iloc[top_diseases_indices]['Disease'].tolist()
        disease_scores = disease_similarities[0][top_diseases_indices].tolist()
        top_drugs = drug_embeddings.iloc[top_drugs_indices]['Drug name'].tolist()
        drug_scores = drug_similarities[0][top_drugs_indices].tolist()
        top_genes_data = list(zip(top_genes, gene_scores))
        top_diseases_data = list(zip(top_diseases, disease_scores))
        top_drugs_data = list(zip(top_drugs, drug_scores))
        enrichment_results = enrich_genes(top_genes)
        if enrichment_results:
            kegg_pathways = [entry for entry in enrichment_results if entry['source'] == 'KEGG' and entry['significant']]
            if kegg_pathways:
                kegg_pathways.sort(key=lambda x: x['p_value'])
                top_kegg_pathway = kegg_pathways[0]
            else:
                top_kegg_pathway = None
        else:
            top_kegg_pathway = None

        return render_template('result.html', top_genes_data=top_genes_data, top_diseases_data=top_diseases_data, top_drugs_data=top_drugs_data, top_kegg_pathway=top_kegg_pathway, text=text)
    except Exception as e:
        return render_template('error.html', error_message=f"An error occurred: {e}")


@app.route('/word_importance', methods=['POST'])
def word_importance():
    text = request.form['text']
    target_name = request.form['target_name']
    target_type = request.form['target_type']
    try:
        word_embeddings, filtered_tokens = compute_word_embeddings(text, model)
        target_embedding = None
        if target_type == 'gene':
            target_row = gene_embeddings[gene_embeddings['Gene name'] == target_name]
        elif target_type == 'disease':
            target_row = disease_embeddings[disease_embeddings['Disease'] == target_name]
        elif target_type == 'drug':
            target_row = drug_embeddings[drug_embeddings['Drug name'] == target_name]
        if not target_row.empty:
            target_embedding = target_row.iloc[:, 2:].values
        else:
            return render_template('error.html', error_message=f"{target_type.capitalize()} '{target_name}' not found.")
        if target_embedding.ndim == 1:
            target_embedding = target_embedding.reshape(1, -1)
        word_cosine_similarities = cosine_similarity(word_embeddings, target_embedding)
        token_similarities = list(zip(filtered_tokens, word_cosine_similarities.flatten()))
        highlighted_text = generate_html_with_gradient_highlight(text, token_similarities, power=2)
        return render_template('analyze.html', target_name=target_name, highlighted_text=Markup(highlighted_text), text=text)
    except Exception as e:
        return render_template('error.html', error_message=f"An error occurred: {e}")

global_list = False
gene_articles = None
disease_articles = None
drug_articles = None
pmc_embeddings_df = None


@app.route('/find_citation', methods=['POST'])
def find_citation():
    text = request.form['text']
    target_name = request.form['target_name']
    target_type = request.form['target_type']
    input_embedding = compute_embeddings([text], model).detach().cpu().numpy()
    # global pmc_embeddings_df, gene_articles, disease_articles, drug_articles, global_list
    if target_type == 'gene':
        target_row = gene_articles[gene_articles['Gene name'] == target_name]
    elif target_type == 'disease':
        target_row = disease_articles[disease_articles['Disease'] == target_name]
    elif target_type == 'drug':
        target_row = drug_articles[drug_articles['Drug name'] == target_name]
    if not target_row.empty:
        pmc_ids = target_row.iloc[0, 1:].values
    else:
        return render_template('error.html', message=f"No articles found for {target_type} '{target_name}'.")
    
    pmc_embeddings = pmc_embeddings_df[pmc_embeddings_df['PMCID'].isin(pmc_ids)]
    article_embeddings = pmc_embeddings.iloc[:, 1:].values
    article_similarities = cosine_similarity(input_embedding, article_embeddings).flatten()
    top_indices = article_similarities.argsort()[-5:][::-1]
    top_articles = pmc_embeddings.iloc[top_indices]
    top_scores = article_similarities[top_indices]
    return render_template('article.html', target_name=target_name, articles=top_articles, text=text, target_type=target_type, target_id=target_name, scores=top_scores)


def compute_word_embeddings(text, model, max_length=512):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    tokenizer = BertTokenizerFast.from_pretrained(model.model_name)
    tokens = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(model.device)
    attention_mask = tokens['attention_mask'].to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.bert(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
    word_embeddings = hidden_states[0, attention_mask[0].bool(), :]
    original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    original_tokens = [token for token, mask in zip(original_tokens, attention_mask[0]) if mask]
    filtered_tokens = [token for token in original_tokens if token.lower() not in stop_words and token not in special_tokens and token.isalpha()]
    return word_embeddings.cpu().numpy(), filtered_tokens

def generate_html_with_gradient_highlight(original_text, token_similarities, power=1):
    adjusted_token_similarities = [(word, value ** power) for word, value in token_similarities]
    adjusted_token_similarities = [(word, value) for word, value in adjusted_token_similarities if not word.startswith('##')]
    min_value = min(value for word, value in adjusted_token_similarities)
    max_value = max(value for word, value in adjusted_token_similarities)
    normalized_token_similarities = [(word, (value - min_value) / (max_value - min_value)) for word, value in adjusted_token_similarities]
    normalized_token_similarities.sort(key=lambda x: x[1], reverse=True)
    n = len(normalized_token_similarities)
    segment_size = n // 4
    segments = [
        normalized_token_similarities[:segment_size],
        normalized_token_similarities[segment_size:2*segment_size],
        normalized_token_similarities[2*segment_size:3*segment_size],
        normalized_token_similarities[3*segment_size:]
    ]
    token_dict = {token: score for token, score in normalized_token_similarities}
    html_paragraph = '<p>'
    for word in original_text.split():
        token = word.lower()
        if token in token_dict:
            score = token_dict[token]
            if token in [w for w, _ in segments[0]] or token in [w for w, _ in segments[2]] or token in [w for w, _ in segments[3]]:
                red_value = int(255)
                green_value = int(255 * (1 - score))
                blue_value = int(255)
                color = f'rgba({red_value}, {green_value}, {blue_value}, 1)'
                html_paragraph += f'<span style="background-color:{color}; color: black">{word}</span> '
            else:
                html_paragraph += f'<span style="background-color:transparent; color: black">{word}</span> '
        else:
            html_paragraph += f'<span style="background-color:transparent; color: black">{word}</span> '
    html_paragraph += '</p>'
    return html_paragraph

def enrich_genes(gene_list):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    data = {
        'organism': 'hsapiens',
        'query': gene_list,
        'sources': ['KEGG'],
        'user_threshold': 0.01,
        'significance_threshold_method': 'fdr',
        'no_evidences': True,
        'no_iea': True,
        'domain_scope': 'annotated'
    }
    headers = {
        'User-Agent': 'FullPythonRequest'
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        try:
            return response.json()['result']
        except ValueError:
            print("Error decoding JSON response")
            print("Response content:", response.content)
            return []
    else:
        print("Error in API request. Status code:", response.status_code)
        print("Response content:", response.content)
        return []

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5000)
