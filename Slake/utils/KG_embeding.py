import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# # Initialize BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Load CSV files
def load_dataframe(file_path, delimiter):
    return pd.read_csv(file_path, delimiter=delimiter)


# Identify delimiter for CSV file
def identify_delimiter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    if ',' in first_line:
        return ','
    elif '#' in first_line:
        return '#'
    else:
        return None

def load_KG(kg_subdir_path='/home/yc/cross-modal-search-demo/datasets/Slake/Slake1.0/KG/'):
    # Load all dataframes
    # kg_subdir_path = '/home/yc/cross-modal-search-demo/datasets/Slake/Slake1.0/KG/'  # Please set this to your actual path
    disease_df = load_dataframe(kg_subdir_path + 'disease.csv', identify_delimiter(kg_subdir_path + 'disease.csv'))
    en_disease_df = load_dataframe(kg_subdir_path + 'en_disease.csv', identify_delimiter(kg_subdir_path + 'en_disease.csv'))
    organ_df = load_dataframe(kg_subdir_path + 'organ.csv', identify_delimiter(kg_subdir_path + 'organ.csv'))
    organ_rel_df = load_dataframe(kg_subdir_path + 'organ_rel.csv', identify_delimiter(kg_subdir_path + 'organ_rel.csv'))
    en_organ_df = load_dataframe(kg_subdir_path + 'en_organ.csv', identify_delimiter(kg_subdir_path + 'en_organ.csv'))
    en_organ_rel_df = load_dataframe(kg_subdir_path + 'en_organ_rel.csv',
                                     identify_delimiter(kg_subdir_path + 'en_organ_rel.csv'))
    # Create a graph from the loaded data
    G = nx.Graph()

    # Add nodes and edges to the graph from the disease and organ data
    # (Note: You might need to adjust the column names based on your actual CSV files)
    for df, entity_col, value_col, relation_col, entity_type in [
        (disease_df, 'disease', 'object', 'relation', 'disease'),
        (en_disease_df, 'disease', 'value', 'attribute', 'disease'),
        (organ_df, 'entity', 'value', 'property', 'organ'),
        (en_organ_df, 'organ', 'value', 'attribute', 'organ'),
        (organ_rel_df, 'entity', 'value', 'property', 'organ'),
        (en_organ_rel_df, 'organ', 'value', 'attribute', 'organ')
    ]:
        for _, row in df.iterrows():
            G.add_node(row[entity_col], type=entity_type)
            G.add_edge(row[entity_col], row[value_col], relation=row[relation_col])
    return G


# Define GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        out = torch.mm(A, X)
        out = self.linear(out)
        return out


# Define GCN model
class SimpleGCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super(SimpleGCN, self).__init__()
        self.layer1 = GCNLayer(n_features, n_hidden)
        self.layer2 = GCNLayer(n_hidden, n_classes)

    def forward(self, X, A):
        X = F.relu(self.layer1(X, A))
        X = self.layer2(X, A)
        return X


def initialize_node_feature_matrix(G, model, tokenizer):
    # Prepare adjacency matrix A and degree matrix D
    A = nx.adjacency_matrix(G)
    N = A.shape[0]
    I = sp.eye(N)
    A_hat = A + I
    D_hat = np.array(A_hat.sum(1))
    D_hat_inv_sqrt = np.power(D_hat, -0.5).flatten()
    D_hat_inv_sqrt[np.isinf(D_hat_inv_sqrt)] = 0.
    D_hat_inv_sqrt = sp.diags(D_hat_inv_sqrt)
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
    A_norm = torch.FloatTensor(A_norm.todense()).to("cuda:0")

    # Initialize feature matrix as identity matrix
    node_list = list(G.nodes())
    node_features = [get_bert_embedding(str(node), model, tokenizer) for node in node_list]
    node_feature_matrix = torch.stack(node_features).squeeze()
    return node_feature_matrix, A_norm


def initialize_GCN_model(n_features=768, n_hidden=768, n_classes=64):
    # Initialize model
    return SimpleGCN(n_features, n_hidden, n_classes)

if __name__ == '__main__':
    kg_subdir_path = '/home/yc/cross-modal-search-demo/datasets/Slake/Slake1.0/KG/'
    # Get G,text_model, text_tokenizer
    G = load_KG(kg_subdir_path)
    # Initialize BERT tokenizer and model
    text_model, text_tokenizer = BertModel.from_pretrained('bert-base-uncased'), BertTokenizer.from_pretrained('bert-base-uncased')
    # Get node_feature_matrix, A_norm
    node_feature_matrix, A_norm = initialize_node_feature_matrix(G,text_model, text_tokenizer)
    # Get GCN model
    GCN_model = initialize_GCN_model(n_features=768, n_hidden=768, n_classes=64)
    # Get embeddings
    embeddings = GCN_model(node_feature_matrix, A_norm).detach().numpy()
    # Done!
