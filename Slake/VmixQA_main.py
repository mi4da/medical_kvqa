# Outline for the complete code for the SLAKE VQA task using ViT for image embedding, BERT for text embedding,
# and GCN for knowledge graph representation with co-attention mechanism for joint representation.

# Import necessary libraries
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
from torchvision import transforms
from PIL import Image
from Medical_KVQA.Slake.utils import load_KG, initialize_node_feature_matrix, initialize_GCN_model
from Medical_KVQA.Slake.utils import SlakeVQADataset, custom_collate, tools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import sys
import timm
from torchsummary import summary

# load data for KG
kg_subdir_path = 'Slake1.0/KG/'
# Initialize BERT , ViT , GCN and feature map
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
# vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').cuda()
vit_model = timm.create_model(
    'maxvit_base_tf_384.in21k_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
).cuda()
GCN_model = initialize_GCN_model(n_features=768, n_hidden=768, n_classes=768).cuda()
G = load_KG(kg_subdir_path)
node_feature_matrix, A_norm = initialize_node_feature_matrix(G, bert_model, bert_tokenizer)


def get_KG_embedding():
    embeddings = GCN_model(node_feature_matrix, A_norm)
    return embeddings


# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


# Function to get ViT embeddings
def get_vit_embedding(image):
    # image already transformed in datasets
    output = vit_model(image)
    # pooled_output = vit_model.forward_head(output, pre_logits=True)
    return output


# Co-Attention Mechanism
class CoAttention(nn.Module):
    def __init__(self, dim):
        super(CoAttention, self).__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, img_feat, text_feat, kg_feat):
        img_mapped = self.W(img_feat)
        text_mapped = self.W(text_feat)
        kg_mapped = self.W(kg_feat)

        co_att_img = F.softmax(torch.matmul(img_mapped, text_mapped.T), dim=-1)
        co_att_text = F.softmax(torch.matmul(text_mapped, img_mapped.T), dim=-1)
        co_att_kg = F.softmax(torch.matmul(kg_mapped, img_mapped.T), dim=-1)

        img_attended = torch.matmul(co_att_img, text_feat)
        text_attended = torch.matmul(co_att_text, img_feat)
        kg_attended = torch.matmul(co_att_kg, img_feat)

        joint_representation = img_attended + text_attended + kg_attended
        return joint_representation


# Classifier
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


# Combine everything into a VQA model
class VQAModel(nn.Module):
    def __init__(self, text_dim, img_dim, kg_dim, num_classes):
        super(VQAModel, self).__init__()
        self.co_attention = CoAttention(text_dim)  # Assume text, image, and kg dimensions are the same
        self.classifier = Classifier(text_dim, num_classes)

    def forward(self, text, img, kg):
        joint_representation = self.co_attention(img, text, kg)
        output = self.classifier(joint_representation)
        return output


def query_KG(base_type_batch, relation_batch, tail_batch, kg_embeddings):
    # 要把batch分开。
    head_embeddings = []
    for var in zip(base_type_batch, relation_batch, tail_batch):
        base_type = var[0]
        relation = var[1]
        tail = var[2]
        # Step 0
        if base_type == "vqa":
        # if base_type:
            head_embeddings.append(torch.zeros(768))
            continue

        # Step 1: Use BERT model to generate embeddings for given relation and tail
        relation_embedding = get_bert_embedding(relation).cpu().detach().numpy()
        tail_embedding = get_bert_embedding(tail).cpu().detach().numpy()

        # Step 2: Get embeddings of entire KG
        # kg_embeddings = GCN_model(node_feature_matrix, A_norm).detach().numpy()

        # Step 3: Find the most similar node embeddings in KG for relation and tail
        relation_similarities = cosine_similarity(relation_embedding, kg_embeddings)
        tail_similarities = cosine_similarity(tail_embedding, kg_embeddings)

        # Here we are assuming that the node that is most similar to both the relation and tail is our desired 'head'.
        # This is a simplification and may need further refinement.
        combined_similarity = relation_similarities + tail_similarities
        head_idx = np.argmax(combined_similarity)

        head_embedding = kg_embeddings[head_idx]
        head_embeddings.append(torch.tensor(head_embedding))
    return torch.stack(head_embeddings)




# Main function
if __name__ == '__main__':

    # Load data (Replace these parts with your actual data loading mechanism)
    device = "cuda:0"
    batch_size = 4
    root_path = "Slake1.0"
    log_dir = "vis_log/"
    tools.recreate_directory(log_dir)
    writer = SummaryWriter(log_dir)
    # load data
    data_config = timm.data.resolve_model_data_config(vit_model)
    transforms = timm.data.create_transform(**data_config, is_training=True)

    img_dir_path = os.path.join(root_path, "imgs")
    train_json_file_path = os.path.join(root_path, "train.json")
    val_json_file_path = os.path.join(root_path, "validate.json")
    test_json_file_path = os.path.join(root_path, "test.json")

    train_dataset = SlakeVQADataset(json_file=train_json_file_path,
                                    img_dir=img_dir_path, transform=transforms)
    val_dataset = SlakeVQADataset(json_file=val_json_file_path,
                                  img_dir=img_dir_path, transform=transforms)
    test_dataset = SlakeVQADataset(json_file=test_json_file_path,
                                   img_dir=img_dir_path, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    train_classes = getattr(train_dataset, "classes")
    val_classes = getattr(val_dataset, "classes")
    test_classes = getattr(test_dataset, "classes")
    all_classes = train_classes + val_classes + test_classes
    # 使用LabelEncoder将字符串映射到整数索引
    label_encoder = LabelEncoder()
    integer_indices = label_encoder.fit_transform(all_classes)
    # 获取类别总数
    num_classes = len(label_encoder.classes_)
    print("总类别数：\n"+100*"*")
    # Initialize VQA model
    vqa_model = VQAModel(768, 768, 768, num_classes=num_classes).to(
        "cuda:0")  # Replace num_classes with actual number of classes

    # Get embeddings
    # kg_embedding = get_KG_embedding()  # Replace with actual KG representation from GCN
    # train
    optimizer = torch.optim.Adam(list(vqa_model.parameters()), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 100
    kg_embedding = get_KG_embedding().cpu().detach().numpy()  # 将变量从gpu下载并且截断梯度传播

    fusion_model_parameters = tools.count_parameters(vqa_model)

    print(vqa_model)
    vit_model_parameters = tools.count_parameters(vit_model)
    print(vit_model)
    bert_model_parameters = tools.count_parameters(bert_model)
    print(bert_model)
    gcn_modl_parameters = tools.count_parameters(GCN_model)
    print(GCN_model)
    print(
        f"可计算参数 ：fusion: {fusion_model_parameters}M\nvit: {vit_model_parameters}M\nbert: {bert_model_parameters}M\nGCN: {gcn_modl_parameters}M")

    for epoch in range(num_epochs):
        # train
        vit_model.train()
        bert_model.train()
        GCN_model.train()
        vqa_model.train()

        total_loss = 0.0
        total_batches = 0
        # train
        for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")):
            optimizer.zero_grad()
            image = batch['image'].to("cuda:0")
            question = batch['question']
            # 将gt转换为整数标签
            answer = torch.tensor(label_encoder.transform(batch['answer'])).to("cuda:0")

            img_embedding = get_vit_embedding(image)
            question_embedding = get_bert_embedding(question)

            base_type = batch['base_type']
            head = batch['head']
            relation = batch['relation']
            tail = batch['tail']
            head_embedding = query_KG(base_type, relation, tail, kg_embedding).to("cuda:0")

            outputs = vqa_model(img_embedding, question_embedding, head_embedding)
            # 计算损失并且反向传播
            loss = loss_function(outputs, answer)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            # tqdm.tqdm.write(f"Step [{step}] - step_loss: {loss.item():.4f}", end='\r')
            print("step-loss:",loss.item(), end="\r")
            sys.stdout.flush()
            writer.add_scalar('Train/Step Loss', loss.item(), step + epoch * len(train_loader))



        average_loss = total_loss / total_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        # 使用SummaryWriter将训练损失写入TensorBoard
        writer.add_scalar('Train/Epoch_Loss', average_loss, epoch)


        if epoch % 1 == 0:
            # validation
            # 将模型设置为评估模式
            vit_model.eval()
            bert_model.eval()
            GCN_model.eval()
            vqa_model.eval()

            total_val_loss = 0.0
            total_val_batches = 0
            all_predictions = []  # 存储所有批次的预测
            all_labels = []  # 存储所有批次的真实标签

            # 验证
            with torch.no_grad():  # 禁用梯度计算
                for batch in tqdm.tqdm(val_loader, desc="Validation"):
                    image = batch['image']
                    question = batch['question']
                    answer = torch.tensor(label_encoder.transform(batch['answer']), dtype=torch.long).to("cuda:0")

                    img_embedding = get_vit_embedding(image)
                    question_embedding = get_bert_embedding(question)

                    base_type = batch['base_type']
                    head = batch['head']
                    relation = batch['relation']
                    tail = batch['tail']
                    head_embedding = query_KG(base_type, relation, tail, kg_embedding).to("cuda:0")

                    outputs = vqa_model(img_embedding, question_embedding, head_embedding)

                    # 计算损失
                    val_loss = loss_function(outputs, answer)
                    total_val_loss += val_loss.item()
                    total_val_batches += 1

                    # 获取预测
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(answer.cpu().numpy())

            # 计算准确度
            accuracy = accuracy_score(all_labels, all_predictions)

            average_val_loss = total_val_loss / total_val_batches
            print(f"Validation Loss: {average_val_loss:.4f}")
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")
            # 使用SummaryWriter将验证损失和准确度写入TensorBoard
            writer.add_scalar('Validation/Loss', average_val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', accuracy, epoch)

