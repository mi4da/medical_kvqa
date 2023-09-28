import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


# Load JSON data from a file
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Custom collate function to skip None values
def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)


# Define the Dataset class
class SlakeVQADataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, select_open=False, select_kvqa=False):
        # TODO: 暂时先不处理开放式
        # 对于所有的问题，一视同仁看作需要使用外部知识，全部使用整个知识图谱作为嵌入
        if select_open:
            self.vqa_data = load_json(json_file)
        else:
            self.vqa_data = []
            self.classes = []
            for i in load_json(json_file):
                if i['answer_type'] == "CLOSED":
                    self.vqa_data.append(i)
                    self.classes.append(i['answer'])
                    self.num_classes = set(self.classes)

        self.img_dir = img_dir
        self.transform = transform
        self.select_open = select_open

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        vqa_item = self.vqa_data[idx]

        folder_name, img_name = os.path.split(vqa_item["img_name"])
        img_path = os.path.join(self.img_dir, folder_name, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"FileNotFoundError for image path: {img_path}. Skipping this sample.")
            return None

        question = vqa_item["question"]
        answer = vqa_item["answer"]
        answer_type = vqa_item['answer_type']
        base_type = vqa_item['base_type']
        triple = vqa_item['triple']
        head, relation, tail = triple[0], triple[1], triple[2]
        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "question": question, "answer": answer, "answer_type": answer_type,
                  "base_type": base_type, "head": head, "relation": relation, "tail": tail}

        return sample




# Example usage
if __name__ == "__main__":
    # Set the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    root_path = "/home/yc/cross-modal-search-demo/datasets/Slake/Slake1.0"
    # Replace these paths with the paths to your actual data
    json_file_path = os.path.join(root_path, "train.json")
    img_dir_path = os.path.join(root_path, "imgs")

    train_dataset = SlakeVQADataset(json_file=json_file_path, img_dir=img_dir_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    # Example to get a single batch of data
    for i, batch in enumerate(train_loader):
        if batch is not None:
            print(f"Image tensor shape: {batch['image'].shape}")
            print(f"questions in batch: {batch['question']}")
            print(f"answers in batch: {batch['answer']}")
            print(f"answers type in batch: {batch['answer_type']}")
        else:
            print("Batch is None, possibly due to missing images.")
        break
