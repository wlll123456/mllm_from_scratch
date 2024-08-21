import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvison.transforms as transforms
from torchvision.datasets import cifar10
from transformers import bertTokenizer, BertModel
import timm
import numpy as np  

class ViT(nn.Module):
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=output_dim) 
        
    def forward(self, x):
        return self.vit(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def forward(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        output = self.model(**encoded_input)
        return output.last_hidden_state[:, 0, :]
        

def load_cifar10_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])
    train_dataset = cifar10.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    classes = train_dataset.classes
    return loader, classes