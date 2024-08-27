import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import cifar
from transformers import BertTokenizer, BertModel
import timm
import numpy as np  
from torchvision.datasets import CIFAR10
from tqdm import tqdm  # 引入 tqdm
from torch.utils.data import SubsetRandomSampler

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
    train_dataset = cifar.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 随机选择一定数量的数据
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    indices = indices[:100]  # 仅使用部分数据
    
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    classes = train_dataset.classes
    return loader, classes

class CLIP(nn.Module):
    def __init__(self, image_output_dim, text_output_dim):
        super(CLIP, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ViT(image_output_dim)

        self.W_i = nn.Parameter(torch.randn(image_output_dim, text_output_dim))
        self.W_t = nn.Parameter(torch.randn(768, text_output_dim))
        
    def forward(self, images, texts):
        I_f = self.image_encoder(images)
        T_f = self.text_encoder(texts)

        I_e = torch.matmul(I_f, self.W_i)
        T_e = torch.matmul(T_f, self.W_t)

        logits = torch.matmul(I_e, T_e.T)
        return logits

def train_clip(clip_model, dataset, classes, num_epochs=10, learning_rate=1e-3):
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        runing_loss = 0.0
        progress_bar = tqdm(dataset, desc=f"Epoch [{epoch+1}/{num_epochs}]") 
        for images, labels in progress_bar:
            texts = [classes[label] for label in labels]
            logits = clip_model(images, texts)
            labels = torch.arange(len(texts))
            loss_i = nn.CrossEntropyLoss()(logits, labels)
            loss_t = nn.CrossEntropyLoss()(logits.T, labels)
            loss = (loss_i + loss_t) /2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            runing_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # 更新进度条的后缀信息

            print(f'Epoch {epoch}, Loss {loss.item()}')
        print(f'Epoch {epoch}, Loss {runing_loss/len(dataset)}')

def predict_clip(clip_model, images, classes):
    clip_model.eval()
    with torch.no_grad():
        texts = [classes[0] for _ in range(len(images))]
        logits = clip_model(images, texts)
        preds = torch.argmax(logits, dim=1)

    return preds

def main():
    dataset, classes = load_cifar10_dataset()
    clip_model = CLIP(image_output_dim=512, text_output_dim=512)

    train_clip(clip_model, dataset, classes)

    test_images, test_labels = next(iter(dataset))
    pre = predict_clip(clip_model, test_images, classes)

    print(pre)    

if __name__ == '__main__':
    main()