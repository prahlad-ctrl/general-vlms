'''
paper - 'AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE'

Building vision transformer from scratch using PyTorch. I wont be coding every single detail like multi head atention, feed forward, etc,
but ill be coding the main structure of the vision transformer
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import dataloader

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
num_classes = 10
num_channels = 1
img_size = 28
patch_size= 7
patch_num = (patch_size// img_size)
attn_head = 4
embed_dim = 64
transformer_blocks = 4
mlp_nodes = 64

train_data = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = dataloader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

''' Patch Embedding - process of converting the image into patches and then embedding those patches into a vector space '''
class PatchEmbed(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

  def forward(self, x):
    x = self.patch_embed(x)
    # going from ([64, 20, 4, 4]) to ([64, 20, 16])
    x = x.flatten(2)
    # then going from ([64, 20, 16]) to ([64, 16, 20])
    x = x.transpose(1,2)
    return x

''' Transformer Encoder - the main structure of the transformer block, which consists of multi head attention and feed forward network '''
class TransformerEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.ln1 = nn.LayerNorm(embed_dim)
    self.mhattn = nn.MultiheadAttention(embed_dim, attn_head, batch_first=True)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_nodes),
                             nn.GELU(),
                             nn.Linear(mlp_nodes, embed_dim))

  def forward(self, x):
    res1 = x
    x = self.ln1(x)
    x = self.mhattn(x, x, x)[0] + res1 # to get the value at 0th index
    res2 = x
    x = self.ln2(x)
    x = self.mlp(x) + res2
    return x

''' MLP Head - the final classification head of the transformer, which consists of a layer norm and a linear layer '''
class MLP_head(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlphead = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, num_classes))

  def forward(self, x):
    x = self.mlphead(x)
    return x

''' Vision Transformer Wrapper - the main structure of the vision transformer, which consists of all classes '''
class VisionTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embed = PatchEmbed()
    self.cls_tokens = nn.Parameter(torch.randn(1,1, embed_dim))
    self.positional_embed = nn.Parameter(torch.randn(1, patch_num + 1, embed_dim))
    self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
    self.mlphead = MLP_head()

  def forward(self, x):
    x = self.patch_embed(x)
    B = x.shape[0]
    cls_tokens = self.cls_tokens.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), 1)
    x = x + self.positional_embed
    x = self.transformer_blocks(x)
    x = x[:, 0] # only the CLS token
    x = self.mlphead(x)
    return x

lr = 0.001
epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
  train_loss = 0.0
  train_acc = 0.0

  model.train()
  total_images = 0
  correct_images = 0

  for images, labels in train_data:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    predicted = torch.argmax(outputs, dim=1)
    total_images += labels.size(0)
    correct_images += (predicted == labels).sum().item()

  print(f"epoch: {epoch+1}/{epochs}, loss: {train_loss/len(train_data):.4f}, accuracy: {100*correct_images/total_images:.2f}%")