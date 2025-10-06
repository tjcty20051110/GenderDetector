from cnn import CNN
import torch.nn as nn
import torch
model = CNN().cuda()        #调用cuda()函数，利用GPU加速训练
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)