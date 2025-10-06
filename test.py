from model_setup import model
import torch
from data_loader import test_loader
from torch.autograd import Variable
from model_setup import loss_fn
model.load_state_dict(torch.load('best_model.pth'))
test_count, test_correct_num = 0., 0.
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    output = model(img)
    loss = loss_fn(output, label)
    test_correct_num += (torch.max(output, dim=1)[1] == label).sum()
    test_count += img.size(0)
print('test_clf_acc:', int(test_correct_num) / test_count)