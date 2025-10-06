from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import image
import cv2
# 定义读取文件的方式，通过文件路径直接读取
def default_loader(path):
    return image.imread(path)

batch_size = 16
img_size = (64, 64)
class MyDataset(Dataset):
    def __init__(self,txt,transform=None,loader=default_loader):
        """
	           在__init__()函数中得到标签文件中所有图像路径和标签，将其存放在列表中
        """
        super(MyDataset, self).__init__()
        f = open(txt, 'r')
        imgs = []
        line = f.readline()
        while line:
            a = line.split()
            line = f.readline()
            imgs.append((a[0], int(a[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        """
        在__getitim__()函数中直接读取图像和标签
        """
        fn, label = self.imgs[index]
        img = self.loader(fn)
        img = cv2.resize(img, img_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

#将图像转化为Tensor，并做归一化处理
transform = transforms.Compose([transforms.ToTensor()])
Train_txt_path = r'.\CelebA\Img\train_txt.txt'
Val_txt_path = r'.\CelebA\Img\val_txt.txt'
Test_txt_path = r'.\CelebA\Img\test_txt.txt'

train_data = MyDataset(txt=Train_txt_path, transform=transform)
val_data = MyDataset(txt=Val_txt_path, transform=transform)
test_data = MyDataset(txt=Test_txt_path, transform=transform)