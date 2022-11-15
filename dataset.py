from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.c_path = join(image_dir, "c")
        self.d_path = join(image_dir, "d")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        c = Image.open(join(self.c_path, self.image_filenames[index])).convert('RGB')
        d = Image.open(join(self.d_path, self.image_filenames[index])).convert('RGB')
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        c = transforms.ToTensor()(c)
        d = transforms.ToTensor()(d)
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        c = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(c)
        d = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(d)
        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)
            c = c.index_select(2, idx)
            d = d.index_select(2, idx)
        if self.direction == "a2b":
            return a, b
        else:
            return b, a , c,d

    def __len__(self):
        return len(self.image_filenames)
