import h5py
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

#called from TrainModel.py to retreive the full dataset in correct form
class PCamDataset(Dataset):
    def __init__(self, h5_path_x, h5_path_y, transform=None):
        self.h5_path_x = h5_path_x
        self.h5_path_y = h5_path_y
        self.transform = transform

        with h5py.File(self.h5_path_x, 'r') as fx:
            self.x = fx['x'][:]
        with h5py.File(self.h5_path_y, 'r') as fy:
            self.y = fy['y'][:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, int(label)
