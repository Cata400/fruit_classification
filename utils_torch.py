import os
import torch
import torchvision


def read_image_torch(path):
    """
    Read image in format CHW for RGB or HW for grayscale from path using PyTorch.

    Parameters
    ----------
    path : string
        Path to image.

    Returns
    -------
    2D or 3D torch.Tensor
        Image in format CHW for RGB or HW for grayscale.
    """
    img = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
    return img


class OneHotEncoding(object): #TODO: better docstrings
    """
    One-hot encode labels.
    """
    def __init__(self, labels):
        """
        Init function for OneHotEncoding.

        Parameters
        ----------
        labels : list
            List of labels.
        """
        self.labels = sorted(labels)
    
    def __call__(self, current_label):
        one_hot_label = torch.zeros(len(self.labels))
        one_hot_label[self.labels.index(current_label)] = 1
        return one_hot_label


class FruitDataset(torch.utils.data.Dataset): #TODO: better docstrings
    """
    Dataset for fruit images. Inherits from torch.utils.data.Dataset.
    Images are read using read_image_torch function.

    Attributes
    ----------
    """
    def __init__(self, dir, transform=None, target_transform=None):
        """
        _summary_

        Parameters
        ----------
        dir : string
            Path to directory with images.
        transform : _type_, optional
            _description_, by default None
        target_transform : _type_, optional
            _description_, by default None
        """
        self.img_dir = dir
        self.img_labels = os.listdir(dir)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        length = 0
        for label in sorted(self.img_labels):
            length += len(os.listdir(os.path.join(self.img_dir, label)))
            
        return length
    
    def __getitem__(self, index):
        previous_length = 0
        for label in sorted(self.img_labels):
            if index > previous_length + len(os.listdir(os.path.join(self.img_dir, label))) - 1:
                previous_length += len(os.listdir(os.path.join(self.img_dir, label)))
            else:
                current_label = label
                break
        
        current_index = index - previous_length
        img = read_image_torch(os.path.join(self.img_dir, current_label, os.listdir(os.path.join(self.img_dir, current_label))[current_index]))
        
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            current_label = self.target_transform(current_label)
            
        return img, current_label
        

