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
    
    Attributes
    ----------
    labels : list or tuple
        List of labels.
    """
    def __init__(self, labels):
        self.labels = sorted(labels)
    
    def __call__(self, current_label):
        one_hot_label = torch.zeros(len(self.labels))
        one_hot_label[self.labels.index(current_label)] = 1
        return one_hot_label


class FruitDataset(torch.utils.data.Dataset): 
    """
    Dataset for fruit images. Inherits from torch.utils.data.Dataset.
    Images are read using read_image_torch function.

    Attributes
    ----------
    img_dir : string
        Path to directory with images.
    img_labels : list or tuple
        List of labels.
    test : bool, optional
        If True, dataset is used for testing, default is False.
    validation : bool, optional
        If True, dataset is used for validation, default is False.
    transform : callable, optional
        Optional transform to be applied on a sample image.
    target_transform : callable, optional
        Optional transform to be applied on a sample label.
    """
    def __init__(self, dir, test=False, validation=False, transform=None, target_transform=None):
        self.img_dir = dir
        self.img_labels = os.listdir(dir)
        self.test = test
        self.validation = validation
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        length = 0
        for i, label in enumerate(sorted(self.img_labels)):
            if not self.test:
                length += len(os.listdir(os.path.join(self.img_dir, label)))
            elif self.validation:                
                length += len(os.listdir(os.path.join(self.img_dir, label))[::2])
            else:
                length += len(os.listdir(os.path.join(self.img_dir, label))[1::2])
            
        return length
    
    def __getitem__(self, index):
        previous_length = 0
        for label in sorted(self.img_labels):
            if not self.test:
                if index > previous_length + len(os.listdir(os.path.join(self.img_dir, label))) - 1:
                    previous_length += len(os.listdir(os.path.join(self.img_dir, label)))
                else:
                    current_label = label
                    break
            elif self.validation:
                if index > previous_length + len(os.listdir(os.path.join(self.img_dir, label))[::2]) - 1:
                    previous_length += len(os.listdir(os.path.join(self.img_dir, label))[::2])
                else:
                    current_label = label
                    break
            else:
                if index > previous_length + len(os.listdir(os.path.join(self.img_dir, label))[1::2]) - 1:
                    previous_length += len(os.listdir(os.path.join(self.img_dir, label))[1::2])
                else:
                    current_label = label
                    break
                
        current_index = index - previous_length
        img = read_image_torch(os.path.join(self.img_dir, current_label, os.listdir(os.path.join(self.img_dir, current_label))[current_index])) / 255
        
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            current_label = self.target_transform(current_label)
            
        return img, current_label
        

class MLP6(torch.nn.Module):
    """
    Multilayer perceptron with 6 hidden layers.

    Attributes
    ----------
    input_size : int
        Size of input.
    no_classes : int
        Number of classes.
    flatten : torch.nn.Flatten
        Flatten layer.
    linear_relu_stack : torch.nn.Sequential
        Sequential layer.
    softmax : torch.nn.Softmax
        Softmax layer.
    """
    def __init__(self, input_size, no_classes):
        super(MLP6, self).__init__()
        self.input_size = input_size
        self.no_classes = no_classes
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.no_classes),
        )
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        # y = self.softmax(y)
        
        return y


def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, device):
    """
    Training loop.

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    val_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    model : torch.nn.Module
        Model to be trained on.
    loss_fn : torch.nn
        Loss function.
    optimizer : torch.optim
        Optimizer.
    device : torch.device
        Device to be used for training.
    """
    train_size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    total_loss, correct = 0, 0
    
    for batch, (x, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f" Batch {batch + 1} / {num_batches} Train loss: {loss:>7f}")
            
    total_loss /= num_batches
    correct /= train_size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
            
    test_loop(val_dataloader, model, loss_fn)


def test_loop(dataloader, model, loss_fn, device):
    """
    Testing loop.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Testing dataloader.
    model : torch.nn.Module
        Model to be tested on.
    loss_fn : torch.nn
        Loss function.
    device : torch.device
        Device to be used for testing.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")