from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.manifold as skm
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

import numpy as np

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

# Adapted from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# Gaussian noise transform for dataset
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(9, 18, 3, 1)
        self.dropout1 = nn.Dropout2d(0.45)
        self.dropout2 = nn.Dropout2d(0.45)
        self.batchnorm1 = nn.BatchNorm2d(9)
        self.batchnorm2 = nn.BatchNorm2d(18)
        self.fc1 = nn.Linear(450, 150)
        self.dropout3 = nn.Dropout2d(0.1)
        self.batchnorm3 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.batchnorm3(x)
        # x = self.dropout3(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, train_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    train_loss = 0
    correct = 0
    train_correct = 0
    test_num = 0
    train_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # truths = pred.eq(target.view_as(pred))
            # misclassified = [i for i, x in enumerate(truths.numpy().flatten()) if not x]
            # for j in misclassified:
            #     plt.imshow(data[j][0].numpy(), cmap='Greys')
            #     plt.show()

            test_num += len(data)

        # for data, target in train_loader:
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #     train_correct += pred.eq(target.view_as(pred)).sum().item()
        #     train_num += len(data)

    test_loss /= test_num
    # train_loss /= train_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss, train_loss

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        for name, layer in model._modules.items():
            layer.register_forward_hook(get_activation(name))
        model.load_state_dict(torch.load(args.load_model))

        # for kernel in model.state_dict()['conv1.weight'].numpy():
        #     plt.imshow(kernel[0] - kernel[0].min(), cmap='Greys')
        #     plt.show()


        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False, **kwargs)

        test(model, device, test_loader, False)

        x_transformed = skm.TSNE(n_components=2).fit_transform(activation['fc1'].numpy())
        print("beep1")

        colors = ['maroon', 'peru', 'gold', 'olive', 'darkolivegreen', 'dodgerblue', 'navy', 'mediumslateblue', 'orchid', 'palegreen']

        for i in range(1000):
            plt.plot(x_transformed[i][0], x_transformed[i][1], color = colors[test_dataset[i][1]], marker='o', linestyle='none')

        print("beep")

        plt.show()

        # activation['fc1']

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    # transforms.RandomErasing(p=0.4, scale=(0.05,0.15), ratio=(0.3, 3.3), value=0),
                    # transforms.Rand
                    AddGaussianNoise(0.1, 0.1),       # this will need to be tweaked
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.

    indices = {}

    for cl in range(10):
        indices[cl] = []

    for idx, d in enumerate(train_dataset):
        indices[d[1]].append(idx)


    subset_indices_train = []
    subset_indices_valid = []

    for cl, idxs in enumerate(indices.values()):
        length = len(idxs)
        val_indices = list(np.random.choice(idxs, int(0.15 * length), replace=False))
        subset_indices_valid += val_indices
        subset_indices_train += [i for i in idxs if i not in val_indices]

    scale_factor = 1

    # subset_indices_valid = list(np.random.choice(subset_indices_valid, int(len(subset_indices_valid) * scale_factor)))
    subset_indices_train = list(np.random.choice(subset_indices_train, int(len(subset_indices_train) * scale_factor)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        te_loss, tr_loss = test(model, device, val_loader, train_loader)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model_full.pt")

    # plt.plot(list(range(len(test_losses))), test_losses, label='Test Loss', color='blue')
    # plt.plot(list(range(len(train_losses))), train_losses, label='Train Loss', color='red')
    # plt.legend()
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training & Test Loss vs. Epoch")
    # plt.show()




if __name__ == '__main__':
    main()
