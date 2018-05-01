import pathlib
from torch.utils.data.dataset import Subset, ConcatDataset
from torchvision import datasets
from torchvision import transforms

MNIST_MEAN = (0.1307, )
MNIST_STD = (0.3081, )
TRAIN_DIR = './mnist/train/'
VALID_DIR = './mnist/valid/'


def restore(img):
    return img * MNIST_STD[0] + MNIST_MEAN[0]


MNISTtrain = datasets.MNIST(TRAIN_DIR,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
)

MNISTvalid = datasets.MNIST(VALID_DIR,
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
)

MNISTvis = ConcatDataset([
    Subset(MNISTtrain, list(range(50))),
    Subset(MNISTvalid, list(range(50))),
])
