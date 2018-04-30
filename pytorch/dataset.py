import pathlib
from torchvision import datasets
from torchvision import transforms

MNIST_MEAN = (0.1307, )
MNIST_STD = (0.3081, )
TRAIN_DIR = './mnist/train/'
VALID_DIR = './mnist/valid/'


def restore(img):
    return img * MNIST_STD + MNIST_MEAN


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

