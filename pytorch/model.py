import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
from tqdm import tqdm

from dataset import restore


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._make_conv(1, 8, kernel=3, stride=2, act='leaky_relu'),
            self._make_conv(8, 32, kernel=3, stride=2, act='leaky_relu'),
            self._make_conv(32, 16, kernel=3, stride=1, act='leaky_relu'),
            self._make_conv(16, 10, kernel=3, stride=1, act='leaky_relu'),
            self._make_conv(10, 10, kernel=2, stride=1),
        )

    def _make_conv(self, in_c, out_c, kernel=1, stride=1, act=None):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride)]
        layers.append(nn.BatchNorm2d(out_c))
        if act == 'relu':
            layers.append(nn.ReLU())
            nn.init.kaiming_normal_(layers[0].weight, nonlinearity='relu')
        elif act == 'leaky_relu':
            layers.append(nn.LeakyReLU())
            nn.init.kaiming_normal_(layers[0].weight, nonlinearity='leaky_relu')
        elif act is None:
            nn.init.xavier_normal_(layers[0].weight)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class RunningAverage(object):
    def __init__(self):
        super().__init__()
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x.item()) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return 'x'
        return f'{self.avg:.3f}'


class MNISTClassifier(object):
    def __init__(self, ckpt_dir, device):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.epoch_dir = None
        self.pbar = None
        self.msg = None

        self.device = device
        self.model = MNISTModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        print(self.model)
        print('CKPT:', self.ckpt_dir)

    def _train(self):
        self.msg.update({
            'loss': RunningAverage(),
            'acc': RunningAverage(),
        })
        self.model.train()
        for img_batch, lbl_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

            self.optimizer.zero_grad()
            out_batch = self.model(img_batch).squeeze()
            loss = self.criterion(out_batch, lbl_batch)
            loss.backward()
            self.optimizer.step()

            prd_batch = torch.argmax(out_batch, dim=1)
            corrects = (prd_batch == lbl_batch).sum().float()
            accuracy = corrects / img_batch.size(0)

            self.msg['loss'].update(loss)
            self.msg['acc'].update(accuracy)
            self.pbar.update(len(img_batch))
            self.pbar.set_postfix(**self.msg)

    def _valid(self):
        self.msg.update({
            'val_loss': RunningAverage(),
            'val_acc': RunningAverage()
        })
        self.model.eval()
        for img_batch, lbl_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

            out_batch = self.model(img_batch).squeeze()
            loss = self.criterion(out_batch, lbl_batch)

            prd_batch = torch.argmax(out_batch, dim=1)
            corrects = (prd_batch == lbl_batch).sum().float()
            accuracy = corrects / img_batch.size(0)

            self.msg['val_loss'].update(loss)
            self.msg['val_acc'].update(accuracy)
        self.pbar.set_postfix(**self.msg)

    def _vis(self):
        self.model.eval()
        idx = 0
        for img_batch, lbl_batch in iter(self.vis_loader):
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            out_batch = self.model(img_batch).squeeze()
            prd_batch = torch.argmax(out_batch, dim=1)
            for img, lbl, prd in zip(img_batch, lbl_batch, prd_batch):
                filename = f'{idx:05d} - pred {prd} - lbl {lbl}.jpg'
                img_path = self.epoch_dir / filename
                save_image(restore(img), str(img_path))
                idx += 1

    def _log(self):
        # log
        new_row = dict((k, v.avg) for k, v in self.msg.items())
        self.log = self.log.append(new_row, ignore_index=True)
        self.log.to_csv(str(self.ckpt_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(dpi=100)
        self.log[['loss', 'val_loss']].plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(self.ckpt_dir / 'loss.jpg'))
        # plot acc
        fig, ax = plt.subplots(dpi=100)
        self.log[['acc', 'val_acc']].plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(self.ckpt_dir / 'acc.jpg'))
        # Close plot to prevent RE
        plt.close()
        # model
        torch.save(self.model, str(self.epoch_dir / 'model.pth'))

    def fit(self, train_dataset, valid_dataset, vis_dataset, epoch=50):
        self.train_loader = DataLoader(train_dataset,
                batch_size=64, shuffle=True, num_workers=3)
        self.valid_loader = DataLoader(valid_dataset,
                batch_size=64, shuffle=False, num_workers=3)
        self.vis_loader = DataLoader(vis_dataset,
                batch_size=32, shuffle=False, num_workers=1)

        self.log = pd.DataFrame()
        for ep in range(epoch):
            self.epoch_dir = (self.ckpt_dir / f'{ep:03d}')
            self.epoch_dir.mkdir()
            self.msg = dict()

            tqdm_args = {
                'total': len(train_dataset),
                'ascii': True,
                'desc': f'Epoch: {ep:03d}',
            }
            with tqdm(**tqdm_args) as self.pbar:
                self._train()
                with torch.no_grad():
                    self._valid()
                    self._vis()
                    self._log()
