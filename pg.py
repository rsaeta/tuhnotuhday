import dataclasses

import torchaudio
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from ds import WavGlobDataset
from pathlib import Path


class MixGaussianNoise(nn.Module):
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.2):
        super().__init__()
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp().to(lms.device)
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio})'
        return format_string


class RandomLinearFader(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

    def forward(self, lms):
        head, tail = self.gain * ((2.0 * np.random.rand(2)) - 1.0) # gain * U(-1., 1) for two ends
        T = lms.shape[2]
        slope = torch.linspace(head, tail, T, dtype=lms.dtype).reshape(1, 1, T).to(lms.device)
        y = lms + slope # add liniear slope to log-scale input
        return y

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(gain={self.gain})'
        return format_string


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.2, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + list(x))[-self.n:]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


class ACNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.sample_rate = 44100
        self.reps_per_second = 70
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, 10, padding=3, stride=3),  # 630
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 10, padding=3, stride=3),  # 207
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 10, padding=3, stride=3),  # 66
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 10, padding=3, stride=3),  # 19
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 10, padding=3, stride=3),  # 2
            nn.BatchNorm1d(256),
            nn.AvgPool1d(2)
        )
        self.dimension = 256

    def forward(self, x):
        """
        Takes a raw waveform tensor and returns audio representations
        :param x:
        :return:
        """
        return self.encoder(x).squeeze()

    def pre_process(self, x):
        """
        Takes a raw waveform tensor and returns second chunks
        :param x:
        :return:
        """
        x = self.pad_to_nearest_second(x)
        c, n = x.shape
        x = x.transpose(0, 1).view(-1, self.sample_rate // self.reps_per_second, c)
        return x.transpose(1, 2)  # (n_chunks, n_channels, n_samples_per_chunk)

    def pad_to_nearest_second(self, x):
        """
        Pads the input tensor to the nearest second
        :param x: waveform tensor of shape (n_channels, n_samples)
        :return:
        """
        n_samples = x.shape[1]
        n_samples_padded = self.sample_rate - (n_samples % self.sample_rate)
        return nn.functional.pad(x, (0, n_samples_padded))

    def chunk(self, x):
        """
        Produces a row for each second/self.reps_per_second
        :param x: waveform tensor of shape (n_channels, n_samples)
        :return:
        """
        num_chunks = x.shape[1] // self.sample_rate
        return x.chunk(num_chunks, dim=1)


@dataclasses.dataclass
class ByolaFuckeryResult:
    encoding: torch.Tensor
    projection: torch.Tensor
    prediction: torch.Tensor


class ByolaFuckery(nn.Module):

    def __init__(self, project_dim=128):
        super().__init__()
        self.encoder = ACNN()
        self.projector = nn.Linear(self.encoder.dimension, project_dim)
        self.predictor = nn.Linear(project_dim, project_dim)

    def forward(self, x):
        enc = self.encoder(x)
        proj = self.projector(enc)
        pred = self.predictor(proj)
        result = ByolaFuckeryResult(enc, proj, pred)
        return result


def loss_fn(x1, x2):
    x1 = F.normalize(x1, dim=-1, p=2)
    x2 = F.normalize(x2, dim=-1, p=2)
    return 2 - 2 * (x1 * x2).sum(dim=-1)


class ByolaWrapper(nn.Module):

    def __init__(self, exp_factor=0.5):
        super().__init__()
        self.online_net = ByolaFuckery()
        self.target_net = ByolaFuckery()
        self.aug = nn.Sequential(
            MixupBYOLA(),
            MixGaussianNoise(),
        )
        self.target_weights = self.online_net.state_dict()
        self.exp_factor = exp_factor

    def update_target_weights(self):
        online_sd = self.online_net.state_dict()
        self.target_weights = {
            k: v + (self.exp_factor * online_sd[k])
            for k, v in self.target_weights.items()
        }

    def forward(self, x: torch.Tensor):
        x1 = self.aug(x)
        x2 = self.aug(x)

        online_res1 = self.online_net(x1)
        online_res2 = self.online_net(x2)

        with torch.no_grad():
            self.target_net.load_state_dict(self.target_weights)
            target_net = self.target_net
            target_res1 = target_net(x1)
            target_res2 = target_net(x2)

        loss1 = loss_fn(online_res1.prediction, target_res1.projection)
        loss2 = loss_fn(online_res2.prediction, target_res2.projection)

        loss = loss1 + loss2
        return loss.mean()


def get_train_dataloader():
    ds = WavGlobDataset(Path('wavs-train'), num_chunks=10)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    return dl


def get_val_dataloader():
    ds = WavGlobDataset(Path('wavs-val'), num_chunks=10)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    return dl


def train():
    exp_rate = 0.5
    wrapper = ByolaWrapper(exp_rate)
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)
    dl = get_train_dataloader()
    for epoch in range(1):
        for waveform in dl:
            optimizer.zero_grad()
            chunks = wrapper.online_net.encoder.pre_process(waveform[0])
            loss = wrapper(chunks)
            del waveform
            del chunks
            print(loss)
            loss.backward()
            optimizer.step()
            wrapper.update_target_weights()

        torch.save(wrapper.state_dict(), f'byola_cnn_{epoch}.pt')

    breakpoint()


def val():
    exp_rate = 0.5
    wrapper = ByolaWrapper(exp_rate)
    dl = get_val_dataloader()
    wrapper.load_state_dict(torch.load('byola_cnn_9.pt'))
    #wrapper.eval()
    for waveform in dl:
        chunks = wrapper.online_net.encoder.pre_process(waveform[0])
        enc = wrapper.online_net.encoder(chunks)
        breakpoint()
        loss = wrapper(chunks)
        print(loss)
        del waveform
        del chunks
    breakpoint()


if __name__ == '__main__':
    train()
    # val()