from __future__ import annotations

from typing import Any, Mapping
import time
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import StatTracker, accuracy, get_output_shape
from dask.distributed import get_worker


logger = logging.getLogger(__name__)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    use_BN: bool = True,
) -> nn.Sequential:
    """Simple convolutional block.

    :param in_channels:
        number of input channels
    :param out_channels:
        number of output channels
    :param kernel_size:
        kernel size
    :param stride:
        Stride of the convolution
    :param padding:
        padded value
    :param use_BN:
        if BN is applied

    :return: a convolutional block layer
    """
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    activation = nn.ReLU(inplace=False)
    if use_BN:
        batch_norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(conv_layer, activation, batch_norm)
    else:
        return nn.Sequential(conv_layer, activation)


class Model(nn.Module):
    """The model to optimize"""

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int],
        config: Mapping[str, Any],
    ):
        super().__init__()
        try:
            self.my_worker_id = get_worker().name
        except ValueError:
            self.my_worker_id = 0

        in_channels = input_shape[0]
        out_channels = config["n_channels_conv_0"]
        n_conv_layers = config["n_conv_layers"]
        kernel_size = config["kernel_size"]
        use_BN = config["use_BN"]
        glob_av_pool = config["global_avg_pooling"]
        dropout_rate = config["dropout_rate"]

        # Compute the dimensions of the convolutional layers
        conv_channels: list[tuple[int, int]] = [(in_channels, out_channels)]
        for layer_i in range(1, n_conv_layers):
            previous_layer_i = layer_i - 1

            previous_dimensions = conv_channels[previous_layer_i]
            _, previous_out = previous_dimensions

            layer_in = previous_out
            layer_out = config.get(f"n_channels_conv_{layer_i}", previous_out * 2)

            conv_channels.append((layer_in, layer_out))

        layers = []
        for (in_channels, out_channels) in conv_channels:
            conv = conv_block(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                use_BN=use_BN,
            )
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.extend([conv, pool])

        self.conv_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1) if glob_av_pool else nn.Identity()
        self.output_size = num_classes

        fc_0_in = get_output_shape(
            self.conv_layers,
            self.pooling,
            shape=input_shape,
        )
        fc_0_out = config["n_channels_fc_0"]
        n_fc_layers = config["n_fc_layers"]

        # Compute the dimenions
        fc_sizes = [(fc_0_in, fc_0_out)]
        for layer_i in range(1, n_fc_layers):
            previous_layer_i = layer_i - 1

            _, fc_previous_out = fc_sizes[previous_layer_i]
            fc_in = fc_previous_out
            fc_out = config.get(f"n_channels_fc_{layer_i}", fc_previous_out // 2)

            fc_sizes.append((fc_in, fc_out))

        self.fc_layers = nn.ModuleList(
            [nn.Linear(int(n_in), int(n_out)) for n_in, n_out in fc_sizes]
        )

        _, last_fc_out = fc_sizes[-1]
        self.last_fc = nn.Linear(int(last_fc_out), self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.time_train = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    def train_fn(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        loader: DataLoader,
        device: str | torch.device,
    ) -> tuple[float, float]:
        """Training method.

        :param optimizer: optimization algorithm
        :criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: (accuracy, loss) on the data
        """
        time_begin = time.time()
        score_tracker = StatTracker()
        loss_tracker = StatTracker()

        self.train()

        # itr = tqdm(loader)
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, labels, topk=(1,))[0]  # accuracy given by top 1
            n = images.size(0)
            loss_tracker.update(loss.item(), n)
            score_tracker.update(acc.item(), n)

            # itr.set_description(f"(=> Training) Loss: {loss_tracker.avg:.4f}")
            if self.my_worker_id:
                logger.debug(f"(=> Worker:{self.my_worker_id} Training) Loss: {loss_tracker.avg:.4f}")
            else:
                logger.debug(f"(=> Training) Loss: {loss_tracker.avg:.4f}")

        self.time_train += time.time() - time_begin
        logger.info(f"Worker:{self.my_worker_id} training time: {self.time_train}")
        return score_tracker.avg, loss_tracker.avg

    def eval_fn(
        self,
        loader: DataLoader,
        device: str | torch.device,
    ) -> float:
        """Evaluation method

        :param loader: data loader for either training or testing set
        :param device: torch device

        :return: accuracy on the data
        """
        score_tracker = StatTracker()
        self.eval()

        # t = tqdm(loader)
        with torch.inference_mode():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc = accuracy(outputs, labels, topk=(1,))[0]
                score_tracker.update(acc.item(), images.size(0))

                # t.set_description(f"(=> Test) Score: {score_tracker.avg:.4f}")
            if self.my_worker_id:
                logger.debug(f"(=> Worker:{self.my_worker_id}) Accuracy: {score_tracker.avg:.4f}")
            else:
                logger.debug(f"Accuracy: {score_tracker.avg:.4f}")

        return score_tracker.avg
