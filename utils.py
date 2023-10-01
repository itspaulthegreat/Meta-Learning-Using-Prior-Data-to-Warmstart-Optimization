from __future__ import annotations
from smac import HyperparameterOptimizationFacade as HPO
from dataclasses import dataclass
from torch.autograd import Variable
import torch
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
)
from pathlib import Path
from smac.runhistory import TrialValue
from smac.runhistory.dataclasses import TrialInfo
import pandas as pd
import logging

METADATA_CONFIG_COLUMNS = {
    "config:n_conv_layers": int,
    "config:use_BN": bool,
    "config:global_avg_pooling": bool,
    "config:n_channels_conv_0": int,
    "config:n_channels_conv_1": pd.Int64Dtype(),
    "config:n_channels_conv_2": pd.Int64Dtype(),
    "config:n_fc_layers": int,
    "config:n_channels_fc_0": int,
    "config:n_channels_fc_1": pd.Int64Dtype(),
    "config:n_channels_fc_2": pd.Int64Dtype(),
    "config:batch_size": int,
    "config:learning_rate_init": float,
    "config:kernel_size": int,
    "config:dropout_rate": float,
}


@dataclass
class WarmstartConfig:
    config: Configuration
    seed: int
    cost: float
    duration: float
    budget: float | None = None

    def as_trial(self) -> tuple[TrialInfo, TrialValue]:
        """Converts this WarmstartConfig into a TrialInfo and TrialValue.

        This can be used with `optimizer.tell(info, value)` to inform SMAC about
        a result before the optimization starts.
        """
        # Since we're not using Multi-fidelity, budget=self.budget it is 20 for all configurations
        trial_info = TrialInfo(config=self.config, instance=None, seed=self.seed, budget=self.budget)
        trial_value = TrialValue(time=self.duration, cost=self.cost)
        return trial_info, trial_value

    @classmethod
    def from_metadata(
            cls,
            path: Path,
            space: ConfigurationSpace,
    ) -> list[WarmstartConfig]:
        metadata = (
            pd.read_csv(path)
            .astype(METADATA_CONFIG_COLUMNS)
            .rename(columns=lambda c: c.replace("config:", ""))
            .drop(
                columns=[
                    "dataset",
                    "datasetpath",
                    "device",
                    "cv_count",
                    "budget_type",
                    "config_id",
                ]
            )
        )

        config_columns = [c.replace("config:", "") for c in METADATA_CONFIG_COLUMNS]

        configs = []
        for _, row in metadata.iterrows():
            config_dict = row[config_columns].to_dict()  # type: ignore
            try:
                configs.append(
                    WarmstartConfig(
                        config=Configuration(
                            configuration_space=space, values=config_dict
                        ),
                        seed=int(row["seed"]),
                        budget=float(row["budget"]),
                        cost=float(row["cost"]),
                        duration=float(row["time"]),
                    )
                )
            except Exception as e:
                logging.warning(f"Skipping config as not in space:\n{row}\n{e}")

        if len(configs) == 0:
            raise RuntimeError("No configs found that are representable in the space")

        return configs


@dataclass
class StatTracker:
    avg: float = 0
    sum: float = 0
    cnt: float = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def get_output_shape(
        *layers: torch.nn.Sequential | torch.nn.Module,
        shape: tuple[int, int, int],
) -> int:
    """Calculate the output dimensions of a stack of conv layer"""
    channels, w, h = shape
    input = Variable(torch.rand(1, channels, w, h))

    seq = torch.nn.Sequential()
    for layer in layers:
        seq.append(layer)

    output_feat = seq(input)

    # Flatten the data out, and get it's size, this will be
    # the size of what's given to a fully connected layer
    n_size = output_feat.data.view(1, -1).size(1)
    return n_size


def prune_bad_configurations(configs: list[WarmstartConfig], smac: HPO) -> None:
    num_configs = len(configs)
    for idx in range(num_configs):
        info, value = configs[idx].as_trial()
        smac.tell(info, value)
# Can generate a grid like this in order to evaluate baseline and check Pareto front
# However, given the dimensionality of the search space, better to use random search

# num_dict = {
#     'batch_size': 3,
#     'dropout_rate': 0.2,
#     'learning_rate_init': 10,
#     'n_channels_conv_0': 3,
#     'n_channels_fc_0': 3,
#     'n_channels_fc_1': 3,
#     'n_channels_fc_2': 3,
#     'n_conv_layers': 3,
#     'n_fc_layers': 3,
#     'n_channels_conv_1': 3,
#     'n_channels_conv_2': 3
# }
# import ConfigSpace
# ConfigSpace.util.generate_grid(cs, num_dict)
