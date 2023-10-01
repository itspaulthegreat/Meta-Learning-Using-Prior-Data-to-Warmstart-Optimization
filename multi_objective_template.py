"""
============================
Multi-objective Optimization
============================
Borrowed from https://automl.github.io/SMAC3/main/examples/3_multi_objective/1_schaffer.html
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import time

import numpy as np
import torch
from thop import profile

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical
)
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs

from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker

from cnn import Model

from datasets import load_deep_woods, load_fashion_mnist


logger = logging.getLogger(__name__)

CV_SPLIT_SEED = 42

# Max fidelities
NUM_EPOCHS = 20
IMG_SIZE = 32
CV_COUNT = 3
DATA_SIZE = 1


def configuration_space(
        device: str,
        dataset: str,
        cv_count: int = 3,
        # budget_type: str = "epochs",
        datasetpath: str | Path = Path("."),
        cs_file: Optional[str | Path] = None
) -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    if cs_file is None:
        # This serves only as an example of how you can manually define a Configuration Space
        # To illustrate different parameter types;
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace(
            {
                "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3),
                "use_BN": Categorical("use_BN", [True, False], default=True),
                "global_avg_pooling": Categorical("global_avg_pooling", [True, False], default=True),
                "n_channels_conv_0": Integer("n_channels_conv_0", (32, 512), default=512, log=True),
                "n_channels_conv_1": Integer("n_channels_conv_1", (16, 512), default=512, log=True),
                "n_channels_conv_2": Integer("n_channels_conv_2", (16, 512), default=512, log=True),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer("n_channels_fc_0", (32, 512), default=512, log=True),
                "n_channels_fc_1": Integer("n_channels_fc_1", (16, 512), default=512, log=True),
                "n_channels_fc_2": Integer("n_channels_fc_2", (16, 512), default=512, log=True),
                "batch_size": Integer("batch_size", (1, 1000), default=200, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1.0),
                    default=1e-3,
                    log=True,
                ),
                "kernel_size": Constant("kernel_size", 3),
                "dropout_rate": Constant("dropout_rate", 0.2),
                "device": Constant("device", device),
                "dataset": Constant("dataset", dataset),
                "cv_count": Constant("cv_count", cv_count),
                # "budget_type": Constant("budget_type", budget_type),
                "datasetpath": Constant("datasetpath", str(datasetpath.absolute())),
            }
        )

        # Add conditions to restrict the hyperparameter space
        use_conv_layer_2 = InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3])
        use_conv_layer_1 = InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3])

        use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
        use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

        # Add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1])
    else:
        with open(cs_file, "r") as fh:
            cs_string = fh.read()
            if cs_file.suffix == ".json":
                cs = cs_json.read(cs_string)
            elif cs_file.suffix in [".pcs", ".pcs_new"]:
                cs = pcs_new.read(pcs_string=cs_string)
        logging.info(f"Loaded configuration space from {cs_file}")

        if "device" not in cs:
            cs.add_hyperparameter(Constant("device", device))
        if "dataset" not in cs:
            cs.add_hyperparameter(Constant("dataset", dataset))
        if "cv_count" not in cs:
            cs.add_hyperparameter(Constant("cv_count", cv_count))
        # if "budget_type" not in cs:
        #     cs.add_hyperparameter(Constant("budget_type", budget_type))
        if "datasetpath" not in cs:
            cs.add_hyperparameter(Constant("datasetpath", str(datasetpath.absolute())))
        logging.debug(f"Configuration space:\n{cs}")

    return cs


def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg(
        cfg: Configuration,
        seed: int
) -> dict[str, float]:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    max_epochs = num_epochs = NUM_EPOCHS
    max_img_size = img_size = IMG_SIZE
    max_cv_splits = cv_count = CV_COUNT
    max_data_size = data_size = DATA_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)
    
    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, _ = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=cv_count, random_state=CV_SPLIT_SEED, shuffle=True)

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    
    training_costs = []
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))
        
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        model = model.to(model_device)

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        start_time = time.time()
        for epoch in range(num_epochs):
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{num_epochs}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")
        training_costs.append(time.time() - start_time)
        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # minimize
    cost = np.mean(training_costs)
    # computing MACs
    # for more info: https://github.com/sovrasov/flops-counter.pytorch/issues/16
    macs, params = profile(model, inputs=(torch.randn(1, 3, img_size, img_size), ))
    
    results = {
        "error": val_error,
        "cost": cost,
        "macs": macs / 1e6  # in Millions
    }
    return results


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg_test(
        cfg: Configuration,
        seed: int
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    max_epochs = num_epochs = NUM_EPOCHS
    max_img_size = img_size = IMG_SIZE
    max_cv_splits = cv_count = CV_COUNT
    max_data_size = data_size = DATA_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)
    
    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(
            datadir=Path(ds_path, "FashionMNIST")
        )
    elif "deepweedsx" in dataset:
        input_shape, train_val, test_data = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset=train_val,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    model = Model(
        config=cfg,
        input_shape=input_shape,
        num_classes=len(train_val.classes),
    )
    model = model.to(model_device)

    model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
    optimizer = model_optimizer(model.parameters(), lr=lr)
    train_criterion = train_criterion().to(device)

    start_time = time.time()
    for epoch in range(num_epochs):
        logging.info(f"Worker:{worker_id} " + "#" * 50)
        logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{num_epochs}]")
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            device=model_device
        )
        logging.info(
            f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}"
        )
    end_time = time.time()

    test_score = model.eval_fn(test_loader, device)
    logging.info(f"Worker:{worker_id} => Test accuracy {test_score:.3f}")
    
    test_error = 1 - test_score  # minimize
    cost = end_time - start_time 
    # computing MACs
    # for more info: https://github.com/sovrasov/flops-counter.pytorch/issues/16
    macs, params = profile(model, inputs=(torch.randn(1, 3, img_size, img_size), ))

    results = {
        "error": test_error,
        "cost": cost,
        "macs": macs / 1e6  # in Millions
    }
    return results


if __name__ == "__main__":
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """

    parser = argparse.ArgumentParser(description="MF example using BOHB.")
    parser.add_argument(
        "--dataset",
        choices=["deepweedsx", "deepweedsx_balanced", "fashion_mnist"],
        default="deepweedsx_balanced",
        help="dataset to use",
    )
    parser.add_argument(
        "--working_dir",
        default="./tmp",
        type=str,
        help="directory where intermediate results are stored",
    )
    parser.add_argument(
        "--runtime",
        default=21600,
        type=int,
        help="Running time (seconds) allocated to run the algorithm",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="num of workers to use with BOHB"
    )
    parser.add_argument(
        "--n_trials", type=int, default=15, help="Number of iterations to run SMAC for"
    )
    parser.add_argument(
        "--cv_count",
        type=int,
        default=3,
        help="Number of cross validations splits to create. "
             "Will not have an effect if the budget type is cv_splits",
    )
    parser.add_argument(
        "--trial_mem_limit",
        type=int,
        default=1024,
        help="memory limit for each trial in MB",
    )
    parser.add_argument(
        "--log_level",
        choices=[
            "NOTSET",
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        default="NOTSET",
        help="Logging level",
    )
    parser.add_argument(
        '--configspace', 
        type=Path, 
        default="multi_objective/configuration_space.json",
        help='Path to file containing the configuration space'
    )
    parser.add_argument(
        '--datasetpath', 
        type=Path, 
        default=Path('./data/'),
        help='Path to directory containing the dataset'
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)

    configspace = configuration_space(
        device=args.device,
        dataset=args.dataset,
        cv_count=args.cv_count,
        datasetpath=args.datasetpath,
        cs_file=args.configspace
    )

    OBJECTIVES = ["error", "cost", "macs"]
    OBJECTIVE_WEIGHTS = [4, 1, 2]

    # Setting up SMAC to run BOHB
    scenario = Scenario(
        name="ExampleMOORun",
        configspace=configspace,
        deterministic=True,
        output_directory=args.working_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        n_workers=args.workers,
        walltime_limit=args.runtime,
        objectives=OBJECTIVES,
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=cnn_from_cfg,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=OBJECTIVE_WEIGHTS,
        ),
        overwrite=True,
        logging_level=args.log_level,  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    # Start optimization
    incumbents = smac.optimize()
    
    # Let's calculate the cost of the default and the incumbent
    logging.info("Optimization done, now calculating the default and incumbent performance.")
    logging.info("------ Summary ------")
    
    logging.info(f"Default configuration: {configspace.get_default_configuration()}")
    default_cost = smac.validate(configspace.get_default_configuration(), seed=args.seed)
    logging.info(f"Default validation accuracy and training time: {1-default_cost[0], default_cost[1]}\n")

    logging.info("Validated objectives from the Pareto front (incumbents): (accuracy, runtime, macs)")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        logging.info(f"--- Validation: {cost}")
# end of file