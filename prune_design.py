from __future__ import annotations

from typing import Any

import warnings

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol
from smac import Scenario
from typing import Iterable
import numpy as np

from smac.initial_design.abstract_initial_design import AbstractInitialDesign


class ProvidedInitialDesign(AbstractInitialDesign):
    """Initial design that uses a user-provided list of configurations."""

    def __init__(self, scenario: Scenario, configs: Iterable[Configuration]):
        self.configs = list(configs)
        super().__init__(scenario=scenario, n_configs=len(self.configs))

    def _select_configurations(self) -> list[Configuration]:
        print('#' * 50)
        print('_select_configurations')
        print('#' * 50)
        for config in self.configs:
            print(config)
            config.origin = "Provided Initial Design"

        return self.configs


def _calculate_config_distance(config1, config2):
    distance = 0
    for param in config1.keys():
        value1 = config1.get(param, 0)
        value2 = config2.get(param, 0)
        if value1 is None:
            if value2 is None:
                value1 = 0
            else:
                value1 = -value2
        if value2 is None:
            if value1 is None:
                value1 = 0
            else:
                value2 = -value1
        distance += (value1 - value2) ** 2
    return distance ** 0.5


class PruneDesign(AbstractInitialDesign):
    """This is modified version of Sobol to use metadata to surpass bottom 80% configuration.See
    https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html for further information.
    In this way we can shrink the configuration space
    """

    def __init__(self, scenario: Scenario, configs: Iterable[Configuration]):
        self.bottom_configs = list(configs)
        super().__init__(scenario=scenario, n_configs=scenario.n_trials, max_ratio=0.25)

        if len(self._configspace.get_hyperparameters()) > 21201:
            raise ValueError(
                "The default initial design Sobol sequence can only handle up to 21201 dimensions. "
                "Please use a different initial design, such as the Latin Hypercube design."
            )

    def _select_configurations(self) -> list[Configuration]:
        params = self._configspace.get_hyperparameters()
        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=True, seed=self._rng.randint(low=0, high=10000000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self._n_configs)

        continues_design = self._transform_continuous_designs(
            design=sobol, origin="Initial Design: Prune Design", configspace=self._configspace
        )

        filtered_configs = []
        for config in continues_design:
            keep_config = True
            for avoid_config in self.bottom_configs:
                bad_config = avoid_config.config
                distance = _calculate_config_distance(bad_config, config)
                if distance < 100.0:
                    keep_config = False
                    break  # No need to check other avoid configurations
            if keep_config:
                filtered_configs.append(config)

        print("#" * 20)
        print(f"from {len(continues_design)} configs {len(filtered_configs)} have been selected")
        print("#" * 20)
        return filtered_configs