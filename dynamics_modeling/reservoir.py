import numpy as np

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    Objective,
    Constraint,
)
from pyomo.core import NonNegativeReals  # type: ignore
from typing import Callable


def init(
    precip: np.ndarray,
    evap_rate: np.ndarray,
    max_volume: float,
    max_area: float,
    max_pump_rate: float,
    area_to_volume: Callable[[float], float],
) -> ConcreteModel:
    model = ConcreteModel()

    num_periods = precip.size
    model.K_periods = RangeSet(0, num_periods - 1)
    model.K_points = RangeSet(0, num_periods)

    # Variables
    model.V = Var(model.K_points, domain=NonNegativeReals)  # Volume
    model.S = Var(model.K_points, domain=NonNegativeReals)  # Storage
    model.A = Var(model.K_points, domain=NonNegativeReals)  # Surface area
    model.E = Var(model.K_periods, domain=NonNegativeReals)  # Evaporation
    model.Q = Var(model.K_periods, domain=NonNegativeReals)  # Pumping
    model.W = Var(model.K_periods, domain=NonNegativeReals)  # Overflow

    # Objective function
    model.min_overflow = Objective(
        expr=lambda model: sum(model.W[k] for k in model.K_periods)
    )

    # Constraints
    model.volumetric_balance = Constraint(
        model.K_periods,
        rule=lambda model, k: model.V[k + 1]
        == model.S[k] + precip[k] - model.E[k] - model.Q[k],
    )  # [1]

    model.overflow = Constraint(
        model.K_periods,
        rule=lambda model, k: model.W[k]
        == (model.V[k + 1] - max_volume + abs(model.V[k + 1] - max_volume)) / 2,
    )  # [2]

    model.evaporation = Constraint(
        model.K_periods,
        rule=lambda model, k: model.E[k]
        == (model.A[k] + model.A[k + 1]) / 2 * evap_rate[k],
    )  # [7]

    model.bound_storage = Constraint(
        model.K_periods,
        rule=lambda model, k: model.S[k + 1] == model.V[k + 1] - model.W[k],
    )  # [3]

    model.storage_to_area = Constraint(
        model.K_points,
        rule=lambda model, k: model.S[k] == area_to_volume(model.A[k]),
    )  # [4]

    model.bound_area = Constraint(
        model.K_points, rule=lambda model, k: model.A[k] <= max_area
    )  # [5]

    model.bound_pumping = Constraint(
        model.K_periods, rule=lambda model, k: model.Q[k] <= max_pump_rate
    )  # [8]

    model.constant = Constraint(rule=lambda model: model.S[0] == 50000)

    return model
