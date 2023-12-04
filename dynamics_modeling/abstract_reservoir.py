import numpy as np
import matplotlib.pyplot as plt
import scienceplots as _

from pyomo.environ import (
    SolverFactory,
    ConcreteModel,
    RangeSet,
    Var,
    Objective,
    Constraint,
)
from pyomo.core import NonNegativeReals  # type: ignore
from dynamics_modeling.utils import power_law_area_to_volume, linear_area_to_volume
from typing import Any
from contextlib import suppress

plt.style.use("ieee")

WIDTH = 0.25


class Solver:
    """
    Simple wrapper around `pyomo.environ.SolverFactory`
    """

    def __init__(
        self, solver="ipopt", options: dict[str, Any] = {"max_iter": 10000}
    ) -> None:
        self._solver = SolverFactory(solver)

        if options:
            for k, v in options.items():
                self._solver.options[k] = v

    def solve(
        self,
        model: ConcreteModel,
        options: dict[str, Any] = {},
        debug=False,
    ) -> Any:
        return self._solver.solve(model, options=options, tee=debug)


class ReservoirModel:
    """
    Optimization model for reservoir dynamics as a wrapper around `pyomo.environ.ConcreteModel` with utility methods
    """

    def __init__(
        self,
        solver: Solver | str,
        g1: float,
        g2: float,
        catchment_area: float,
        max_area: float,
        max_pump_rate: float,
        precip_rate: np.ndarray,
        pan_evap_rate: np.ndarray,
        evap_coeff=0.8,
        infil_coeff=0.3,
        is_linear=True,
    ) -> None:
        """
        Initializes a reservoir model.

        Parameters
        ----------
        `solver` : `Solver` or `str`
            Solver to use for optimization.
        `g1` : `float`
            Power-law coefficient (`gamma_1`).
        `g2` : `float`
            Power-law coefficient (`gamma_2`).
        `catchment_area` : `float`
            Catchment surface area (`A_c`).
        `max_area` : `float`
            Maximum surface area (`A`).
        `max_pump_rate` : `float`, optional
            Maximum pumping rate (`Q`), by default `0`.
        `precip_rate` : `np.ndarray`
            1D-array of the precipitation rate over each period (same shape as `pan_evap_rate`).
        `pan_evap_rate` : `np.ndarray`
            1D-array of the pan evaporation rate over each period (same shape as `precip_rate`).
        `evap_coeff` : `float`, optional
            Coefficient for the pan evaporation rate (`C_p`), by default `0.8`.
        `infil_coeff` : `float`, optional
            Coefficient for the infiltration rate (`C_i`), by default `0.3`.
        `is_linear` : `bool`, optional
            Determines whether to model with only linear formulas or not (e.g. linear or power-law volume-area relationship), by default `True`.
        """

        # Validate args
        assert (
            catchment_area > 0
        ), f"Catchment surface area must be positive: {catchment_area}"
        assert max_area > 0, f"Maximum surface area must be positive: {max_area}"
        assert (
            precip_rate.shape == pan_evap_rate.shape and len(precip_rate.shape) == 1
        ), f"Precipitation and pan evaporation rates should be defined over the same set of time periods: {precip_rate.shape} != {pan_evap_rate.shape}"
        assert (
            max_pump_rate >= 0
        ), f"Maximum pumping rate must be nonnegative: {max_pump_rate}"

        self._solver = solver if isinstance(solver, Solver) else Solver(solver)

        self._is_linear = is_linear
        self._num_points = precip_rate.size  # TODO: Shouldn't this be `_num_periods`?
        self._g1 = g1
        self._g2 = g2
        self._Ac = catchment_area
        self._A = max_area
        self._S = self._est_volume(self._A)  # Maximum storage volume
        self._Q = max_pump_rate
        self._Cp = evap_coeff
        self._Ci = infil_coeff

        self._p = precip_rate
        self._ep = pan_evap_rate
        self._e = self._Cp * self._ep  # Reservoir evaporation rate
        self._r = self._Ac * self._p  # Rainfall inflow

        self._construct()
        self._results = None

    def _est_volume(self, area: float) -> float:
        """
        Estimates volume from area, depending on `ReservoirModel._is_linear`.
         - `utils.power_law_area_to_volume`
         - `utils.linear_area_to_volume`

        If `ReservoirModel._S` does not exist, defaults to the power-law formula (usually to estimate `ReservoirModel._S` first, regardless of `ReservoirModel._is_linear`).
        """
        if self._is_linear:
            with suppress(AttributeError):
                return linear_area_to_volume(area, self._A, self._S)

        return power_law_area_to_volume(area, self._g1, self._g2)

    def _construct_vars(self) -> None:
        assert (model := self._model), "`ConcreteModel` not initialized"

        model.V = Var(model.K_points, domain=NonNegativeReals)  # Volume
        model.S = Var(model.K_points, domain=NonNegativeReals)  # Storage
        model.A = Var(model.K_points, domain=NonNegativeReals)  # Surface area
        model.E = Var(model.K_periods, domain=NonNegativeReals)  # Evaporation
        model.Q = Var(model.K_periods, domain=NonNegativeReals)  # Pumping
        model.W = Var(model.K_periods, domain=NonNegativeReals)  # Overflow

    def _construct_contraints(self) -> None:
        assert (model := self._model), "`ConcreteModel` not initialized"

        # Constraints
        model.balance_volume = Constraint(
            model.K_periods,
            rule=lambda model, k: model.V[k + 1]
            == model.S[k] + self._r[k] - model.E[k] - model.Q[k],
        )  # Volumetric balance equation [1]

        model.overflow = Constraint(
            model.K_periods,
            rule=lambda model, k: model.W[k]
            == (model.V[k + 1] - self._S + abs(model.V[k + 1] - self._S)) / 2,
        )  # Overflow [2]

        model.evaporation = Constraint(
            model.K_periods,
            rule=lambda model, k: model.E[k]
            == (model.A[k] + model.A[k + 1]) / 2 * self._e[k],
        )  # Evaporation [7]

        model.bound_storage = Constraint(
            model.K_periods,
            rule=lambda model, k: model.S[k + 1] == model.V[k + 1] - model.W[k],
        )  # Bound storage [3]

        model.storage_to_area = Constraint(
            model.K_points,
            rule=lambda model, k: model.S[k] == self._est_volume(model.A[k]),
        )  # Relate storage to area [4]

        # model.bound_area = Constraint(
        #     model.K_points, rule=lambda model, k: model.A[k] <= self._A
        # )  # Bound surface area [5] (TODO: Check if actually redundant)

        model.bound_pumping = Constraint(
            model.K_periods, rule=lambda model, k: model.Q[k] <= self._Q
        )  # Bound pumping [8]

    def _construct(self) -> None:
        self._model = model = ConcreteModel()

        # Set `K` of period indices
        model.K_periods = RangeSet(0, self._num_points - 1)
        model.K_points = RangeSet(0, self._num_points)

        self._construct_vars()

        # Objective function
        model.min_overflow = Objective(
            expr=lambda model: sum(model.W[k] for k in model.K_periods)
        )  # Minimize overflow

        self._construct_contraints()

    def solve(self, options: dict[str, Any] = {}, debug=False) -> None:
        self._results = self._solver.solve(self._model, options=options, debug=debug)

    def plot(self, debug=False) -> None:
        model = self._model

        rainfall = self._r
        volume = np.array([model.V[i].value for i in model.K_points])
        storage = np.array([model.S[i].value for i in model.K_points])
        evaporation = np.array([model.E[i].value for i in model.K_periods])
        pumping = np.array([model.Q[i].value for i in model.K_periods])
        overflow = np.array([model.W[i].value for i in model.K_periods])

        assert (
            np.equal(volume.shape, storage.shape)
            and np.unique(
                [rainfall.shape, evaporation.shape, pumping.shape, overflow.shape]
            ).size
            == 1
        )

        x_periods = np.arange(rainfall.size)

        if debug:
            print("Volume:", volume)
            print("Storage:", storage)
            print("Rainfall:", rainfall)
            print("Evaporation:", evaporation)
            print("Pumping:", pumping)
            print("Overflow:", overflow)

        plt.rcParams["figure.figsize"] = [8, 5]
        plt.title("Hydraulic Storage Model")
        plt.plot(
            volume, "-bo", alpha=0.5, label="Volume", markersize=2
        )  # Could omit last/12-th point (volume[:-1])
        plt.plot(storage, "-ro", alpha=0.5, label="Storage", markersize=2)  # sim.
        plt.bar(
            x_periods,
            rainfall,
            width=WIDTH,
            color="y",
            alpha=0.5,
            label="Rainfall",
        )
        plt.bar(
            x_periods + WIDTH,
            evaporation,
            width=WIDTH,
            color="g",
            alpha=0.5,
            label="Evaporation",
        )
        plt.bar(
            x_periods + 2 * WIDTH,
            pumping,
            color="c",
            width=WIDTH,
            alpha=0.5,
            label="Pumping",
        )
        plt.bar(
            x_periods + 3 * WIDTH,
            overflow,
            width=WIDTH,
            color="m",
            alpha=0.5,
            label="Overflow",
        )
        plt.hlines(
            y=self._S,
            xmin=0,
            xmax=self._num_points,
            alpha=0.5,
            linestyles="dashed",
            label="Max Storage",
        )

        plt.xlabel(f"Time (periods 0-{self._num_points})")
        plt.ylabel("Volume (m$^3$)")
        plt.legend(prop={"size": 8}, ncol=3)
        plt.show()

    def pprint(self) -> None:
        self._model.pprint()
