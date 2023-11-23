import numpy as np
import pandas as pd

from typing import Literal
from scipy.optimize import curve_fit


def power_law_area_to_volume(area: float, c1: float, c2: float, mu=1e-8) -> float:
    """
    Estimates volume from area using a power-law relationship.
    To avoid non-differentiablity when `area = 0`, offsets area by `mu` in calculation.
    """
    return c1 * (mu + area) ** c2


def linear_area_to_volume(area: float, max_area: float, max_volume) -> float:
    """
    Estimates volume from area using a linear relationship.
    """
    return max_volume / max_area * area


def load_mesonet(path: str, debug=False) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["valid", "tmpf", "dwpf", "sknt", "p01i"],
        header=0,
        dtype={
            "valid": "string",  # Datetime. Consider "datetime64[ns, UTC]"
            "tmpf": "Float32",  # Temperature (°F)
            "dwpf": "Float32",  # Dewpoint (°F)
            "sknt": "Float32",  # Wind speed (knots)
            "p01i": "Float32",  # Precipitation in last hr (mm)
        },
    )

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["valid"] = df["valid"].dt.tz_localize(None)  # Remove timezone information
    df = df.convert_dtypes()

    df = df.dropna()
    df["tmpc"] = df["tmpf"].apply(f_to_c)
    df["dwpc"] = df["dwpf"].apply(f_to_c)
    df["smps"] = df["sknt"].apply(knots_to_mps)

    if debug:
        print(df.describe())

    return df


def load_pvgis(path: str, debug=False) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["time", "G(i)", "H_sun", "T2m", "WS10m"],
        header=0,
        dtype={
            "time": "string",  # Datetime. Consider "datetime64[ns, UTC]"
            "G(i)": "Float32",  # Global in-plane irradiance (W / m^2)
            "H_sun": "Float32",  # Sun height (°)
            "T2m": "Float32",  # Air temperature (°C)
            "WS10m": "Float32",  # Wind speed (m/s)
        },
    )

    # Convert YYYYMMDD:HHMM format to Pandas datetime
    df["time"] = pd.to_datetime(
        df["time"].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:8] + " " + x[9:]),
        utc=True,
    )
    df["time"] = df["time"].dt.tz_localize(None)  # Remove timezone information
    df = df.convert_dtypes()

    if debug:
        print(df.describe())

    return df


def get_mean(
    df: pd.DataFrame,
    time_col: str,
    columns: str | list[str],
    interval: Literal["D", "M", "Y"],
) -> pd.DataFrame:
    """
    Returns the daily, monthly, or yearly mean of the given column(s) as a `DataFrame` with additional `day`, `month`, and `year` columns where applicable
    """

    avg_df = df.groupby(df[time_col].dt.to_period(interval))[columns].mean()

    avg_df["year"] = avg_df.index.year
    if interval != "Y":
        avg_df["month"] = avg_df.index.month
    if interval == "D":
        avg_df["day"] = avg_df.index.day

    return avg_df


def get_sum(
    df: pd.DataFrame,
    time_col: str,
    columns: str | list[str],
    interval: Literal["D", "M", "Y"],
) -> pd.DataFrame:
    """
    Returns the daily, monthly, or yearly sum of the given column(s) as a `DataFrame` with additional `day`, `month`, and `year` columns where applicable
    """

    avg_df = df.groupby(df[time_col].dt.to_period(interval))[columns].sum()

    avg_df["year"] = avg_df.index.year
    if interval != "Y":
        avg_df["month"] = avg_df.index.month
    if interval == "D":
        avg_df["day"] = avg_df.index.day

    return avg_df


def get_range(
    df: pd.DataFrame,
    time_col: str,
    columns: str | list[str],
    interval: Literal["D", "M", "Y"],
) -> pd.DataFrame:
    """
    Returns the daily, monthly, or yearly range of the given column(s) as a `DataFrame` with additional `day`, `month`, and `year` columns where applicable
    """

    interval_df = pd.DataFrame(
        df.groupby(df[time_col].dt.to_period(interval))[columns].max()
        - df.groupby(df[time_col].dt.to_period(interval))[columns].min()
    )

    interval_df["year"] = interval_df.index.year
    if interval != "Y":
        interval_df["month"] = interval_df.index.month
    if interval == "D":
        interval_df["day"] = interval_df.index.day

    return interval_df


def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def knots_to_mps(knots: float) -> float:
    return knots * 0.514444


def get_evaporation_rate(
    mean_temp: float, dewp_temp: float, elev: float, solar_irr: float, wind_speed: float
) -> float:
    return (0.015 + 0.00042 * mean_temp + 10**-6 * elev) * (
        0.8 * solar_irr
        - 40
        + 2.5 * (1.0 - 8.7 * 10**-5 * elev) * wind_speed * (mean_temp - dewp_temp)
    )


def evaporation_rate_curve(
    X: np.ndarray,
    # Output/fitting variables (unknown)
    elevation=1815,
    c1=0.015,
    c2=0.00042,
    c3=10e-6,
    c4=0.8,
    c5=-40,
    c6=2.5,
    c7=1.0,
    c8=8.7 * 10**-5,
) -> np.ndarray:
    mean_temp, dewp_temp, solar_irr, wind_speed = X.T

    return (c1 + c2 * mean_temp + c3 * elevation) * (
        c4 * solar_irr
        + c5
        + c6 * (c7 - c8 * elevation) * wind_speed * (mean_temp - dewp_temp)
    )


def get_fit_parameters(res_m_data: np.ndarray, pan_m_evap: np.ndarray) -> np.ndarray:
    return curve_fit(
        evaporation_rate_curve,
        res_m_data,
        pan_m_evap,
        p0=[
            0.015,
            0.00042,
            10e-6,
            0.8,
            -40,
            2.5,
            1.0,
            8.7 * 10**-5,
        ],  # Seeding parameters from reference
    )[0]
