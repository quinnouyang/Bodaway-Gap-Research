import pandas as pd

from typing import Literal


def get_mean(
    df: pd.DataFrame,
    time_col: str,
    columns: str | list[str],
    interval: Literal["D", "M", "Y"],
) -> pd.DataFrame:
    """
    Returns the daily, monthly, or yearly mean of the given column(s) as a `DataFrame` with additional `day`, `month`, and `year` columns where applicable
    """

    avg_df = pd.DataFrame(
        df.groupby(df[time_col].dt.to_period(interval))[columns].mean()
    )

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
