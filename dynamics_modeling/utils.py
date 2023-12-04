import pandas as pd


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


def load_daily_mesonet(path: str) -> pd.DataFrame:
    """
    Parses a CSV of [daily observations from the Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/daily.phtml).
    Filters to precipitation, standardizing to metric units. Drops rows with any `NaN` values.
    """

    df = pd.read_csv(
        path,
        usecols=["day", "precip_in"],
        header=0,
        dtype={
            "day": "string",
            "p01i": "Float32",  # (in)
        },
        parse_dates=["day"],
    )

    df["day"] = pd.to_datetime(df["day"], utc=True)
    df["day"] = df["day"].dt.tz_localize(None)  # Remove timezone information
    df = df.convert_dtypes()

    df.dropna(inplace=True)

    df["precip_in"] = df["precip_in"].apply(lambda i: i * 0.0254)  # (in -> m)
    df.rename(columns={"day": "time", "precip_in": "precip"}, inplace=True)

    return df


def load_hourly_mesonet(path: str) -> pd.DataFrame:
    """
    Parses a CSV of [hourly observations from the Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml).
    Filters to precipitation, standardizing to metric units. Drops rows with any `NaN` values.
    """

    df = pd.read_csv(
        path,
        usecols=["valid", "p01i"],
        header=0,
        dtype={
            "valid": "string",  # Datetime. Consider "datetime64[ns, UTC]"
            "p01i": "Float32",  # Precipitation in last hr (mm)
        },
    )

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["valid"] = df["valid"].dt.tz_localize(None)  # Remove timezone information
    df = df.convert_dtypes()
    df.dropna(inplace=True)

    return df
