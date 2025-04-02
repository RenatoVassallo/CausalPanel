import pandas as pd
import numpy as np
from params import raw_input_path

# Load isocode-to-continent mapping
df_isocodes = pd.read_csv(raw_input_path + "country_isocode_regions.csv")
isocodes = df_isocodes['isocode'].unique()

def make_mask(target_start, target_end, include_continent=False):
    # 1. Create monthly period range (as YYYYMM integers)
    sdate = pd.to_datetime(f"{str(target_start)[:4]}-{str(target_start)[4:]}")
    edate = pd.to_datetime(f"{str(target_end)[:4]}-{str(target_end)[4:]}")
    periods = pd.date_range(start=sdate, end=edate, freq='MS').strftime('%Y%m').astype(int)
    
    # 2. Create cartesian product of isocode x period
    df_mask = pd.MultiIndex.from_product(
        [isocodes, periods],
        names=['isocode', 'period']
    ).to_frame(index=False)
    
    # 3. Optionally merge continent info
    if include_continent:
        df_mask = df_mask.merge(df_isocodes[['isocode', 'continent']], on='isocode', how='left')
    
    return df_mask.sort_values(['isocode', 'period']).reset_index(drop=True)

def first_value(series, value=1):
    # Find the index of the first occurrence of 1
    first_one_index = series.eq(value).idxmax() if value in series.values else None

    # Return a boolean series where only the first 1 is True
    return series.index == first_one_index


def dict_to_dataframe(input_dict, months_offset=18):
    # Initialize an empty list to store the rows
    rows = []

    # Iterate over each key in the dictionary
    for key, date_range in input_dict.items():
        # Generate a date range from 'min' to 'max' + months_offset
        dates = pd.date_range(
            start=date_range["min"],
            end=date_range["max"] + pd.DateOffset(months=months_offset),
            freq="MS",
        )

        # Determine the original maximum date
        original_max_date = date_range["max"]

        # For each date in the range, create a row with the key, date, and artificial_period
        for date in dates:
            is_artificial = int(date > original_max_date)
            rows.append(
                {
                    "conflict_id": key,
                    "year_mo": date,
                    "extension": is_artificial,
                }
            )

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df


# Function to find the most frequent string
def most_frequent(x):
    mode_result = x.mode()
    if len(mode_result) > 0:
        return mode_result[0]
    else:
        return None


def exponential_weighted_moving_average(series, alpha=0.5):
    """
    Computes the Exponential Weighted Moving Average (EWMA) of a time series.

    Parameters:
    series (pd.Series): The time series data for a group.
    alpha (float): The smoothing factor, between 0 and 1.

    Returns:
    pd.Series: The EWMA values of the series.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha should be between 0 and 1.")

    ewma = np.zeros_like(series, dtype=np.float64)
    ewma[0] = series.iloc[0]  # Initializing the first value with iloc

    for t in range(1, len(series)):
        ewma[t] = alpha * series.iloc[t] + (1 - alpha) * ewma[t - 1]

    return pd.Series(ewma, index=series.index)


# # function to generate since a variable and until a a variable
def gen_since(series: pd.Series):
    series = series == False
    groups = (series == 0).cumsum()
    result = series.groupby(groups).cumsum()
    result[groups == 0] = np.nan
    return result


def gen_until(series: pd.Series) -> pd.Series:
    """
    Generate the 'until' series indicating the number of periods until the next event.

    Parameters:
    -----------
    series : pd.Series
        A boolean series where True indicates the event.

    Returns:
    --------
    pd.Series
        A series indicating the number of periods until the next event.
    """
    series = series[::-1]
    series = series == 0
    groups = (series == 0).cumsum()
    result = series.groupby(groups).cumsum()[::-1]
    result[groups == 0] = np.nan
    return result
