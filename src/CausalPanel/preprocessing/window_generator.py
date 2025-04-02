import pandas as pd
import numpy as np
from typing import Union, List


class WindowGenerator:
    """
    Generate treatment and control windows for time-series data analysis.

    Parameters
    ----------
    DataFrame : pd.DataFrame
        The input dataframe containing the data.
    unit_column : str
        The column name representing the unit of analysis.
    time_column : str
        The column name representing the time variable.
    treatment_column : str
        The column name indicating treatment status.
    matching_column : str
        The column name used for matching observations.
    frame_size : int or list of int
        The size of the window frame. If an integer is provided, it is used for both left and right frames.
        If a list is provided, it should be in the form [left_frame, right_frame].
    buffer_size : int or list of int
        The size of the buffer around treatment events. Works similarly to frame_size.
    method : str, optional
        The matching method to use. Currently supported: 'nn_propensity_score'.
        Default is 'nn_propensity_score'.

Attributes
    ----------
    df : pd.DataFrame
        A copy of the original dataframe.
    treat_col : str
        The column name indicating treatment status.
    unit_col : str
        The column name representing the unit of analysis.
    time_col : str
        The column name representing the time variable.
    match_col : str
        The column name used for matching observations.
    max_times_by_unit : dict
        A dictionary mapping each unit to its maximum time value.
    method : str
        The matching method in use.
    left_frame : int
        The left frame size for windows.
    right_frame : int
        The right frame size for windows.
    left_buffer : int
        The left buffer size.
    right_buffer : int
        The right buffer size.
    treated_obs : list
        A list of tuples (time, unit) for treated observations.
    treated_windows : pd.DataFrame
        DataFrame containing generated treatment windows.
    control_windows : pd.DataFrame
        DataFrame containing generated control windows.
    windows : pd.DataFrame
        Combined DataFrame of treatment and control windows.
    """

    def __init__(
        self,
        DataFrame: pd.DataFrame,
        unit_column: str,
        time_column: str,
        treatment_column: str,
        matching_column: str,
        frame_size: Union[int, List[int]],
        buffer_size: Union[int, List[int]],
        method: str = 'nn_propensity_score',
    ):
        """
        Initialize the WindowGenerator with the provided dataframe and configuration.

        Parameters
        ----------
        DataFrame : pd.DataFrame
            The input dataframe containing the data.
        unit_column : str
            The column name representing the unit of analysis.
        time_column : str
            The column name representing the time variable.
        treatment_column : str
            The column name indicating treatment status.
        matching_column : str
            The column name used for matching observations.
        frame_size : int or list of int
            The size of the window frame.
        buffer_size : int or list of int
            The size of the buffer around treatment events.
        method : str, optional
            The matching method to use. Currently only supports 'nn_propensity_score'.
            Default is 'nn_propensity_score'.

        Raises
        ------
        ValueError
            If an unknown matching method is provided.
        """
        self.df = DataFrame.copy()
        self.treat_col = treatment_column
        self.unit_col = unit_column
        self.time_col = time_column
        self.match_col = matching_column
        self.max_times_by_unit = (
            self.df.groupby(self.unit_col)[self.time_col].max().to_dict()
        )
        self.method = method
        self.df["max_time_by_unit"] = self.df.groupby(self.unit_col)[
            self.time_col
        ].transform("max")
        self.left_frame, self.right_frame = self._gen_frame_size(frame_size)
        self.left_buffer, self.right_buffer = self._gen_frame_size(buffer_size)

        if self.method not in ["nn_propensity_score"]:
            raise ValueError(
                f"Unknown matching method: {self.method}. Please select nn_propensity_score."
            )

    def _gen_frame_size(self, frame_size: Union[int, List[int]]) -> (int, int):
        """
        Generate left and right frame sizes from the provided frame_size.

        Parameters
        ----------
        frame_size : int or list of int
            If an int is provided, both left and right frames are set to that value.
            If a list is provided, it should be in the form [left_frame, right_frame].

        Returns
        -------
        tuple of int
            A tuple (left_frame, right_frame).
        """
        if isinstance(frame_size, list):
            return frame_size[0], frame_size[1]
        else:
            return frame_size, frame_size

    def gen_treatment_windows(self, buffer_treated_windows: bool = True) -> None:
        """
        Generate treatment windows based on specified conditions.

        Parameters
        ----------
        buffer_treated_windows : bool, optional
            If True, applies additional buffer conditions around treatment events.
            Default is True.
        """
        self.df["since_treatment"] = self.df.groupby(self.unit_col)[
            self.treat_col
        ].transform(self.gen_since)
        self.df["until_treatment"] = self.df.groupby(self.unit_col)[
            self.treat_col
        ].transform(self.gen_until)
        self.df["since_treatment_1"] = self.df.groupby(
            self.unit_col
        ).since_treatment.transform(lambda x: x.shift(1))
        self.df["until_treatment_-1"] = self.df.groupby(
            self.unit_col
        ).until_treatment.transform(lambda x: x.shift(-1))

        condition = (
            (self.df[self.treat_col] == 1)
            & (self.df[self.time_col] >= self.left_frame)
            & (
                self.df[self.time_col]
                <= (self.df["max_time_by_unit"] - self.right_frame)
            )
        )

        if buffer_treated_windows:
            condition &= (self.df["since_treatment_1"] >= self.left_buffer) | (
                self.df["since_treatment_1"].isna()
            )
            condition &= (self.df["until_treatment_-1"] >= self.right_buffer) | (
                self.df["until_treatment_-1"].isna()
            )

        # For each method, omit the rows where there are NA values in the columns they rely on.
        if self.method == "bins":
            condition &= ~self.df[self.match_col + "_bin"].isna()
        elif self.method == "nn_propensity_score":
            condition &= ~self.df[self.match_col].isna()

        self.treated_obs = list(
            self.df.copy()
            .loc[condition, [self.time_col, self.unit_col]]
            .itertuples(index=False, name=None)
        )

        treated_windows_list = []
        for i, (time, unit) in enumerate(self.treated_obs):
            window_condition = (
                (self.df[self.unit_col] == unit)
                & (self.df[self.time_col] >= time - self.left_frame)
                & (self.df[self.time_col] <= time + self.right_frame)
            )
            window_df = self.df.loc[window_condition].assign(
                window_id=np.random.sample(1)[0]
            )
            treated_windows_list.append(window_df)

        self.treated_windows = pd.concat(treated_windows_list, ignore_index=True)
        self.treated_windows["window_t"] = (
            self.treated_windows.groupby(["window_id"])[self.time_col].transform(
                lambda x: pd.factorize(x, sort=True)[0]
            )
            - self.left_frame
        )

    def gen_since(self, series: pd.Series) -> pd.Series:
        """
        Generate a series representing the number of periods since the last treatment event.

        Parameters
        ----------
        series : pd.Series
            A series where False indicates a non-event and True indicates a treatment event.

        Returns
        -------
        pd.Series
            A series with counts of periods since the last treatment event.
        """
        series = series == False
        groups = (series == 0).cumsum()
        result = series.groupby(groups).cumsum()
        return result

    def gen_until(self, series: pd.Series) -> pd.Series:
        """
        Generate a series representing the number of periods until the next treatment event.

        Parameters
        ----------
        series : pd.Series
            A series where False indicates a non-event and True indicates a treatment event.

        Returns
        -------
        pd.Series
            A series with counts of periods until the next treatment event.
        """
        series = series[::-1]
        series = series == 0
        groups = (series == 0).cumsum()
        result = series.groupby(groups).cumsum()[::-1]
        result[groups == 0] = np.nan
        return result

    def nearest_neighbors(self, series1, series2, k: int = 3, d: float = 0.15, t: int = 20):
        """
        Find the k-nearest neighbors in series2 for each value in series1, subject to a distance
        threshold and maximum usage constraints.

        Parameters
        ----------
        series1 : array-like or pd.Series
            The first series of values to find neighbors for.
        series2 : array-like or pd.Series
            The second series of values to match against.
        k : int, optional
            The number of nearest neighbors to find. Default is 3.
        d : float, optional
            The distance threshold. Neighbors with a distance larger than this are not considered.
            Default is 0.15.
        t : int, optional
            The maximum number of times an observation in series2 can be used as a neighbor.
            Default is 20.

        Returns
        -------
        tuple
            - flattened_valid_indices : list
                A flattened list of indices in series2 corresponding to valid neighbors.
            - series1_indices_to_drop : list
                A list of indices in series1 for which no valid neighbors were found.
        """
        _series2 = series2.copy()
        valid_indices = []
        series1_indices_to_drop = []
        usage_count = np.zeros(len(series2), dtype=int)
        large_value = 1e6  # A large value to represent invalid distances

        for i, val in enumerate(series1):
            # if usage count is above threshold, mark as invalid.
            _series2[usage_count >= t] = large_value
            distances = abs(_series2 - val)
            distances[distances > d] = large_value

            neighbors = np.argsort(distances)[:k]
            valid_neighbors = [
                neighbor
                for neighbor in neighbors
                if distances.iloc[neighbor] < large_value
            ]
            valid_indices.append(valid_neighbors)

            for neighbor in valid_neighbors:
                usage_count[neighbor] += 1

            if not valid_neighbors:
                series1_indices_to_drop.append(i)

        flattened_valid_indices = [obs for sublist in valid_indices for obs in sublist]

        return flattened_valid_indices, series1_indices_to_drop

    def nearest_neighbors_arrays(
        self, array1, array2, k: int = 3, d: float = 0.15, t: int = 20, large_value: float = 1e6
    ):
        """
        Find the k-nearest neighbors in array2 for each row in array1 using normalized Euclidean distance,
        subject to a distance threshold and a maximum usage constraint.

        Parameters
        ----------
        array1 : array-like
            The first array (or series) of observations (rows) to find neighbors for.
        array2 : array-like
            The second array (or series) of observations (rows) to match against.
        k : int, optional
            The number of nearest neighbors to find. Default is 3.
        d : float, optional
            The distance threshold. Neighbors with a distance larger than this are not considered.
            Default is 0.15.
        t : int, optional
            The maximum number of times an observation in array2 can be used as a neighbor.
            Default is 20.
        large_value : float, optional
            A large value used to mark invalid distances. Default is 1e6.

        Returns
        -------
        tuple
            - flattened_valid_indices : list
                A flattened list of indices in array2 corresponding to valid neighbors.
            - array1_indices_to_drop : list
                A list of indices in array1 for which no valid neighbors were found.
        """
        _array2 = array2.copy()
        valid_indices = []
        array1_indices_to_drop = []
        usage_count = np.zeros(len(array2), dtype=int)

        def normalized_euclidean_distance(row, array2):
            return np.linalg.norm(array2 - row, axis=1) / np.sqrt(array2.shape[1])

        for i, row in enumerate(array1):
            _array2[usage_count >= t] = large_value
            distances = normalized_euclidean_distance(row, _array2)
            distances[distances > d] = large_value

            neighbors = np.argsort(distances)[:k]
            valid_neighbors = [neighbor for neighbor in neighbors if distances[neighbor] < large_value]
            valid_indices.append(valid_neighbors)

            for neighbor in valid_neighbors:
                usage_count[neighbor] += 1

            if not valid_neighbors:
                array1_indices_to_drop.append(i)

        flattened_valid_indices = [obs for sublist in valid_indices for obs in sublist]
        return flattened_valid_indices, array1_indices_to_drop

    def gen_control_windows(
        self,
        buffer_size: Union[int, List[int]],
        k: int = 10,
        sample_with_replacement: bool = False,
        d: float = 1,
        t: int = 200,
        verbose: bool = True,
    ) -> None:
        """
        Generate control windows based on specified conditions and matching criteria.

        Parameters
        ----------
        buffer_size : int or list of int
            The size of the buffer zone around treatment events.
        k : int, optional
            The number of control samples for each treated sample. Default is 10.
        sample_with_replacement : bool, optional
            Whether to sample with replacement. Default is False.
        d : float, optional
            The distance threshold for nearest neighbor matching. Default is 1.
        t : int, optional
            The maximum usage threshold for a control observation. Default is 200.
        verbose : bool, optional
            If True, prints additional information when control observations have no support.
            Default is True.
        """
        self.left_buffer, self.right_buffer = self._gen_frame_size(buffer_size)

        mask = (
            (self.df[self.treat_col] == 0)
            & (
                (self.df["since_treatment"] >= self.left_buffer + 1)
                | (self.df["since_treatment"].isna())
            )
            & (
                (self.df["until_treatment"] >= self.right_buffer + 1)
                | (self.df["until_treatment"].isna())
            )
            & (self.df[self.time_col] >= self.left_frame)
            & (
                self.df[self.time_col]
                <= (self.df["max_time_by_unit"] - self.right_frame)
            )
        )
        mask &= ~self.df[self.match_col].isna()

        self.df_subset = self.df.loc[mask].copy()
        self.treated_matching_col = self.df.set_index(
            [self.time_col, self.unit_col]
        ).loc[self.treated_obs][self.match_col]

        self.control_obs, self.treated_indices_to_drop = self.nearest_neighbors(
            self.treated_matching_col,
            self.df_subset[self.match_col],
            k=k,
            d=d,
            t=t,
        )

        if len(self.treated_indices_to_drop) > 0:
            treated_obs_to_drop = self.treated_matching_col.iloc[
                self.treated_indices_to_drop
            ].index

            self.treated_ids_to_drop = (
                self.treated_windows.query("window_t == 0")
                .set_index([self.time_col, self.unit_col])
                .loc[treated_obs_to_drop]["window_id"]
            )
            if verbose:
                print(
                    f"There are {len(self.treated_ids_to_drop)} treated observations that do not have support and their windows are being dropped. The following are a sample:"
                )
                print(
                    self.treated_windows.loc[
                        self.treated_windows["window_id"].isin(self.treated_ids_to_drop)
                    ]
                    .query("window_t == 0")[
                        [self.unit_col, self.time_col, self.match_col]
                    ]
                    .sample(len(self.treated_ids_to_drop))
                    .head()
                    .to_string()
                )

            self.treated_windows = self.treated_windows.loc[
                ~self.treated_windows["window_id"].isin(self.treated_ids_to_drop)
            ]

        self.control_obs = list(
            self.df_subset.iloc[self.control_obs][[self.unit_col, self.time_col]].itertuples(index=False, name=None)
        )

        control_windows = []
        for i, (unit, time) in enumerate(self.control_obs):
            window_condition = (
                (self.df[self.unit_col] == unit)
                & (self.df[self.time_col] >= time - self.left_frame)
                & (self.df[self.time_col] <= time + self.right_frame)
            )
            window_df = self.df.loc[window_condition].assign(
                window_id=np.random.sample(1)[0]
            )
            control_windows.append(window_df)

        self.control_windows = pd.concat(control_windows, ignore_index=True)
        self.control_windows["window_t"] = (
            self.control_windows.groupby(["window_id"])[self.time_col].transform(
                lambda x: pd.factorize(x, sort=True)[0]
            )
            - self.left_frame
        )

    def combine_groups(self) -> None:
        """
        Combine treated and control windows into a single dataframe.

        The treated windows are labeled with treated=1 and control windows with treated=0.
        """
        self.windows = pd.concat(
            [
                self.treated_windows.assign(treated=1),
                self.control_windows.assign(treated=0),
            ],
            ignore_index=True,
        )
        self.windows["window_id"] = self.windows["window_id"].factorize()[0]

    def gen_sample_size(self, level: str) -> str:
        """
        Generate a summary string of sample sizes for treated and control groups.

        Parameters
        ----------
        level : str
            A label indicating the level (e.g., "Unit" or "Group") for the summary.

        Returns
        -------
        str
            A formatted string summarizing the number of treated and control windows.
        """
        Treatment = self.treat_col.capitalize()
        n_treated = self.windows.query("window_t == 0 & treated == 1").window_id.nunique()
        n_control = self.windows.query("window_t == 0 & treated == 0").window_id.nunique()

        return f"{level} {Treatment} & {n_treated} & {n_control}"