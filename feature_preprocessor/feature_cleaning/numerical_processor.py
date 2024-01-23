from typing import Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.tree import ExtraTreeRegressor


class NumericalDataProcessor:
    def __init__(self):
        # Столбцы с нулевой предиктивной способностью
        self.zero_significance_features = None
        # Столбцы со смешанными типами
        self.mixed_type_columns = None
        # Столбцы со численными типами
        self.numerical_columns = None

        self.normolizer = MinMaxScaler()
        self.standardizer = StandardScaler()
        self.imputer = None

        self.impute_strategies = {
            'mean': SimpleImputer(),
            'median': SimpleImputer(strategy='median'),
            'moda': SimpleImputer(strategy='most_frequent'),
            'iterative': IterativeImputer(estimator=ExtraTreeRegressor),
            'knn': KNNImputer(),
        }

    def find_mixed_type_columns(
            self,
            frame: pd.DataFrame,
            threshold: float = 0.5
    ) -> list:
        """Finds columns where most of the data is numeric, but
        contains values of other types.

        :param frame: DataFrame to analyze.
        :param threshold: threshold value, based on which the column
            data type is checked.

        :return: A list of columns that match the criterion.
        """
        mixed_type_columns: list = []

        for column in frame.columns:
            numeric_percentage: float = frame[column].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ).notnull().mean()

            if 0 < numeric_percentage < 1 and numeric_percentage >= threshold:
                mixed_type_columns.append(column)

        self.mixed_type_columns: list = mixed_type_columns
        return self.mixed_type_columns

    def normalize_columns(
            self,
            frame: pd.DataFrame,
            numerical_columns: tuple = None,
            inplace: bool = True
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """Bringing the data to a range from 0 to 1.

        :param frame: DataFrame to analyze.
        :param numerical_columns: Columns with numerical data.
        :param inplace: If True, the function returns the same
            dataset, but with normalized numeric columns.

        :return: A tuple of dataframe and fitted normalizer.
        """
        if numerical_columns is None:
            numerical_columns = []

            for column in frame.columns:
                if is_numeric_dtype(frame[column]):
                    numerical_columns.append(column)

        self.numerical_columns = numerical_columns

        frame = self._transform(frame, self.normolizer, inplace)

        return frame, self.normolizer

    def standardize_columns(
            self,
            frame: pd.DataFrame,
            numerical_columns: tuple = None,
            inplace: bool = True
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """Bringing dataframe to data with zero mean and
        unit standard deviation.

        :param frame: DataFrame to analyze.
        :param numerical_columns: Columns with numerical data.
        :param inplace: If True, the function returns the same
            dataset, but with normalized numeric columns.

        :return: A tuple of dataframe and fitted standardizer.
        """
        if numerical_columns is None:
            numerical_columns = []

            for column in frame.columns:
                if is_numeric_dtype(frame[column]):
                    numerical_columns.append(column)

        self.numerical_columns = numerical_columns

        frame = self._transform(frame, self.standardizer, inplace)

        return frame, self.standardizer

    def impute_columns(
            self,
            frame: pd.DataFrame,
            numerical_columns: tuple = None,
            strategy: str = 'mean',
            inplace: bool = True
    ) -> tuple[pd.DataFrame, Union[SimpleImputer, IterativeImputer, KNNImputer]]:
        if numerical_columns is None:
            numerical_columns = []

            for column in frame.columns:
                if is_numeric_dtype(frame[column]):
                    numerical_columns.append(column)

        self.numerical_columns = numerical_columns

        self.imputer = self.impute_strategies[strategy]

        frame = self._imputate(frame, self.imputer, inplace)

        return frame, self.imputer

    def _transform(
            self,
            frame: pd.DataFrame,
            scaler: Union[MinMaxScaler, StandardScaler],
            inplace: bool = True,
    ):
        if len(self.numerical_columns) == 0:
            # TODO
            raise ValueError

        if inplace:
            frame[self.numerical_columns]: pd.DataFrame = pd.DataFrame(
                scaler.fit_transform(frame[self.numerical_columns]),
                index=frame.index,
                columns=self.numerical_columns,
            )
            return frame

        processed_frame: pd.DataFrame = pd.DataFrame(
            scaler.fit_transform(frame[self.numerical_columns]),
            index=frame.index,
            columns=self.numerical_columns,
        )

        return processed_frame

    def _imputate(
            self,
            frame: pd.DataFrame,
            imputer: Union[SimpleImputer, IterativeImputer, KNNImputer],
            inplace: bool = True,
    ):
        if len(self.numerical_columns) == 0:
            # TODO
            raise ValueError

        if inplace:
            frame[self.numerical_columns]: pd.DataFrame = pd.DataFrame(
                imputer.fit_transform(frame[self.numerical_columns]),
                index=frame.index,
                columns=self.numerical_columns,
            )
            return frame

        processed_frame: pd.DataFrame = pd.DataFrame(
            imputer.fit_transform(frame[self.numerical_columns]),
            index=frame.index,
            columns=self.numerical_columns,
        )

        return processed_frame


