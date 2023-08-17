import pandas as pd
from typing import Union


class ProgPyDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ProgPyDataFrame

    def timestamps(self, times: list[float] = None):
        self.insert(0, 'time', times)
        self.set_index('time', inplace=True, drop=True)

    def add_timestamp_row(self, time: float = None, data=None):
        self.loc[time] = data

    # def add_row

    def get_progpy_dict(self):
        if self.empty:
            return []
        else:
            return self.to_dict('records')[0]

    def add_row(self, row):
        if isinstance(row, ProgPyDataFrame):
            row_check = row.empty
        else:
            row_check = row
        if not row_check:
            if self.empty:
                for col in row.columns:
                    self[col] = row[col]
                    self.reset_index(drop=True)
            else:
                if isinstance(row, ProgPyDataFrame):
                    row = row.get_progpy_dict()
                self.loc[len(self)] = row

    # def __repr__(self) -> str:
    #     """
    #     represents object as string
    #
    #     returns: a string of dictionaries containing all the keys and associated matrix values
    #     """
    #     return str(self.to_dict('record')[0])


InputContainer = ProgPyDataFrame

StateContainer = ProgPyDataFrame

OutputContainer = ProgPyDataFrame
