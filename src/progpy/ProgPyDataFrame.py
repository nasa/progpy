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
        return self.to_dict('records')[0]

    def add_row(self, row):
        if isinstance(row, ProgPyDataFrame):
            row = row.get_progpy_dict()
        if self.empty and not row.empty:
            self.loc[0] = row
        elif not self.empty and not row.empty:
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
