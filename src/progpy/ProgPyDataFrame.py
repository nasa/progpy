import pandas as pd


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


InputContainer = ProgPyDataFrame

StateContainer = ProgPyDataFrame

OutputContainer = ProgPyDataFrame
