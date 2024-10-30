import pandas as pd

from request import Request


class Trace():
    def __init__(self, df):
        self.num_requests = len(df)
        self.requests = []
        self.populate_requests(df)

    def populate_requests(self, df):
        for idx, request_dict in df.iterrows():
            request = Request.from_dict(request_dict)
            self.requests.append(request)

    @classmethod
    def from_csv(cls, path):
        print(f">>>Loading CSV from path: {path}")  # 打印文件路径
        df = pd.read_csv(path)
        return Trace(df)
