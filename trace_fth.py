import pandas as pd  # 导入 pandas 库，用于数据处理

from request import Request  # 从 request 模块中导入 Request 类

class Trace():  # 定义 Trace 类
    def __init__(self, df):  # 初始化方法，接收一个 DataFrame 作为参数
        self.num_requests = len(df)  # 计算请求数量并存储在 num_requests 属性中
        self.requests = []  # 初始化一个空列表，用于存储请求
        self.populate_requests(df)  # 调用 populate_requests 方法，传入 DataFrame 参数

    def populate_requests(self, df):  # 定义 populate_requests 方法，将每行数据转换为 Request 对象
        for idx, request_dict in df.iterrows():  # 使用 iterrows() 方法逐行遍历 DataFrame
            request = Request.from_dict(request_dict)  # 调用 Request 的 from_dict 类方法，将每行数据转换为 Request 对象
            self.requests.append(request)  # 将创建的 Request 对象添加到 requests 列表中

    @classmethod  # 定义类方法 from_csv，可以直接从 CSV 文件创建 Trace 对象
    def from_csv(cls, path):  # 接收类对象 cls 和 CSV 文件路径 path 作为参数
        df = pd.read_csv(path)  # 使用 pandas 的 read_csv() 方法读取 CSV 文件，生成 DataFrame
        return Trace(df)  # 返回一个 Trace 实例，传入读取的 DataFrame
