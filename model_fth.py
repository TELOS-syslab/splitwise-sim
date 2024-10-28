import logging  # 导入日志模块

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass，用于简化数据类的定义


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 ModelArchitecture 数据类，设置为仅关键字参数
class ModelArchitecture():
    name: str  # 模型名称，类型为字符串
    num_layers: int  # 模型层数，类型为整数


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 LLMArchitecture 数据类，继承自 ModelArchitecture
class LLMArchitecture(ModelArchitecture):  # LLM 表示大语言模型（Large Language Model）结构
    hidden_size: int  # 隐藏层大小，类型为整数
    num_heads: int  # 注意力头数量，类型为整数


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 ModelParallelism 数据类，描述模型并行方式
class ModelParallelism():
    """
    Captures the different parallelisms of a Model. 捕获一个模型的不同并行性。
    """
    pipeline_parallelism: int  # 管道并行的数量，类型为整数
    tensor_parallelism: int  # 张量并行的数量，类型为整数

    @property  # 使用 property 装饰器定义只读属性
    def num_processors(self):
        """
        The number of GPUs required is the product of the parallelisms. 所需的gpu的数量是并行性的乘积。
        """
        return self.pipeline_parallelism * self.tensor_parallelism  # 返回所需 GPU 的数量，即并行数量的乘积


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 ModelSize 数据类，表示模型的大小
class ModelSize():
    """
    Captures the various sizes of a Model. 捕获模型的各种大小。
    """
    weights: int  # 模型权重的大小，类型为整数
    dtype_size: int  # 数据类型大小（如浮点数的位数），类型为整数

    @property  # 使用 property 装饰器定义只读属性
    def total_size(self):
        return self.weights  # 返回模型的总大小，这里等于权重的大小


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 Model 数据类，描述完整的模型
class Model():
    name: str  # 模型名称，类型为字符串
    architecture: ModelArchitecture  # 模型架构，类型为 ModelArchitecture
    parallelism: ModelParallelism  # 模型并行方式，类型为 ModelParallelism
    size: ModelSize  # 模型大小，类型为 ModelSize

    @property  # 使用 property 装饰器定义只读属性
    def size_per_processor(self):
        return self.size.total_size / self.parallelism.num_processors  # 返回每个处理器的模型大小


@dataclass(kw_only=True)  # 使用 dataclass 装饰器定义 GenerativeLLM 数据类，表示生成式大语言模型
class GenerativeLLM(Model):
    """
    Generative Large Language Model.
    NOTE: We currently don't capture embeddings, variable context lengths, etc.
    生成大语言模型。
    注意：我们目前不捕获嵌入、可变的上下文长度等。
    """
    context_size: int = 0  # 上下文大小，默认为 0
