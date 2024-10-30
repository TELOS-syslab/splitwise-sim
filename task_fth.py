import logging  # 导入日志模块

from dataclasses import dataclass, field  # 从dataclasses模块导入数据类和字段
from enum import IntEnum  # 从enum模块导入整型枚举

from metrics import TaskMetrics, TaskSLO  # 从metrics模块导入任务指标和任务SLO
from node import Node  # 从node模块导入节点类
from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从simulator模块导入时钟和事件管理功能


class TaskType(IntEnum):  # 定义任务类型的枚举类
    COMPUTE = 0  # 计算任务
    PROMPT = 1  # 提示任务
    TOKEN = 2  # 令牌任务


@dataclass(kw_only=True)  # 定义一个数据类，启用关键字参数
class Task(Node):  # 任务类继承自Node类
    """
    Tasks are computation nodes in the Request DAG.  # 任务是请求有向无环图中的计算节点。
    Tasks execute on Instances.  # 任务在实例上执行。

    Tasks are the computational counterparts of Flows.  # 任务是流的计算对应物。
    """
    task_type: TaskType  # 任务类型
    batch_size: int = 1  # 批大小，默认为1
    duration: float = 0.  # 持续时间，默认为0
    remaining_duration: float = 0.  # 剩余时间，默认为0
    cleanup_memory: bool = True  # 是否清理内存，默认为True
    metrics: TaskMetrics = field(default_factory=TaskMetrics)  # 任务指标
    slo: TaskSLO = field(default_factory=TaskSLO)  # 任务服务水平目标
    executor: 'Executor' = None  # 执行器，默认为None
    instances = []  # 实例列表
    _instance = None  # 内部实例，默认为None

    def __hash__(self):  # 重写哈希函数
        return hash(self.node_id)  # 返回节点ID的哈希值

    @property
    def instance(self):  # 实例属性
        return self._instance  # 返回内部实例

    @instance.setter
    def instance(self, instance):  # 实例属性的设置器
        if instance is self._instance:  # 如果设置的实例和当前实例相同
            return  # 直接返回
        self._instance = instance  # 更新内部实例
        if instance is not None:  # 如果实例不为None
            self.instances.append(instance)  # 将实例添加到实例列表

    @property
    def memory(self):  # 内存属性
        return 0  # 返回0，表示默认内存占用

    @classmethod
    def from_type(cls, task_type, **kwargs):  # 从任务类型创建任务实例的类方法
        if task_type == TaskType.COMPUTE:  # 如果任务类型是计算
            return ComputeTask(**kwargs)  # 返回计算任务实例
        elif task_type == TaskType.PROMPT:  # 如果任务类型是提示
            return PromptTask(**kwargs)  # 返回提示任务实例
        elif task_type == TaskType.TOKEN:  # 如果任务类型是令牌
            return TokenTask(**kwargs)  # 返回令牌任务实例
        else:  # 如果任务类型无效
            raise ValueError(f"Invalid TaskType {task_type}")  # 抛出错误


@dataclass(kw_only=True)  # 定义一个数据类，启用关键字参数
class ComputeTask(Task):  # 计算任务类继承自Task类
    """
    Compute tasks represent arbitrary computation.  # 计算任务表示任意计算。
    """
    task_type: TaskType = TaskType.COMPUTE  # 任务类型默认为计算

    def __hash__(self):  # 重写哈希函数
        return hash(self.node_id)  # 返回节点ID的哈希值

    @property
    def memory(self):  # 内存属性
        return 0  # 返回0，表示默认内存占用


@dataclass(kw_only=True)  # 定义一个数据类，启用关键字参数
class PromptTask(Task):  # 提示任务类继承自Task类
    """
    Prompt tasks are the prompt (prefill) computation in a generative LLM.  # 提示任务是生成式LLM中的提示计算。
    They are typically the root task in a GenerativeLLMRequest.  # 它们通常是生成式LLM请求中的根任务。
    """
    prompt_size: int  # 提示大小
    tokens_per_iteration: int = 0  # 每次迭代的令牌数，默认为0
    processing_tokens: int = 0  # 正在处理的令牌数，默认为0
    processed_tokens: int = 0  # 已处理的令牌数，默认为0
    generating_tokens: int = 0  # 正在生成的令牌数，默认为0
    generated_tokens: int = 0  # 已生成的令牌数，默认为0
    task_type: TaskType = TaskType.PROMPT  # 任务类型默认为提示
    cleanup_memory: bool = False  # 是否清理内存，默认为False

    def __post_init__(self):  # 后初始化方法
        self.tokens_per_iteration = self.prompt_size  # 每次迭代的令牌数等于提示大小

    def __hash__(self):  # 重写哈希函数
        return hash(self.node_id)  # 返回节点ID的哈希值

    @property
    def memory(self):  # 内存属性
        num_tokens = self.prompt_size + 1  # 计算令牌数
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,  # 返回估算的KV缓存大小
                                                   model=self.instance.model)  # 传入令牌数和模型

    def max_memory(self, instance):  # 最大内存方法
        num_tokens = self.prompt_size + 1  # 计算令牌数
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,  # 返回估算的KV缓存大小
                                                   model=instance.model)  # 传入令牌数和模型

    def run(self):  # 执行方法
        super().run()  # 调用父类的run方法

        # manage memory  # 管理内存
        self.instance.alloc_memory(self.request, self.memory)  # 为请求分配内存
        self.request.memory += self.memory  # 更新请求的内存占用

    def complete_iteration(self):  # 完成迭代方法
        # tokens processing  # 令牌处理
        # TODO: finer-grained memory management  # TODO: 更精细的内存管理
        self.processed_tokens += self.processing_tokens  # 更新已处理的令牌数
        self.request.processed_tokens += self.processing_tokens  # 更新请求的已处理令牌数
        self.generated_tokens += self.generating_tokens  # 更新已生成的令牌数
        self.request.generated_tokens += self.generating_tokens  # 更新请求的已生成令牌数
        self.processing_tokens = 0  # 重置正在处理的令牌数
        self.generating_tokens = 0  # 重置正在生成的令牌数

    def is_complete(self):  # 判断任务是否完成
        return self.generated_tokens == 1  # 如果已生成的令牌数等于1，返回True

    def complete(self):  # 完成方法
        super().complete()  # 调用父类的complete方法

        # update scheduler bookkeeping  # 更新调度器的账本
        self.instance.sched_pending_tokens -= self.prompt_size  # 减少调度器的待处理令牌数

        # update the TTFT  # 更新TTFT
        self.request.metrics.prompt_end_timestamp = clock()  # 记录提示结束时间
        self.request.metrics.TTFT = clock() - \
                                self.request.metrics.router_arrival_timestamp  # 减去路由到达时间   # 计算TTFT

        # ensure that we processed and generated all tokens  # 确保已处理和生成所有令牌
        assert self.processed_tokens == self.prompt_size  # 断言已处理的令牌数等于提示大小
        assert self.request.processed_tokens == self.request.prompt_size  # 断言请求已处理的令牌数等于请求的提示大小
        assert self.generated_tokens == 1  # 断言已生成的令牌数等于1

        # manage memory  # 管理内存
        if self.cleanup_memory:  # 如果需要清理内存
            self.instance.free_memory(self.request, self.request.memory)  # 释放请求的内存
            self.request.memory = 0  # 将请求的内存重置为0


@dataclass(kw_only=True)  # 定义一个数据类，启用关键字参数
class TokenTask(Task):  # 令牌任务类继承自Task类
    """
    Token tasks represent the token (decode) phase in a generative LLM.  # 令牌任务表示生成式LLM中的令牌解码阶段。
    """
    token_size: int  # 令牌大小
    tokens_per_iteration: int = 1  # 每次迭代的令牌数，默认为1
    processing_tokens: int = 0  # 正在处理的令牌数，默认为0
    processed_tokens: int = 0  # 已处理的令牌数，默认为0
    generating_tokens: int = 0  # 正在生成的令牌数，默认为0
    generated_tokens: int = 0  # 已生成的令牌数，默认为0
    task_type: TaskType = TaskType.TOKEN  # 任务类型默认为令牌

    def __hash__(self):  # 重写哈希函数
        return hash(self.node_id)  # 返回节点ID的哈希值

    @property
    def memory(self):  # 内存属性
        num_tokens = self.token_size  # 计算令牌数
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,  # 返回估算的KV缓存大小
                                                   model=self.instance.model)  # 传入令牌数和模型

    def max_memory(self, instance):  # 最大内存方法
        num_tokens = self.token_size  # 计算令牌数
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,  # 返回估算的KV缓存大小
                                                   model=instance.model)  # 传入令牌数和模型

    def run(self):  # 执行方法
        super().run()  # 调用父类的run方法

        # manage memory  # 管理内存
        self.instance.alloc_memory(self.request, self.memory)  # 为请求分配内存
        self.request.memory += self.memory  # 更新请求的内存占用

    def complete_iteration(self):  # 完成迭代方法
        # tokens processing  # 令牌处理
        self.processed_tokens += self.processing_tokens  # 更新已处理的令牌数
        self.request.processed_tokens += self.processing_tokens  # 更新请求的已处理令牌数
        self.generated_tokens += self.generating_tokens  # 更新已生成的令牌数
        self.request.generated_tokens += self.generating_tokens  # 更新请求的已生成令牌数
        self.processing_tokens = 0  # 重置正在处理的令牌数
        self.generating_tokens = 0  # 重置正在生成的令牌数

    def is_complete(self):  # 判断任务是否完成
        return self.generated_tokens == self.token_size  # 如果已生成的令牌数等于令牌大小，返回True

    def complete(self):  # 完成方法
        super().complete()  # 调用父类的complete方法

        # update scheduler bookkeeping  # 更新调度器的账本
        self.instance.sched_pending_tokens -= 1  # 减少调度器的待处理令牌数

        # ensure that we generated all tokens  # 确保已生成所有令牌
        assert self.processed_tokens == self.token_size  # 断言已处理的令牌数等于令牌大小
        assert self.generated_tokens == self.token_size  # 断言已生成的令牌数等于令牌大小
        assert self.request.generated_tokens == self.request.token_size  # 断言请求已生成的令牌数等于请求的令牌大小
        assert self.request.processed_tokens == self.request.prompt_size + \
                                                self.request.token_size - 1   # 断言请求已处理的令牌数等于提示大小加令牌大小减去1

        # manage memory  # 管理内存
        if self.cleanup_memory:  # 如果需要清理内存
            self.instance.free_memory(self.request, self.request.memory)  # 释放请求的内存
            self.request.memory = 0  # 将请求的内存重置为0


if __name__ == "__main__":  # 如果是主模块
    pass  # 不执行任何操作
