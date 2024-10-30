import logging  # 导入 logging 库，用于日志记录

from dataclasses import dataclass, field  # 从 dataclasses 模块中导入 dataclass 和 field
from enum import IntEnum  # 导入 IntEnum 类，用于定义枚举
from itertools import count  # 导入 count 迭代器，用于生成连续的整数

import networkx as nx  # 导入 networkx 库，用于创建和操作有向无环图 (DAG)

from executor import Executor  # 从 executor 模块中导入 Executor 类
from flow import Flow  # 从 flow 模块中导入 Flow 类
from metrics import RequestMetrics, GenerativeLLMRequestMetrics, RequestSLO  # 从 metrics 模块导入请求相关的度量类
from node import Node  # 从 node 模块中导入 Node 类
from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从 simulator 模块导入时间和事件相关的函数
from task import Task, TaskType  # 从 task 模块导入 Task 类和 TaskType 枚举

class RequestState(IntEnum):  # 定义请求状态的枚举类
    """
    RequestState 描述了请求的不同状态。
    """
    NONE = 0  # 无状态
    QUEUED_AT_ROUTER = 1  # 已排队到达路由器
    QUEUED_AT_SCHEDULER = 2  # 已排队到达调度器
    RUNNING_ON_EXECUTOR = 3  # 在执行器上运行
    COMPLETED_AT_SCHEDULER = 4  # 在调度器处完成
    COMPLETED_AT_ROUTER = 5  # 在路由器处完成
    ABORTED = 6  # 已中止

class RequestType(IntEnum):  # 定义请求类型的枚举类
    COMPUTE = 0  # 计算型请求（未实现）
    DNN = 1  # 深度神经网络请求（未实现）
    GENERATIVE_LLM = 2  # 生成式 LLM 请求

@dataclass(kw_only=True)
class Request():  # 定义 Request 类，用于表示一个包含任务和流的请求
    """
    Request 是一个面向应用程序的包含任务和流的有向无环图 (DAG)。
    请求必须有一个根节点。
    """
    request_id: int  # 请求的唯一 ID
    node_id: count = field(default_factory=count)  # 节点 ID 计数器
    application_id: int  # 请求对应的应用程序 ID
    request_type: RequestType  # 请求类型
    batch_size: int = 1  # 批处理大小
    arrival_timestamp: float = 0.  # 到达时间戳
    state: RequestState = field(default=RequestState.NONE)  # 当前状态
    dag: nx.DiGraph = field(default_factory=nx.DiGraph)  # 有向无环图，用于存储任务和流
    root_node: Node = None  # 根节点
    nodes: dict = field(default_factory=dict)  # 存储节点的字典
    metrics: RequestMetrics = field(default_factory=RequestMetrics)  # 请求的度量
    slo: RequestSLO = field(default_factory=RequestSLO)  # 服务等级目标
    executor: Executor = None  # 执行器

    def __post_init__(self):
        pass  # 初始化后操作，目前无内容

    def __hash__(self):  # 定义哈希方法
        """
        注意：子类中哈希函数被重写为 None
        """
        return hash(self.request_id)

    def __eq__(self, other):  # 定义相等方法
        return self.request_id == other.request_id

    def successors(self, node):  # 获取某节点的后继节点
        """
        返回节点之后要执行的下一个任务或流。
        """
        return self.dag.successors(node)

    def predecessors(self, node):  # 获取某节点的前驱节点
        """
        返回节点之前要执行的任务或流。
        """
        return self.dag.predecessors(node)

    def get_node(self, node_id):  # 获取指定 ID 的节点
        """
        从 DAG 中返回具有 node_id 的节点。
        """
        return self.nodes[node_id]

    def get_node_metrics(self, node_id):  # 获取节点的度量
        """
        返回具有 node_id 的节点的度量。
        """
        node = self.get_node(node_id)
        if isinstance(node, Task):  # 如果是任务节点
            node_type = node.task_type.name
            runner = f"{node.instance.name}_{node.instance.instance_id}"
        elif isinstance(node, Flow):  # 如果是流节点
            node_type = node.flow_type.name
            runner = node.link.name
        else:
            raise ValueError("不支持的节点类型")
        data = {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "node_id": node_id,
            "node_type": node_type,
            "runner": runner,
            "start_timestamp": node.metrics.start_timestamp,
            "completion_timestamp": node.metrics.completion_timestamp,
        }
        return data

    def get_all_node_metrics(self):  # 获取所有节点的度量
        data = []
        for node_id in self.nodes:
            data.append(self.get_node_metrics(node_id))
        return data

    def arrive_at_router(self):  # 请求到达路由器
        assert self.state == RequestState.NONE
        self.metrics.router_arrival_timestamp = clock()
        self.state = RequestState.QUEUED_AT_ROUTER

    def arrive_at_scheduler(self):  # 请求到达调度器
        """
        注意：我们不跟踪路由开销。
        """
        assert self.state == RequestState.QUEUED_AT_ROUTER
        self.metrics.scheduler_arrival_timestamp = clock()
        self.metrics.router_queue_time = clock() - self.metrics.router_arrival_timestamp
        self.state = RequestState.QUEUED_AT_SCHEDULER

    def run_on_executor(self):  # 请求在执行器上运行
        assert self.state == RequestState.QUEUED_AT_SCHEDULER
        self.metrics.executor_start_timestamp = clock()
        self.metrics.scheduler_queue_time = clock() - self.metrics.scheduler_arrival_timestamp
        self.state = RequestState.RUNNING_ON_EXECUTOR

    def complete_at_scheduler(self):  # 请求在调度器上完成
        """
        注意：我们不跟踪执行器 <--> 调度器的通信开销。
        """
        assert self.state == RequestState.RUNNING_ON_EXECUTOR
        self.metrics.scheduler_completion_timestamp = clock()
        self.metrics.service_time += clock() - self.metrics.executor_start_timestamp
        self.metrics.scheduler_response_time = clock() - self.metrics.scheduler_arrival_timestamp
        self.state = RequestState.COMPLETED_AT_SCHEDULER

    def complete_at_router(self):  # 请求在路由器上完成
        """
        注意：我们不跟踪调度器 <--> 路由器的通信开销。
        """
        assert self.state == RequestState.COMPLETED_AT_SCHEDULER
        self.metrics.router_completion_timestamp = clock()
        self.metrics.router_response_time = clock() - self.metrics.router_arrival_timestamp
        self.state = RequestState.COMPLETED_AT_ROUTER

    def abort(self):  # 中止请求
        if self.state == RequestState.QUEUED_AT_ROUTER:
            self.metrics.router_queue_time += clock() - self.metrics.router_arrival_timestamp
        elif self.state == RequestState.QUEUED_AT_SCHEDULER:
            self.metrics.scheduler_queue_time += clock() - self.metrics.scheduler_arrival_timestamp
        elif self.state == RequestState.RUNNING_ON_EXECUTOR:
            self.metrics.service_time += clock() - self.metrics.executor_start_timestamp
        elif self.state == RequestState.COMPLETED_AT_SCHEDULER:
            pass
        self.state = RequestState.ABORTED

    def get_results(self):  # 获取请求结果
        pass

    def create_task(self, task_type, **kwargs):  # 创建任务并添加到 DAG
        task = Task.from_type(task_type=task_type, node_id=next(self.node_id), request=self, **kwargs)
        self.dag.add_node(task)
        self.nodes[task.node_id] = task
        return task

    def create_flow(self, flow_type, **kwargs):  # 创建流并添加到 DAG
        flow = Flow.from_type(flow_type=flow_type, node_id=next(self.node_id), request=self, **kwargs)
        self.dag.add_node(flow)
        self.nodes[flow.node_id] = flow
        return flow

    def remove_node(self, node):  # 从 DAG 中删除节点
        self.dag.remove_node(node)
        del self.nodes[node.node_id]

    @classmethod
    def from_dict(cls, request_dict):  # 从字典创建 Request
        """
        从 Pandas 字典返回一个 Request 实例。
        """
        if request_dict["request_type"] == RequestType.GENERATIVE_LLM:
            request = GenerativeLLMRequest(**request_dict)
        else:
            raise ValueError(f"不支持的请求类型: {request_dict['request_type']}")
        return request

@dataclass(kw_only=True)
class GenerativeLLMRequest(Request):  # 定义 Generative
