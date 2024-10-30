import logging  # 导入日志模块

from dataclasses import dataclass, field  # 从 dataclasses 模块导入数据类和字段
from enum import IntEnum  # 从 enum 模块导入整型枚举类

from metrics import NodeMetrics  # 从 metrics 模块导入节点指标类
from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从 simulator 模块导入时钟和事件调度相关函数


class NodeState(IntEnum):  # 定义节点状态的整型枚举类
    NONE = 0  # 未定义状态
    QUEUED = 1  # 队列状态
    RUNNING = 2  # 运行状态
    BLOCKED = 3  # 被阻塞状态
    COMPLETED = 4  # 完成状态
    ABORTED = 5  # 中止状态


@dataclass(kw_only=True)  # 定义节点类，启用关键字参数
class Node():
    """
    请求中任务和节点的基类
    请求有向无环图（DAG）中最简单的元素
    """
    node_id: int  # 节点 ID
    num_preemptions: int = 0  # 预抢占次数，默认值为 0
    request: 'Request' = None  # 请求对象，默认为 None
    state: NodeState = NodeState.NONE  # 节点状态，默认为 NONE
    metrics: NodeMetrics = field(default_factory=NodeMetrics)  # 节点指标，默认实例化 NodeMetrics
    # 必须连续执行的节点链
    # 仅在链的第一个节点中存储
    chain: list = field(default_factory=list)  # 节点链，默认为空列表

    def __hash__(self):
        """
        注意：哈希函数在子类中被重写为 None
        """
        return hash(self.node_id)  # 返回节点 ID 的哈希值

    def __eq__(self, other):
        return self.node_id == other.node_id  # 比较节点 ID 是否相等

    def arrive(self):
        assert self.state == NodeState.NONE  # 确保状态为 NONE
        self.metrics.arrival_timestamp = clock()  # 记录到达时间戳
        self.state = NodeState.QUEUED  # 状态变更为 QUEUED

    def run(self):
        assert self.state == NodeState.QUEUED  # 确保状态为 QUEUED
        self.metrics.run_timestamp = clock()  # 记录运行时间戳
        self.metrics.start_timestamp = clock()  # 记录开始时间戳
        self.metrics.queue_time += clock() - self.metrics.arrival_timestamp  # 更新排队时间
        if self.request.root_node is self:  # 如果当前节点是请求的根节点
            self.request.metrics.prompt_start_timestamp = clock()  # 记录提示开始时间戳
            self.request.metrics.queue_time = clock() - \
                            self.request.metrics.router_arrival_timestamp  # 更新请求排队时间
        self.state = NodeState.RUNNING  # 状态变更为 RUNNING

    def run_after_preempt(self):
        assert self.state == NodeState.BLOCKED  # 确保状态为 BLOCKED
        self.metrics.run_timestamp = clock()  # 记录运行时间戳
        self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp  # 更新被阻塞时间
        self.state = NodeState.RUNNING  # 状态变更为 RUNNING

    def complete(self):
        assert self.state == NodeState.RUNNING  # 确保状态为 RUNNING
        self.metrics.completion_timestamp = clock()  # 记录完成时间戳
        self.metrics.service_time += clock() - self.metrics.run_timestamp  # 更新服务时间
        self.metrics.response_time = clock() - self.metrics.arrival_timestamp  # 更新响应时间
        self.state = NodeState.COMPLETED  # 状态变更为 COMPLETED

    def preempt(self):
        assert self.state == NodeState.RUNNING  # 确保状态为 RUNNING
        self.metrics.preempt_timestamp = clock()  # 记录抢占时间戳
        self.metrics.service_time += clock() - self.metrics.run_timestamp  # 更新服务时间
        self.state = NodeState.BLOCKED  # 状态变更为 BLOCKED

    def abort(self):
        if self.state == NodeState.QUEUED:  # 如果状态为 QUEUED
            self.metrics.queue_time += clock() - self.metrics.arrival_timestamp  # 更新排队时间
            if self.request.root_node is self:  # 如果当前节点是请求的根节点
                self.request.metrics.queue_time = clock() - \
                                self.request.metrics.router_arrival_timestamp  # 更新请求排队时间
        elif self.state == NodeState.RUNNING:  # 如果状态为 RUNNING
            self.metrics.service_time += clock() - self.metrics.run_timestamp  # 更新服务时间
        elif self.state == NodeState.BLOCKED:  # 如果状态为 BLOCKED
            self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp  # 更新被阻塞时间
        self.state = NodeState.ABORTED  # 状态变更为 ABORTED
