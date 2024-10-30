import logging
import os
import time

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import utils

from executor import Executor, ExecutorType
from interconnect import DummyLink
from performance_model import get_duration
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import Task, TaskType
from flow import FlowType

# 调度器基类，用于调度请求到实例，并生成执行器来处理它们
class Scheduler(ABC):
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        self.application = application
        self.router = router
        self.overheads = overheads
        self.executor_overheads = executor_overheads
        self.debug = debug

        # 实例列表
        self.instances = []

        # 请求队列
        self.pending_queue = []
        self.executing_queue = []
        self.completed_queue = []

        # 执行器
        self.executor_type = ExecutorType.CentralExecutor
        self.executors = {}

        # 调度器动作的日志记录器
        logger_name = f"schedulers/{self.application.application_id}"
        level = logging.DEBUG if self.debug else logging.INFO
        os.makedirs("schedulers", exist_ok=True)
        self.scheduler_logger = utils.file_logger(logger_name, level=level)
        self.scheduler_logger.info("time,action,info")

    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, application):
        self._application = application

    def add_instance(self, instance):
        """
        在调度器级别跟踪实例。
        有助于维护调度器特定的实例视图。
        """
        self.instances.append(instance)

    @abstractmethod
    def schedule(self, request, *args, **kwargs):
        """
        主要的调度器逻辑，将请求分配给实例。
        当请求运行时被调用。
        为请求创建计划。
        """
        raise NotImplementedError

    def request_arrival(self, request):
        """
        处理新请求的到来。
        """
        request.arrive_at_scheduler()
        self.pending_queue.append(request)
        if len(self.pending_queue) == 1:
            self.run_request(request)

    def request_completion(self, request):
        """
        处理请求的完成。
        """
        request.complete_at_scheduler()
        self.executing_queue.remove(request)
        self.completed_queue.append(request)
        self.router.request_completion(request)

    def run_request(self, request):
        """
        通过调度它并生成一个执行器来运行请求。
        """
        request.run_on_executor()
        # 测量调度开销
        start = time.time()
        self.schedule(request)
        end = time.time()
        self.scheduler_logger.debug('%s,sched_overhead,%s', clock(), end-start)
        self.spawn_executor(ExecutorType.CentralExecutor,
                            request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    def spawn_executor(self, executor_type, request):
        """
        为请求生成一个执行器。
        执行器可以在逻辑上在任何地方执行。
        我们不在模拟中模拟它们运行的位置。
        """
        executor = Executor.create(executor_type,
                                   request,
                                   self,
                                   self.executor_overheads)
        self.executors[request.request_id] = executor
        executor.run()

    def notify_busy_instance(self, instance):
        """
        通知调度器实例忙碌。
        """

    def notify_free_instance(self, instance):
        """
        通知调度器实例空闲。
        """

    def terminate_executor(self, executor):
        """
        从调度器中删除执行器。
        """
        del self.executors[executor.request.request_id]

    def save_all_request_metrics(self):
        """
        保存所有请求节点的开始和结束时间戳。
        有助于甘特图。
        """
        node_metrics = []
        for request in self.completed_queue:
            node_metrics.extend(request.get_all_node_metrics())
        node_metrics_df = pd.DataFrame(node_metrics)
        node_metrics_df.to_csv("request_nodes.csv", index=False)

    def get_results(self):
        """
        返回所有已完成请求的结果。
        """
        array_results = {}

        request_ids = [r.request_id for r in self.completed_queue]
        array_results["request_ids"] = np.array(request_ids)

        response_times = [r.metrics.router_response_time for r in self.completed_queue]
        array_results["response_times"] = np.array(response_times)

        queue_times = [r.metrics.queue_time for r in self.completed_queue]
        array_results["queue_times"] = np.array(queue_times)

        ttft_times = [r.metrics.TTFT for r in self.completed_queue]
        array_results["ttft_times"] = np.array(ttft_times)

        tbt_times = [(r.metrics.router_response_time - r.metrics.TTFT) / (r.token_size)
                     for r in self.completed_queue]
        array_results["tbt_times"] = np.array(tbt_times)

        nth_token_overhead = [r.get_nth_token_overhead() for r in self.completed_queue]
        array_results["nth_token_overheads"] = np.array(nth_token_overhead)

        prompt_sizes = [r.prompt_size for r in self.completed_queue]
        array_results["prompt_sizes"] = np.array(prompt_sizes)

        token_sizes = [r.token_size for r in self.completed_queue]
        array_results["token_sizes"] = np.array(token_sizes)

        return array_results

# KVScheduler是调度器的基类，用于发送KV缓存。
# 它不实现调度方法。
class KVScheduler(Scheduler):
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.prompt_processors = prompt_processors
        self.token_processors = token_processors
        self.prompt_instances = []
        self.token_instances = []

    def add_instance(self, instance):
        """
        分别跟踪提示和令牌实例。
        注意：假设实例标签是区分符，而不是硬件本身
        TODO: 使这个更灵活和健壮
        """
        self.instances.append(instance)
        if instance.tag == "prompt":
            self.prompt_instances.append(instance)
        elif instance.tag == "token":
            self.token_instances.append(instance)
        else:
            # 另一种区分实例的方法
            if isinstance(self.prompt_processors, list):
                if instance.name in self.prompt_processors:
                    self.prompt_instances.append(instance)
                elif instance.name in self.token_processors:
                    self.token_instances.append(instance)
                else:
                    raise ValueError(f"Unsupported instance type: \
                                        {instance.processors[0].name}")

    def add_kv_cache_transfer(self, request, src_instance, dest_instance, bandwidth):
        """
        通过在请求图中添加一个流节点，将prompt->token请求转换为prompt->kvtransfer->token请求
        """
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # 创建新任务和流
        flow_size = request.estimate_kv_cache_size(
                                        num_tokens=prompt_task.prompt_size,
                                        model=src_instance.model)
        kv_transfer_flow = request.create_flow(FlowType.KVCacheTransfer,
                                               size=flow_size,
                                               src=src_instance,
                                               dest=dest_instance)
        kv_transfer_flow.notify = True

        # 更新请求DAG
        request.flow_node = kv_transfer_flow
        request.dag.remove_edge(prompt_task, token_task)
        request.dag.add_edge(prompt_task, kv_transfer_flow)
        request.dag.add_edge(kv_transfer_flow, token_task)

        # 为实例和链接分配任务和流
        prompt_task.instance = src_instance
        token_task.instance = dest_instance
        # 注意：通过添加一个可配置带宽的链接来模拟延迟
        kv_transfer_flow.link = DummyLink(name="DummyLink",
                                          bandwidth=bandwidth)

# RandomScheduler随机调度请求到实例。
class RandomScheduler(Scheduler):
    def schedule(self, request, *args, **kwargs):
        """
        将请求中的所有节点分配给随机实例
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # 通过链式调用启用运行到完成
        prompt_task.chain = [token_task]

        instance = np.random.choice(self.instances)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

# RoundRobinScheduler以轮询方式调度请求跨所有实例。
class RoundRobinScheduler(Scheduler):
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        将请求中的所有节点分配给下一个实例
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

       