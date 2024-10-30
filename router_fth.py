import logging  # 导入 logging 库，用于日志记录

from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 和 abstractmethod，创建抽象基类

from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从 simulator 模块导入模拟所需的函数

class Router(ABC):  # 定义 Router 类，继承自 ABC 类，表示这是一个抽象类
    """
    Router 类用于将请求路由到应用程序的调度器。
    """
    def __init__(self, cluster, overheads):  # 初始化方法，接受集群和开销参数
        self.cluster = cluster  # 记录集群对象
        self.overheads = overheads  # 存储路由开销
        self.applications = []  # 初始化应用程序列表
        self.schedulers = {}  # 初始化调度器字典

        # 请求队列
        self.pending_queue = []  # 初始化挂起队列，用于存储待处理请求
        self.executing_queue = []  # 初始化执行队列，用于存储正在执行的请求
        self.completed_queue = []  # 初始化完成队列，用于存储已完成的请求

    def add_application(self, application):  # 添加应用程序及其调度器到路由器
        self.applications.append(application)  # 将应用程序添加到应用程序列表中
        self.schedulers[application.application_id] = application.scheduler  # 将应用程序 ID 与其调度器对应存储

    def run(self):  # 运行方法，未实现的占位符
        pass  # 占位符方法，无操作

    @abstractmethod  # 定义抽象方法，子类需要实现该方法
    def route(self, *args):  # 路由请求的主逻辑
        """
        路由的主逻辑
        """
        raise NotImplementedError  # 如果未实现该方法，抛出未实现异常

    def request_arrival(self, request):  # 请求到达方法
        request.arrive_at_router()  # 调用请求的 arrive_at_router 方法
        self.pending_queue.append(request)  # 将请求添加到挂起队列
        self.route_request(request)  # 调用 route_request 方法以路由请求

    def request_completion(self, request):  # 请求完成方法
        request.complete_at_router()  # 调用请求的 complete_at_router 方法
        self.executing_queue.remove(request)  # 从执行队列中移除该请求
        self.completed_queue.append(request)  # 将请求添加到完成队列

    def route_request(self, request):  # 路由请求方法
        self.route(request)  # 调用 route 方法，将请求发送到合适的调度器
        self.pending_queue.remove(request)  # 从挂起队列中移除该请求
        self.executing_queue.append(request)  # 将请求添加到执行队列

    def save_results(self):  # 保存结果方法
        #results = []  # 创建一个空列表，用于存储结果
        #for request in self.completed_queue:  # 遍历完成队列中的请求
        #    times = request.time_per_instance_type()  # 获取每种实例类型的时间
        #    results.append(times)  # 将时间添加到结果列表
        #utils.save_dict_as_csv(results, "router.csv")  # 保存结果为 CSV 文件
        pass  # 占位符方法，当前未实现


class NoOpRouter(Router):  # 定义 NoOpRouter 类，继承自 Router，表示无操作路由器
    """
    将请求直接转发到相应的调度器，无任何额外开销。
    """
    def route(self, request):  # 路由方法
        scheduler = self.schedulers[request.application_id]  # 获取请求对应的调度器
        f = lambda scheduler=scheduler, request=request: \
            scheduler.request_arrival(request)  # 调用调度器的 request_arrival 方法，将请求传递给调度器   # 定义 lambda 函数
        schedule_event(self.overheads.routing_delay, f)  # 调度一个事件，延迟 routing_delay 后执行 f
