import heapq  # 导入 heapq 模块，用于操作优先队列
import logging  # 导入日志模块，用于记录日志信息

from collections import defaultdict  # 导入 defaultdict，用于创建默认值的字典

import utils  # 导入 utils 模块，包含一些实用函数


# global simulator that drives the simulation
# bad practice, but it works for now
sim = None  # 全局模拟器对象（不推荐，但当前这样使用）


class Event:
    """
    Events are scheduled actions in the simulator.
    """
    def __init__(self, time, action):  # 初始化事件
        self.time = time  # 事件发生时间
        self.action = action  # 事件对应的动作

    def __str__(self):  # 将事件转换为字符串表示
        return f"Event with time {self.time} and action {self.action}"

    def __lt__(self, other):  # 小于运算符重载，使事件可按时间排序  用于heapq优先队列
        return self.time < other.time


class Simulator:
    """
    A discrete event simulator that schedules and runs Events.
    """
    def __init__(self, end_time):  # 初始化模拟器
        global sim # 在 Simulator 类的 __init__ 方法中，使用 global sim 的目的是告诉 Python 在该方法内使用全局变量 sim，而不是创建一个新的局部变量 sim。随后代码 sim = self 把当前创建的 Simulator 对象赋值给全局变量 sim。
        sim = self  # 将当前模拟器对象赋值给全局 sim 变量 在 Python 中，赋值 sim = self 是 引用传递，也就是将当前对象 self 的引用赋给全局变量 sim。这样，sim 和 self 实际上指向同一个对象的内存地址。因此，之后通过 self 对实例内部成员变量进行修改，sim 也会反映这些修改，因为 sim 和 self 指向的是同一个对象。
        self.time = 0  # 初始化模拟时间为 0
        self.end_time = end_time  # 模拟结束时间
        self.events = []  # 事件队列
        self.deleted_events = []  # 被取消的事件列表
        logging.info("Simulator initialized")  # 记录初始化日志

        # logger for simulator events
        self.logger = utils.file_logger("simulator")  # 创建日志记录器
        self.logger.info("time,event")  # 记录事件日志的标题

    def schedule(self, delay, action):  # 定义 schedule 方法，用于安排事件
        """
        Schedule an event by specifying delay and an action function.
        """
        # run immediately if delay is 0
        if delay == 0:  # 如果延迟为 0，则立即执行动作
            action()
            return None
        event = Event(self.time + delay, action)  # 创建新的事件，延迟时间为 delay
        heapq.heappush(self.events, event)  # 将事件加入优先队列
        return event

    def cancel(self, event):  # 定义 cancel 方法，用于取消事件
        """
        Cancel an event.
        """
        self.deleted_events.append(event)  # 将事件加入取消列表

    def reschedule(self, event, delay):  # 定义 reschedule 方法，用于重新安排事件
        """
        Reschedule an event by cancelling and scheduling it again.
        """
        self.cancel(event)  # 先取消事件
        return self.schedule(delay, event.action)  # 重新安排事件

    def run(self):  # 定义 run 方法，运行模拟器
        """
        Run the simulation until the end time.
        """
        while self.events and self.time < self.end_time:  # 在事件队列不为空且未到结束时间前
            event = heapq.heappop(self.events)  # 获取最近的事件
            if event in self.deleted_events:  # 如果事件已被取消
                self.deleted_events.remove(event)  # 从取消列表中移除
                continue
            self.time = event.time  # 更新当前时间
            event.action()  # 执行动作
            self.logger.debug(f"{event.time},{event.action}")  # 记录事件信息


class TraceSimulator(Simulator):
    """
    A discrete event simulator that processes Request arrivals from a Trace.一种处理来自跟踪的请求到达者的离散事件模拟器。
    """
    def __init__(self,
                 trace,
                 cluster,
                 applications,
                 router,
                 arbiter,
                 end_time):  # 初始化 TraceSimulator
        super().__init__(end_time)  # 调用父类初始化
        self.trace = trace  # 请求追踪记录
        self.cluster = cluster  # 集群对象
        self.applications = applications  # 应用程序字典
        self.router = router  # 路由器对象
        self.arbiter = arbiter  # 仲裁器对象
        logging.info("TraceSimulator initialized")  # 记录初始化日志
        self.load_trace()  # 加载追踪记录

    def load_trace(self):  # 定义 load_trace 方法，加载追踪请求
        """
        Load requests from the trace as arrival events.从跟踪中加载请求作为到达事件。
        """
        for request in self.trace.requests:  # 遍历追踪中的每个请求
            self.schedule(request.arrival_timestamp,
                          lambda request=request: self.router.request_arrival(request))  # 将请求到达事件安排到日程表

    def run(self):  # 重写 run 方法，运行追踪模拟器
        # start simulation by scheduling a cluster run
        self.schedule(0, self.cluster.run)  # 安排集群的运行事件
        self.schedule(0, self.router.run)  # 安排路由器的运行事件
        self.schedule(0, self.arbiter.run)  # 安排仲裁器的运行事件

        # run simulation
        super().run()  # 调用父类的 run 方法，开始模拟
        self.logger.info(f"{self.time},end")  # 记录模拟结束时间
        logging.info(f"TraceSimulator completed at {self.time}")  # 记录模拟结束日志

        self.save_results()  # 保存模拟结果

    def save_results(self, detailed=True):  # 定义 save_results 方法，保存结果
        """
        Save results at the end of the simulation.
        """
        self.router.save_results()  # 保存路由器的结果

        sched_results = {}  # 计划结果字典
        alloc_results = {}  # 分配结果字典
        for application_id, application in self.applications.items():  # 遍历应用程序
            allocator_results, scheduler_results = application.get_results()  # 获取分配和计划结果
            alloc_results[application_id] = allocator_results  # 存储分配结果
            sched_results[application_id] = scheduler_results  # 存储计划结果

        # summary sched results
        summary_results = defaultdict(list)  # 创建一个默认值为列表的字典，用于汇总结果
        for application_id, results_dict in sched_results.items():  # 遍历计划结果
            summary_results["application_id"].append(application_id)  # 添加应用程序 ID
            for key, values in results_dict.items():  # 遍历结果字典
                summary = utils.get_statistics(values)  # 计算统计信息
                # merge summary into summary_results
                for metric, value in summary.items():  # 将统计信息合并到汇总结果中
                    summary_results[f"{key}_{metric}"].append(value)

        # save summary results
        utils.save_dict_as_csv(summary_results, "summary.csv")  # 将汇总结果保存到 CSV 文件

        if detailed:  # 如果需要详细结果
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():  # 遍历计划结果
                utils.save_dict_as_csv(result, f"detailed/{application_id}.csv")  # 保存详细的计划结果
            for application_id, result in alloc_results.items():  # 遍历分配结果
                utils.save_dict_as_csv(result, f"detailed/{application_id}_alloc.csv")  # 保存详细的分配结果


# Convenience functions for simulator object

def clock():  # 获取当前模拟时间
    """
    Return the current time of the simulator. 返回模拟器的当前时间。
    """
    return sim.time  # 返回全局模拟器的当前时间

def schedule_event(*args):  # 在模拟器中安排事件
    """
    Schedule an event in the simulator at desired delay. 在期望的延迟时调度模拟器中的事件。
    """
    return sim.schedule(*args)  # 调用全局模拟器的 schedule 方法

def cancel_event(*args):  # 取消模拟器中的事件
    """
    Cancel existing event in the simulator.
    """
    return sim.cancel(*args)  # 调用全局模拟器的 cancel 方法

def reschedule_event(*args):  # 重新安排模拟器中的事件
    """
    Reschedule existing event in the simulator.
    Equivalent to cancelling and scheduling a new event. 
    重新调度模拟器中的现有事件。
    相当于取消和调度一个新的事件。
    """
    return sim.reschedule(*args)  # 调用全局模拟器的 reschedule 方法
