import logging  # 导入日志模块

from hydra.utils import instantiate  # 从 hydra.utils 导入实例化函数

import utils  # 导入工具模块
import hardware_repo  # 导入硬件仓库模块

from power_model import get_server_power  # 从 power_model 导入获取服务器功耗的函数
from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从 simulator 导入时钟和事件调度相关函数

# 用于 hydra 实例化
import processor  # 导入处理器模块
import interconnect  # 导入互连模块


class Server:  # 定义服务器类
    """
    服务器是一个处理器集合，可以通过本地互连连接。
    服务器本身也通过互连相互连接。
    服务器运行实例（部分或全部）。

    属性:
        server_id (str): 服务器的唯一 ID。
        processors (list): 处理器列表。
        interconnects (list[Link]): 直接连接到该服务器的对等设备。
    """
    servers = {}  # 服务器字典，用于存储所有服务器实例
    # 所有服务器的日志记录器
    logger = None  # 初始化日志记录器为 None

    def __init__(self,
                 server_id,  # 服务器 ID
                 name,  # 服务器名称
                 processors,  # 处理器列表
                 interconnects):  # 互连列表
        if server_id in Server.servers:  # 如果服务器 ID 已存在于服务器字典中
            # 注意：这是一个 hacky 的解决方案，用于 Hydra
            # Hydra 多次运行有一个 bug，尝试用相同的类再次实例化集群
            # 触发此路径。这可能是因为 Hydra 多次运行在不同线程间重用相同的类
            Server.servers = {}  # 清空服务器字典
            Server.logger = None  # 重置日志记录器
        self.server_id = server_id  # 设置服务器 ID
        self.name = name  # 设置服务器名称
        self.processors = processors  # 设置处理器列表
        for proc in self.processors:  # 遍历处理器列表
            proc.server = self  # 设置每个处理器的服务器属性
        self.interconnects = interconnects  # 设置互连列表
        for intercon in self.interconnects:  # 遍历互连列表
            intercon.server = self  # 设置每个互连的服务器属性
        self.cluster = None  # 初始化集群为 None
        Server.servers[server_id] = self  # 将当前服务器实例添加到服务器字典中
        self.instances = []  # 初始化实例列表
        self.power = 0  # 初始化功耗为 0
        self.update_power(0)  # 更新功耗

        # 初始化服务器日志记录器
        if Server.logger is None:  # 如果日志记录器为 None
            self.logger = utils.file_logger("server")  # 创建新的文件日志记录器
            Server.logger = self.logger  # 更新类级别的日志记录器
            self.logger.info("time,server")  # 记录表头信息
        else:
            self.logger = Server.logger  # 使用现有的日志记录器

    def __str__(self):  # 定义字符串表示
        return f"Server:{self.server_id}"  # 返回服务器的字符串表示

    def __repr__(self):  # 定义可打印表示
        return self.__str__()  # 返回字符串表示

    @property
    def instances(self):  # 实例属性的 getter
        return self._instances  # 返回实例列表

    @instances.setter
    def instances(self, instances):  # 实例属性的 setter
        self._instances = instances  # 设置实例列表

    def update_power(self, power):  # 更新功耗
        old_power = self.power  # 保存旧的功耗
        self.power = get_server_power(self) + \
                        sum(processor.power for processor in self.processors)  # 计算所有处理器的总功耗  # 获取服务器的功耗并加上所有处理器的功耗
        if self.cluster:  # 如果集群存在
            self.cluster.update_power(self.power - old_power)  # 更新集群的功耗

    def run(self):  # 运行服务器（未实现）
        pass  # 占位符

    @classmethod
    def load(cls):  # 类方法，用于加载服务器（未实现）
        pass  # 占位符

    @classmethod
    def from_config(cls, *args, server_id, **kwargs):  # 从配置创建服务器实例
        sku_cfg = args[0]  # 获取 SKU 配置
        processors_cfg = sku_cfg.processors  # 获取处理器配置
        interconnects_cfg = sku_cfg.interconnects  # 获取互连配置

        processors = []  # 初始化处理器列表
        for processor_cfg in processors_cfg:  # 遍历处理器配置
            for n in range(processor_cfg.count):  # 根据计数添加处理器
                processor = hardware_repo.get_processor(processor_cfg.name)  # 从硬件仓库获取处理器
                processors.append(processor)  # 添加处理器到列表

        # TODO: 添加更好的网络拓扑/配置支持
        interconnects = []  # 初始化互连列表
        for interconnect_name in interconnects_cfg:  # 遍历互连配置
            intercon = hardware_repo.get_interconnect(interconnect_name)  # 从硬件仓库获取互连
            interconnects.append(intercon)  # 添加互连到列表

        return cls(server_id=server_id,  # 创建服务器实例并返回
                   name=sku_cfg.name,
                   processors=processors,
                   interconnects=interconnects)


if __name__ == "__main__":  # 如果作为主程序运行
    pass  # 占位符
