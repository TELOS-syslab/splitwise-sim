"""
Utility functions for initializing the simulation environment.
"""
# 描述了文件的作用：提供初始化模拟环境的实用函数。

import logging
import os

from hydra.utils import instantiate
from hydra.utils import get_original_cwd

from application import Application
from cluster import Cluster
from hardware_repo import HardwareRepo
from model_repo import ModelRepo
from orchestrator_repo import OrchestratorRepo
from start_state import load_start_state
from trace import Trace
# import logging：导入日志模块，用于记录日志信息。
# import os：导入 os 模块，提供文件和目录操作的功能。
# from hydra.utils import instantiate：从 hydra.utils 导入 instantiate 函数，用于实例化配置中的类。
# from hydra.utils import get_original_cwd：从 hydra.utils 导入 get_original_cwd 函数，返回 Hydra 运行的原始目录。
# from application import Application 等：导入模拟环境中所需的类，如 Application、Cluster、HardwareRepo 等。
# 中文注释：导入所需的库和模块，包括日志模块、文件操作模块以及各个类和函数，用于初始化模拟环境。

# 定义了 init_trace 函数，用于初始化追踪记录。
# trace_path = os.path.join(get_original_cwd(), cfg.trace.path)：构建追踪文件的完整路径。
# trace = Trace.from_csv(trace_path)：从 CSV 文件中加载追踪数据，生成 Trace 对象。
# return trace：返回 Trace 对象。
def init_trace(cfg):
    trace_path = os.path.join(get_original_cwd(), cfg.trace.path)
    trace = Trace.from_csv(trace_path)
    return trace


def init_hardware_repo(cfg):  # 初始化硬件仓库
    processors_path = os.path.join(get_original_cwd(),  # 获取处理器配置文件的绝对路径
                                   cfg.hardware_repo.processors)
    interconnects_path = os.path.join(get_original_cwd(),  # 获取互连配置文件的绝对路径
                                      cfg.hardware_repo.interconnects)
    skus_path = os.path.join(get_original_cwd(),  # 获取 SKU 配置文件的绝对路径
                             cfg.hardware_repo.skus)
    hardware_repo = HardwareRepo(processors_path,  # 创建 HardwareRepo 对象，传入路径参数
                                 interconnects_path,
                                 skus_path)
    return hardware_repo  # 返回硬件仓库对象


def init_model_repo(cfg):  # 初始化模型仓库
    model_architectures_path = os.path.join(get_original_cwd(),  # 获取模型架构配置文件的绝对路径
                                            cfg.model_repo.architectures)
    model_sizes_path = os.path.join(get_original_cwd(),  # 获取模型尺寸配置文件的绝对路径
                                    cfg.model_repo.sizes)
    model_repo = ModelRepo(model_architectures_path, model_sizes_path)  # 创建 ModelRepo 对象
    return model_repo  # 返回模型仓库对象


def init_orchestrator_repo(cfg):  # 初始化调度器仓库
    allocators_path = os.path.join(get_original_cwd(),  # 获取分配器配置文件的绝对路径
                                   cfg.orchestrator_repo.allocators)
    schedulers_path = os.path.join(get_original_cwd(),  # 获取调度器配置文件的绝对路径
                                   cfg.orchestrator_repo.schedulers)
    orchestrator_repo = OrchestratorRepo(allocators_path, schedulers_path)  # 创建 OrchestratorRepo 对象
    return orchestrator_repo  # 返回调度器仓库对象


def init_performance_model(cfg):  # 初始化性能模型
    performance_model = instantiate(cfg.performance_model)  # 根据配置实例化性能模型
    return performance_model  # 返回性能模型对象


def init_power_model(cfg):  # 初始化功耗模型
    power_model = instantiate(cfg.power_model)  # 根据配置实例化功耗模型
    return power_model  # 返回功耗模型对象


def init_cluster(cfg):  # 初始化集群
    cluster = Cluster.from_config(cfg.cluster)  # 使用配置生成集群对象
    return cluster  # 返回集群对象


def init_router(cfg, cluster):  # 初始化路由器
    router = instantiate(cfg.router, cluster=cluster)  # 根据配置实例化路由器对象，并传入集群对象
    return router  # 返回路由器对象


def init_arbiter(cfg, cluster):  # 初始化仲裁器
    arbiter = instantiate(cfg.arbiter, cluster=cluster)  # 根据配置实例化仲裁器对象，并传入集群对象
    return arbiter  # 返回仲裁器对象


def init_applications(cfg, cluster, router, arbiter):  # 初始化应用程序
    applications = {}  # 创建空字典存储应用程序
    for application_cfg in cfg.applications:  # 遍历配置中的每个应用程序配置
        application = Application.from_config(application_cfg,  # 使用配置创建应用对象
                                              cluster=cluster,
                                              router=router,
                                              arbiter=arbiter)
        applications[application_cfg.application_id] = application  # 将应用对象添加到字典
    return applications  # 返回应用程序字典


def init_start_state(cfg, **kwargs):  # 初始化启动状态
    load_start_state(cfg.start_state, **kwargs)  # 加载启动状态


if __name__ == "__main__":  # 主程序入口
    pass  # 占位符，表示什么也不执行
