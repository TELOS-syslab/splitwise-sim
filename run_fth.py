import logging  # 导入日志模块，用于记录程序运行信息
import os  # 导入操作系统接口模块，用于文件和目录操作
import random  # 导入随机数模块，用于生成随机数
import sys  # 导入系统模块，用于与 Python 解释器进行交互

import hydra  # 导入 Hydra 配置管理库

from hydra.utils import instantiate  # 导入 instantiate，用于根据配置实例化类
from hydra.utils import get_original_cwd, to_absolute_path  # 导入路径相关函数
from omegaconf import DictConfig, OmegaConf  # 导入配置类型 DictConfig 和 OmegaConf 管理库

from simulator import TraceSimulator  # 从 simulator 模块导入 TraceSimulator 类
from initialize import *  # 从 initialize 模块导入所有函数


# register custom hydra resolver
OmegaConf.register_new_resolver("eval", eval)  # 将 eval 函数注册为 OmegaConf 的自定义解析器，用于在配置文件中动态计算表达式


def run_simulation(cfg):  # 定义 run_simulation 函数，用于运行模拟
    hardware_repo = init_hardware_repo(cfg)  # 初始化硬件仓库
    model_repo = init_model_repo(cfg)  # 初始化模型仓库
    orchestrator_repo = init_orchestrator_repo(cfg)  # 初始化调度器仓库
    performance_model = init_performance_model(cfg)  # 初始化性能模型
    power_model = init_power_model(cfg)  # 初始化功耗模型
    cluster = init_cluster(cfg)  # 初始化集群
    router = init_router(cfg, cluster)  # 初始化路由器，并传入集群
    arbiter = init_arbiter(cfg, cluster)  # 初始化仲裁器，并传入集群
    applications = init_applications(cfg, cluster, router, arbiter)  # 初始化应用程序，并传入集群、路由器和仲裁器
    trace = init_trace(cfg)  # 初始化追踪记录
    for application in applications.values():  # 遍历应用程序并添加到路由器和仲裁器中
        router.add_application(application)  # 将应用程序添加到路由器
        arbiter.add_application(application)  # 将应用程序添加到仲裁器
    sim = TraceSimulator(trace=trace,  # 创建 TraceSimulator 对象，传入追踪记录和其他组件
                         cluster=cluster,
                         applications=applications,
                         router=router,
                         arbiter=arbiter,
                         end_time=cfg.end_time)
    init_start_state(cfg,  # 初始化模拟的起始状态
                     cluster=cluster,
                     applications=applications,
                     router=router,
                     arbiter=arbiter)
    sim.run()  # 运行模拟器
    

@hydra.main(config_path="configs", config_name="config", version_base=None)  # Hydra 的装饰器，指定配置文件路径和名称
def run(cfg: DictConfig) -> None:  # 定义主函数 run，接受 DictConfig 类型的配置参数
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))  # 注释掉的代码，用于打印配置内容
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # 获取 Hydra 配置（注释掉）
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))  # 注释掉的代码，用于打印 Hydra 配置

    # initialize random number generator
    random.seed(cfg.seed)  # 初始化随机数生成器，设置种子确保结果可重复

    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):  # 检查是否存在 oom.csv 文件
        os.remove("oom.csv")  # 如果存在，删除该文件

    run_simulation(cfg)  # 调用 run_simulation 函数，运行模拟程序



if __name__ == "__main__":  # 主程序入口
    run()  # 调用主函数 run，执行程序
