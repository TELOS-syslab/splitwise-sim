"""
Utility functions to initialize the Cluster with a starting state. 用于启动状态初始化集群的实用程序函数。
"""

import logging  # 导入日志模块

from model import ModelParallelism  # 从 model 模块导入 ModelParallelism 类
from simulator import clock, schedule_event, cancel_event, reschedule_event  # 从 simulator 模块导入调度函数


def load_start_state(start_state_cfg, **kwargs):  # 定义 load_start_state 函数，用于加载并初始化集群的起始状态;    与* args 类似，** kwargs 也是 Python 中的特殊语法，它可以让我们接受任意数量的关键字参数。在函数声明中，** kwargs 其实是一个字典，其中包含了所有传入的关键字参数。
    """
    Load the start state configuration and initialize the cluster.  加载启动状态配置并初始化集群。
    """
    state_type = start_state_cfg.state_type  # 获取起始状态的类型
    if state_type == "unallocated":  # 如果状态类型为 "unallocated"
        pass  # 不做任何操作
    elif state_type == "orca":  # 如果状态类型为 "orca"
        uniform(start_state_cfg, **kwargs)  # 调用 uniform 函数初始化
    elif state_type == "baseline":  # 如果状态类型为 "baseline"
        uniform(start_state_cfg, **kwargs)  # 调用 uniform 函数初始化
    elif "splitwise" in state_type:  # 如果状态类型包含 "splitwise"
        splitwise(start_state_cfg, **kwargs)  # 调用 splitwise 函数初始化
    else:
        raise ValueError(f"Unknown start state type: {state_type}")  # 如果状态类型未知，抛出异常


def uniform(start_state_cfg, cluster, applications, **kwargs):  # 定义 uniform 函数，用于所有服务器启动单个应用实例
    """
    Initialize all servers with a single instance of the application. 使用一个应用程序实例初始化所有服务器。
    """
    application = applications[start_state_cfg.application_id]  # 获取指定应用程序
    allocator = application.allocator  # 获取应用程序的分配器
    servers = cluster.servers  # 获取集群中的所有服务器

    instance_cfg = start_state_cfg.instance  # 获取实例配置
    parallelism = ModelParallelism(pipeline_parallelism=instance_cfg.pipeline_parallelism,
                                   tensor_parallelism=instance_cfg.tensor_parallelism)  # 设置模型并行度

    for sku_name in servers:  # 遍历所有服务器 SKU
        for server in servers[sku_name]:  # 遍历 SKU 下的所有服务器
            allocator.start_spin_up_instance(instance_cfg=instance_cfg,
                                             processors=server.processors,
                                             parallelism=parallelism,
                                             pre_start=True)  # 启动应用实例


def splitwise(start_state_cfg, cluster, applications, **kwargs):  # 定义 splitwise 函数，用于按并行类型初始化服务器
    """
    Initialize all servers with a single instance of the application.
    Separate prompt and token instances with different kinds of parallelism.
    TODO: use preferences and constraints within scheduler instead
    使用一个应用程序实例初始化所有服务器。
    使用不同类型的并行性分隔提示实例和标记实例。
    TODO：在调度程序中使用首选项和约束来代替
    """
    application = applications[start_state_cfg.application_id]  # 获取指定应用程序
    allocator = application.allocator  # 获取应用程序的分配器
    servers = cluster.servers  # 获取集群中的所有服务器

    prompt_cfg = start_state_cfg.prompt  # 获取提示配置
    token_cfg = start_state_cfg.token  # 获取令牌配置
    prompt_parallelism = ModelParallelism(pipeline_parallelism=prompt_cfg.pipeline_parallelism,
                                          tensor_parallelism=prompt_cfg.tensor_parallelism)  # 设置提示并行度
    token_parallelism = ModelParallelism(pipeline_parallelism=token_cfg.pipeline_parallelism,
                                         tensor_parallelism=token_cfg.tensor_parallelism)  # 设置令牌并行度

    split_type = start_state_cfg.split_type  # 获取分割类型

    if split_type == "homogeneous":  # 同质分割类型
        n_prompts = prompt_cfg.num_instances  # 获取提示实例数量
        n_tokens = token_cfg.num_instances  # 获取令牌实例数量
        # allocate n_prompt instance of prompt
        all_servers = [server for sku_name in servers for server in servers[sku_name]]  # 获取所有服务器的列表
        for server in all_servers[:n_prompts]:  # 分配提示实例到前 n_prompts 台服务器
            for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                 processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                 parallelism=prompt_parallelism,
                                                 pre_start=True,
                                                 tag="prompt")  # 启动提示实例
        for server in all_servers[n_prompts:n_prompts+n_tokens]:  # 分配令牌实例到接下来的 n_tokens 台服务器
            for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                 processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                 parallelism=token_parallelism,
                                                 pre_start=True,
                                                 tag="token")  # 启动令牌实例

    if split_type == "heterogeneous":  # 异质分割类型
        prompt_instances = prompt_cfg.instance_names  # 获取提示实例的 SKU 列表
        token_instances = token_cfg.instance_names  # 获取令牌实例的 SKU 列表
        for sku_name in servers:  # 遍历所有服务器 SKU
            for server in servers[sku_name]:  # 遍历每个 SKU 下的服务器
                if sku_name in prompt_instances:  # 如果 SKU 名在提示实例列表中
                    # allocate as many prompt instances as possible
                    for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                         processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                         parallelism=prompt_parallelism,
                                                         pre_start=True,
                                                         tag="prompt")  # 启动提示实例
                elif sku_name in token_instances:  # 如果 SKU 名在令牌实例列表中
                    # allocate as many token instances as possible
                    for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                         processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                         parallelism=token_parallelism,
                                                         pre_start=True,
                                                         tag="token")  # 启动令牌实例
                else:
                    raise ValueError(f"Unsupported sku_name: {sku_name}")  # 如果 SKU 名不支持，抛出异常
