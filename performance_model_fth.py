import math  # 导入数学模块，用于数学运算
import os  # 导入操作系统接口模块，用于文件和目录操作

from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器

import pandas as pd  # 导入 pandas 库，用于数据处理

from hydra.utils import get_original_cwd  # 导入获取原始当前工作目录的函数
from scipy.interpolate import interp1d  # 导入一维线性插值函数

from task import TaskType, PromptTask, TokenTask  # 从 task 模块导入任务类型和任务类


performance_model = None  # 初始化性能模型变量


class PerformanceModel(ABC):  # 定义性能模型的抽象基类
    """
    PerformanceModel 帮助估计任务或迭代的持续时间，
    在给定硬件、模型和并行配置下。
    抽象类，必须被子类化。
    """
    def __init__(self):
        global performance_model  # 声明全局性能模型变量
        performance_model = self  # 将当前实例赋值给性能模型变量

    @abstractmethod
    def get_duration(self, task, batch, instance, *args, **kwargs):
        """
        返回任务的执行时间。
        """
        raise NotImplementedError  # 抛出未实现错误

    @abstractmethod
    def get_iteration_duration(self, batch, instance, *args, **kwargs):
        """
        返回连续迭代的执行时间。
        """
        raise NotImplementedError  # 抛出未实现错误


class ConstantPerformanceModel(PerformanceModel):  # 定义常量性能模型
    """
    PerformanceModel 返回一个常量值，无论其他参数如何。
    用于测试目的。
    """
    def __init__(self, prompt_time, token_time):
        super().__init__()  # 调用父类构造函数
        self.prompt_time = prompt_time  # 设置提示时间
        self.token_time = token_time  # 设置令牌时间

    def get_duration(self, task, batch, instance, *args, **kwargs):
        if task.task_type == TaskType.PROMPT:  # 如果任务类型是提示
            return self.prompt_time  # 返回提示时间
        elif task.task_type == TaskType.TOKEN:  # 如果任务类型是令牌
            return self.token_time  # 返回令牌时间
        else:
            raise NotImplementedError  # 抛出未实现错误

    def get_iteration_duration(self, batch, instance, *args, **kwargs):
        raise NotImplementedError  # 抛出未实现错误


class DatabasePerformanceModel(PerformanceModel):  # 定义基于数据库的性能模型
    """
    PerformanceModel 基于 CSV 数据库的特征运行。
    在数据点之间插值，并相应更新数据库。
    底层预测器可以更改以实现不同的插值策略。
    """
    def __init__(self, db_path):
        super().__init__()  # 调用父类构造函数
        self.db = pd.read_csv(os.path.join(get_original_cwd(), db_path),  # 从 CSV 文件加载数据
                              dtype={"model": "category", "hardware": "category"})

        # 确保数据库具有正确的列，并移除多余的列
        self.db = self.db[["model",
                           "hardware",
                           "tensor_parallel",
                           "prompt_size",
                           "batch_size",
                           "token_size",
                           "prompt_time",
                           "token_time"]]

        # 转换为秒
        self.db["prompt_time"] = self.db["prompt_time"] / 1000  # 将提示时间转换为秒
        self.db["token_time"] = self.db["token_time"] / 1000  # 将令牌时间转换为秒

        self.init_predictor()  # 初始化预测器

    def init_predictor(self):
        """
        使用批量中的令牌数进行预测。
        """
        self.prompt_time_predictors = {}  # 提示时间预测器字典
        self.token_time_predictors = {}  # 令牌时间预测器字典
        self.prompt_time_cache = {}  # 提示时间缓存字典
        self.token_time_cache = {}  # 令牌时间缓存字典

        for model in self.db["model"].unique():  # 遍历每个模型
            for hardware in self.db["hardware"].unique():  # 遍历每种硬件
                for tensor_parallel in self.db["tensor_parallel"].unique():  # 遍历每种张量并行配置
                    mask = (self.db["model"] == model) & \
                            (self.db["hardware"] == hardware) & \
                            (self.db["tensor_parallel"] == tensor_parallel)  # 创建掩码
                    db_subset = self.db[mask].copy()  # 选择子集
                    if len(db_subset) == 0:  # 如果子集为空，继续下一个循环
                        continue
                    db_subset["batch_tokens"] = db_subset["prompt_size"] * db_subset["batch_size"]  # 计算批量令牌数
                    x = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median().index  # 获取批量令牌数
                    y = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median()["prompt_time"]  # 获取提示时间
                    self.prompt_time_predictors[(model, hardware, tensor_parallel)] = interp1d(  # 创建插值函数
                                                                    x, y, fill_value="extrapolate")
                    x = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median().index  # 获取批量令牌数
                    y = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median()["token_time"]  # 获取令牌时间
                    self.token_time_predictors[(model, hardware, tensor_parallel)] = interp1d(  # 创建插值函数
                                                                    x, y, fill_value="extrapolate")

    def _match(self, **kwargs):
        """
        返回数据库中符合 kwargs 条件的布尔掩码。
        """
        mask = True  # 初始化掩码为 True
        for k, v in kwargs.items():  # 遍历所有关键字参数
            mask &= (self.db[k] == v)  # 更新掩码
        return mask  # 返回最终掩码

    def predict_new_row(self, **kwargs):
        """
        预测新行的提示和令牌时间。
        将新行插入数据库。
        """
        model = kwargs["model"]  # 获取模型
        hardware = kwargs["hardware"]  # 获取硬件
        tensor_parallel = kwargs["tensor_parallel"]  # 获取张量并行配置
        batch_tokens = kwargs["batch_tokens"]  # 获取批量令牌数
        new_row = pd.DataFrame(kwargs, index=[0])  # 创建新行的 DataFrame

        prompt_time = self.prompt_time_predictors[(model, hardware, tensor_parallel)](batch_tokens)  # 预测提示时间
        token_time = self.token_time_predictors[(model, hardware, tensor_parallel)](batch_tokens)  # 预测令牌时间

        new_row["prompt_time"] = prompt_time  # 将预测的提示时间添加到新行
        new_row["token_time"] = token_time  # 将预测的令牌时间添加到新行
        self.db = pd.concat([self.db, new_row], ignore_index=True)  # 将新行添加到数据库
        return new_row  # 返回新行

    def get_prompt_time(self, **kwargs):
        """
        从数据库返回提示时间。
        """
        prompt_time = self.db[self._match(**kwargs)]["prompt_time"].median()  # 获取中位数提示时间
        # 如果未找到，进行预测
        if math.isnan(prompt_time):  # 如果提示时间是 NaN
            new_row = self.predict_new_row(**kwargs)  # 预测新行
            prompt_time = new_row["prompt_time"][0]  # 获取预测的提示时间
        return prompt_time  # 返回提示时间

    def get_token_time(self, **kwargs):
        """
        从数据库返回令牌时间。
        """
        token_time = self.db[self._match(**kwargs)]["token_time"].median()  # 获取中位数令牌时间
        # 如果未找到，进行预测
        if math.isnan(token_time):  # 如果令牌时间是 NaN
            new_row = self.predict_new_row(**kwargs)  # 预测新行
            token_time = new_row["token_time"][0]  # 获取预测的令牌时间
        return token_time  # 返回令牌时间

    def get_duration(self,
                     task,
                     batch,
                     instance,
                     *args,
                     **kwargs):
        model = instance.model.name  # 获取模型名称
        hardware = instance.processors[0].name  # 获取硬件名称
        pipeline_parallel = instance.model.parallelism.pipeline_parallelism  # 获取管道并行度
        tensor_parallel = instance.model.parallelism.tensor_parallelism  # 获取张量并行度
        if task.task_type == TaskType.PROMPT:  # 如果任务类型是提示
            prompt_size = task.request.prompt_size  # 获取提示大小
            token_size = task.request.token_size  # 获取令牌大小
            batch_size = len(batch)  # 获取批量大小
            prompt_time = self.get_prompt_time(model=model,
                                               hardware=hardware,
                                               tensor_parallel=tensor_parallel,
                                               prompt_size=prompt_size,
                                               batch_size=batch_size,
                                               token_size=token_size,
                                               batch=batch)  # 获取提示时间
            return prompt_time  # 返回提示时间
        elif task.task_type == TaskType.TOKEN:  # 如果任务类型是令牌
            prompt_size = task.request.prompt_size  # 获取提示大小
            token_size = task.request.token_size  # 获取令牌大小
            batch_size = len(batch)  # 获取批量大小
            token_time = self.get_token_time(model=model,
                                             hardware=hardware,
                                             tensor_parallel=tensor_parallel,
                                             prompt_size=prompt_size,
                                             batch_size=batch_size,
                                             token_size=token_size,
                                             batch=batch)  # 获取令牌时间
            return token_time * task.token_size  # 返回令牌时间乘以令牌大小
        else:
            raise NotImplementedError  # 抛出未实现错误

    def get_iteration_duration(self,
                               batch,
                               instance,
                               *args,
                               **kwargs):
        """
        注意：假设提示始终完全处理。
        即，我们当前不支持提示分块。
        """
        model = instance.model.name  # 获取模型名称
        hardware = instance.processors[0].name  # 获取硬件名称
        pipeline_parallel = instance.model.parallelism.pipeline_parallelism  # 获取管道并行度
        tensor_parallel = instance.model.parallelism.tensor_parallelism  # 获取张量并行度

        prompt_tasks = []  # 初始化提示任务列表
        token_tasks = []  # 初始化令牌任务列表
        batch_tokens = 0  # 初始化批量令牌计数
        for task in batch:  # 遍历批量中的任务
            if isinstance(task, PromptTask):  # 如果任务是提示任务
                prompt_tasks.append(task)  # 将任务添加到提示任务列表
                batch_tokens += task.request.prompt_size  # 增加批量令牌计数
            elif isinstance(task, TokenTask):  # 如果任务是令牌任务
                token_tasks.append(task)  # 将任务添加到令牌任务列表
                batch_tokens += 1  # 增加批量令牌计数
            else:
                raise NotImplementedError  # 抛出未实现错误

        iteration_time = None  # 初始化迭代时间为 None
        cache_key = (model, hardware, tensor_parallel, batch_tokens)  # 创建缓存键
        predictors_key = (model, hardware, tensor_parallel)  # 创建预测器键

        if len(prompt_tasks) == len(batch):  # 如果所有任务都是提示任务
            iteration_time = self.prompt_time_cache.get(cache_key)  # 从缓存获取迭代时间
            if iteration_time is None:  # 如果缓存中没有迭代时间
                iteration_time = float(self.prompt_time_predictors[predictors_key](batch_tokens))  # 预测迭代时间
                self.prompt_time_cache[cache_key] = float(iteration_time)  # 将预测结果存入缓存
        elif len(token_tasks) == len(batch):  # 如果所有任务都是令牌任务
            iteration_time = self.token_time_cache.get(cache_key)  # 从缓存获取迭代时间
            if iteration_time is None:  # 如果缓存中没有迭代时间
                iteration_time = float(self.token_time_predictors[predictors_key](batch_tokens))  # 预测迭代时间
                self.token_time_cache[cache_key] = float(iteration_time)  # 将预测结果存入缓存
        else:  # 如果有混合任务
            iteration_time = self.prompt_time_cache.get(cache_key)  # 从缓存获取迭代时间
            if iteration_time is None:  # 如果缓存中没有迭代时间
                iteration_time = float(self.prompt_time_predictors[predictors_key](batch_tokens))  # 预测迭代时间
                self.prompt_time_cache[cache_key] = float(iteration_time)  # 将预测结果存入缓存
            iteration_time *= 1.1  # 将迭代时间乘以 1.1

        assert iteration_time > 0  # 确保迭代时间大于 0
        return iteration_time  # 返回迭代时间


def get_duration(*args, **kwargs):
    """
    返回任务的执行时间。
    """
    return performance_model.get_duration(*args, **kwargs)  # 调用性能模型的 get_duration 方法


def get_iteration_duration(*args, **kwargs):
    """
    返回连续迭代的执行时间。
    """
    return performance_model.get_iteration_duration(*args, **kwargs)  # 调用性能模型的 get_iteration_duration 方法
