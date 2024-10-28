"""
Misc. utility functions
"""

import logging  # 导入日志模块，用于记录日志
import os  # 导入操作系统接口模块，用于文件和目录操作

from pathlib import Path  # 导入 Path，用于处理文件路径

import numpy as np  # 导入 numpy，用于数值计算
import pandas as pd  # 导入 pandas，用于数据操作

from hydra.utils import get_original_cwd  # 从 hydra.utils 导入 get_original_cwd，获取当前工作目录
from omegaconf import OmegaConf  # 导入 OmegaConf，用于处理配置文件
from scipy import stats  # 导入 scipy.stats，用于统计分析


def file_logger(name, level=logging.INFO):  # 定义 file_logger 函数，返回自定义文件日志记录器
    """
    returns a custom logger that logs to a file
    """
    logger = logging.getLogger(name)  # 创建名为 name 的日志记录器
    logger.setLevel(level)  # 设置日志记录器的记录等级

    # don't print to console (don't propagate to root logger)
    logger.propagate = False  # 不向根日志记录器传递日志信息

    # create a file handler
    handler = logging.FileHandler(f"{name}.csv", mode="w")  # 创建文件处理器，将日志写入 name.csv 文件
    handler.setLevel(level)  # 设置文件处理器的记录等级

    # add the handlers to the logger
    logger.addHandler(handler)  # 将文件处理器添加到日志记录器

    return logger  # 返回自定义日志记录器


def read_all_yaml_cfgs(yaml_cfg_dir):  # 定义 read_all_yaml_cfgs 函数，读取目录中的所有 YAML 配置文件
    """
    Read all yaml config files in a directory
    Returns a dictionary of configs keyed by the yaml filename
    """
    yaml_cfgs = {}  # 创建空字典用于存储配置
    yaml_cfg_files = os.listdir(yaml_cfg_dir)  # 列出目录中的所有文件
    for yaml_cfg_file in yaml_cfg_files:  # 遍历文件列表
        if not yaml_cfg_file.endswith((".yaml", ".yml")):  # 跳过非 YAML 文件
            continue
        yaml_cfg_path = os.path.join(yaml_cfg_dir, yaml_cfg_file)  # 构建 YAML 文件路径
        yaml_cfg = OmegaConf.load(yaml_cfg_path)  # 加载 YAML 配置文件
        yaml_cfg_name = Path(yaml_cfg_path).stem  # 获取文件名（不含扩展名）
        yaml_cfgs[yaml_cfg_name] = yaml_cfg  # 将配置添加到字典，以文件名为键
    return yaml_cfgs  # 返回包含所有 YAML 配置的字典


def get_statistics(values, statistics=None):  # 定义 get_statistics 函数，计算指标的统计信息
    """
    Compute statistics for a metric
    """
    if statistics is None:  # 如果未指定统计项，则使用默认统计项
        statistics = ["mean",
                      "std",
                      "min",
                      "max",
                      "median",
                      "p50",
                      "p90",
                      "p95",
                      "p99",
                      "p999",
                      "geomean"]
    results = {}  # 创建空字典用于存储统计结果
    if "mean" in statistics:
        results["mean"] = np.mean(values)  # 计算均值
    if "std" in statistics:
        results["std"] = np.std(values)  # 计算标准差
    if "min" in statistics:
        results["min"] = np.min(values)  # 计算最小值
    if "max" in statistics:
        results["max"] = np.max(values)  # 计算最大值
    if "median" in statistics:
        results["median"] = np.median(values)  # 计算中位数
    if "p50" in statistics:
        results["p50"] = np.percentile(values, 50)  # 计算 50 分位数
    if "p90" in statistics:
        results["p90"] = np.percentile(values, 90)  # 计算 90 分位数
    if "p95" in statistics:
        results["p95"] = np.percentile(values, 95)  # 计算 95 分位数
    if "p99" in statistics:
        results["p99"] = np.percentile(values, 99)  # 计算 99 分位数
    if "p999" in statistics:
        results["p999"] = np.percentile(values, 99.9)  # 计算 99.9 分位数
    if "geomean" in statistics:
        results["geomean"] = stats.gmean(values)  # 计算几何均值
    return results  # 返回统计结果字典


def save_dict_as_csv(d, filename):  # 定义 save_dict_as_csv 函数，将字典保存为 CSV 文件
    dirname = os.path.dirname(filename)  # 获取文件所在目录
    if dirname != "":  # 如果目录不为空
        os.makedirs(dirname, exist_ok=True)  # 创建目录（如果不存在）
    df = pd.DataFrame(d)  # 将字典转换为 DataFrame
    df.to_csv(filename, index=False)  # 将 DataFrame 保存为 CSV 文件，不写入行索引
