import os                                          # 导入操作系统模块

from collections import namedtuple                 # 从collections模块导入namedtuple

import requests                                    # 导入requests模块

import numpy as np                                 # 导入NumPy库，简写为np

import pandas as pd                                # 导入Pandas库，简写为pd

from scipy import stats                            # 从scipy库导入stats模块


Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])         # 定义名为Distributions的namedtuple，包含指定的字段
Distribution = namedtuple('Distribution', ['name', 'params'])       # 定义名为Distribution的namedtuple，包含'name'和'params'字段


def generate_samples(distribution, params, size):                   # 定义函数generate_samples，生成指定分布的随机样本
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":                                  # 如果分布类型是"constant"
        return np.ones(size) * params["value"]                      # 返回值为指定值的数组
    elif distribution == "normal":                                  # 如果分布类型是"normal"
        return stats.norm(**params).rvs(size=size)                  # 返回正态分布的随机样本
    elif distribution == "truncnorm":                               # 如果分布类型是"truncnorm"
        return stats.truncnorm(**params).rvs(size=size)             # 返回截断正态分布的随机样本
    elif distribution == "randint":                                 # 如果分布类型是"randint"
        return stats.uniform(**params).rvs(size=size)               # 返回均匀分布的随机样本（应为整数分布，可能需要修正）
    elif distribution == "uniform":                                 # 如果分布类型是"uniform"
        return stats.uniform(**params).rvs(size=size)               # 返回均匀分布的随机样本
    elif distribution == "exponential":                             # 如果分布类型是"exponential"
        return stats.expon(**params).rvs(size=size)                 # 返回指数分布的随机样本
    elif distribution == "poisson":                                 # 如果分布类型是"poisson"
        return stats.poisson(**params).rvs(size=size)               # 返回泊松分布的随机样本
    elif distribution == "trace":                                   # 如果分布类型是"trace"
        df = pd.read_csv(params["filename"])                        # 读取指定文件为DataFrame
        return df[params["column"]].sample(size, replace=True).values   # 从指定列中有放回地采样
    else:
        raise ValueError(f"Invalid distribution: {distribution}")   # 如果分布类型无效，抛出异常


def generate_trace(max_requests, distributions, end_time=None):     # 定义函数generate_trace，生成请求的追踪数据
    """
    Generate a trace of requests based on the given distributions.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)                           # 生成请求ID数组

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)             # 生成到达时间间隔的样本
    arrival_timestamps = np.cumsum(arrival_timestamps)              # 计算到达时间戳（累积和）
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)                # 生成应用程序ID的样本
    application_ids = map(int, application_ids)                     # 将应用程序ID转换为整数
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)                    # 生成批量大小的样本
    batch_sizes = map(int, batch_sizes)                             # 将批量大小转换为整数
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)                   # 生成提示大小的样本
    prompt_sizes = map(int, prompt_sizes)                           # 将提示大小转换为整数
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)                    # 生成生成的令牌大小的样本
    token_sizes = map(int, token_sizes)                             # 将令牌大小转换为整数
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)               # 生成请求类型ID的样本
    request_type_ids = map(int, request_type_ids)                   # 将请求类型ID转换为整数

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,                                  # 请求ID
        "request_type": request_type_ids,                           # 请求类型
        "application_id": application_ids,                          # 应用程序ID
        "arrival_timestamp": arrival_timestamps,                    # 到达时间戳
        "batch_size": batch_sizes,                                  # 批量大小
        "prompt_size": prompt_sizes,                                # 提示大小
        "token_size": token_sizes,                                  # 令牌大小
    })

    if end_time is not None:                                        # 如果指定了结束时间
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]   # 过滤掉超过结束时间的请求

    return trace_df                                                 # 返回生成的追踪数据


def get_exponential_scale(num_servers, utilization, request_duration):  # 定义函数get_exponential_scale，计算指数分布的尺度参数
    """
    assumes that request_duration is in seconds
    """
    interarrival_time = request_duration / (1.0 * utilization)      # 计算请求的平均到达间隔时间
    exponential_scale = interarrival_time / num_servers             # 计算指数分布的尺度参数
    return exponential_scale                                        # 返回尺度参数


def generate_trace_from_utilization(
    max_requests,
    end_time,
    num_servers,
    utilization,
    request_duration,
    pt_distributions_file):                                         # 定义函数generate_trace_from_utilization，根据利用率生成追踪数据
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    exponential_scale = get_exponential_scale(num_servers, utilization, request_duration)  # 计算指数分布的尺度参数
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),      # 应用程序ID为常数0
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference   # 请求类型为常数2（用于LLM推理）
        arrival_process=Distribution("exponential", {"scale": exponential_scale}),        # 到达过程为指数分布
        prompt_size=Distribution("trace", {"filename": pt_distributions_file,
                                           "column": "ContextTokens"}),                  # 提示大小从文件中采样
        token_size=Distribution("trace", {"filename": pt_distributions_file,
                                          "column": "GeneratedTokens"}),                 # 令牌大小从文件中采样
        batch_size=Distribution("constant", {"value": 1}),                               # 批量大小为1
    )

    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)                                         # 生成追踪数据
    return trace_df                                                                     # 返回追踪数据


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):                                                         # 定义函数generate_trace_from_prompt_token_size_distributions，根据请求率生成追踪数据
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),                          # 应用程序ID为常数0
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference   # 请求类型为常数2（用于LLM推理）
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),     # 到达过程为指数分布，尺度为1/请求率
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),                 # 提示大小从文件中采样
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),                  # 提示大小使用截断正态分布（注释掉）
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),                # 令牌大小从文件中采样
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),                     # 令牌大小使用截断正态分布（注释掉）
        batch_size=Distribution("constant", {"value": 1}),                              # 批量大小为1
    )
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)                                        # 生成追踪数据
    return trace_df                                                                    # 返回追踪数据


def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):                                           # 定义函数generate_traces，生成多个请求率下的追踪数据
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:                                                  # 遍历请求率列表
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)                                                      # 生成指定请求率下的追踪数据
        trace_filename = trace_filename_template.format(request_rate)                   # 生成文件名
        trace_df.to_csv(trace_filename, index=False)                                    # 保存追踪数据到CSV文件


def generate_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    trace_filename_template="traces/rr_code_{}.csv"):                                   # 定义函数generate_code_traces，生成代码请求的追踪数据
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]): # 如果追踪文件目录不存在
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])        # 创建目录

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    code_distributions_file,
                    trace_filename_template)                                            # 调用generate_traces函数生成追踪数据


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):                                   # 定义函数generate_conv_traces，生成对话请求的追踪数据
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]): # 如果追踪文件目录不存在
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])        # 创建目录

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)                                            # 调用generate_traces函数生成追踪数据


def download_file(url, filename):                                                       # 定义函数download_file，下载指定URL的文件
    """
    Download a file from the given URL.
    """
    response = requests.get(url)                                                        # 发送GET请求
    with open(filename, "wb") as f:                                                     # 以二进制写模式打开文件
        f.write(response.content)                                                       # 写入文件内容


def download_azure_llm_traces():                                                        # 定义函数download_azure_llm_traces，下载Azure LLM的追踪数据
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):                                                      # 如果data目录不存在
        os.makedirs("data")                                                             # 创建data目录

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"# 定义基础URL

    if not os.path.exists("data/code_distributions.csv"):                               # 如果代码分布文件不存在
        url = url_base + "AzureLLMInferenceTrace_code.csv"                              # 定义代码分布文件的URL
        download_file(url, "data/code_distributions.csv")                               # 下载代码分布文件
        print("Downloaded code traces")                                                 # 打印下载完成信息

    if not os.path.exists("data/conv_distributions.csv"):                               # 如果对话分布文件不存在
        url = url_base + "AzureLLMInferenceTrace_conv.csv"                              # 定义对话分布文件的URL
        download_file(url, "data/conv_distributions.csv")                               # 下载对话分布文件
        print("Downloaded conv traces")                                                 # 打印下载完成信息


if __name__ == "__main__":                                                              # 主程序入口
    # download prompt and token size distributions
    download_azure_llm_traces()                                                         # 下载Azure LLM的追踪数据

    # generate request traces
    generate_code_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        code_distributions_file="data/code_distributions.csv")                          # 生成代码请求的追踪数据
    print("Generated code traces")                                                      # 打印生成完成信息

    generate_conv_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        conv_distributions_file="data/conv_distributions.csv")                          # 生成对话请求的追踪数据
    print("Generated conv traces")                                                      # 打印生成完成信息

    # generate request traces for 2 min
    generate_code_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        code_distributions_file="data/code_distributions.csv",
        trace_filename_template="traces/rr_code_{}_2min.csv")                           # 生成持续2分钟的代码请求追踪数据
    print("Generated code 2min traces")                                                 # 打印生成完成信息

    generate_conv_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        conv_distributions_file="data/conv_distributions.csv",
        trace_filename_template="traces/rr_conv_{}_2min.csv")                           # 生成持续2分钟的对话请求追踪数据
    print("Generated conv 2min traces")                                                 # 打印生成完成信息
