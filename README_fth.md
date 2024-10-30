# fth
pip install hydra-core

https://github.com/TELOS-syslab/splitwise-sim


# SplitwiseSim: LLM Serving Cluster Simulator

SplitwiseSim is a discrete event simulator that helps evaluate model serving in LLM inference clusters. It was built to evaluate [Splitwise](#reference), a generative LLM inference serving technique that splits LLM inference phases across different machines. SplitwiseSim can easily be extended to other applications and use cases.

SplitwiseSim：LLM 服务集群模拟器
SplitwiseSim 是一个离散事件模拟器，帮助评估大型语言模型（LLM）推理集群中的模型服务。它用于评估 Splitwise，这是一种生成式 LLM 推理服务技术，通过在不同机器间分割 LLM 推理阶段。SplitwiseSim 也易于扩展至其他应用和使用场景。



## Setup

You can set up SplitwiseSim by installing its Python dependencies. We recommend starting with a fresh Python environment.
安装
可以通过安装 Python 依赖来设置 SplitwiseSim。推荐使用一个新的 Python 环境。

```python
# Create and activate new Python environment
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim

# Install dependencies
pip install -r requirements.txt
```

**NOTE**: SplitwiseSim has only been tested with Python 3.11. However, it will likely also work with other Python versions.
注意：SplitwiseSim 仅在 Python 3.11 上测试过，但可能也支持其他 Python 版本。

## Inputs and Outputs

SplitwiseSim takes in a hierarchical set of YAML configuration files as input, and it produces several CSV files as output. It uses [Hydra](https://hydra.cc/) for configuration management. You can learn more about configuration management from the [Hydra docs](https://hydra.cc/docs/intro/).
输入和输出
SplitwiseSim 以一组层级化的 YAML 配置文件作为输入，并输出多个 CSV 文件。它使用 Hydra 管理配置，可以通过 Hydra 文档了解更多配置管理信息。

The top-level configuration file for SplitwiseSim is [`config.yaml`](configs/config.yaml), which points to lower-level configurations specified by other files in the `configs/` directory. Specifically, `config.yaml` captures the following key components:
SplitwiseSim 的顶层配置文件为 config.yaml，它引用了 configs/ 目录中的其他文件以实现更细粒度的配置，主要配置项包括：

- [cluster](configs/cluster/): the provisioned server SKUs in the cluster, along with their respective counts.
- [trace](#request-traces): request trace that specifies the set of requests that arrive into the cluster.
- [router](configs/router/): the cluster-level router that routes incoming requests to application-level schedulers; currently a no-op.
- [arbiter](configs/arbiter/): the cluster-level arbiter that manages compute resources between applications to support autoscaling; currently a no-op.
- [application](configs/applications/): the logical endpoint that the requests target, which specifies the model and the set of instances on which the request runs; currently, we support only one application.
- [model_repo](configs/model_repo/): the set of models (LLMs) available to run in the cluster; used for dynamic model instantiation.
- [orchestrator_repo](configs/orchestrator_repo/): the set of application resource orchestrators (i.e., schedulers and allocators) in the cluster; used for dynamic application management.
- [hardware_repo](configs/hardware_repo/): the set of available SKUs that can be provisioned in the cluster; used for dynamic server instantiation.
- [performance_model](#performance-model): an analytical model that helps estimate request runtimes with different batch, model, and hardware configurations.
- [start_state](configs/start_state/): starting state for the cluster, which helps simplify evaluation.

Several other aspects can be configured; please see [`config.yaml`](configs/config.yaml) for details.

!cluster：集群中的服务器 SKU 及其数量。
trace：定义进入集群的请求集的请求跟踪。
router：集群级路由器，将传入请求路由到应用级调度器（目前为无操作）。
arbiter：集群级仲裁器，在应用间管理计算资源以支持自动扩展（目前为无操作）。
application：请求的逻辑终端，指定模型和实例集，目前只支持一个应用。
!model_repo：可在集群中运行的模型集（LLM），用于动态模型实例化。
!orchestrator_repo：集群中应用资源的编排器（即调度器和分配器），用于动态应用管理。
hardware_repo：可在集群中预配的 SKU 集，用于动态服务器实例化。
!performance_model：帮助估算不同批处理、模型和硬件配置下请求运行时的分析模型。
!start_state：集群的初始状态，用于简化评估。
配置的其他部分请见 config.yaml 了解详细信息。

SplitwiseSim generates the following key outputs:

- Summary of application-level metrics (`summary.csv`)
- Per-request metrics for each completed request for each application (`detailed/{application_id}.csv`)
- Request node-level metrics (`request_nodes.csv`)
- Instance-level execution metrics (in `instances/`, with `debug` enabled)

We provide various [utility functions](notebooks/utils.py) to process outputs, as shown in [`notebooks/example.ipynb`](notebooks/example.ipynb) and [`notebooks/plots.ipynb`](notebooks/plots.ipynb).

SplitwiseSim 生成的主要输出包括：
应用层级的指标摘要（summary.csv）
每个应用的每个完成请求的请求指标（detailed/{application_id}.csv）
请求节点层级指标（request_nodes.csv）
实例层级的执行指标（存储在 instances/ 中，需开启 debug 模式）
我们提供了多种实用函数来处理输出，具体使用见 notebooks/example.ipynb 和 notebooks/plots.ipynb。

## Example Run

The simplest way to run SplitwiseSim is to execute [`run.py`](run.py), which runs with the default configuration parameters specified in [`config.yaml`](configs/config.yaml). The default configurations can be overridden by specifying appropriate command line parameters using Hydra. Below is an example script, [`scripts/run_baseline_h_example.sh`](scripts/run_baseline_h_example.sh), which overrides the default configuration to execute a simple `Baseline-H100` configuration with a single DGX-H100 server.

运行示例
运行 SplitwiseSim 最简单的方法是执行 run.py，它使用 config.yaml 中指定的默认配置参数。可以使用 Hydra 命令行参数覆盖默认配置。以下是一个示例脚本scripts/run_baseline_h_example.sh，用于在单个 DGX-H100 服务器上执行简单的 Baseline-H100 配置：

```bash
# scripts/run_baseline_h_example.sh
# ./scripts/run_baseline_h_example.sh

SCHEDULER=token_jsq
NUM_DGX_A100=0
NUM_DGX_H100=1
START_STATE=baseline
TRACE=test_trace

python run.py \
    applications.0.scheduler=$SCHEDULER \
    cluster=half_half \
    cluster.servers.0.count=$NUM_DGX_A100 \
    cluster.servers.1.count=$NUM_DGX_H100 \
    start_state=$START_STATE \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0
```
```
SCHEDULER=token_jsq：调度器设定为 token_jsq。
NUM_DGX_A100=0：集群中 DGX-A100 服务器的数量为 0。
NUM_DGX_H100=1：集群中 DGX-H100 服务器的数量为 1。
START_STATE=baseline：集群的初始状态设定为 baseline。
TRACE=test_trace：请求跟踪文件名设定为 test_trace。


python run.py \：运行 run.py 文件。
applications.0.scheduler=$SCHEDULER：将第一个应用的调度器设为 token_jsq。
cluster=half_half：将集群配置从默认的 dgx-a100 改为 half_half，包含一个 DGX-A100 和一个 DGX-H100。
cluster.servers.0.count=$NUM_DGX_A100：集群中的 DGX-A100 服务器数量为 0。
cluster.servers.1.count=$NUM_DGX_H100：集群中的 DGX-H100 服务器数量为 1。
start_state=$START_STATE：集群初始状态设为 baseline。
performance_model=db：使用性能模型 db，基于 A100 和 H100 的真实性能数据。
trace.filename=$TRACE：请求跟踪文件名设为 test_trace。
debug=True：启用调试模式。
seed=0：设置随机种子为 0。
```

Specifically, each configuration override changes a corresponding default from `config.yaml` as follows:
具体来说，每个配置覆盖项会改变 config.yaml 中对应的默认配置，如下：

- `cluster=half_half` overrides the cluster default from [`dgx-a100`](configs/cluster/dgx-a100.yaml) to [`half_half`](configs/cluster/half_half.yaml), which has 1 DGX-A100 and 1 DGX-H100 server SKU by default. 
cluster=half_half：将集群默认值从 dgx-a100 修改为 half_half，即默认包含 1 个 DGX-A100 和 1 个 DGX-H100。

- `cluster.servers.*` replace the number of DGX-A100 and DGX-H100 servers within the [`half_half`](configs/cluster/half_half.yaml) cluster to 0 and 1, respectively.
cluster.servers.*：更改 half_half 中的 DGX-A100 和 DGX-H100 数量为 0 和 1。

- `applications.0.scheduler=token_jsq` switches the default [`round_robin`](configs/orchestrator_repo/schedulers/round_robin.yaml) scheduler, as specified in [`configs/applications/solo.yaml`](configs/applications/solo.yaml), to the [`token_jsq`](configs/orchestrator_repo/schedulers/token_jsq.yaml) scheduler.
applications.0.scheduler=token_jsq：将调度器从 round_robin 切换为 token_jsq。

- `start_state=baseline` overrides the starting state from [`orca`](configs/start_state/orca.yaml) to [`baseline`](configs/start_state/baseline.yaml).
start_state=baseline：将起始状态从 orca 修改为 baseline。

- `performance_model=db` overrides the performance model to [`db`](configs/performance_model/db.yaml) instead of the default [`constant`](configs/performance_model/constant.yaml).
performance_model=db：将性能模型修改为 db。

- `trace.filename=test_trace` changes the trace file name (same as default, so no effect).
trace.filename=test_trace：修改跟踪文件名。

- `debug=True` enables the debug flag (changed from `False`)
debug=True：启用调试模式。

- `seed=0` sets the seed to `0` (same as default, so no effect).
seed=0：设置随机种子为 0。

Several of the above overrides configure objects of classes specified by the `_target_` field in the corresponding configuration files.
以上覆盖项中的几个配置了由相应配置文件中的`_target_` 字段指定的类对象。

To simulate this simple Baseline-H100 configuration with a single DGX-H100 on [`test_trace.csv`](traces/test_trace.csv), we can simply run the bash script:
为了使用单个 DGX-H100 服务器配置简单的 Baseline-H100 模型模拟，并以 test_trace.csv 作为请求跟踪文件，我们可以直接运行以下 bash 脚本：


```bash
# run simple Baseline-H100 example  运行单个 DGX-H100 上的简单 Baseline-H100 配置示例：
./scripts/run_baseline_h_example.sh
```

Similarly, we could run a simple Splitwise-HA configuration, which simulates KV-cache transfers from a DGX-H100 machine to DGX-A100 machine (see [paper](#reference) for more details):
可以运行简单的 Splitwise-HA 配置，模拟从 DGX-H100 向 DGX-A100 传输 KV 缓存的过程（详情见 论文）。

```bash

# run simple Splitwise-HA example
./scripts/run_splitwise_ha_example.sh
```

**NOTE**: Scripts must be run from the top-level directory.
注意：脚本必须从顶层目录运行。

Results will be generated in the `results/` directory according to the output path template specified by the `output_dir` field in [`config.yaml`](configs/config.yaml). Open [`notebooks/example.ipynb`](notebooks/example.ipynb) using Jupyter Notebook to see an example of how to easily extract the associated outputs.
结果会按照 config.yaml 中 output_dir 字段指定的输出路径模板生成在 results/ 目录中。在 Jupyter Notebook 中打开 notebooks/example.ipynb，查看提取输出的示例。

## Request Traces 请求Trace



SplitwiseSim expects request traces in a CSV file that contains the following fields for each request:
SplitwiseSim 需要包含以下字段的 CSV 请求Trace文件：

- `request_id`: ID of the request, typically a monotonically increasing number.
- `request_type`: Type of the request (e.g., DL inference, LLM inference, etc.). Use `2` for generative LLM inference, which is the only supported type at present.
- `application_id`: ID of the application / endpoint that the request targets. Default to `0` for a single application.
- `arrival_timestamp`: Timestamp at which the request arrives into the cluster.
- `batch_size`: If the request is already batched when it arrives, that can be specified here (currently not used).
- `prompt_size`: Number of tokens in the input prompt of the request.
- `token_size`: Number of tokens to be generated as output by the request.
request_id：请求 ID，通常递增。
request_type：请求类型，例如 DL 推理、LLM 推理等。对于生成式 LLM 推理，使用 2。
application_id：请求目标应用的 ID，默认为 0。
arrival_timestamp：请求到达集群的时间戳。
batch_size：到达时已分批处理的请求大小（当前未使用）。
prompt_size：请求输入提示的 token 数量。
token_size：请求生成的输出 token 数量。

Many of these fields have limited configurability at present. A typical new trace would change the `request_id`, `arrival_timestamp`, `prompt_size`, and `token_size`. An example trace can be found in [`traces/test_trace.csv`](traces/test_trace.csv).
目前，许多这些领域的可配置性都很有限。一个典型的新跟踪将改变`request_id`、`到达_时间戳`、`提示符_size`和`令牌_size`。在[`traces/test_trace.csv`]（traces/test_trace.csv）.中可以找到一个示例



### Production Traces and Trace Generation
生产Trace和Trace生成

'''
python generate_traces.py
'''

[Splitwise](#reference) was evaluated with request traces that were based off [production traces](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) from LLM inference services at Microsoft Azure. The [`generate_trace.py`](generate_trace.py) script can automatically download the production traces and use the corresponding prompt/token size distributions to generate request traces with different request rates. It can also help generate custom traces with different kinds of distributions. Modify and run `generate_trace.py` with desired request rates and other parameters. By default, all generated traces are expected to reside in the `traces/` directory.
Splitwise 的评估基于来自 Microsoft Azure LLM 推理服务的 生产Trace。generate_trace.py 脚本可以自动下载这些生产Trace，并使用相应的提示/输出 token 大小分布生成具有不同请求率的请求Trace。它还可以帮助生成具有不同分布类型的自定义Trace。修改 generate_trace.py 中的请求率和其他参数以生成不同的跟踪数据。默认情况下，生成的所有Trace文件都保存在 traces/ 目录中。

作用：这一部分允许用户使用真实生产数据或自定义生成的请求流量对模拟器进行负载测试，以便于评估模型在不同负载条件下的表现。



## Request Processing
请求处理流程

SplitwiseSim processes request traces as follows:
SplitwiseSim 按以下流程处理请求跟踪：

- All requests first arrive at a [Cluster](cluster.py)-level [Router](router.py), which forwards them to their target [Application](application.py). The Cluster also has an [Arbiter](arbiter.py) which helps reallocate [Servers](server.py) or [Processors](processor.py) between Applications. Currently, the Router and Arbiter act as no-ops, but they could be modified in the future to include smarter routing and autoscaling strategies with overheads.
所有请求首先到达 Cluster (cluster.py) 级别的 Router (router.py)，该路由器将请求转发至目标 Application (application.py)。集群还包含一个 Arbiter (arbiter.py)，负责在应用之间重新分配 Servers 或 Processors。当前路由器和仲裁器均为无操作，但将来可以升级为支持智能路由和自动扩展。

- Each [Request](request.py) targets a specific [Application](application.py), which may have one or more [Instances](instance.py) that run [Models](model.py). [Applications](application.py) use [Allocators](allocator.py) to spin-up/spin-down Instances on [Processors](processor.py), and they use [Schedulers](scheduler.py) to load balance Requests across Instances. Currently, we do not support dynamic Instance spin-up/spin-down, but rather use [start states](start_state.py) for specifying the initial set of Cluster Instances.
每个 Request (request.py)都会针对一个特定 Application (application.py)，应用可能包含一个或多个运行 Models 的实例 (instance.py)。应用(application.py)会使用 Allocators(allocator.py) 在 Processors(processor.py) 上启动/停止实例，使用 Schedulers(scheduler.py) 在实例间平衡请求负载。当前版本不支持动态实例启动/停止，而是使用 start states(start_state.py) 设定初始实例集。

- [Requests](request.py) are specified as a Directed Acyclic Graph (DAG) of [Nodes](node.py) for flexibility. Request nodes may either be [Tasks](task.py) and [Flows](flow.py). Requests are processed on [Instances](instance.py), which run on [Servers](server.py); specifically, Tasks are run on [Processors](processor.py) and Flows are run on [Links](interconnect.py).
Requests(request.py) 通过有向无环图（DAG）形式的 Nodes(node.py) 进行指定。请求节点可以是 Tasks(task.py) 或 Flows(flow.py)。请求会在 Instances(instance.py) 上处理，实例运行在 Servers(server.py) 上，特定任务在 Processors(processor.py) 上执行，流量在 Links(interconnect.py) 上传输。

Note that all simulation times are assumed to be specified in seconds.
作用：该流程展示了请求在模拟集群中的详细执行流程，确保在集群级、应用级和实例级都有适当的负载均衡和资源管理。



## Performance Model
性能模型

The [performance_model](performance_model.py) helps SplitwiseSim estimate how long requests run on diverse input, output, hardware, batch, etc. configurations. `performance_model.PerformanceModel` is an interface class which exposes the following two estimation functions to the simulator:
performance_model 估算请求在不同输入、输出、硬件、批量配置下的运行时间。performance_model.PerformanceModel 是一个接口类，提供了两个用于估算的函数：

1. `get_duration()`: used to estimate the runtime of prompt and token tasks.
2. `get_iteration_duration()`: used to estimate the runtime of each batching iteration (e.g., from continuous batching).
get_duration()：估算提示prompt和 token 任务的运行时间。
get_iteration_duration()：估算每次批处理迭代的运行时间（例如，连续批处理时 from continuous batching）。

Since modern LLM serving typically uses [iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu), we primarily rely on `get_iteration_duration` in the [Instance](instance.py) implementation (e.g., ORCAInstance and SplitwiseInstance).
现代 LLM 服务通常使用 iteration-level scheduling，因此在 Instance 实现中主要依赖 get_iteration_duration 函数（如在 ORCAInstance 和 SplitwiseInstance 中）。

Currently, SplitwiseSim provides two concrete performance models:
当前，SplitwiseSim 提供两个具体的性能模型：

1. `performance_model=constant`: This model assumes that all prompt and token tasks take a constant duration. While unrealistic, it is helpful for testing / debugging purposes.
performance_model=constant：假设所有提示和 token 任务的持续时间恒定，适用于测试/调试。

2. `performance_model=db`: This model uses extensive profiling data from the DGX-A100 and DGX-H100 machines and is the preferable model to use for realistic simulations. The associated raw data can be found in [`data/perf-model.csv`](data/perf-model.csv). The `performance_model.DatabasePerformanceModel` class reads this raw data to build a simple linear predictor, which serves as the performance model. To extend SplitwiseSim to different LLMs/platforms, please add your profiling data to the data file and potentially update the performance model predictor.
performance_model=db：利用 DGX-A100 和 DGX-H100 的配置数据，使用线性预测器作为性能模型。
作用：此模块为不同硬件配置和模型提供灵活的时间估算，以便于用户在不同场景下优化 LLM 服务的延迟和吞吐量。












## Experiments Workflow 实验工作流

This section describes how to run larger-scale simulations spanning a variety of configurations.
这一部分描述了如何运行跨多种配置的大规模模拟。

### Parallel Simulations 并行模拟


SplitwiseSim can be run on multiple cores (on one or more machines) to evaluate many different configurations in parallel. Each simulation configuration is run in a single process on a single core. SplitwiseSim uses [Ray](https://github.com/ray-project/ray) via the [Hydra Ray plugin](https://hydra.cc/docs/plugins/ray_launcher/) for parallelization.
SplitwiseSim 可在多核（单机或多机）上运行，以并行评估多种配置。每个模拟配置在单核单进程中运行。SplitwiseSim 使用 Ray 及 Hydra Ray 插件 实现并行化。

To start a Ray cluster, run:
启动 Ray 集群的步骤：

- `ray start --head` on the head machine.
- `ray start --address=xxx` on each of the worker machines.
在主节点运行：ray start --head
在每个工作节点运行：ray start --address=xxx

See [Ray docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html) for more details.

If you do not want to use Ray, you may alternatively use the Hydra [joblib](https://hydra.cc/docs/plugins/joblib_launcher/) launcher, which only supports multicore parallelization on a single machine.
如果不希望使用 Ray，可以使用 Hydra 的 joblib 启动器，仅支持单机多核并行。

Running a Hydra configuration in parallel requires the `--multirun` flag. For example, to sweep over multiple seed values in parallel, use `python --multirun run.py seed=0,1,2,3,4,5,6,7,8,9` after starting the Ray cluster.
要并行运行 Hydra 配置，需使用 --multirun 标志。例如，启动 Ray 集群后可运行：python --multirun run.py seed=0,1,2,3,4,5,6,7,8,9。

Output from multi-machine runs is stored on different machines corresponding to where each simulation configuration runs. Subsequently, you may need to manually collect results back into the same machine using sync scripts. Example sync scripts can be found in the `sync_scripts` folder.
多机器运行的输出存储在不同的机器上，对应于每个模拟配置运行的位置。随后，您可能需要使用同步脚本手动将结果收集回同一台机器中。示例同步脚本可以在`sync_shorits`文件夹中找到。

作用：并行化支持更大规模的模拟，使用户能够在不同配置下高效地评估集群表现。



### Experiment Runs 实验运行

The `scripts/` directory provides several scripts to run larger experiments, including parallel sweeps over different cluster configurations:
scripts/ 目录中提供了多个脚本用于运行大规模实验，包括并行的集群配置遍历：

- To run a baseline configuration, run `./scripts/run_baseline_a.sh` (Baseline-A100) or `./scripts/run_baseline_h.sh` (Baseline-H100).
运行基线配置：./scripts/run_baseline_a.sh（Baseline-A100）或 ./scripts/run_baseline_h.sh（Baseline-H100）。

- To run a Splitwise configuration, run the appropriate Splitwise-XX file under the scripts directory. For example, to run Splitwise-HA, run `./scripts/run_splitwise_ha.sh`.
运行 Splitwise 配置：运行对应的 Splitwise-XX 脚本，例如 ./scripts/run_splitwise_ha.sh 可运行 Splitwise-HA。

- Various experiment configurations used in the [Splitwise paper](#reference) are specified in the `configs/experiment/` folder. For example, to run a sweep of iso-cost clusters, you can run `./scripts/run_isocost.sh` which corresponds to `configs/experiment/*_isocost.yaml` with the appropriate sweep parameters (warning: running this may spin up many configurations in parallel and take a long time; try smaller configurations to begin with).
在 configs/experiment/ 文件夹中指定了 Splitwise 论文 中使用的各种实验配置。例如，要运行等成本（iso-cost）集群的遍历（sweep），可以执行 ./scripts/run_isocost.sh，该脚本对应 configs/experiment/*_isocost.yaml 中的合适遍历参数。（注意：运行此脚本可能会同时启动许多配置，并且可能需要较长时间；建议从较小的配置开始。）


作用：实验脚本简化了复杂实验的启动，使用户可以快速测试不同集群配置。




### Experiment Plots and Gantt Charts

Outputs from experiment sweeps can be visualized by using the plotting scripts provided in `notebooks/plots.ipynb`. These scripts were used to plot some of the graphs in the [Splitwise paper](#reference).

If the `debug` flag is enabled, SplitwiseSim additionally outputs iteration-level metadata per instance (including start/end timestamps), which can be visualized as Gantt charts for analysis and debugging. Check out `notebooks/example.ipynb` for a simple example. Custom markers can be added by modifying the simulator.
实验图表和甘特图
可使用 notebooks/plots.ipynb 中的绘图脚本可视化实验遍历的输出。这些脚本用于绘制 Splitwise 论文 中的一些图表。

如果启用了 debug 标志，SplitwiseSim 还会输出每个实例的迭代级元数据（包括开始/结束时间戳），可用于生成甘特图以进行分析和调试。详见 notebooks/example.ipynb 中的示例。

作用：可视化工具帮助用户分析集群运行过程中的细节，识别瓶颈，优化性能。

## Reference

If you use SplitwiseSim in your work, please cite the accompanying [paper](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/):

> Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", in Proceedings of the International Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.
参考文献
如果您在工作中使用 SplitwiseSim，请引用相关 论文：

Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", in Proceedings of the International Symposium on Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.

作用：为使用此工具的研究人员提供适当的学术引用，确保研究成果的可追溯性和公正性。