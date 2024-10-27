NUM_DGX_A100=0
NUM_DGX_H100=1
SCHEDULER=token_jsq
START_STATE=baseline
TRACE=test_trace

python run.py \
    cluster=half_half \
    cluster.servers.0.count=$NUM_DGX_A100 \
    cluster.servers.1.count=$NUM_DGX_H100 \
    applications.0.scheduler=$SCHEDULER \
    start_state=$START_STATE \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0

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