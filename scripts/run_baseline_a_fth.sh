python run.py \                                     # 运行 Python 脚本
     applications.0.scheduler=token_jsq \          # 设置调度器为 token_jsq
     cluster=half_half \                            # 设置集群类型为 half_half
     cluster.servers.0.count=70 \                   # 设置第一个服务器的数量为 70
     cluster.servers.1.count=0 \                    # 设置第二个服务器的数量为 0
     start_state=baseline \                          # 设置初始状态为 baseline
     performance_model=db \                          # 设置性能模型为数据库
     trace.filename=rr_conv_80 \                    # 设置跟踪文件名为 rr_conv_80
     seed=0                                         # 设置随机种子为 0
     #+experiment=traces_light \                    # 选择实验 traces_light（被注释掉，暂时不使用）
