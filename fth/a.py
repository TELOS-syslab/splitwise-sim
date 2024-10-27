# from hydra.main import Hydra
import hydra
from omegaconf import DictConfig



# 定义主函数，并用 @hydra.main 装饰器
@hydra.main(config_name=None)
def main(cfg: DictConfig):
    # 任务逻辑
    name = cfg.name if "name" in cfg else "World"
    return f"Hello, {name}!"

# 运行函数
if __name__ == "__main__":
    result = main()  # `main` 会返回 `Hello, World!` 或其他结果
    print(result)    # 输出结果