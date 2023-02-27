import argparse
import torch
import yaml
from yacs.config import CfgNode

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="path to load saved checkpoint."
    )
    return parser

def main():
    # 设置调试时返回更多信息
    torch.autograd.set_detect_anomaly(True)

    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件
    cfg = None
    with open(args.config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # 加载数据集
    



if __name__ == "__main__":
    main()