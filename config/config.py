# # # config/config.py
# # import os
# # from dataclasses import dataclass
# # from typing import Optional
# # import torch
# #
# # # 获取当前文件路径
# # current_file_dir = os.path.dirname(os.path.abspath(__file__))
# # # 获取获取 上一级目录（父目录）
# # parent_dir = os.path.dirname(current_file_dir)
# # # print("当前文件路径",current_file_dir)
# # # print("父目录",parent_dir)
# # @dataclass           # 装饰器
# # class ModelConfig:
# #     """模型配置 - 改为 BGE 文本嵌入"""
# #     # 移除 clip_model_name（不再适用）
# #     embedding_model: str = "BAAI/bge-small-zh-v1.5"  # 或 bge-small-zh-v1.5
# #     embedding_dim: int = 512  # bge-base 是 768；bge-small 是 512！注意匹配
# #     device: str = "cuda" if torch.cuda.is_available() else "cpu"
# #     project_root = os.path.dirname(os.path.dirname(__file__))  # 假设当前文件在 core/ 下
# #     local_model_path: str = os.path.join(project_root, "models", "BAAI_bge-small-zh-v1.5")
# #     # local_model_path: str = os.path.join(parent_dir, "models", "BAAI_bge-small-zh-v1.5")
# #     # local_model_path: str = "D:/Code/data_process/models/chinese-clip"
# #
# # @dataclass
# # class MilvusConfig:
# #     """Milvus 配置"""
# #     # 👇 关键修改：从环境变量读取 host/port，但默认值仍是 localhost（适合独立开发）
# #     #    这样既支持 docker-compose 设置，也支持本地直接运行
# #     host: str = os.getenv("MILVUS_HOST", "192.168.223.10")
# #     port: int = int(os.getenv("MILVUS_PORT", "19530"))
# #     collection_name: str = "RAG_data1"
# #
# #     # 索引配置
# #     index_type: str = "IVF_FLAT"    # 索引类型
# #     metric_type: str = "IP"         # 内积，适合CLIP归一化向量
# #     nlist: int = 1024               # nlist：聚类中心数量（Number of Clusters）
# #     nprobe: int = 10                # nprobe：查询时扫描的簇数量
# #
# #     # 分片配置
# #     # shards_num: int = 2 # 可以在MIlvus集群时用，这里是（单机）模式，不是集群（Cluster）模式
# #     """
# #     分片
# #     分片（Sharding） 是把同一个集合（collection）中的数据（比如你提到的“数据A”——即所有景点向量）水平切分成多份。
# #     这些分片可以分布到不同的物理节点（在 Milvus 集群模式下）。
# #     查询时，Milvus 会并行地在所有分片上执行搜索，然后合并结果。
# #     所以，分片的主要目的之一就是提升检索（和写入）的吞吐与速度，尤其是在大规模数据或高并发场景下
# #     """
# #
# # @dataclass
# # class DataConfig:
# #     """数据配置"""
# #     # data_root: str = "D:/RAG_img/data"
# #     data_root: str = os.path.join(parent_dir,"data")
# #     # print(data_root)
# #
# #
# # # 全局配置实例
# # MODEL_CONFIG = ModelConfig()
# # MILVUS_CONFIG = MilvusConfig()
# # DATA_CONFIG = DataConfig()
# #
#
# # config/config.py
# import os
# from dataclasses import dataclass
# from typing import Optional
# import torch
#
# # 获取当前文件路径
# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取获取 上一级目录（父目录）
# parent_dir = os.path.dirname(current_file_dir)
# # print("当前文件路径",current_file_dir)
# # print("父目录",parent_dir)
# @dataclass           # 装饰器
# class ModelConfig:
#     """模型配置 - 改为 BGE 文本嵌入"""
#     # 移除 clip_model_name（不再适用）
#     embedding_model: str = "/home/junh/yolo_RAG/yolo_rag2/models/BAAI/bge-small-zh-v1.5"  # 或 bge-small-zh-v1.5
#     embedding_dim: int = 512  # bge-base 是 768；bge-small 是 512！注意匹配
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     project_root = os.path.dirname(os.path.dirname(__file__))  # 假设当前文件在 core/ 下
#     # local_model_path: str = os.path.join(project_root, "models", "BAAI_bge-small-zh-v1.5")
#     local_model_path: str = "/home/junh/yolo_RAG/yolo_rag2/models/BAAI_bge-small-zh-v1.5"
#     # local_model_path: str = os.path.join(parent_dir, "models", "BAAI_bge-small-zh-v1.5")
#     # local_model_path: str = "D:/Code/data_process/models/chinese-clip"
#
# @dataclass
# class MilvusConfig:
#     """Milvus 配置"""
#     # 👇 关键修改：从环境变量读取 host/port，但默认值仍是 localhost（适合独立开发）
#     #    这样既支持 docker-compose 设置，也支持本地直接运行
#     host: str = os.getenv("MILVUS_HOST", "192.168.223.10")
#     port: int = int(os.getenv("MILVUS_PORT", "19530"))
#     collection_name: str = "RAG_data1"
#
#     # 索引配置
#     index_type: str = "IVF_FLAT"    # 索引类型
#     metric_type: str = "IP"         # 内积，适合CLIP归一化向量
#     nlist: int = 1024               # nlist：聚类中心数量（Number of Clusters）
#     nprobe: int = 10                # nprobe：查询时扫描的簇数量
#
#     # 分片配置
#     # shards_num: int = 2 # 可以在MIlvus集群时用，这里是（单机）模式，不是集群（Cluster）模式
#     """
#     分片
#     分片（Sharding） 是把同一个集合（collection）中的数据（比如你提到的“数据A”——即所有景点向量）水平切分成多份。
#     这些分片可以分布到不同的物理节点（在 Milvus 集群模式下）。
#     查询时，Milvus 会并行地在所有分片上执行搜索，然后合并结果。
#     所以，分片的主要目的之一就是提升检索（和写入）的吞吐与速度，尤其是在大规模数据或高并发场景下
#     """
#
# @dataclass
# class DataConfig:
#     """数据配置"""
#     # data_root: str = "D:/RAG_img/data"
#     data_root: str = os.path.join(parent_dir,"data")
#     # print(data_root)
#
#
# # 全局配置实例
# MODEL_CONFIG = ModelConfig()
# MILVUS_CONFIG = MilvusConfig()
# DATA_CONFIG = DataConfig()
#
#
# config/config.py
import os
from dataclasses import dataclass
from typing import Optional
import torch

# 获取当前文件路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 获取获取 上一级目录（父目录）
parent_dir = os.path.dirname(current_file_dir)
# print("当前文件路径",current_file_dir)
# print("父目录",parent_dir)
@dataclass           # 装饰器
class ModelConfig:
    """模型配置 - 改为 BGE 文本嵌入"""
    # 移除 clip_model_name（不再适用）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"  # 或 bge-small-zh-v1.5
    embedding_dim: int = 512  # bge-base 是 768；bge-small 是 512！注意匹配
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(__file__))  # 假设当前文件在 core/ 下
    local_model_path: str = os.path.join(project_root, "models", "BAAI_bge-small-zh-v1.5")
    # local_model_path: str = os.path.join(parent_dir, "models", "BAAI_bge-small-zh-v1.5")
    # local_model_path: str = "D:/Code/data_process/models/chinese-clip"

@dataclass
class MilvusConfig:
    """Milvus 配置"""
    # 👇 关键修改：从环境变量读取 host/port，但默认值仍是 localhost（适合独立开发）
    #    这样既支持 docker-compose 设置，也支持本地直接运行
    # host: str = os.getenv("MILVUS_HOST", "192.168.255.6")
    host: str = os.getenv("MILVUS_HOST", "192.168.110.217")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name: str = "RAG_data2"

    # 索引配置
    index_type: str = "IVF_FLAT"    # 索引类型
    metric_type: str = "IP"         # 内积，适合CLIP归一化向量
    nlist: int = 1024               # nlist：聚类中心数量（Number of Clusters）
    nprobe: int = 10                # nprobe：查询时扫描的簇数量

    # 分片配置
    # shards_num: int = 2 # 可以在MIlvus集群时用，这里是（单机）模式，不是集群（Cluster）模式
    """
    分片
    分片（Sharding） 是把同一个集合（collection）中的数据（比如你提到的“数据A”——即所有景点向量）水平切分成多份。
    这些分片可以分布到不同的物理节点（在 Milvus 集群模式下）。
    查询时，Milvus 会并行地在所有分片上执行搜索，然后合并结果。
    所以，分片的主要目的之一就是提升检索（和写入）的吞吐与速度，尤其是在大规模数据或高并发场景下
    """

@dataclass
class DataConfig:
    """数据配置"""
    # data_root: str = "D:/RAG_img/data"
    data_root: str = os.path.join(parent_dir,"data")
    # print(data_root)


# 全局配置实例
MODEL_CONFIG = ModelConfig()
MILVUS_CONFIG = MilvusConfig()
DATA_CONFIG = DataConfig()

