import os
import sys
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility
)
import logging
from typing import List

# ========== 添加路径设置 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置
try:
    from config.config import MILVUS_CONFIG, MODEL_CONFIG
except ImportError as e:
    print(f"导入配置失败: {e}")
    raise

logger = logging.getLogger(__name__)

class MilvusDataManager:
    """Milvus 数据管理器 - 存储部件名称向量和描述"""

    def __init__(self):
        self.host = MILVUS_CONFIG.host
        self.port = MILVUS_CONFIG.port
        self.collection_name = MILVUS_CONFIG.collection_name
        self.index_type = MILVUS_CONFIG.index_type
        self.metric_type = MILVUS_CONFIG.metric_type
        self.nlist = MILVUS_CONFIG.nlist
        self.nprobe = MILVUS_CONFIG.nprobe

        logger.info(f"初始化 MilvusDataManager:")
        logger.info(f"  主机: {self.host}:{self.port}")
        logger.info(f"  集合: {self.collection_name}")

        self._connect()

    def _connect(self):
        """连接 Milvus"""
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"已连接到 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    def create_collection(self):
        """创建用于部件检索的集合"""
        if utility.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在，跳过创建")
            return True

        try:
            fields = [
                FieldSchema(
                    name="component_id",  # 部件唯一标识/名称
                    dtype=DataType.VARCHAR,
                    max_length=128,
                    is_primary=True
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=MODEL_CONFIG.embedding_dim
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                )
                # 移除了 component_name 字段
            ]

            schema = CollectionSchema(fields=fields, description="部件名称向量 + 文本描述")
            collection = Collection(name=self.collection_name, schema=schema)

            # 创建向量索引
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": {"nlist": self.nlist}
                }
            )
            logger.info(f"集合 {self.collection_name} 创建成功")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def insert_component(self, component_id: str, vector: List[float], description: str):
        """
        插入一个部件的数据
        Args:
            component_id (str): 部件唯一标识/名称，如 "比亚迪汉中控屏"
            vector (List[float]): 512 维归一化向量
            description (str): 对应的中文文本描述
        """
        try:
            if not utility.has_collection(self.collection_name):
                self.create_collection()

            collection = Collection(self.collection_name)
            data = [
                [component_id],      # component_id
                [vector],           # vector
                [description]       # description
                # 移除了 component_name 数据
            ]

            mr = collection.upsert(data)
            collection.flush()
            logger.info(f"插入部件: {component_id}, 描述长度: {len(description)}")
            return mr

        except Exception as e:
            logger.error(f"插入部件失败 (component_id={component_id}): {e}")
            return None

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect(alias="default")
            logger.info("已断开 Milvus 连接")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")