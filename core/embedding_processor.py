# # core/embedding_processor.py
# import os
# import sys
# import torch
# from transformers import AutoTokenizer, AutoModel
# from typing import List
# import logging
#
# # ========== 路径设置 ==========
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# from config.config import MODEL_CONFIG
#
# logger = logging.getLogger(__name__)
#
# class BgeTextEmbedder:
#     def __init__(self, model_path=None):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_path = model_path or MODEL_CONFIG.local_model_path  # 自动读取配置
#
#         print(f"加载 BGE 模型: {model_path}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModel.from_pretrained(model_path).to(self.device)
#         self.model.eval()
#
#     def encode_text(self, text: str) -> list:
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(self.device)
#
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             embeddings = outputs.last_hidden_state[:, 0]  # [CLS]
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
#
#         return embeddings.cpu().numpy().flatten().tolist()

# core/embedding_processor.py
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
import logging

# ========== 路径设置 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class BgeTextEmbedder:
    def __init__(self, model_path=None, verbose=False):
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or MODEL_CONFIG.local_model_path  # 自动读取配置

        if self.verbose:
            print(f"加载 BGE 模型: {self.model_path}")
            print(f"使用设备: {self.device}")

        try:
            # 先加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # 方法1：尝试直接加载到设备
            try:
                self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            except NotImplementedError as e:
                if "meta tensor" in str(e):
                    print("检测到meta tensor，使用方法2加载...")
                    # 方法2：使用 to_empty() 处理meta tensor
                    self.model = AutoModel.from_pretrained(self.model_path)
                    if any(param.is_meta for param in self.model.parameters()):
                        print("使用to_empty()处理meta tensor...")
                        self.model.to_empty(device=self.device)
                        # 重新从文件加载权重
                        from safetensors.torch import load_file
                        import glob

                        # 查找权重文件
                        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
                        bin_files = glob.glob(os.path.join(self.model_path, "*.bin"))

                        if safetensors_files:
                            state_dict = load_file(safetensors_files[0])
                        elif bin_files:
                            state_dict = torch.load(bin_files[0], map_location="cpu")
                        else:
                            raise FileNotFoundError("未找到权重文件")

                        self.model.load_state_dict(state_dict, strict=True)
                    else:
                        self.model.to(self.device)
                else:
                    raise e

            self.model.eval()
            print(f"✓ 模型加载成功")

        except Exception as e:
            print(f"模型加载失败: {e}")
            # 方法3：尝试使用CPU加载
            try:
                print("尝试使用CPU加载...")
                self.device = torch.device("cpu")
                self.model = AutoModel.from_pretrained(self.model_path)
                self.model.eval()
                print(f"✓ 模型在CPU上加载成功")
            except Exception as e2:
                print(f"CPU加载也失败: {e2}")
                raise

    def encode_text(self, text: str) -> list:
        """编码文本为向量"""
        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # [CLS]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            return embeddings.cpu().numpy().flatten().tolist()

        except Exception as e:
            print(f"文本编码失败: {e}")
            raise