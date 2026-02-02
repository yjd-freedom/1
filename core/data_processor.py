import os
import sys
import json
import logging
import traceback
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

# ========== 添加路径设置 ==========
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config import DATA_CONFIG, MODEL_CONFIG
    from core.embedding_processor import BgeTextEmbedder
    from core.milvus_manager import MilvusDataManager
except ImportError as e:
    print(f"导入模块失败: {e}")
    raise

logger = logging.getLogger(__name__)


def check_model_exists(model_path: str) -> bool:
    """检查模型文件是否存在"""
    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        print(f"模型路径不存在: {model_path}")
        return False

    # 检查关键文件
    required_files = ["config.json"]
    for file in required_files:
        if not (model_path_obj / file).exists():
            print(f"缺少文件: {file}")
            return False

    # 检查权重文件
    has_weights = (
            (model_path_obj / "model.safetensors").exists() or
            (model_path_obj / "pytorch_model.bin").exists()
    )
    if not has_weights:
        print(f"缺少权重文件")
        return False

    return True

class TextDataProcessor:
    def __init__(self, data_root: Optional[str] = None):
        """
        初始化文本数据处理器
        仅处理文本：部件名称作为向量，txt文件作为描述
        """
        # 使用配置中的模型路径
        model_path_str = MODEL_CONFIG.local_model_path

        print("=" * 50)
        print(f"模型路径: {model_path_str}")
        print(f"检查模型文件...")

        # 检查模型是否存在
        if not check_model_exists(model_path_str):
            print(f"✗ 模型文件不完整或不存在")
            raise FileNotFoundError(f"模型文件不完整，请检查 {model_path_str}")
        else:
            print("✓ 模型文件检查通过")

        # 初始化文本编码器
        print("初始化文本编码器...")
        self.text_embedder = BgeTextEmbedder()  # 由内部自动读取 config 中的 local_model_path

        # 初始化 Milvus 管理器
        print("初始化 Milvus 管理器...")
        self.milvus_manager = MilvusDataManager()

        # 设置数据根目录
        self.data_root = data_root if data_root is not None else DATA_CONFIG.data_root
        print(f"数据根目录: {self.data_root}")
        print("=" * 50)

    def _create_processed_marker(self, folder_path: str) -> bool:
        """
        在处理完成的文件夹中创建隐藏标记文件

        Args:
            folder_path: 文件夹路径

        Returns:
            是否成功创建标记文件
        """
        try:
            marker_file = os.path.join(folder_path, ".processed")
            print(f"  正在创建标记文件: {marker_file}")

            with open(marker_file, 'w', encoding='utf-8') as f:
                f.write(f"Processed at: {self._get_current_time()}\n")
                f.write(f"Component processed successfully\n")

            # 如果是Windows系统，设置为隐藏文件
            if os.name == 'nt':  # Windows
                try:
                    import ctypes
                    FILE_ATTRIBUTE_HIDDEN = 0x02
                    ctypes.windll.kernel32.SetFileAttributesW(marker_file, FILE_ATTRIBUTE_HIDDEN)
                    print(f"  ✓ 已设置隐藏属性")
                except Exception as e:
                    print(f"  ⚠️  设置隐藏属性失败: {e}")

            # 验证文件是否创建成功
            if os.path.exists(marker_file):
                print(f"  ✓ 标记文件创建成功")
                return True
            else:
                print(f"  ✗ 标记文件创建失败")
                return False

        except Exception as e:
            print(f"  ✗ 创建处理标记文件失败: {e}")
            traceback.print_exc()
            return False

    def _check_processed_marker(self, folder_path: str) -> bool:
        """
        检查文件夹是否有处理标记文件

        Args:
            folder_path: 文件夹路径

        Returns:
            是否存在处理标记文件
        """
        marker_file = os.path.join(folder_path, ".processed")
        exists = os.path.exists(marker_file)
        if exists:
            print(f"  标记文件存在: {marker_file}")
        else:
            print(f"  标记文件不存在: {marker_file}")
        return exists

    def _remove_processed_marker(self, folder_path: str) -> bool:
        """
        删除处理标记文件（用于强制重新处理的情况）

        Args:
            folder_path: 文件夹路径

        Returns:
            是否成功删除标记文件
        """
        try:
            marker_file = os.path.join(folder_path, ".processed")
            print(f"  尝试删除标记文件: {marker_file}")

            if os.path.exists(marker_file):
                print(f"  标记文件存在，准备删除...")

                # 如果是Windows系统，可能需要先去除隐藏属性
                if os.name == 'nt':  # Windows
                    try:
                        import ctypes
                        FILE_ATTRIBUTE_NORMAL = 0x80
                        # 去除隐藏属性
                        ctypes.windll.kernel32.SetFileAttributesW(marker_file, FILE_ATTRIBUTE_NORMAL)
                        print(f"  ✓ 已去除隐藏属性")
                    except Exception as e:
                        print(f"  ⚠️  去除隐藏属性失败: {e}")

                # 尝试删除文件
                try:
                    os.remove(marker_file)
                    print(f"  ✓ 文件删除命令已执行")
                except Exception as e:
                    print(f"  ✗ 文件删除失败: {e}")
                    return False

                # 验证文件是否真的被删除
                if os.path.exists(marker_file):
                    print(f"  ✗ 文件删除后仍然存在！")
                    print(f"  ℹ️  尝试强制删除...")

                    # 尝试其他删除方法
                    try:
                        import stat
                        os.chmod(marker_file, stat.S_IWRITE)  # 确保有写权限
                        os.remove(marker_file)
                    except Exception as e2:
                        print(f"  ✗ 强制删除也失败: {e2}")
                        return False

                # 再次验证
                if not os.path.exists(marker_file):
                    print(f"  ✓ 标记文件删除成功")
                    return True
                else:
                    print(f"  ✗ 标记文件仍然存在")
                    return False
            else:
                print(f"  ℹ️  标记文件不存在，无需删除")
                return True  # 文件不存在也算删除成功

        except Exception as e:
            print(f"  ✗ 删除处理标记文件失败: {e}")
            traceback.print_exc()
            return False

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _read_text_file(self, file_path: str) -> str:
        """读取文本文件，支持 utf-8 / gbk / gb2312"""
        encodings = ['utf-8', 'gbk', 'gb2312']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if content:
                        print(f"  使用编码 {encoding} 成功读取文件")
                        return content
            except Exception:
                continue
        print(f"  无法读取文件: {file_path}")
        return ""

    def process_single_component(self, component_path: str, component_name: str,
                                 skip_if_processed: bool = True) -> Dict:
        """
        处理单个部件（仅文本）

        Args:
            component_path: 部件文件夹路径
            component_name: 部件名称，如 "比亚迪汉中控屏"
            skip_if_processed: 如果已处理则跳过，默认为True

        Returns:
            处理结果字典
        """
        print(f"\n{'=' * 30}")
        print(f"处理部件: {component_name}")
        print(f"路径: {component_path}")
        print(f"skip_if_processed 参数: {skip_if_processed}")

        results = {
            "component_name": component_name,
            "status": "success",
            "errors": [],
            "skipped": False,
            "skip_reason": ""
        }

        # 检查是否已处理（根据标记文件）
        has_marker = self._check_processed_marker(component_path)
        print(f"  标记文件检查结果: {has_marker}")

        if skip_if_processed and has_marker:
            results["status"] = "skipped"
            results["skipped"] = True
            results["skip_reason"] = "已处理标记文件存在"
            print(f"  ⏭️  跳过已处理的部件（skip_if_processed={skip_if_processed}, 标记文件存在={has_marker}）")
            return results
        else:
            print(f"  → 继续处理（skip_if_processed={skip_if_processed}, 标记文件存在={has_marker}）")

        # 查找 txt 文件
        txt_files = [f for f in os.listdir(component_path) if f.endswith('.txt')]
        print(f"  找到 {len(txt_files)} 个 txt 文件")

        if not txt_files:
            results["errors"].append("未找到 .txt 文件")
            results["status"] = "failed"
            print(f"  ✗ 未找到 .txt 文件")
            return results

        txt_file = os.path.join(component_path, txt_files[0])
        print(f"  读取文件: {txt_file}")
        description = self._read_text_file(txt_file)
        if not description:
            results["errors"].append("文本文件为空")
            results["status"] = "failed"
            print(f"  ✗ 文本文件为空")
            return results

        print(f"  ✓ 读取文本文件，长度: {len(description)} 字符")

        try:
            # 将部件名称编码为向量
            print(f"  编码部件名称: {component_name}")
            vector = self.text_embedder.encode_text(component_name)
            print(f"  ✓ 向量编码完成，维度: {len(vector)}")

            # 插入到 Milvus
            print(f"  插入到 Milvus...")
            result = self.milvus_manager.insert_component(
                component_id=component_name,  # 使用部件名称作为ID
                vector=vector,  # 部件名称的向量
                description=description  # txt文件内容作为描述
            )

            if result:
                print(f"  ✓ 成功插入部件: {component_name}")
                # 创建处理标记文件
                marker_result = self._create_processed_marker(component_path)
                if not marker_result:
                    print(f"  ⚠️  标记文件创建失败，但数据已处理")
            else:
                raise Exception("Milvus 插入失败")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            traceback.print_exc()
            results["errors"].append(str(e))
            results["status"] = "failed"

        return results

    def process_all_components(self, force_reprocess: bool = False) -> Dict:
        """
        处理所有部件

        Args:
            force_reprocess: 是否强制重新处理所有部件
                - True: 强制重新处理，删除标记文件并重新处理所有部件（默认）
                - False: 不强制处理，已标记的部件将被跳过

        Returns:
            统计信息字典
        """
        print("\n" + "=" * 50)
        print("开始处理所有部件（文本向量化）")
        print(f"force_reprocess 参数值: {force_reprocess}")

        # 根据force_reprocess参数显示不同的提示信息
        if force_reprocess:
            print("⚠️  强制重新处理模式已启用，将删除已有标记文件并重新处理所有部件")
        else:
            print("正常处理模式，已标记的部件将被跳过")
        print("=" * 50)

        stats = {
            "total_components": 0,
            "processed_components": 0,
            "skipped_components": 0,
            "failed_components": [],
            "skipped_details": [],
            "components_details": []
        }

        data_root = Path(self.data_root)
        if not data_root.exists():
            raise ValueError(f"data_root 不存在: {self.data_root}")

        # 查找 "部件介绍" 文件夹
        parts_folder = data_root
        if not parts_folder.exists() or not parts_folder.is_dir():
            raise FileNotFoundError(f"未找到 '部件介绍' 文件夹: {parts_folder}")

        # 获取所有部件文件夹
        components = [item for item in parts_folder.iterdir() if item.is_dir()]
        print(f"找到 {len(components)} 个部件文件夹")

        # 统计初始标记文件数量
        initial_markers = 0
        for item in components:
            if self._check_processed_marker(str(item)):
                initial_markers += 1
        print(f"初始已处理标记文件数量: {initial_markers}/{len(components)}")

        # 遍历处理每个部件
        for idx, item in enumerate(components, 1):
            component_name = item.name  # 文件夹名称就是部件名称
            component_path = str(item)

            try:
                print(f"\n{'=' * 40}")
                print(f"[{idx}/{len(components)}] 处理部件: {component_name}")
                print(f"文件夹路径: {component_path}")

                # 处理前检查标记文件状态
                marker_file_before = os.path.join(component_path, ".processed")
                marker_exists_before = os.path.exists(marker_file_before)
                print(f"处理前标记文件状态: {'存在' if marker_exists_before else '不存在'}")

                # 重要：根据force_reprocess参数决定是否删除标记文件
                if force_reprocess:
                    # 强制重新处理模式：删除标记文件
                    print(f"执行强制重新处理逻辑...")
                    delete_result = self._remove_processed_marker(component_path)
                    print(f"删除标记文件结果: {delete_result}")

                    # 验证删除是否成功
                    marker_file_after = os.path.join(component_path, ".processed")
                    marker_exists_after = os.path.exists(marker_file_after)
                    print(f"删除后标记文件状态: {'存在' if marker_exists_after else '不存在'}")

                    # 如果标记文件仍然存在且是强制模式，尝试其他删除方法
                    if marker_exists_after and force_reprocess:
                        print(f"  ⚠️  警告：强制模式下标记文件仍然存在！")
                        print(f"  ℹ️  尝试使用其他方法删除...")
                        try:
                            if os.path.exists(marker_file_after):
                                import stat
                                os.chmod(marker_file_after, 0o777)  # 设置完全权限
                                os.remove(marker_file_after)
                                print(f"  ✓ 直接删除成功")
                        except Exception as e:
                            print(f"  ✗ 直接删除失败: {e}")
                else:
                    # 非强制模式：保留标记文件
                    print(f"非强制模式，保留标记文件")

                # 重要：根据force_reprocess参数决定是否跳过已处理的部件
                # force_reprocess=True 时，skip_if_processed=False（不跳过）
                # force_reprocess=False 时，skip_if_processed=True（跳过）
                skip_if_processed = not force_reprocess
                print(f"skip_if_processed 计算值: not {force_reprocess} = {skip_if_processed}")

                # 在调用前再次检查标记文件状态
                final_check = os.path.join(component_path, ".processed")
                final_exists = os.path.exists(final_check)
                print(f"调用 process_single_component 前标记文件状态: {'存在' if final_exists else '不存在'}")

                # 处理单个部件
                result = self.process_single_component(component_path, component_name, skip_if_processed)

                # 根据处理结果更新统计信息
                if result["status"] == "success":
                    stats["processed_components"] += 1
                    stats["components_details"].append({
                        "name": component_name,
                        "status": "success",
                        "timestamp": self._get_current_time()
                    })
                    print(f"  ✅ 处理成功")
                elif result["status"] == "skipped":
                    stats["skipped_components"] += 1
                    stats["skipped_details"].append({
                        "name": component_name,
                        "reason": result["skip_reason"]
                    })
                    stats["components_details"].append({
                        "name": component_name,
                        "status": "skipped",
                        "reason": result["skip_reason"]
                    })
                    print(f"  ⏭️ 跳过处理")
                else:  # failed
                    stats["failed_components"].append({
                        "name": component_name,
                        "errors": result["errors"]
                    })
                    stats["components_details"].append({
                        "name": component_name,
                        "status": "failed",
                        "errors": result["errors"]
                    })
                    print(f"  ❌ 处理失败")

                stats["total_components"] += 1

            except Exception as e:
                print(f"  ❌ 处理部件失败: {component_name}, 错误: {e}")
                traceback.print_exc()
                stats["failed_components"].append({
                    "name": component_name,
                    "errors": [str(e)]
                })
                stats["components_details"].append({
                    "name": component_name,
                    "status": "error",
                    "errors": [str(e)]
                })

        # 输出统计信息
        print("\n" + "=" * 50)
        print("处理完成!")
        print(f"总部件数: {stats['total_components']}")
        print(f"成功处理: {stats['processed_components']}")
        print(f"跳过处理: {stats['skipped_components']}")
        print(f"失败部件数: {len(stats['failed_components'])}")

        if stats['skipped_components'] > 0:
            print("\n跳过详情:")
            for skipped in stats['skipped_details']:
                print(f"  {skipped['name']}: {skipped['reason']}")

        if stats['failed_components']:
            print("\n失败详情:")
            for failed in stats['failed_components']:
                print(f"  {failed['name']}: {failed['errors']}")

        # 处理完成后再次统计标记文件数量
        final_markers = 0
        for item in components:
            if self._check_processed_marker(str(item)):
                final_markers += 1
        print(f"\n处理完成后标记文件数量: {final_markers}/{len(components)}")

        return stats

    def force_process_all_components(self) -> Dict:
        """
        强制重新处理所有部件（忽略标记文件）
        这是备用的强制处理方法，直接调用process_all_components并传递force_reprocess=True
        """
        print("\n" + "=" * 50)
        print("使用 force_process_all_components 方法强制重新处理")
        print("=" * 50)
        return self.process_all_components(force_reprocess=True)

    def debug_marker_files(self) -> None:
        """
        调试标记文件状态
        """
        print("\n" + "=" * 50)
        print("调试标记文件状态")
        print("=" * 50)

        data_root = Path(self.data_root)
        parts_folder = data_root

        if not parts_folder.exists():
            print(f"部件介绍文件夹不存在: {parts_folder}")
            return

        components = [item for item in parts_folder.iterdir() if item.is_dir()]
        print(f"找到 {len(components)} 个部件文件夹")

        for item in components:
            component_name = item.name
            component_path = str(item)
            marker_file = os.path.join(component_path, ".processed")

            print(f"\n部件: {component_name}")
            print(f"标记文件路径: {marker_file}")
            print(f"文件是否存在: {os.path.exists(marker_file)}")

            if os.path.exists(marker_file):
                try:
                    # 获取文件信息
                    import stat
                    file_stat = os.stat(marker_file)
                    print(f"文件大小: {file_stat.st_size} 字节")
                    print(f"文件权限: {oct(file_stat.st_mode)[-3:]}")
                    print(
                        f"是否是隐藏文件: {bool(file_stat.st_file_attributes & 2) if hasattr(file_stat, 'st_file_attributes') else 'N/A'}")

                    # 尝试读取内容
                    with open(marker_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"文件内容: {content.strip()}")
                except Exception as e:
                    print(f"读取文件信息失败: {e}")


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="文本数据处理器")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制重新处理所有部件，忽略已有的处理标记（默认已经是强制处理）")
    parser.add_argument("--skip", "-s", action="store_true",
                        help="跳过已处理的部件（覆盖默认的强制处理）")
    parser.add_argument("--data-root", type=str,
                        help="指定数据根目录，覆盖配置文件中的设置")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细调试信息")
    parser.add_argument("--debug-markers", action="store_true",
                        help="调试标记文件状态")

    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        processor = TextDataProcessor(data_root=args.data_root)

        if args.debug_markers:
            # 调试标记文件
            processor.debug_marker_files()
            return 0

        # 重要：根据命令行参数决定处理模式
        # 1. 如果指定了--skip，则跳过已处理的（force_reprocess=False）
        # 2. 如果指定了--force或不指定任何参数，则强制处理（force_reprocess=True）
        if args.skip:
            print("使用跳过模式（--skip 参数生效，跳过已处理的部件）")
            force_param = False
        else:
            # 默认情况（包括--force或不指定参数）都使用强制处理
            print("使用强制处理模式（默认或--force参数）")
            force_param = True

        print(f"传递给 process_all_components 的 force_reprocess 参数: {force_param}")

        # 调用主处理方法
        stats = processor.process_all_components(force_reprocess=force_param)

        print("\n" + "=" * 50)
        print("最终统计结果:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n程序执行失败: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())