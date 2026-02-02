"""
智能对话模块 - 专业汽车直播解说版（支持部件多次介绍）
优化：基于部件介绍次数的差异化解说 + 全局历史记录 + 自然衔接
"""

import asyncio
import time
import logging
import re
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("smart_dialogue")


class SmartDialogueClient:
    """
    专业汽车直播解说客户端
    支持基于部件介绍次数的差异化解说
    """

    def __init__(self, llm_client):
        """
        初始化专业解说客户端

        Args:
            llm_client: 已有的LLM客户端实例
        """
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, Any]] = []  # 全局对话历史记录
        self.max_history_length = 15
        self.total_parts_introduced = 0

        # 🆕 新增：部件介绍次数计数器
        self.part_introduction_counts = {}  # 格式: {"整车": 2, "车轮": 1, "前灯": 1}

        self.conversation_templates = self._init_conversation_templates()
        self.functional_connections = self._init_functional_connections()

        logger.info("✅ 专业汽车直播解说客户端初始化完成（支持部件多次介绍）")
        logger.info(f"📊 最大历史记录长度: {self.max_history_length}")
        logger.info("🎯 核心特性：部件介绍次数统计 + 差异化解说 + 全局历史记录")

    def _init_conversation_templates(self) -> Dict[str, Dict[str, str]]:
        """
        初始化专业直播解说模板
        🆕 修改：为不同介绍次数准备不同的模板

        Returns:
            语言代码到对话模板的映射字典
        """
        # 🆕 新增：根据介绍次数选择不同模板
        return {
            "zh-CN": {
                "first_introduction": """【专业汽车直播解说员 - 首次介绍】

你是一名经验丰富的汽车直播解说员，正在为观众进行一场连续的专业车辆部件解说。

【重要禁止规则 - 必须严格遵守】
1. 绝对不要预测或提及下一个部件是什么
2. 绝对不要使用"下个部件，咱们聊聊..."这样的预测性语句
3. 绝对不要使用"下一张图片，我们继续..."这样的语句
4. 绝对不要使用"这就是本次解说的内容，感谢您的关注！"这样的结束语
5. 直播还在持续进行中，不要使用任何类似结束直播的语句

【直播历史回顾（按介绍顺序）】
{history_context}

【当前要解说的部件 - 首次介绍】
部件名称：{current_part_name}
部件基本信息：{current_rag_result}

【本次解说任务】
基于整个直播历史，用专业直播的风格首次介绍当前部件。

【解说规则 - 必须遵守】

1. 开场方式：
   - 如果是今天解说的第1个部件：使用"首先，让我们从...开始今天的解说！"开头
   - 如果是第2个及以上部件：必须使用"刚才我们介绍了...，基于...的设计/功能，现在我们来看..."的句式

2. 历史关联 - 重要！
   - 必须引用前面至少1个部件的信息进行扩展介绍
   - 建立功能、设计或体验上的逻辑关联

3. 直播风格要求：
   - 口语化、亲切、有互动感，像和朋友聊天
   - 使用设问句与观众互动
   - 长度：180-250字
   - 以互动问句或中性总结结尾

4. 内容要点：
   - 重点介绍该部件的基本特征和核心功能
   - 建立与历史部件的联系
   - 不预测未来的部件

现在，请基于直播历史，开始你的专业解说！""",

                "repeat_introduction": """【专业汽车直播解说员 - 再次介绍】

你是一名经验丰富的汽车直播解说员，正在为观众进行一场连续的专业车辆部件解说。

【重要禁止规则 - 必须严格遵守】
1. 绝对不要预测或提及下一个部件是什么
2. 绝对不要使用"下个部件，咱们聊聊..."这样的预测性语句
3. 绝对不要使用"下一张图片，我们继续..."这样的语句
4. 绝对不要使用"这就是本次解说的内容，感谢您的关注！"这样的结束语
5. 直播还在持续进行中，不要使用任何类似结束直播的语句

【直播历史回顾（按介绍顺序）】
{history_context}

【当前要解说的部件 - 再次介绍】
部件名称：{current_part_name}
部件基本信息：{current_rag_result}
介绍状态：这是第{introduction_count}次介绍该部件

【历史介绍回顾】
{previous_introductions}

【本次解说任务】
基于整个直播历史，用专业直播的风格再次介绍当前部件。这是第{introduction_count}次介绍该部件，需要从不同角度或补充信息进行解说。

【解说规则 - 必须遵守】

1. 开场方式 - 重要！
   - 绝对不能使用"首先，让我们从...开始"的开头
   - 必须使用"让我们再次聚焦..."、"基于之前我们了解的...，现在让我们从另一个角度看看..."、"我们继续深入了解..."等句式
   - 明确表明这是再次介绍，例如："让我们再次聚焦..."或"回到...这个话题"

2. 内容差异化 - 关键！
   - 不能简单重复之前介绍过的内容
   - 必须提供新的视角、补充信息或深入分析
   - 可以从不同功能点、设计细节、用户体验等方面进行补充
   - 示例：第一次介绍强调设计，第二次可以强调性能或实际体验

3. 历史关联：
   - 既要引用之前对该部件的介绍，也要关联其他部件
   - 可以对比本次介绍与之前介绍的侧重点差异

4. 直播风格要求：
   - 口语化、亲切、有互动感
   - 使用设问句与观众互动
   - 长度：180-250字
   - 以互动问句或中性总结结尾

【解说示例】
第一次介绍整车："首先，让我们从整车开始今天的解说！"
第二次介绍整车："让我们再次聚焦整车，基于之前我们了解的设计理念，现在从性能角度进一步了解..."
第三次介绍整车："回到整车这个话题，这次我们换个视角，看看它在日常使用中的实际表现..."

现在，请基于直播历史和之前的介绍，开始你的专业解说！"""
            },

            "en-US": {
                "first_introduction": """【Professional Car Livestream Commentator - First Introduction】

You are an experienced car livestream commentator.

【Livestream History Review】
{history_context}

【Current Part to Commentate - First Introduction】
Part Name: {current_part_name}
Basic Info: {current_rag_result}

【Commentary Task】
Based on the livestream history, introduce the current part for the first time.

【Commentary Rules】
- Use appropriate transitions based on part order
- Reference historical parts
- Provide interactive questions
- No predictions about future parts
- No ending phrases

Begin your commentary now!""",

                "repeat_introduction": """【Professional Car Livestream Commentator - Repeat Introduction】

You are an experienced car livestream commentator.

【Livestream History Review】
{history_context}

【Current Part to Commentate - Repeat Introduction】
Part Name: {current_part_name}
Basic Info: {current_rag_result}
Introduction Status: This is the {introduction_count} time introducing this part

【Previous Introductions Review】
{previous_introductions}

【Commentary Task】
Based on the livestream history, introduce the current part again. This is the {introduction_count} time introducing this part, so provide new perspectives or additional information.

【Commentary Rules】
- Do NOT use "First, let's begin with..." openings
- Use "Let's focus again on..." or "Building on our previous discussion of..."
- Provide different perspectives or additional information
- Do not simply repeat previous content
- Include interactive questions
- No predictions about future parts
- No ending phrases

Begin your commentary now!"""
            }
        }

    def _init_functional_connections(self) -> Dict[str, List[str]]:
        """
        初始化部件功能关联库
        """
        return {
            "方向盘": ["方向盘的控制需要通过转向系统传递到车轮", "方向盘的手感直接影响驾驶体验"],
            "车轮": ["车轮的抓地力影响方向盘的控制精度", "车轮的尺寸与车辆稳定性密切相关"],
            "刹车系统": ["刹车系统与车轮的配合确保制动效果", "刹车盘的大小影响制动距离"],
            "发动机": ["发动机的动力输出需要变速箱合理匹配", "发动机的布局影响车辆重心"],
            "变速箱": ["变速箱的换挡逻辑影响驾驶平顺性", "变速箱与发动机的匹配度影响油耗"],
            "仪表盘": ["仪表盘显示的信息与车辆各系统状态相关", "仪表盘的布局影响驾驶信息获取效率"],
            "中控屏": ["中控屏集成了车辆大部分控制功能", "中控屏的响应速度影响用户体验"],
            "座椅": ["座椅的舒适性与长途驾驶疲劳度相关", "座椅的包裹性与车辆操控性相辅相成"],
            "大灯": ["大灯的照明效果影响夜间行车安全", "大灯的设计语言与整车造型统一"],
            "尾灯": ["尾灯的设计与前大灯形成呼应", "尾灯的亮度影响后车识别度"],
            "整车": ["整车的设计语言体现在各个部件的协调统一", "整车的性能是各个部件协同工作的结果"],
            "车门": ["车门的设计与整车流线型造型相协调", "车门的开闭感受影响使用体验"],
            "车窗": ["车窗的设计影响车内采光和视野", "车窗的隔音性能影响车内静谧性"],
            "前脸": ["前脸的设计体现车辆的品牌特征", "前脸的造型影响空气动力学性能"],
            "车尾": ["车尾的设计与前脸形成视觉平衡", "车尾的造型影响后方视野和空气动力学"]
        }

    def add_to_history(self, part_name: str, part_description: str,
                       conversation_result: Optional[str] = None):
        """
        添加部件介绍到全局历史记录
        🆕 修改：更新部件介绍次数

        Args:
            part_name: 部件名称
            part_description: 部件描述
            conversation_result: 对话结果
        """
        self.total_parts_introduced += 1

        # 🆕 更新部件介绍次数
        if part_name in self.part_introduction_counts:
            self.part_introduction_counts[part_name] += 1
        else:
            self.part_introduction_counts[part_name] = 1

        # 获取当前介绍次数
        introduction_count = self.part_introduction_counts[part_name]

        # 提取关键特征
        key_features = self._extract_key_features(part_description)

        history_entry = {
            "part_number": self.total_parts_introduced,
            "part_name": part_name,
            "part_description": part_description,
            "key_features": key_features,
            "introduction_count": introduction_count,  # 🆕 记录这是第几次介绍该部件
            "conversation_result": conversation_result or part_description,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.conversation_history.append(history_entry)

        # 保持历史记录长度
        if len(self.conversation_history) > self.max_history_length:
            removed = self.conversation_history.pop(0)
            logger.debug(f"🗑️ 移除最旧历史记录（第{removed['part_number']}个）: {removed['part_name']}")

        logger.info(f"📝 添加到历史（第{self.total_parts_introduced}个）: {part_name}")
        logger.info(f"📊 当前历史记录数: {len(self.conversation_history)}")
        logger.info(f"🔢 {part_name}的介绍次数: {introduction_count}")

    def _extract_key_features(self, description: str, max_features: int = 3) -> List[str]:
        """
        从描述中提取关键特征
        """
        features = []
        feature_keywords = [
            "设计", "材质", "功能", "性能", "尺寸", "颜色", "形状",
            "科技", "智能", "安全", "舒适", "豪华", "运动", "经济",
            "响应", "精准", "稳定", "高效", "耐用", "美观", "实用"
        ]

        sentences = re.split(r'[。！？!?.]', description)
        for sentence in sentences:
            if any(keyword in sentence for keyword in feature_keywords):
                simplified = sentence.strip()
                if len(simplified) > 5 and len(simplified) < 50:
                    features.append(simplified)
                    if len(features) >= max_features:
                        break

        if not features and description:
            features.append(description[:50].strip() + "...")

        return features

    def _get_previous_introductions(self, part_name: str, limit: int = 3) -> str:
        """
        获取该部件之前的介绍记录
        🆕 新增：用于重复介绍时提供历史参考

        Args:
            part_name: 部件名称
            limit: 返回的历史记录数量限制

        Returns:
            格式化的历史介绍记录
        """
        previous_entries = []

        # 从历史记录中查找该部件的介绍
        for entry in reversed(self.conversation_history):
            if entry['part_name'] == part_name:
                intro_count = entry.get('introduction_count', 1)
                key_features = entry.get('key_features', [])
                features_text = "；".join(key_features[:2]) if key_features else "无关键特征记录"

                previous_entries.append(
                    f"第{intro_count}次介绍: {features_text}"
                )

                if len(previous_entries) >= limit:
                    break

        if not previous_entries:
            return "这是第一次介绍该部件，没有历史记录。"

        return "\n".join(previous_entries)

    def _build_history_context(self, current_part_name: str, current_part_number: int,
                               introduction_count: int) -> str:
        """
        构建丰富的历史上下文
        🆕 修改：加入部件介绍次数信息

        Args:
            current_part_name: 当前部件名称
            current_part_number: 当前部件序号
            introduction_count: 当前部件的介绍次数

        Returns:
            格式化的历史上下文字符串
        """
        # 检查是否是第一次介绍该部件
        is_first_introduction = introduction_count == 1

        if not self.conversation_history:
            return "这是第一个介绍的部件，还没有历史对话。可以直接开始解说。"

        # 构建历史回顾
        history_texts = []
        history_texts.append("【直播历史回顾 - 按介绍顺序排列】")

        # 显示最近部件的历史（最多8个）
        display_count = min(len(self.conversation_history), 8)

        for i, entry in enumerate(self.conversation_history[-display_count:], 1):
            part_num = entry['part_number']
            part_name = entry['part_name']

            # 如果是当前部件，特别标记
            if part_name == current_part_name:
                intro_count = entry.get('introduction_count', 1)
                part_name_display = f"{part_name}（已介绍过{intro_count - 1}次）"
            else:
                part_name_display = part_name

            # 获取关键特征
            features = entry.get('key_features', [])
            if features:
                features_text = "；".join(features[:2])
                feature_desc = f"，特点：{features_text}"
            else:
                feature_desc = ""

            history_texts.append(
                f"{i}. 第{part_num}个部件：{part_name_display}{feature_desc}"
            )

        # 添加当前部件信息
        history_texts.append(f"\n【当前部件信息】")
        history_texts.append(f"- 部件名称：{current_part_name}")
        history_texts.append(f"- 这是第{current_part_number}个介绍的部件")
        history_texts.append(f"- 这是第{introduction_count}次介绍该部件")

        if is_first_introduction:
            history_texts.append("- 这是第一次介绍该部件，请全面介绍其核心特征")
        else:
            history_texts.append(f"- 该部件之前已经介绍过{introduction_count - 1}次，请提供新的视角或补充信息")

        # 添加功能关联提示
        if current_part_name in self.functional_connections:
            connections = self.functional_connections[current_part_name]
            if connections:
                history_texts.append(f"\n【功能关联提示】")
                history_texts.append(f"- {current_part_name}的功能关联：{connections[0]}")

        history_texts.append(f"\n【解说提示】")
        history_texts.append(f"- 请基于整个历史进行自然衔接")
        history_texts.append(f"- 不要预测下一个部件")
        history_texts.append(f"- 不要使用结束语")

        return "\n".join(history_texts)

    def _find_historical_reference(self, current_part_name: str) -> Dict[str, Any]:
        """
        为当前部件查找最相关历史部件进行引用

        Args:
            current_part_name: 当前部件名称

        Returns:
            引用信息字典
        """
        if not self.conversation_history:
            return {"found": False}

        # 优先查找功能相关部件（但不是当前部件自身）
        for history_entry in reversed(self.conversation_history):
            hist_part_name = history_entry['part_name']

            # 跳过当前部件自身的历史记录
            if hist_part_name == current_part_name:
                continue

            # 检查是否在关联库中
            if hist_part_name in self.functional_connections:
                for connection in self.functional_connections[hist_part_name]:
                    if any(keyword in connection for keyword in ["配合", "联动", "相关", "影响", "协同"]):
                        return {
                            "found": True,
                            "part_name": hist_part_name,
                            "part_number": history_entry['part_number'],
                            "connection": connection,
                            "features": history_entry.get('key_features', [])
                        }

        # 如果没有找到功能关联，使用最近的一个非当前部件
        for history_entry in reversed(self.conversation_history):
            if history_entry['part_name'] != current_part_name:
                return {
                    "found": True,
                    "part_name": history_entry['part_name'],
                    "part_number": history_entry['part_number'],
                    "connection": f"刚才介绍的{history_entry['part_name']}",
                    "features": history_entry.get('key_features', [])
                }

        # 如果所有历史都是当前部件，返回空
        return {"found": False}

    async def generate_connected_description(self,
                                             current_part_name: str,
                                             current_rag_result: str,
                                             target_language: str = "zh-CN") -> str:
        """
        生成关联的描述（基于全局历史记录）
        🆕 修改：根据介绍次数选择不同模板

        Args:
            current_part_name: 当前部件名称
            current_rag_result: 当前部件的RAG结果
            target_language: 目标语言代码

        Returns:
            经过对话衔接处理后的最终描述
        """
        # 计算当前是第几个部件
        current_part_number = self.total_parts_introduced + 1

        # 🆕 获取当前部件的介绍次数（当前还未添加，所以次数是已介绍过的次数）
        introduction_count = self.part_introduction_counts.get(current_part_name, 0) + 1
        is_first_introduction = introduction_count == 1

        has_history = len(self.conversation_history) > 0

        logger.info(f"🎤 开始专业解说: {current_part_name}")
        logger.info(f"🔢 部件序号: 第{current_part_number}个部件")
        logger.info(f"📊 介绍次数: 第{introduction_count}次介绍该部件")
        logger.info(f"📚 历史状态: {'有' if has_history else '无'}历史记录")

        # 查找历史引用
        historical_ref = None
        if has_history:
            historical_ref = self._find_historical_reference(current_part_name)
            if historical_ref['found']:
                logger.info(f"📎 找到历史引用: {historical_ref['part_name']} (第{historical_ref['part_number']}个部件)")

        # 构建历史上下文（传入介绍次数）
        history_context = self._build_history_context(
            current_part_name, current_part_number, introduction_count
        )
        logger.debug(f"📚 历史上下文（前300字符）:\n{history_context[:300]}...")

        # 🆕 根据介绍次数选择模板类型
        if is_first_introduction:
            template_type = "first_introduction"
            logger.info(f"📝 使用模板: 首次介绍模板")
        else:
            template_type = "repeat_introduction"
            logger.info(f"📝 使用模板: 重复介绍模板（第{introduction_count}次）")

        # 获取对话模板
        language_templates = self.conversation_templates.get(
            target_language,
            self.conversation_templates["zh-CN"]
        )

        template = language_templates.get(
            template_type,
            language_templates["first_introduction"]  # 默认使用首次介绍模板
        )

        # 如果是重复介绍，获取之前的介绍记录
        previous_introductions = ""
        if not is_first_introduction:
            previous_introductions = self._get_previous_introductions(current_part_name)
            logger.debug(f"📚 之前介绍记录:\n{previous_introductions}")

        # 格式化提示词
        prompt = template.format(
            history_context=history_context,
            current_part_name=current_part_name,
            current_rag_result=current_rag_result,
            introduction_count=introduction_count,
            previous_introductions=previous_introductions if not is_first_introduction else ""
        )

        logger.debug(f"📝 解说提示词（前300字符）:\n{prompt[:300]}...")

        # 调用LLM生成专业解说
        logger.info(f"🔄 调用LLM生成专业解说")
        dialogue_start_time = time.time()

        try:
            conversation_result = await self.llm_client.generate_summary(
                context=prompt,
                target_language=target_language
            )

            dialogue_end_time = time.time()
            dialogue_time = (dialogue_end_time - dialogue_start_time) * 1000
            logger.info(f"✅ 专业解说生成完成，耗时: {dialogue_time:.2f}ms")
            logger.info(f"📝 解说结果预览: {conversation_result[:80]}...")

            # 后处理：确保符合直播解说风格
            conversation_result = self._post_process_for_professional_livestream(
                conversation_result,
                current_part_name,
                current_part_number,
                introduction_count,
                has_history,
                historical_ref
            )

        except Exception as e:
            logger.error(f"❌ 专业解说生成失败: {e}")
            # 失败时返回基于历史的对话
            conversation_result = self._create_professional_dialogue(
                current_part_name, current_rag_result, current_part_number,
                introduction_count, has_history, historical_ref
            )

        # 更新全局历史记录
        self.add_to_history(
            part_name=current_part_name,
            part_description=current_rag_result,
            conversation_result=conversation_result
        )

        return conversation_result

    def _post_process_for_professional_livestream(self, text: str, current_part_name: str,
                                                  current_part_number: int, introduction_count: int,
                                                  has_history: bool, historical_ref: Optional[Dict] = None) -> str:
        """
        后处理中文解说文本
        🆕 修改：根据介绍次数进行不同的后处理

        Args:
            text: 原始解说文本
            current_part_name: 当前部件名称
            current_part_number: 当前部件序号
            introduction_count: 当前部件的介绍次数
            has_history: 是否有历史记录
            historical_ref: 历史引用信息

        Returns:
            处理后的解说文本
        """
        if not text or len(text.strip()) < 20:
            return self._create_professional_dialogue(
                current_part_name, "", current_part_number, introduction_count, has_history, historical_ref
            )

        processed = text.strip()

        # 🆕 根据介绍次数进行不同的处理
        is_first_introduction = introduction_count == 1

        # 1. 检查开头是否恰当
        if not is_first_introduction and introduction_count > 1:
            # 重复介绍：绝对不能使用"首先"开头
            if processed.startswith("首先"):
                logger.warning(f"⚠️ 检测到不当开头（第{introduction_count}次介绍不应使用'首先'）")

                # 构建适合重复介绍的开头
                repeat_openings = [
                    f"让我们再次聚焦{current_part_name}，从另一个角度深入了解...",
                    f"基于之前对{current_part_name}的介绍，现在让我们看看它的另一个重要方面...",
                    f"我们之前已经了解了{current_part_name}的基本特点，现在让我们进一步探索...",
                    f"回到{current_part_name}这个话题，这次我们从不同的视角来看...",
                    f"继续深入{current_part_name}的细节，这次我们关注...",
                    f"让我们重新审视{current_part_name}，这次侧重...",
                    f"关于{current_part_name}，我们之前了解了一部分，现在补充更多信息..."
                ]

                opening = random.choice(repeat_openings)
                processed = processed[2:].lstrip("，。,. ")  # 移除"首先"及其后的标点
                processed = opening + " " + processed
                logger.info(f"🔄 已替换不当开头为重复介绍的开头")

        elif is_first_introduction and has_history:
            # 首次介绍但有历史：检查是否包含历史引用
            history_ref_patterns = [
                r"刚才我们介绍了",
                r"基于前面提到的",
                r"延续.*的设计",
                r"在.*的基础上"
            ]

            has_ref = any(re.search(pattern, processed[:150]) for pattern in history_ref_patterns)

            if not has_ref and historical_ref and historical_ref['found']:
                # 添加历史引用
                ref_part = historical_ref['part_name']
                additions = [
                    f"刚才我们介绍了{ref_part}，基于这个部件的功能特点，",
                    f"延续前面部件的解说，基于{ref_part}的设计理念，",
                    f"基于刚才对{ref_part}的详细介绍，"
                ]

                addition = random.choice(additions)
                processed = addition + processed
                logger.info(f"🔄 已添加历史引用：基于{ref_part}")

        # 2. 确保有互动问句
        if "？" not in processed and "?" not in processed:
            questions = [
                f"大家觉得这个{current_part_name}的设计怎么样？",
                f"想象一下，这样的{current_part_name}在实际使用中会有什么样的体验？",
                f"您对这样的{current_part_name}设计有什么看法？",
                f"这样的{current_part_name}，是否符合您的期待？",
                f"看到这里，您对{current_part_name}有什么想了解的吗？"
            ]

            sentences = re.split(r'[。！？]', processed)
            if len(sentences) > 2:
                insert_index = max(1, len(sentences) - 2)
                question = random.choice(questions)
                sentences.insert(insert_index, question)
                processed = "。".join([s for s in sentences if s.strip()]) + "。"
            else:
                processed += " " + random.choice(questions)

        # 3. 移除预测性语句
        predictive_patterns = [
            r'下个部件[，。！,]?\s*咱们聊聊.*?！',
            r'下一张图片[，。！,]?\s*我们继续.*?！',
            r'接下来我们.*?内饰.*?',
            r'下面我们.*?外观.*?',
            r'待会儿我们.*?',
            r'等一下我们.*?',
            r'稍后我们.*?',
            r'下一个部件[，。！,]?',
            r'下个图片[，。！,]?'
        ]

        for pattern in predictive_patterns:
            processed = re.sub(pattern, '', processed)

        # 4. 移除直播结束语句
        ending_phrases = [
            "这就是本次解说的内容，感谢您的关注！",
            "本次直播到此结束",
            "感谢大家的观看",
            "直播结束",
            "本次解说结束",
            "感谢您的关注"
        ]

        for phrase in ending_phrases:
            if phrase in processed:
                neutral_endings = [
                    "我们继续为大家带来精彩解说！",
                    "这就是这个部件的精彩之处！",
                    "这样的设计是不是很用心？",
                    "我们继续发现更多精彩细节！"
                ]
                replacement = random.choice(neutral_endings)
                processed = processed.replace(phrase, replacement)
                logger.info(f"🔄 已替换结束语: {phrase} → {replacement}")

        # 5. 确保结尾自然
        neutral_endings = [
            "这就是工程与设计的完美结合！",
            "大家有什么想了解的吗？",
            "这样的设计确实很用心！",
            "我们继续探索车辆的更多奥秘！",
            "这就是精湛工艺的体现！"
        ]

        current_end = processed[-20:] if len(processed) > 20 else processed
        has_good_ending = any(
            pattern in current_end for pattern in ["？", "！", "怎么样", "如何", "什么看法", "体验"]
        )

        if not has_good_ending and len(processed) > 50:
            processed += " " + random.choice(neutral_endings)

        # 6. 清理文本
        processed = re.sub(r'[。！？]{2,}', '。', processed)
        processed = re.sub(r'[,. ]{2,}', '，', processed)
        processed = re.sub(r'\s+', ' ', processed)
        processed = processed.strip()

        if processed and processed[-1] not in ['。', '！', '？', '.', '!', '?']:
            processed += '。'

        return processed

    def _create_professional_dialogue(self, part_name: str, description: str,
                                      part_number: int, introduction_count: int,
                                      has_history: bool, historical_ref: Optional[Dict] = None) -> str:
        """
        创建专业的解说对话（备用方案）
        🆕 修改：根据介绍次数创建不同的对话

        Args:
            part_name: 部件名称
            description: 部件描述
            part_number: 部件序号
            introduction_count: 介绍次数
            has_history: 是否有历史记录
            historical_ref: 历史引用信息

        Returns:
            专业的解说文本
        """
        # 🆕 根据介绍次数选择不同的开头
        if introduction_count == 1:
            # 第一次介绍
            if part_number == 1:
                beginnings = [f"首先，让我们从第{part_number}个部件{part_name}开始今天的解说！"]
            else:
                beginnings = [f"现在，让我们来看第{part_number}个部件{part_name}。"]
        else:
            # 重复介绍
            beginnings = [
                f"让我们再次聚焦{part_name}，这是第{part_number}个介绍的部件。",
                f"回到{part_name}这个话题，这是第{part_number}次介绍。",
                f"我们继续深入了解{part_name}，这是第{part_number}个部件。"
            ]

        beginning = random.choice(beginnings)

        # 中间部分
        if description:
            middle = description
        else:
            middle = f"这个{part_name}体现了车辆的精湛工艺和设计理念。"

        # 结尾部分
        endings = [
            f"大家觉得这个{part_name}的设计怎么样？",
            f"您对这样的{part_name}有什么看法？",
            f"想象一下，这样的{part_name}在实际使用中会有什么样的体验？"
        ]

        ending = random.choice(endings)

        return f"{beginning}{middle}{ending}"

    def generate_connected_description_sync(self, current_part_name: str, current_rag_result: str,
                                            target_language: str = "zh-CN") -> str:
        """
        同步版本的智能对话生成（避免事件循环冲突）

        Args:
            current_part_name: 当前部件名称
            current_rag_result: RAG生成的结果
            target_language: 目标语言代码

        Returns:
            智能对话处理后的最终描述
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def _run_in_thread():
            """在独立线程中运行异步代码"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _async_task():
                    return await self.generate_connected_description(
                        current_part_name=current_part_name,
                        current_rag_result=current_rag_result,
                        target_language=target_language
                    )

                return loop.run_until_complete(_async_task())
            finally:
                loop.close()

        # 在线程池中执行
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            try:
                return future.result(timeout=45)
            except Exception as e:
                logger.error(f"⚠️ 同步智能对话执行失败: {e}")
                # 返回基于历史的专业对话
                current_part_number = self.total_parts_introduced + 1
                has_history = len(self.conversation_history) > 0
                introduction_count = self.part_introduction_counts.get(current_part_name, 0) + 1
                historical_ref = self._find_historical_reference(current_part_name) if has_history else None
                return self._create_professional_dialogue(
                    current_part_name, current_rag_result, current_part_number,
                    introduction_count, has_history, historical_ref
                )

    def clear_history(self):
        """清空全局历史记录"""
        history_count = len(self.conversation_history)
        total_count = self.total_parts_introduced
        self.conversation_history.clear()
        self.total_parts_introduced = 0
        self.part_introduction_counts.clear()  # 🆕 清空介绍次数统计
        logger.info(f"🧹 已清空全局历史记录，清除了 {history_count} 条记录（共介绍了{total_count}个部件）")
        return {"history_cleared": history_count, "total_introduced": total_count}

    def get_history_stats(self) -> Dict[str, Any]:
        """
        获取历史统计信息
        🆕 修改：包含部件介绍次数统计

        Returns:
            历史统计字典
        """
        # 获取最近部件列表
        recent_parts = []
        for entry in self.conversation_history[-5:]:
            intro_count = entry.get('introduction_count', 1)
            recent_parts.append({
                "part_number": entry["part_number"],
                "part_name": entry["part_name"],
                "introduction_count": intro_count,  # 🆕 添加介绍次数
                "timestamp": entry["timestamp"],
                "datetime": entry["datetime"]
            })

        # 获取最近部件
        recent_part = None
        if self.conversation_history:
            recent = self.conversation_history[-1]
            intro_count = recent.get('introduction_count', 1)
            recent_part = {
                "part_number": recent["part_number"],
                "name": recent["part_name"],
                "introduction_count": intro_count,  # 🆕 添加介绍次数
                "time": recent["datetime"]
            }

        # 🆕 统计介绍次数最多的部件
        top_parts = sorted(
            self.part_introduction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        stats = {
            "total_introduced": self.total_parts_introduced,
            "history_count": len(self.conversation_history),
            "max_capacity": self.max_history_length,
            "recent_parts": recent_parts,
            "recent_part": recent_part,
            "history_enabled": True,
            "global_history": True,
            "part_introduction_stats": {  # 🆕 新增部件介绍统计
                "unique_parts": len(self.part_introduction_counts),
                "top_introduced_parts": top_parts,
                "total_introductions": sum(self.part_introduction_counts.values())
            },

            # 兼容旧版本键名
            "total_entries": len(self.conversation_history),
            "parts_list": [entry["part_name"] for entry in self.conversation_history],
            "recent_part_name": recent["part_name"] if self.conversation_history else None,
        }

        return stats

    def get_full_history(self) -> List[Dict[str, Any]]:
        """
        获取完整的历史记录

        Returns:
            完整历史记录列表
        """
        return self.conversation_history.copy()

    def get_part_introduction_count(self, part_name: str) -> int:
        """
        获取指定部件的介绍次数
        🆕 新增：外部查询部件介绍次数

        Args:
            part_name: 部件名称

        Returns:
            介绍次数，0表示从未介绍过
        """
        return self.part_introduction_counts.get(part_name, 0)


# 全局对话客户端实例
dialogue_client = None


def init_dialogue_client(llm_client):
    """
    初始化全局对话客户端

    Args:
        llm_client: LLM客户端实例
    """
    global dialogue_client
    dialogue_client = SmartDialogueClient(llm_client)
    logger.info("✅ 专业汽车直播解说客户端初始化完成（支持部件多次介绍）")
    logger.info("🎯 核心特性：")
    logger.info("  1. 部件介绍次数统计（区分首次与重复介绍）")
    logger.info("  2. 差异化解说模板（首次介绍 vs 重复介绍）")
    logger.info("  3. 全局历史记录（程序不重启就不清空）")
    logger.info("  4. 移除预测性语句和不适当的结束语")
    logger.info("  5. 基于功能的部件关联扩展")
    return dialogue_client


def get_dialogue_client():
    """
    获取全局对话客户端

    Returns:
        SmartDialogueClient实例
    """
    global dialogue_client
    if dialogue_client is None:
        raise RuntimeError("智能对话客户端未初始化，请先调用 init_dialogue_client()")
    return dialogue_client