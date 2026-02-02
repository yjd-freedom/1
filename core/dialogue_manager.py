# dialogue_manager.py
"""
真正的LLM驱动对话管理器
使用LLM生成自然衔接，而不是固定模板
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("dialogue_manager")

class SmartDialogueManager:
    """智能对话管理器 - 真正使用LLM生成自然衔接"""
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.conversation_history = []  # 完整的对话历史
        self.components_history = []  # 讲解过的部件历史
        self.max_history = 8  # 最多记住8个部件

        logger.info("🤖 智能对话管理器已初始化")
        if llm_client:
            logger.info("✅ LLM客户端已连接")

    # ================== 核心方法：LLM生成衔接 ==================

    async def generate_intelligent_transition(self, new_component: Dict[str, Any],
                                              target_language: str = "zh-CN") -> Tuple[str, str]:
        """
        使用LLM智能生成衔接和讲解

        Args:
            new_component: 新部件信息
            target_language: 目标语言

        Returns:
            (transition_text, full_narrative) - 衔接文本和完整讲解
        """
        component_name = new_component.get('label', '')
        component_desc = new_component.get('description', '')

        # 1. 构建LLM提示词（关键！）
        prompt = self._build_intelligent_prompt(
            component_name=component_name,
            component_description=component_desc,
            conversation_history=self.conversation_history,
            target_language=target_language
        )

        logger.info(f"🧠 使用LLM生成智能衔接 - 语言: {target_language}")

        try:
            # 2. 调用LLM生成自然衔接的讲解
            llm_response = await self.llm_client.generate_summary(
                context=prompt,
                target_language=target_language,
                question=None  # 您的llm_client可能需要这个参数
            )

            # 3. 解析LLM响应（可能包含衔接和讲解）
            transition, narrative = self._parse_llm_response(
                llm_response, component_name, target_language
            )

            logger.info(f"✅ LLM生成完成 - 衔接长度: {len(transition)} 字符")

            return transition, narrative

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            # 回退到简单衔接
            return self._generate_fallback_transition(component_name), component_desc

    def _build_intelligent_prompt(self, component_name: str, component_description: str,
                                  conversation_history: List, target_language: str) -> str:
        """
        构建智能提示词，让LLM理解上下文并生成自然衔接

        这是关键！不是固定模板，而是让LLM理解整个对话
        """
        # 1. 提取历史信息
        history_text = self._format_conversation_history(conversation_history)

        # 2. 构建提示词（根据不同语言调整）
        if target_language == "zh-CN":
            prompt = f"""你是一名专业的汽车销售顾问，正在直播介绍车辆。

## 对话历史回顾：
{history_text if history_text else "这是第一次介绍车辆。"}

## 现在要介绍的新部件：
**{component_name}**

## 这个部件的技术信息：
{component_description}

## 你的任务：
请生成一段自然流畅的直播销售讲解，要求：

1. **衔接自然**：如果之前介绍过其他部件，请自然地从这个话题过渡到新部件
2. **突出亮点**：用生动有趣的方式介绍这个部件的优点
3. **语言风格**：像朋友聊天一样自然，有直播的感染力
4. **结构建议**：
   - 开场衔接（如果之前有内容）
   - 部件介绍（突出1-2个核心优点）
   - 结束过渡（为下一步留有余地）

记住：你不是在读说明书，而是在和朋友分享好东西！

请开始你的讲解："""

        elif target_language == "en-US":
            prompt = f"""You are a professional car sales consultant doing a live stream.

## Conversation History:
{history_text if history_text else "This is the first introduction to the vehicle."}

## New component to introduce:
**{component_name}**

## Technical information about this component:
{component_description}

## Your task:
Generate a natural, engaging live stream sales pitch:

1. **Smooth transition**: If you've introduced other components before, naturally transition to this new one
2. **Highlight benefits**: Focus on 1-2 key benefits in an engaging way
3. **Language style**: Conversational, friendly, like sharing with friends
4. **Suggested structure**:
   - Opening transition (if applicable)
   - Component introduction
   - Closing transition

Remember: You're not reading a manual, you're sharing something cool with friends!

Start your pitch:"""

        else:
            # 其他语言的通用提示词
            prompt = f"""You are introducing car components in a live stream.

History: {history_text}
New component: {component_name}
Info: {component_description}

Please introduce this component naturally, connecting it to previous topics if any."""

        return prompt

    def _format_conversation_history(self, history: List) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        # 只取最近3次讲解
        recent = history[-3:] if len(history) > 3 else history

        formatted = []
        for i, entry in enumerate(recent):
            # 提取关键信息
            component = entry.get('component', '')
            narrative = entry.get('narrative', '')

            # 只取前100个字符
            preview = narrative[:100] + "..." if len(narrative) > 100 else narrative

            formatted.append(f"{i + 1}. {component}: {preview}")

        return "\n".join(formatted)

    def _parse_llm_response(self, llm_response: str, component_name: str,
                            target_language: str) -> Tuple[str, str]:
        """
        解析LLM响应，提取衔接部分和完整讲解

        策略：如果LLM响应很长，智能分割出衔接部分
        """
        # 清理响应
        response = llm_response.strip()

        # 对于中文，尝试找自然的分割点
        if target_language == "zh-CN":
            # 常见的中文衔接词位置
            transition_markers = ["接下来", "我们再看", "除了", "另外", "同时", "说到"]

            for marker in transition_markers:
                if marker in response[:50]:  # 在前50个字符内找
                    # 找到衔接词的结束位置（通常到第一个句号）
                    end_pos = response.find("。", response.find(marker))
                    if end_pos > 0:
                        transition = response[:end_pos + 1]
                        narrative = response
                        return transition, narrative

        # 默认：前1-2句作为衔接
        sentences = self._split_sentences(response, target_language)

        if len(sentences) >= 2:
            # 前1-2句作为衔接
            transition = "".join(sentences[:2])
            narrative = response
        else:
            # 只有一句，整个作为讲解
            transition = f"我们来看看{component_name}，"
            narrative = response

        return transition, narrative

    def _split_sentences(self, text: str, language: str) -> List[str]:
        """按语言分句"""
        if language == "zh-CN":
            # 中文分句
            import re
            sentences = re.split(r'[。！？!?]', text)
            return [s.strip() + "。" for s in sentences if s.strip()]
        else:
            # 英文分句
            import re
            sentences = re.split(r'[.!?]', text)
            return [s.strip() + "." for s in sentences if s.strip()]

    # ================== 历史管理 ==================

    async def process_new_component(self, component_info: Dict[str, Any],
                                    target_language: str = "zh-CN") -> Dict[str, Any]:
        """
        处理新部件，生成智能讲解

        Returns:
            包含衔接、讲解和历史信息的完整结果
        """
        component_name = component_info.get('label', '')

        # 1. 使用LLM生成智能衔接和讲解
        transition, narrative = await self.generate_intelligent_transition(
            component_info, target_language
        )

        # 2. 更新历史
        history_entry = {
            'component': component_name,
            'narrative': narrative,
            'transition': transition,
            'timestamp': time.time(),
            'language': target_language,
            'component_info': {
                'confidence': component_info.get('confidence', 0),
                'score': component_info.get('score', 0),
                'description': component_info.get('description', '')
            }
        }

        self.conversation_history.append(history_entry)
        self.components_history.append(component_name)

        # 3. 限制历史长度
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            self.components_history = self.components_history[-self.max_history:]

        # 4. 构建返回结果
        result = {
            'component': component_name,
            'transition': transition,
            'narrative': narrative,
            'full_presentation': f"{transition} {narrative}",
            'history_length': len(self.conversation_history),
            'recent_components': self.components_history[-3:] if self.components_history else [],
            'llm_generated': True,
            'timestamp': time.time()
        }

        logger.info(f"📝 已讲解 {len(self.conversation_history)} 个部件，最新: {component_name}")

        return result

    def get_conversation_flow(self) -> str:
        """获取对话流程摘要"""
        if not self.conversation_history:
            return "对话尚未开始"

        flow = []
        for i, entry in enumerate(self.conversation_history[-4:]):  # 最近4个
            flow.append(f"{i + 1}. {entry['component']}")

        return " → ".join(flow)

    def clear_history(self) -> None:
        """清空历史"""
        self.conversation_history = []
        self.components_history = []
        logger.info("🔄 对话历史已清空")

    # ================== 回退方案 ==================

    def _generate_fallback_transition(self, component_name: str) -> str:
        """LLM失败时的回退衔接"""
        transitions = [
            f"我们接着看看{component_name}，这个配置很有意思，",
            f"接下来介绍一下{component_name}，",
            f"再看车辆的{component_name}，",
            f"我们继续介绍{component_name}，"
        ]
        import random
        return random.choice(transitions)

    # ================== 部件选择策略 ==================

    def select_component_to_narrate(self, detections: List[Dict[str, Any]],
                                    strategy: str = "smart") -> Optional[Dict[str, Any]]:
        """
        智能选择要讲解的部件

        Args:
            detections: 所有检测到的部件
            strategy: 选择策略

        Returns:
            选择的部件
        """
        if not detections:
            return None

        if strategy == "smart":
            # 智能策略：避免重复，考虑重要性
            return self._smart_selection(detections)
        elif strategy == "score":
            # 按分数选择
            return max(detections, key=lambda x: x.get('score', 0))
        else:
            # 默认：第一个
            return detections[0]

    def _smart_selection(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """智能选择策略"""
        # 按分数排序
        sorted_dets = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)

        # 获取已讲解的部件
        narrated = set(self.components_history)

        # 优先选择未讲解过的高分部件
        for det in sorted_dets:
            if det.get('label', '') not in narrated:
                return det

        # 如果都讲解过，选择最久没讲的高分部件
        # 这里简化：选择分数最高的
        return sorted_dets[0] if sorted_dets else None


# ================== 高级功能：多部件连贯讲解 ==================
class MultiComponentNarrator:
    """多部件连贯讲解生成器"""

    def __init__(self, dialogue_manager: SmartDialogueManager):
        self.dialogue_manager = dialogue_manager
        self.batch_history = []  # 批次处理历史

    async def generate_coherent_presentation(self, components: List[Dict[str, Any]],
                                             target_language: str = "zh-CN") -> Dict[str, Any]:
        """
        为多个部件生成连贯的讲解

        Args:
            components: 部件列表
            target_language: 目标语言

        Returns:
            连贯的讲解结果
        """
        if not components:
            return {'success': False, 'message': '没有部件'}

        # 1. 智能排序部件
        ordered_components = self._order_components(components)

        # 2. 为每个部件生成讲解
        narratives = []
        transitions = []

        for i, component in enumerate(ordered_components):
            # 生成讲解
            result = await self.dialogue_manager.process_new_component(
                component, target_language
            )

            narratives.append(result['narrative'])
            transitions.append(result['transition'])

            # 如果是第一个，不需要特殊处理
            # 后续的会基于历史自动生成衔接

        # 3. 组合成完整讲解
        full_presentation = " ".join(narratives)

        # 4. 添加开场白和结束语
        enhanced_presentation = self._enhance_presentation(
            full_presentation, ordered_components, target_language
        )

        return {
            'success': True,
            'components': [comp['label'] for comp in ordered_components],
            'narratives': narratives,
            'transitions': transitions,
            'full_presentation': enhanced_presentation,
            'component_count': len(ordered_components)
        }

    def _order_components(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """智能排序部件"""
        # 这里可以有很多排序策略，比如：
        # 1. 按分数从高到低
        # 2. 按汽车系统逻辑
        # 3. 按视觉位置

        # 简单实现：按分数排序
        return sorted(components, key=lambda x: x.get('score', 0), reverse=True)

    def _enhance_presentation(self, presentation: str, components: List[Dict[str, Any]],
                              language: str) -> str:
        """增强讲解，添加开场和结束"""
        component_names = [comp['label'] for comp in components]

        if language == "zh-CN":
            opening = f"让我为大家介绍这款车的几个亮点配置，包括{self._join_names(component_names)}。"
            closing = "从这些配置可以看出，这款车在设计上非常用心。"
        else:
            opening = f"Let me introduce several highlight features of this vehicle, including {self._join_names(component_names, english=True)}."
            closing = "These features show the careful design of this vehicle."

        return f"{opening} {presentation} {closing}"

    def _join_names(self, names: List[str], english: bool = False) -> str:
        """连接部件名称"""
        if not names:
            return ""

        if len(names) == 1:
            return names[0]

        if english:
            if len(names) == 2:
                return f"{names[0]} and {names[1]}"
            return f"{', '.join(names[:-1])}, and {names[-1]}"
        else:
            if len(names) == 2:
                return f"{names[0]}和{names[1]}"
            return f"{'、'.join(names[:-1])}和{names[-1]}"


# ================== 全局实例 ==================

_global_smart_manager = None


def get_smart_dialogue_manager(llm_client=None) -> SmartDialogueManager:
    """获取智能对话管理器"""
    global _global_smart_manager

    if _global_smart_manager is None:
        _global_smart_manager = SmartDialogueManager(llm_client=llm_client)

    return _global_smart_manager


def get_multi_narrator(llm_client=None) -> MultiComponentNarrator:
    """获取多部件讲解器"""
    manager = get_smart_dialogue_manager(llm_client)
    return MultiComponentNarrator(manager)


# ================== 测试代码 ==================

async def test_smart_dialogue():
    """测试智能对话管理器"""

    # 模拟LLM客户端
    class MockLLMClient:
        async def generate_summary(self, context, target_language, question=None):
            # 模拟LLM生成自然的衔接
            if "发动机" in context:
                return "刚才我们看了外观，现在来看看车辆的心脏——发动机。这款2.0T涡轮增压发动机真的很给力！"
            elif "变速箱" in context:
                return "有了强劲的发动机，当然需要一台聪明的变速箱来匹配。这台7速双离合变速箱换挡特别平顺。"
            else:
                return "我们再来看看这个配置，它设计得很不错。"

    # 创建管理器
    manager = SmartDialogueManager(llm_client=MockLLMClient())

    # 模拟部件
    engine = {
        'label': '发动机',
        'description': '2.0T涡轮增压发动机，最大功率150kW',
        'confidence': 0.95,
        'score': 92.5
    }

    transmission = {
        'label': '变速箱',
        'description': '7速双离合变速箱，换挡迅速平顺',
        'confidence': 0.92,
        'score': 88.3
    }

    # 处理第一个部件
    result1 = await manager.process_new_component(engine, "zh-CN")
    print("第一次讲解（发动机）：")
    print(f"  衔接: {result1['transition'][:50]}...")
    print(f"  历史流程: {manager.get_conversation_flow()}")
    print()

    # 处理第二个部件（会自动基于历史生成衔接）
    result2 = await manager.process_new_component(transmission, "zh-CN")
    print("第二次讲解（变速箱）：")
    print(f"  衔接: {result2['transition'][:50]}...")
    print(f"  历史流程: {manager.get_conversation_flow()}")
    print(f"  完整讲解: {result2['full_presentation'][:100]}...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_smart_dialogue())