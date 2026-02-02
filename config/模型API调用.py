import asyncio
import time
from turtle import goto
from openai import AsyncOpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


async def call_qwen_api():
    client = AsyncOpenAI(
        base_url="http://192.168.110.217:8091/v1",
        api_key="EMPTY"
    )
    gold_info = [
        "厂商：一汽红旗级别：中型车能源类型：电动车上市时间2025.05电动机：纯电动 150马力纯电续航里程（KM）：475快充时间：快充0.43小时快充电量（%）：20-80",
        "最大功率（KW）:110(150Ps)最大扭矩(N·m)：205变速箱：电动车单速变速箱长x宽x高(mm)：5040x1910x1569车身结构：4门5座三厢车最高车速(km/h)：130百公里耗电量(kWh/100km)：12.3电能当量燃料消耗量(L/100km)：1.43",
        "长(mm)：5040宽(mm):1910高(mm):1569轴距(mm):2990前轮距(mm):1654后轮距(mm):1655车身结构：三厢车"]

    for i, gold in enumerate(gold_info):
        logging.info(f"RAG原文:{gold}")
        if i == 1:
            wenti = "车的长度"
        else:
            wenti = "用法语简要介绍一下这个车型的特点，不要超过100个字。"
        # 纯文本提示（简洁写法）
        messages = [
            {
                "role": "user",
                "content": f"{gold}，根据这句话，简要回答问题，{wenti}"
            }
        ]
        logging.info(f"问题:{wenti}")

        try:
            start_time = time.time()

            chat_completion = await client.chat.completions.create(
                model="/data/ai/model/models/cpatonn-mirror/Qwen3-VL-4B-Instruct-AWQ-4bit",
                # 必须与 --served-model-name 一致
                messages=messages,
                temperature=0.1,  # 低温度 → 更确定、简洁
                max_tokens=100,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )

            end_time = time.time()
            response_time = end_time - start_time
            logging.info(f"响应时间: {response_time:.2f} 秒")

            reply = chat_completion.choices[0].message.content.strip()
            print(f"模型回答:\n{reply}")

            usage = chat_completion.usage
            if usage:
                print(
                    f"\nToken Usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")

        except Exception as e:
            print(f"API 调用出错: {e}")


if __name__ == "__main__":
    asyncio.run(call_qwen_api())

# import asyncio
# from openai import AsyncOpenAI
# import time
# import requests
# import logging
# import struct
#
# logging.basicConfig(level=logging.INFO)
# # ========== 配置 ==========
# SERVER_URL = 'http://localhost:8001'  # 改为你的服务器地址，例如: 'http://192.168.1.100:8001'
#
# # ========== 调用代码 ==========
# def call_tts(text, language='en', gender='female', emotion='neutral', output_file='output.wav'):
#     """调用TTS服务"""
#
#     url = f"{SERVER_URL}/tts"
#     params = {'text': text, 'language': language, 'gender': gender, 'emotion': emotion, 'stream': True}
#
#     response = requests.post(url, params=params, stream=True, timeout=60)
#     if response.status_code != 200:
#         raise Exception(f"请求失败: {response.status_code} - {response.text}")
#
#     sample_rate = int(response.headers.get('X-Sample-Rate', 24000))
#     pcm_data = b''.join(response.iter_content(chunk_size=8192))
#
#     # 创建WAV文件
#     data_size = len(pcm_data)
#     wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
#         b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, 1,
#         sample_rate, sample_rate * 2, 2, 16, b'data', data_size)
#
#     with open(output_file, 'wb') as f:
#         f.write(wav_header + pcm_data)
#
#     print(f"✅ 音频已保存到: {output_file}")
#
#
# async def call_qwen_api():
#     client = AsyncOpenAI(
#         base_url="http://192.168.223.10:8091/v1",
#         api_key="EMPTY"
#     )
#     gold_info = [
#         "厂商：一汽红旗级别：中型车能源类型：电动车上市时间2025.05电动机：纯电动 150马力纯电续航里程（KM）：475快充时间：快充0.43小时快充电量（%）：20-80",
#         "最大功率（KW）:110(150Ps)最大扭矩(N·m)：205变速箱：电动车单速变速箱长x宽x高(mm)：5040x1910x1569车身结构：4门5座三厢车最高车速(km/h)：130百公里耗电量(kWh/100km)：12.3电能当量燃料消耗量(L/100km)：1.43",
#         "长(mm)：5040宽(mm):1910高(mm):1569轴距(mm):2990前轮距(mm):1654后轮距(mm):1655车身结构：三厢车"
#     ]
#
#     loop = asyncio.get_event_loop()
#
#     for i, gold in enumerate(gold_info):
#         logging.info(f"RAG原文: {gold}")
#         wenti = '''
# # Role
# 你是一个汽车解说文本改写助手。你需要将技术参数重构成短小精悍的口语化文案。
#
# # Task Steps
# 1. **信息筛选**：从给定的原文中剔除无意义的符号（如：x, (mm), (%), ：）。
# 2. **结构重组**：严禁按照原文的顺序输出。请采用 [赞美词] + [核心参数] + [生活化解释] 的结构。
# 3. **口语化转换**：将“轴距 2990mm”转换为“接近三米的超长轴距”；将“4门5座”转换为“宽敞的五座空间”。
#
# # Compulsory Rules (硬性规则)
# - 严禁原封不动地输出原文中的短句。
# - 字数限制：必须在 80 字以内完成。
#
# # Format Template
# [亮点评价]！[核心参数描述]。[用户利益点]。
# 用英文回答
# '''
#
#         messages = [
#             {"role": "user", "content": f"{gold}，根据这句话，简要回答问题，{wenti}"}
#         ]
#
#         try:
#             start_time = time.time()
#             chat_completion = await client.chat.completions.create(
#                 model="/home/junh/models/Qwen3-VL-4B-Instruct-AWQ-4bit",  # ⚠️ 重要：改为服务器注册的模型名！
#                 messages=messages,
#                 temperature=0.1,
#                 max_tokens=1000,
#             )
#             end_time = time.time()
#             response_time = end_time - start_time
#             logging.info(f"响应时间: {response_time:.2f} 秒")
#
#             reply = chat_completion.choices[0].message.content.strip()
#             print(f"多模态模型回答:\n{reply}")
#
#             # ====== 关键修复：在这里调用 TTS ======
#             output_file = f"output_{i+1}.wav"
#             start_time = time.time()
#             await loop.run_in_executor(None, call_tts, reply, 'en', 'female', 'neutral', output_file)
#             end_time = time.time()
#             response_time = end_time - start_time
#             logging.info(f"TTS 响应时间: {response_time:.2f} 秒")
#             usage = chat_completion.usage
#             if usage:
#                 print(f"Token Usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}")
#
#         except Exception as e:
#             print(f"API 调用出错: {e}")
#
# async def main():
#     while True:
#         print("🕒 触发 Qwen + TTS 流程...")
#         await call_qwen_api()

# import asyncio
# import time
# from turtle import goto
# from openai import AsyncOpenAI
# import logging
# #
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
#
#
# async def call_qwen_api():
#     client = AsyncOpenAI(
#         base_url="http://192.168.223.10:8091/v1",
#         api_key="EMPTY"
#     )
#     gold_info = [
#         "厂商：一汽红旗级别：中型车能源类型：电动车上市时间2025.05电动机：纯电动 150马力纯电续航里程（KM）：475快充时间：快充0.43小时快充电量（%）：20-80",
#         "最大功率（KW）:110(150Ps)最大扭矩(N·m)：205变速箱：电动车单速变速箱长x宽x高(mm)：5040x1910x1569车身结构：4门5座三厢车最高车速(km/h)：130百公里耗电量(kWh/100km)：12.3电能当量燃料消耗量(L/100km)：1.43",
#         "长(mm)：5040宽(mm):1910高(mm):1569轴距(mm):2990前轮距(mm):1654后轮距(mm):1655车身结构：三厢车"]
#
#     for i, gold in enumerate(gold_info):
#         logging.info(f"RAG原文:{gold}")
#         if i == 1:
#             wenti = "车的长度"
#         else:
#             wenti = "用法语简要介绍一下这个车型的特点，不要超过100个字。"
#         # 纯文本提示（简洁写法）
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"{gold}，根据这句话，简要回答问题，{wenti}"
#             }
#         ]
#         logging.info(f"问题:{wenti}")
#
#         try:
#             start_time = time.time()
#
#             chat_completion = await client.chat.completions.create(
#                 model="/home/junh/models/Qwen3-VL-4B-Instruct-AWQ-4bit",
#                 # 必须与 --served-model-name 一致
#                 messages=messages,
#                 temperature=0.1,  # 低温度 → 更确定、简洁
#                 max_tokens=100,
#                 presence_penalty=0.0,
#                 frequency_penalty=0.0,
#             )
#
#             end_time = time.time()
#             response_time = end_time - start_time
#             logging.info(f"响应时间: {response_time:.2f} 秒")
#
#             reply = chat_completion.choices[0].message.content.strip()
#             print(f"模型回答:\n{reply}")
#
#             usage = chat_completion.usage
#             if usage:
#                 print(
#                     f"\nToken Usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")
#
#         except Exception as e:
#             print(f"API 调用出错: {e}")
#
# if __name__ == "__main__":
#     asyncio.run(call_qwen_api())