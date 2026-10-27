import os
from openai import OpenAI


# 导入模块时即检查环境变量并创建客户端；不在代码中保存密钥。
api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("未检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def ask_llm(prompt):
    """把含问题和检索资料的提示词发送给 DeepSeek，返回首个候选回答文本。"""

    # system 约束回答范围；user 携带本次问题及检索上下文。
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个教材问答助手。"
                "请仅根据用户提供的参考资料回答问题。"
                "如果资料不足以回答，请明确说明资料不足，不要自行补充事实。"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        temperature=0.2,  # 较低采样温度，减少回答的随机性。
        max_tokens=500,  # 单次生成上限，较长回答可能被截断。
        stream=False,  # 等待完整响应后一次性返回。
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    answer = response.choices[0].message.content

    return answer