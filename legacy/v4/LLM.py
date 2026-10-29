import os
from openai import OpenAI


api_key = os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("未检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def ask_llm(prompt):

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
        temperature=0.2,
        max_tokens=500,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    answer = response.choices[0].message.content

    return answer