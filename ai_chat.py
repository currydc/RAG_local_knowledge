from openai import OpenAI
import time
client = OpenAI(
    api_key="ollama", # 请替换成您的ModelScope Access Token
    base_url="http://127.0.0.1:11434/v1/"
)
# client = OpenAI(
#     api_key="", # 请替换成您的ModelScope Access Token
#     base_url=""
# )
start_time=time.time()
print(time.time())

response = client.chat.completions.create(
    # model="Qwen/Qwen3-32B", # ModelScope Model-Id
    model="qwen3:0.6b",  # ModelScope Model-Id
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '你是谁'
        }
    ],
    stream=True
)

for chunk in response:
    content=chunk.choices[0].delta.content
    think=chunk.choices[0].delta.model_extra.get('reasoning_content')
    if think:
        print(think, end='', flush=True)
    if content:
        print(content, end='', flush=True)
