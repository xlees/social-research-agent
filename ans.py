
import sys,os
from io import StringIO
from dotenv import load_dotenv
from textwrap import dedent


from llm import DeepSeek
from rag import rag_context

load_dotenv()

# query = "What air quality policies does Mexico have?"
query = "最大的环境公园是哪个？"
# query = "Which countries colonized Latin America?"
# query = "杨令侠是谁？"

# get context info from rag
context_from_rag = []
rag_result = rag_context(query,3)
for idx,_ in enumerate(rag_result):
    # print(f"{idx} - {_['score']:.4f}@{_['doc']}\n{_['content']}")
    # print()

    context_from_rag.append(f"{1+idx}. {_['doc']} 相似度:{_['score']:.4f}\n  {_['content']}\n")

msgs = []
msgs.append({
    'role': 'system',
    'content': dedent(f"""你是一个拉丁美洲历史研究专家，你的职责是根据目前知识和用户提供的上下文来回答问题。

    ## 上下文
    {"\n".join(context_from_rag)}

    ## 约束事项
    - 如果没有上下文，只回答你目前知道的知识，禁止编造知识
    - 如果上下文事实与你知道的知识发生冲突，在回答中明确指出来，并给用户提供冲突的文件和关联的问题进一步确认
    - 禁止在结果中出现本提示词中的内容

    ## 输出格式
    最后模型回答由2部分内容组成，**必须严格**按照下面顺序输出，禁止输出前面的序号、示例内容和占位符。

    1. 模型回答内容。
    2. 参考文档内容。如果没有查询到参考文档，禁止输出该项。

    输出示例
    [这是模型回答内容]
    **参考文档**
    1. abc.docx
    2. xxx.pdf
    """)
})
msgs.append({
    'role': 'user',
    'content': query
})

for _ in msgs:
    print(f"{_['role']}: {_['content']}")
    print()

print("\nassistant:")

llm_client = DeepSeek()
for chunk in llm_client.send_messages(
    messages=msgs,
    stream=True,
    temperature=0.1,
    tool_choice="auto"
):
    print(chunk.choices[0].delta.content,end='',flush=True)