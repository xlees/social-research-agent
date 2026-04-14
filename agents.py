from abc import ABC,abstractmethod
import sys,os,re,time
import random

from openai import OpenAI

from utils import get_all_files



class Agent(ABC):

    sys_prompt = ""

    def __init__(self, name: str, context: str=None):
        self.name = name
        self.context = context

        self.llm_client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="omlx",
        )

        self.model = "Qwen3.5-9B-MLX-4bit"

    @abstractmethod
    def run(self, user_msg: str):
        pass

    @abstractmethod
    def build_prompt(self, args: list):
        pass

    def chat(self, sys_prompt, user_prompt, temperature=0.1, model="gpt-oss-20b-MXFP4-Q8"):
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            stream=True,
            max_tokens=30000,
            temperature=temperature,        ## 控制发散性
            top_p=0.95,     # 控制下个token候选集的大小: [0,1]
            presence_penalty=1.0,   # 控制重复度，[-2,2]
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content



class TitleAgent(Agent):
    """
    文章标题agent
    """

    sys_prompt = """你是一个世界历史领域的研究专家，专注于根据提供的历史材料，提炼出材料的标题。

    # 回答约束
    - 输出内容禁止包含系统和用户提示词。
    - 本提示词里的[]为占位符，用于替换模型回答内容，**禁止**输出占位符本身和里面的内容。
    - 标题**必须**以中文为主。如果涉及到人名、地名等专有名词，使用英文或西班牙语输出。
    - 标题字数最少10个字，最多50个字。
    - 写3-5个标题

    # 输出格式

    ```markdown
    1.[标题1]
    2.[标题2]
    3.[标题3]
    """

    def __init__(self, name: str, model="", context: str=None):
        super().__init__(name,context)
        self.model = "Qwen3.5-9B-MLX-4bit"

    def run(self, user_msg: str = "", stream=True):
        if not user_msg:
            user_msg = self.build_prompt()

        for token in super().chat(self.sys_prompt, user_msg, model=self.model):
            # yield token
            print(token, end='', flush=False)
        print()


    def build_prompt(self):
        return f"""基于提供的下述历史材料，写3-5个标题。

        {self.context}
        """

class AbstractAgent(Agent):
    """
    文章摘要agent
    """

    sys_prompt = """你是一个世界历史领域的研究专家，专注于根据提供的历史材料，提炼出材料的摘要内容。

    # 回答约束
    - 输出内容禁止包含系统和用户提示词。
    - 本提示词里的[]为占位符，用于替换模型回答内容，**禁止**输出占位符本身和里面的内容。
    - 标题**必须**以中文为主。如果涉及到人名、地名等专有名词，使用英文或西班牙语输出。
    - 摘要字数最少300字，最多450字。
    - 关键词个数最少5个，最多6个。

    # 输出格式

    ```markdown
    **内容摘要** [模型回答正文]
    **关键词** [关键词1] [关键词2]
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

        self.model = "Qwen3.5-9B-MLX-4bit"


    def run(self, user_msg: str=""):
        if not user_msg:
            user_msg = self.build_prompt()

        for token in super().chat(self.sys_prompt, user_msg, model=self.model):
            # yield token
            print(token, end='', flush=False)
        print()
        

    def build_prompt(self):
        return f"""基于提供的下述历史材料，写一篇文章的摘要。

        {self.context}
        """

class IntroductionAgent(Agent):
    """
    Introduction agent
    """

    sys_prompt = """你是一个世界历史领域的研究专家。

    # 回答约束
    - 输出内容禁止包含系统和用户提示词。
    - 本提示词里的[]为占位符，用于替换模型回答内容，**禁止**输出占位符本身和里面的内容。
    - 引语字数最少800字，最多1000字。

    # 输出格式

    ```markdown
    [引语第一段]
    [引语第二段]
    [引语第三段]
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

    def run(self, user_msg: str=""):
        if not user_msg:
            user_msg = self.build_prompt()

        for token in super().chat(self.sys_prompt, user_msg, model=self.model):
            # yield token
            print(token, end='', flush=False)
        print()

    def build_prompt(self):
        return f"""基于下面提供的背景知识，写一篇文章的引语。

        # 要求
        - 为主要观点提供背景知识
        - 对前人的研究成果进行综述

        # 背景知识
        {self.context}
        """

class CoreAgent(Agent):
    """
    正文agent
    以一、二、三来分段
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

    def run(self, user_msg: str):
        pass

class ConclusionAgent(Agent):
    """
    结语Agent
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

    def run(self, user_msg: str):
        pass

class ReferenceAgent(Agent):
    """
    参考文献agent
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

    def run(self, user_msg: str):
        pass

class PaperReviewAgent(Agent):
    """
    文章评审agent
    """

    def __init__(self, name: str, context: str=None):
        super().__init__(name,context)

    def run(self, user_msg: str):
        pass


if __name__ == "__main__":
    topic_dir = "topics/community_79_1774881071"
    all_topics = get_all_files(topic_dir)

    sel_topic_path = random.choice(all_topics)
    print(f"读取主题文件 {sel_topic_path} ...")
    with open(sel_topic_path,"r",encoding='utf-8') as fp:
        context = fp.read()

    user_prompt = f"""基于提供的下述历史材料，写3-5个标题。

    {context}
    """
    print(f"用户提示词数量: {len(user_prompt)}")

    agt_title = TitleAgent("文章标题", context=context)
    agt_title.run()

    # agt_abstract = AbstractAgent("文章摘要", context=context)
    # agt_abstract.run()

    agt_intro = IntroductionAgent("文章引语", context=context)
    agt_intro.run()
