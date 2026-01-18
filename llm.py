

from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import (
    # ChatCompletionAudioParam,
    # completion_list_params,
    completion_create_params,
    # completion_update_params,
)

import sys,os
from abc import ABC, abstractmethod

load_dotenv()

class LLM(ABC):
    @abstractmethod
    def send_messages(self,
        messages: List[Dict[str, str]],
        tools: List[str]
    ):
        pass

class DeepSeek(LLM):
    """

    """
    def __init__(self):

        self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def send_messages(self,
        messages: List[Dict[str, str]],
        tools: List[str] = None,
        stream: bool = False,
        temperature: float = 0.4,
        parallel_tool_calls: bool = False,
        max_tokens: int = 8192,
        tool_choice: str = "auto",
        response_format: completion_create_params.ResponseFormat = None
    ) -> ChatCompletion:
        """
        发送多轮对话消息给llm, 需限定历史对话次数，否则会很浪费token.
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        - id
        - choices
        - created
        - model
        - object
        - service_tier
        - system_fingerprint
        - usage
        """

        try:
            return self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                stream=stream,
                temperature=temperature,
                tool_choice=tool_choice,
                presence_penalty=1.6,
                max_tokens=max_tokens,
                response_format=response_format,
                parallel_tool_calls=parallel_tool_calls
            )

        except Exception as e:
            raise Exception(f"Failed to create message: {str(e)}")
        
        
class Qwen(LLM):
    """
    阿里千问大模型
    """
    endpoints = {
        'bj': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'sg': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    }
    
    def __init__(self):
        # 北京: https://dashscope.aliyuncs.com/compatible-mode/v1
        # 新加坡：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        # http endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
        self.client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url=Qwen.endpoints['sg'])

    def send_messages(self,
        messages: List[Dict[str, str]],
        tools: List[str] = None,
        temperature: float = 0.4
    ) -> ChatCompletion:
        """
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        - id
        - choices
        - created
        - model
        - object
        - service_tier
        - system_fingerprint
        - usage
        """
        try:
            return self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=tools,
                temperature=temperature,
                tool_choice="auto"
            )

        except Exception as e:
            raise Exception(f"Failed to create message: {str(e)}")
