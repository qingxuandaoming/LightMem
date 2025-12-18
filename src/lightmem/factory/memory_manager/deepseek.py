from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any
import json
import concurrent.futures
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response
from .base import BaseMemoryManager

class DeepseekManager(BaseMemoryManager):
    def __init__(self, config: BaseMemoryManagerConfig):
        super().__init__(config)
        self.config = config
        if not self.config.model:
            self.config.model = "deepseek-chat"
        self.api_key = self.config.api_key
        # Config is an object, not a dict, so use getattr or dot access. 
        # BaseMemoryManagerConfig has deepseek_base_url
        self.base_url = self.config.deepseek_base_url or "https://api.deepseek.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using DeepSeek.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # DeepSeek API might not support response_format in the same way as OpenAI json_object,
        # but if it does, we pass it. If not, we might need to handle it.
        # Assuming compatibility for now.
        if response_format:
             params["response_format"] = response_format

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.
        """
        if not extract_list:
            return []
        
        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(api_call_segments: List[List[Dict]]):
            """Process one API call (multiple topic segments inside)"""
            try:
                user_prompt_parts = []
                for idx, topic_segment in enumerate(api_call_segments, start=1):
                    topic_text = self._concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {idx} ---\n{topic_text}")

                user_prompt = "\n".join(user_prompt_parts)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                # DeepSeek might not support json_object response format perfectly, 
                # but we try. If it fails, clean_response should handle markdown json blocks.
                raw_response = self.generate_response(
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                cleaned_result = clean_response(raw_response)
                return {
                    "input_prompt": messages,
                    "output_prompt": raw_response,
                    "cleaned_result": cleaned_result
                }
            except Exception as e:
                print(f"Error processing API call: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, extract_list))
                return results
            except Exception as e:
                print(f"Error in parallel execution: {e}")
                return [None] * len(extract_list)