from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, Any
import json
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig

class BaseMemoryManager(ABC):
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        pass

    @abstractmethod
    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        pass

    def _concatenate_messages(self, segment: List[Dict], messages_use: str) -> str:
        """Concatenate messages based on usage strategy"""
        role_filter = {
            "user_only": {"user"},
            "assistant_only": {"assistant"},
            "hybrid": {"user", "assistant"}
        }

        if messages_use not in role_filter:
            raise ValueError(f"Invalid messages_use value: {messages_use}")

        allowed_roles = role_filter[messages_use]
        message_lines = []

        for mes in segment:
            if mes.get("role") in allowed_roles:
                sequence_id = mes.get("sequence_number", "")
                role = mes["role"]
                content = mes.get("content", "")
                prefix = f"{sequence_id}." if sequence_id != "" else ""
                message_lines.append(f"{prefix}{role}: {content}")

        return "\n".join(message_lines)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.
        This is a common helper for API-based models (OpenAI/DeepSeek).
        """
        # Note: This assumes 'response' is an object with choices[0].message
        # compatible with OpenAI SDK.
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content
