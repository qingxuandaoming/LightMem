import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Literal, Any, Union
import json
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response
from .base import BaseMemoryManager

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

class LocalHFLoraManager(BaseMemoryManager):
    def __init__(self, config: BaseMemoryManagerConfig):
        super().__init__(config)
        self.config = config
        
        if not self.config.model:
            raise ValueError("Model name must be provided for LocalHFLoraManager")

        print(f"Loading model: {self.config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        
        # Determine torch_dtype
        torch_dtype = "auto"
        if self.config.torch_dtype:
            if self.config.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.config.torch_dtype == "float16":
                torch_dtype = torch.float16
            elif self.config.torch_dtype == "float32":
                torch_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            device_map=self.config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        if self.config.adapter_path:
            if PeftModel is None:
                raise ImportError("peft library is required for LoRA. Please install it with `pip install peft`.")
            print(f"Loading adapter from: {self.config.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.adapter_path,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left' # For batch generation
            
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> str:
        """
        Generate a response based on the given messages using local HF model.
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": self.config.max_tokens,
        }
        
        if self.config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
        else:
            gen_kwargs["do_sample"] = False
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response_text

    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments.
        Optimized with batch processing for local model.
        """
        if not extract_list:
            return []
            
        prompts = []
        metadata_list = [] # Stores (messages, api_call_segments) for result construction
        
        for api_call_segments in extract_list:
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
                
                # Apply chat template here to get raw text for batching
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                metadata_list.append(messages)
            except Exception as e:
                print(f"Error preparing prompt: {e}")
                # We need to maintain index alignment, so we append a placeholder
                # But if preparation fails, we can't batch it. 
                # This is tricky. Let's just skip batching for failed ones?
                # For simplicity, if one fails, we might have issues.
                # Let's assume prompt preparation is safe enough.
                prompts.append("") # Placeholder
                metadata_list.append(None)

        # Filter out invalid prompts
        valid_indices = [i for i, p in enumerate(prompts) if p]
        valid_prompts = [prompts[i] for i in valid_indices]
        
        if not valid_prompts:
            return [None] * len(extract_list)
            
        # Batch inference
        batch_size = 4 # Configurable?
        all_outputs = []
        
        gen_kwargs = {
            "max_new_tokens": self.config.max_tokens,
        }
        if self.config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
        else:
            gen_kwargs["do_sample"] = False
            
        # Ensure padding side is left for batch generation
        self.tokenizer.padding_side = "left"

        for i in range(0, len(valid_prompts), batch_size):
            batch = valid_prompts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            all_outputs.extend(decoded)
            
        # Reconstruct results
        results = []
        output_idx = 0
        for i in range(len(extract_list)):
            if i in valid_indices:
                raw_response = all_outputs[output_idx]
                output_idx += 1
                messages = metadata_list[i]
                cleaned_result = clean_response(raw_response)
                results.append({
                    "input_prompt": messages,
                    "output_prompt": raw_response,
                    "cleaned_result": cleaned_result
                })
            else:
                results.append(None)
                
        return results
