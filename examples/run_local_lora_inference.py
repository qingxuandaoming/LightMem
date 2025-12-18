import os
import sys
import logging
import json
from lightmem.factory.memory_manager.factory import MemoryManagerFactory
from lightmem.configs.memory_manager.base import MemoryManagerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Example configuration for Local HF LoRA
    # NOTE: You must have a trained adapter or use a base model without adapter
    config = MemoryManagerConfig(
        model_name="local_hf_lora",
        configs={
            "model": "Qwen/Qwen2.5-7B-Instruct", # Replace with your base model
            # "adapter_path": "./adapters/lightmem_mem", # Uncomment if you have an adapter
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "max_tokens": 512,
            "temperature": 0.1
        }
    )

    try:
        manager = MemoryManagerFactory.from_config(config)
        logger.info("LocalHFLoraManager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize manager: {e}")
        return

    # Test generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    
    logger.info("Testing simple generation...")
    response = manager.generate_response(messages)
    print(f"\nResponse:\n{response}\n")

    # Test metadata extraction (Batch)
    logger.info("Testing metadata extraction (Batch)...")
    
    system_prompt = "Extract key facts from the conversation as JSON."
    
    # Simulate two parallel API calls, each with multiple topic segments
    extract_list = [
        [ # Call 1
            [ # Segment 1
                {"role": "user", "content": "My name is Alice.", "sequence_number": 1},
                {"role": "assistant", "content": "Hi Alice.", "sequence_number": 2}
            ],
            [ # Segment 2
                {"role": "user", "content": "I like apples.", "sequence_number": 3}
            ]
        ],
        [ # Call 2
            [ # Segment 1
                {"role": "user", "content": "My name is Bob.", "sequence_number": 1},
                {"role": "assistant", "content": "Hi Bob.", "sequence_number": 2}
            ]
        ]
    ]
    
    results = manager.meta_text_extract(system_prompt, extract_list, messages_use="hybrid")
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        if res:
            print(f"Cleaned Result: {json.dumps(res['cleaned_result'], indent=2, ensure_ascii=False)}")
        else:
            print("Failed.")

if __name__ == "__main__":
    main()
