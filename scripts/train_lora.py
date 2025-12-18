import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse

def train_lora(
    base_model_name: str,
    data_path: str,
    output_dir: str,
    eval_data_path: str = None,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    batch_size: int = 4,
    epochs: int = 3,
    learning_rate: float = 2e-4,
):
    print(f"Loading model: {base_model_name}")
    
    # QLoRA config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    eval_dataset = None
    if eval_data_path:
        print(f"Loading evaluation dataset from: {eval_data_path}")
        eval_dataset = load_dataset("json", data_files=eval_data_path, split="train")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA adapter for LightMem")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--eval_data_path", type=str, help="Path to evaluation JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./adapters/lightmem_mem", help="Output directory for adapter")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    train_lora(
        base_model_name=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_quant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
