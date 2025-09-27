import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Mxfp4Config
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
torch.manual_seed(42)

from huggingface_hub import login
login(token = "") # ADD HUGGINGFACE TOKEN HERE

import transformers
import peft
import pandas
import sklearn
import accelerate
import pandas as pd
import openai
print("openai:", openai.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("wandb:", wandb.__version__)
print("accelerate:", accelerate.__version__)


IGNORE_INDEX = -100

def main(args):
    if args.report_to_wandb:
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # --- 1. GPU Setup ---
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        num_gpus = len(args.gpu_ids.split(','))
        print(f"Using {num_gpus} specific GPUs: {args.gpu_ids}")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Using all available {num_gpus} GPUs.")

    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Please check your setup.")
    print(f"Detected {torch.cuda.device_count()} CUDA-capable devices.")
    print(f"Actual number of GPUs to be used by Trainer: {num_gpus}")

    # --- 2. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.model_max_length = args.max_seq_length

    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
        elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            print("Warning: pad_token is None. Using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Warning: pad_token is None and eos_token is None. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"Tokenizer pad token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer EOS token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")

    # --- 3. Quantization Configuration (BitsAndBytes) ---
    bnb_config = None
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Using 4-bit NF4 quantization with compute_dtype: {compute_dtype}.")

    # --- 4. Load Base Model ---
    print(f"Loading base model: {args.model_name_or_path}")
    if args.use_gpt_oss:
        quantization_config = Mxfp4Config(dequantize=True)
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            use_cache=False,
            device_map="auto",
        )
    else:
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    model.config.use_cache = False
    # model.config.pretraining_tp = 1 # Might be needed if model was saved with tensor parallelism

    # --- 5. PEFT (LoRA) Configuration ---
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.use_gpt_oss:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            target_parameters=[
                "7.mlp.experts.gate_up_proj",
                "7.mlp.experts.down_proj",
                "15.mlp.experts.gate_up_proj",
                "15.mlp.experts.down_proj",
                "23.mlp.experts.gate_up_proj",
                "23.mlp.experts.down_proj",
            ],
        )
    else:
        target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 6. Load and Preprocess Dataset ---
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    if args.dataset_subset_size > 0:
        print(f"Subsetting dataset to {args.dataset_subset_size} examples for quick testing.")
        dataset = dataset.select(range(args.dataset_subset_size))

    if 0 < args.validation_split_percentage < 100:
        dataset_split = dataset.train_test_split(test_size=args.validation_split_percentage / 100.0, seed=args.seed)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        print(f"Using {len(train_dataset)} for training and {len(eval_dataset)} for evaluation.")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Using {len(train_dataset)} for training. No validation set (or invalid split percentage).")

    # Should work for both Llama2 and DeepSeek-Llama Distill
    def tokenize_function(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for prompt, response in zip(prompts, responses):
            # 1. Create the prompt part using the chat template
            prompt_messages = [
                #{"role": "system", "content": "You are an expert assistant in compliance checking of documents based on a given policy."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply template to the prompt section only
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # 2. Create the full example (prompt + response)
            full_messages = prompt_messages + [{"role": "assistant", "content": response}]
            
            formatted_full_example = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
            )
            # Add the End-Of-Sequence token, which is crucial for the model to learn when to stop.
            formatted_full_example += tokenizer.eos_token

            # 3. Tokenize the full example
            tokenized_full = tokenizer(
                formatted_full_example,
                max_length=args.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            # 4. Tokenize the prompt-only part to find its length for masking
            # We don't pad this because we only need its length.
            tokenized_prompt = tokenizer(
                formatted_prompt,
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            prompt_len = tokenized_prompt.input_ids.shape[1]

            # The labels are a copy of the input_ids, but with the prompt tokens masked out.
            labels = tokenized_full.input_ids.clone().squeeze(0)
            labels[:prompt_len] = IGNORE_INDEX
            labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX
            model_inputs["input_ids"].append(tokenized_full.input_ids.squeeze(0))
            model_inputs["attention_mask"].append(tokenized_full.attention_mask.squeeze(0))
            model_inputs["labels"].append(labels)
            
        return model_inputs

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval_dataset = None
    if eval_dataset:
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
        )

    # --- 7. Data Collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        return_tensors="pt"
    )

    # --- 8. Training Arguments ---
    report_to_list = []
    if args.report_to_wandb:
        report_to_list.append("wandb")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size if eval_dataset else args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optim,
        fp16=not args.use_bf16 and args.use_fp16,
        bf16=args.use_bf16,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to_list,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        # load_best_model_at_end=True if eval_dataset and args.report_to_wandb else False, # wandb can track best model too
        # metric_for_best_model="eval_loss" if eval_dataset else None, # common metric
    )

    # --- 9. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    torch.cuda.empty_cache()

    print("Starting training...")
    trainer.train()

    # --- Save Model ---
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_dir}")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    if args.save_merged_model:
        merged_model_dir = os.path.join(args.output_dir, "final_merged_model")
        print(f"Merging LoRA adapters and saving full model to {merged_model_dir}...")
        try:
            del model
            torch.cuda.empty_cache()

            base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
                trust_remote_code=True,
                device_map="cpu"
            )
            from peft import PeftModel
            merged_model = PeftModel.from_pretrained(base_model_for_merge, final_adapter_dir)
            merged_model = merged_model.merge_and_unload()

            merged_model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"Full merged model saved to {merged_model_dir}")
        except Exception as e:
            print(f"Error merging and saving model: {e}")
            print("Please ensure you have enough CPU RAM for merging.")

    if args.report_to_wandb and wandb.run:
        wandb.finish()

    print("Finetuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a Llama-based 8B model with LoRA, 4-bit quantization, and WandB.")

    # Model and Data
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model ID or path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to .jsonl dataset.")
    parser.add_argument("--dataset_subset_size", type=int, default=0, help="Use N instances for testing (0 for full).")
    parser.add_argument("--validation_split_percentage", type=int, default=5, help="Percentage for validation (0-99). 0 for no validation.")
    parser.add_argument("--output_dir", type=str, default="./deepseek_finetuned_8b", help="Output directory for adapters.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--use_gpt_oss", action="store_true", default=False, help="Finetune GPT-OSS.")

    # GPU and Hardware
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs.")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization.")
    parser.add_argument("--no_use_4bit", action="store_false", dest="use_4bit", help="Disable 4-bit quantization.")
    parser.add_argument("--use_bf16", action="store_true", default=None, help="Use bfloat16. Auto-detected.")
    parser.add_argument("--use_fp16", action="store_true", default=None, help="Use float16. Auto-detected if bf16 not available.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU (train).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per GPU (eval).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing.")

    # LoRA Specific
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every X steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep.")
    parser.add_argument("--save_merged_model", action="store_true", help="Save full merged model.")

    # Reporting (WandB)
    parser.add_argument("--report_to_wandb", action="store_true", default=False, help="Enable reporting to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="deepseek-finetuning", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (generates one if None).")

    # Dataloader
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="Num workers for preprocessing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Num workers for dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parsed_args = parser.parse_args()

    # Auto-detect precision
    if parsed_args.use_bf16 is None and parsed_args.use_fp16 is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("BF16 is supported. Enabling BF16 training.")
            parsed_args.use_bf16 = True
            parsed_args.use_fp16 = False
        elif torch.cuda.is_available():
            print("BF16 not supported. Enabling FP16 training.")
            parsed_args.use_bf16 = False
            parsed_args.use_fp16 = True
        else:
            parsed_args.use_bf16 = False
            parsed_args.use_fp16 = False
    elif parsed_args.use_bf16:
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("Warning: BF16 was requested but is not supported.")
        parsed_args.use_fp16 = False
    elif parsed_args.use_fp16:
         parsed_args.use_bf16 = False

    if parsed_args.preprocessing_num_workers is None:
        parsed_args.preprocessing_num_workers = os.cpu_count()

    main(parsed_args)