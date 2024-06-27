import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_training_samples", type=int, help="Number of training samples", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=2e-4)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r= 64,          
        lora_alpha= 16,
        lora_dropout=0.05, #0.1
        bias="none",
        task_type="CAUSAL_LM",
    #target_modules= ["Wqkv", "out_proj"] #["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
    )


    # Model
    base_model = "microsoft/phi-2"
    #base_model = "openai-community/gpt2"

    new_model = "phi-2-cnn_news"

    # Dataset
    dataset_path = "cnn_dailymail"
    raw_dataset = load_dataset(dataset_path, "3.0.0")["train"]
    raw_dataset_eval = load_dataset(dataset_path, "3.0.0")["validation"]

    # sample 1000 examples 
    raw_dataset = raw_dataset.shuffle(seed=42)
    raw_dataset = raw_dataset.select(range(args.nb_training_samples))

    raw_dataset_eval = raw_dataset_eval.shuffle(seed=42)
    raw_dataset_eval = raw_dataset_eval.select(range(args.nb_training_samples // 10))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Tokenize
    def tokenize_function(example):
        return tokenizer(example["article"], padding="max_length", truncation=True)


    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
    #    quantization_config=bnb_config,
        device_map={"": 0}
    )

    training_arguments = TrainingArguments(
        output_dir = "./phi_finetuning_news_results",
        num_train_epochs = 1,
        #fp16 = False,
        #bf16 = True,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.1,
        #max_grad_norm = 0.3,
        save_steps = 0,
        optim="paged_adamw_32bit",
        logging_steps = 0.005,
        evaluation_strategy="steps",
        eval_steps=0.05
    )


    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset,
        eval_dataset=raw_dataset_eval,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        peft_config=peft_config,
        dataset_text_field="article",
    )

    # Train model
    trainer.train()

    # save full model
    model = trainer.model.merge_and_unload()

    # Save model
    model.save_pretrained(f"trained_models/{new_model}_peft", safe_serialization=False)



