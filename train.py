import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import yaml
from huggingface_hub import login

def main():
    

    login(token=os.environ['HF_TOKEN'])

    config_file = "training/config.yml"
    cfg = yaml.safe_load(open(config_file))

    print(f"Using model: {cfg['model_name']}")

    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], legacy=False)
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="cuda")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(tokenizer))

    # add LoRA
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj","v_proj"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Loading PEFT Model...")
    model = get_peft_model(base_model, lora_config)

    print(f"Loading dataset...")
    data_file = cfg["train_file"]
    dataset_raw = load_dataset("json", data_files=data_file)["train"]
    tokenize = make_tokenize(tokenizer)
    dataset = dataset_raw.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    
    print("Loading Data Collector...")
    data_collector=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Creating Training Args...")
    learning_rate = float(cfg["learning_rate"])
    batch_size = cfg["batch_size"]
    gradiant_accumulation_steps = cfg["gradient_accumulation_steps"]
    num_train_epochs = cfg["num_train_epochs"]
    print(f"\t learning_rate: {learning_rate}")
    print(f"\t batch_size: {batch_size}")
    print(f"\t gradiant_accumulation_steps: {gradiant_accumulation_steps}")
    print(f"\t num_train_epochs: {num_train_epochs}")
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradiant_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=10,
        save_strategy="epoch"
    )

    print("Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collector
    )

    trainer.train()

    print("Saving Trainer Model...")
    out_dir = cfg["output_dir"]
    model_merged = model.merge_and_unload()
    model_merged.save_pretrained(out_dir)
    
    print("Saving Tokenizer...")
    tokenizer.save_pretrained(out_dir)
    
    

def make_tokenize(tokenizer):
    # tokenizer.pad_token = tokenizer.eos_token
    def tokenize_wrapper(example):
        prompt = example["prompt"]
        completion = example["completion"]
        input_text = f"prompt: {prompt} completes: {completion}"
        tokens = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask
        }
    return tokenize_wrapper     

main()
