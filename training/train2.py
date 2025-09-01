from unsloth import get_chat_template
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import yaml
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig


def main():
    

    login(token=os.environ['HF_TOKEN'])

    config_file = "training/config.yml"
    cfg = yaml.safe_load(open(config_file, encoding='utf-8'))

    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], legacy=False)
    chat_template = get_chat_template(tokenizer, chat_template = cfg["chat_template"])

    print(f"Loading dataset...")
    dataset = (
        load_dataset("json", data_files=cfg["train_file"])["train"]
            .map(generate_conversation, batched = True)
            .map(make_formatting_prompts_func(chat_template), batched = True)
    )

    print(f"Using model: {cfg['model_name']}")
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="cuda")
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

    print("Creating Training Args...")
    learning_rate = float(cfg["learning_rate"])
    batch_size = cfg["batch_size"]
    gradiant_accumulation_steps = cfg["gradient_accumulation_steps"]
    num_train_epochs = cfg["num_train_epochs"]
    print(f"\t learning_rate: {learning_rate}")
    print(f"\t batch_size: {batch_size}")
    print(f"\t gradiant_accumulation_steps: {gradiant_accumulation_steps}")
    print(f"\t num_train_epochs: {num_train_epochs}")
    
    print("Training...")
    trainer = SFTTrainer(
        dataset_num_proc=1,
        model = model,
        tokenizer = chat_template,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_num_proc=1,
            dataset_text_field = "text",
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradiant_accumulation_steps, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs=num_train_epochs,
            max_steps = 60,
            learning_rate=learning_rate,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer.train()

    print("Saving Trainer Model...")
    out_dir = cfg["output_dir"]
    model_merged = model.merge_and_unload()
    model_merged.save_pretrained(out_dir)
    
    print("Saving Tokenizer...")
    tokenizer.save_pretrained(out_dir)
    

def make_formatting_prompts_func(chat_template):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [chat_template.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    return formatting_prompts_func

def generate_conversation(examples):
    prompts  = examples["prompt"]
    completions = examples["completion"]
    conversations = []
    for prompt, completion in zip(prompts, completions):
        conversations.append([
            {"role" : "user",      "content" : prompt},
            {"role" : "assistant", "content" : completion},
        ])
    return { "conversations": conversations, }

main()
