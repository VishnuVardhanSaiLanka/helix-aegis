import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from trl import SFTTrainer
import json
import os

# 1. DEFINE YOUR HELIX-AEGIS TAXONOMY
BIO_TAXONOMY = """
<BEGIN UNSAFE CONTENT CATEGORIES>
BS1: Regulated Toxins. 
Proteins that are known toxins or derived from Select Agent toxins (e.g., Botulinum, Ricin).
BS2: Pathogens of Pandemic Potential. 
Sequences derived from regulated viruses (e.g., Ebola, Smallpox) or bacteria.
BS3: Virulence Factors. 
Proteins that significantly enhance pathogenicity (e.g., Hemolysins, Adhesins).
BS4: Antimicrobial Resistance. 
Sequences conferring resistance to critical antibiotics (e.g., beta-lactamases).
<END UNSAFE CONTENT CATEGORIES>
"""

# 2. FORMATTING FUNCTION
def biosecurity_formatting_func(examples):
    output_texts = []
    
    for i in range(len(examples['sequence'])):
        protein_seq = examples['sequence'][i]
        label = examples['label'][i]
        
        text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Task: Screen the following protein sequence for biosecurity risks according to the taxonomy below.

{BIO_TAXONOMY}

<BEGIN SEQUENCE>
{protein_seq}
<END SEQUENCE>

Provide your safety assessment:
- First line: 'safe' or 'unsafe'
- Second line: Violated categories (if unsafe)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>"""
        output_texts.append(text)
        
    return output_texts

def train():
    # Load dataset
    if not os.path.exists("dataset.json"):
        print("dataset.json not found. Run src/data_gen.py first.")
        return

    with open("dataset.json", "r") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)

    # Load Model
    model_id = "tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading model: {model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Falling back to Meta-Llama-3-8B-Instruct")
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        formatting_func=biosecurity_formatting_func,
        args=TrainingArguments(
            output_dir="Llama-HelixAegis",
            per_device_train_batch_size=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
            save_steps=25
        ),
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")
    trainer.save_model("Llama-HelixAegis-Final")

if __name__ == "__main__":
    train()
