import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils import translate_dna_to_protein
import argparse

BIO_TAXONOMY = """
<BEGIN UNSAFE CONTENT CATEGORIES>
BS1: Regulated Toxins. 
BS2: Pathogens of Pandemic Potential. 
BS3: Virulence Factors. 
BS4: Antimicrobial Resistance. 
<END UNSAFE CONTENT CATEGORIES>
"""

def load_model(base_model_id, adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return model, tokenizer

def screen_sequence(model, tokenizer, protein_seq):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Task: Screen the following protein sequence for biosecurity risks according to the taxonomy below.

{BIO_TAXONOMY}

<BEGIN SEQUENCE>
{protein_seq}
<END SEQUENCE>

Provide your safety assessment:
- First line: 'safe' or 'unsafe'
- Second line: Violated categories (if unsafe)<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant's response part
    response_body = response.split("assistant")[-1].strip()
    return response_body

def main():
    parser = argparse.ArgumentParser(description="Helix-Aegis Screening Tool")
    parser.add_argument("--dna", type=str, help="DNA sequence to screen")
    parser.add_argument("--file", type=str, help="File containing DNA sequence")
    parser.add_argument("--model_path", type=str, default="Llama-HelixAegis-Final", help="Path to fine-tuned adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model ID")
    
    args = parser.parse_args()
    
    dna_seq = ""
    if args.dna:
        dna_seq = args.dna
    elif args.file:
        with open(args.file, "r") as f:
            dna_seq = f.read().strip()
    else:
        print("Please provide a DNA sequence via --dna or --file")
        return

    model, tokenizer = load_model(args.base_model, args.model_path)
    
    print("\nTranslating DNA to 6 frames...")
    frames = translate_dna_to_protein(dna_seq)
    
    unsafe_found = False
    results = {}
    
    for frame_name, protein_seq in frames.items():
        print(f"Screening {frame_name}...")
        result = screen_sequence(model, tokenizer, protein_seq)
        results[frame_name] = result
        if "unsafe" in result.lower():
            unsafe_found = True
            
    print("\n=== SCREENING REPORT ===")
    if unsafe_found:
        print("🚨 UNSAFE CONTENT DETECTED 🚨")
        for name, res in results.items():
            if "unsafe" in res.lower():
                print(f"[{name}] {res}")
    else:
        print("✅ No threats detected in any reading frame.")

if __name__ == "__main__":
    main()
