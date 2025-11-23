# Helix-Aegis: AI-Powered Biosecurity Screening

Helix-Aegis is a hackathon project designed to detect hazardous biological sequences (toxins, pathogens, virulence factors) using a fine-tuned Large Language Model (LLM). It addresses the "Common Mechanism" gap by using a model that understands protein semantics, allowing it to detect obfuscated or fragmented threats that traditional BLAST searches might miss.

## Features
- **"Describe to Detect"**: Fine-tunes `protein2text` to classify safety instead of just describing proteins.
- **6-Frame Translation**: Automatically translates input DNA into all 6 reading frames to detect hidden coding regions.
- **Bio-Safety Taxonomy**: Classifies threats into NIST/IGSC aligned categories (BS1-BS4).

## Taxonomy
| Code | Category | Description |
|------|----------|-------------|
| **BS1** | Regulated Toxins | Functional protein toxins (e.g., Ricin, Botulinum). |
| **BS2** | Select Agents (Viral) | Sequences unique to regulated viral pathogens (e.g., Ebola). |
| **BS3** | Virulence Factors | Proteins that enable a pathogen to harm a host. |
| **BS4** | Hazardous Function | Proteins with inherent hazardous activity (e.g., drug resistance). |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Generation
Fetch training data from UniProt (requires internet connection):
```bash
python src/data_gen.py
```
This will create `dataset.json`.

### 2. Training
Fine-tune the model (requires GPU):
```bash
python src/train.py
```
This will save the adapter to `Llama-HelixAegis-Final`.

### 3. Inference (Screening)
Screen a DNA sequence:
```bash
python src/inference.py --dna "ATGCGT..."
```
Or from a file:
```bash
python src/inference.py --file sequence.fasta
```

## Model
We use [tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M](https://huggingface.co/tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M) as the base model, which combines ESM-2 (protein encoder) with Llama 3.1 (text decoder).
