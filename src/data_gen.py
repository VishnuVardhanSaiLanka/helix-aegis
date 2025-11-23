import requests
import json
import time
from Bio import SeqIO
from io import StringIO

def fetch_uniprot_data(query, limit=50):
    """
    Fetches data from UniProt based on a query.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "fasta",
        "size": limit
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return []
    
    sequences = []
    for record in SeqIO.parse(StringIO(response.text), "fasta"):
        sequences.append({
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq)
        })
    print(f"Found {len(sequences)} sequences for query: {query}")
    return sequences

def generate_dataset(output_file="dataset.json"):
    print("Fetching Unsafe Examples (Toxins)...")
    # BS1: Regulated Toxins
    unsafe_query = "toxin AND reviewed:true"
    unsafe_data = fetch_uniprot_data(unsafe_query, limit=100)
    if not unsafe_data:
        print("No unsafe data found. Check query.")
    
    print("Fetching Safe Examples (Housekeeping)...")
    # Safe: Housekeeping genes
    safe_query = "polymerase AND reviewed:true"
    safe_data = fetch_uniprot_data(safe_query, limit=100)
    if not safe_data:
        print("No safe data found. Check query.")
    
    dataset = []
    
    for item in unsafe_data:
        dataset.append({
            "sequence": item['sequence'],
            "label": "unsafe\nBS1",
            "metadata": {"id": item['id'], "desc": item['description'], "type": "unsafe"}
        })
        
    for item in safe_data:
        dataset.append({
            "sequence": item['sequence'],
            "label": "safe",
            "metadata": {"id": item['id'], "desc": item['description'], "type": "safe"}
        })
        
    print(f"Generated {len(dataset)} examples.")
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
