from Bio.Seq import Seq

def translate_dna_to_protein(dna_seq):
    """
    Translates a DNA sequence into 6 protein reading frames (3 forward, 3 reverse).
    """
    dna_seq = dna_seq.upper().replace("\n", "").replace(" ", "")
    seq_obj = Seq(dna_seq)
    
    frames = {}
    
    # Forward frames
    for i in range(3):
        frames[f"Forward_{i+1}"] = str(seq_obj[i:].translate())
        
    # Reverse frames
    rev_seq = seq_obj.reverse_complement()
    for i in range(3):
        frames[f"Reverse_{i+1}"] = str(rev_seq[i:].translate())
        
    return frames

if __name__ == "__main__":
    # Test with a dummy sequence
    dummy_dna = "ATGCGTACGTTAGCTAGCTAG"
    print(f"Input DNA: {dummy_dna}")
    frames = translate_dna_to_protein(dummy_dna)
    for name, seq in frames.items():
        print(f"{name}: {seq}")
