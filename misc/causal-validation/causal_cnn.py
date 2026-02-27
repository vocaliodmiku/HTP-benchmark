import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from models import CausalCNNModel
 


# Helper function to test the causal property
def verify_causality(model, batch_size=2, seq_len=100, n_mel=80):
    """
    Verify that the model is truly causal by checking that
    outputs don't change when future inputs are modified.
    
    Args:
        model: The causal CNN model to test
        batch_size: Batch size for test
        seq_len: Sequence length for test
        n_mel: Number of mel frequency bins
    
    Returns:
        Boolean indicating if causality holds
    """
    model.eval()
    
    # Create two identical sequences
    x1 = torch.randn(batch_size, seq_len, n_mel)
    x2 = x1.clone()
    
    # Modify future part of the second sequence
    # (from time step seq_len//2 to the end)
    modify_start = seq_len // 2
    x2[:, modify_start:, :] = torch.randn(batch_size, seq_len - modify_start, n_mel)
    
    # Get outputs
    with torch.no_grad():
        out1 = model((x1,), None)
        out2 = model((x2,), None)
    
    # Check that outputs are identical up to the modification point
    # (allowing for small numerical differences)
    output_key = "word"  # or "phone_out" depending on what you want to check
    tolerance = 1e-5
    
    for t in range(modify_start):
        if not torch.allclose(out1[output_key][:, t, :], 
                            out2[output_key][:, t, :], 
                            atol=tolerance):
            print(f"Causality violation at time step {t}")
            return False
    
    print("Causality test passed!")
    return True


# Example usage
if __name__ == "__main__":
    # Mock config class for testing
    class Config:
        def __init__(self):
            self.inp_size = 256
            self.cnn_out_size = 256
            self.rnn_drop = 0.2
            self.num_phonemes = 40
            self.pretrained_word_embeddings_dim = 300
    
    # Create model instance
    config = Config()
    model = CausalCNNModel(config)
    
    # Print model architecture
    print("Causal CNN Model Architecture:")
    print(model)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 150
    test_input = torch.randn(batch_size, seq_len, 256)
    
    output = model((test_input,), torch.tensor([seq_len]*batch_size))
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if value is not None:
            print(f"{key}: {value.shape}")
    
    # Verify causality
    print("\nVerifying causality...")
    verify_causality(model, batch_size=2, seq_len=100, n_mel=256)