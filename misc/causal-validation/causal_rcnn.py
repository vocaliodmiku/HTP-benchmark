import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from models import CausalConv1dWrapper

class CausalRCNNModel(nn.Module):
    """
    Alternative implementation using the CausalConv1dWrapper from previous implementation.
    This provides cleaner code by encapsulating the causal logic in the wrapper.
    """
    def __init__(self, config):
        super(CausalRCNNModel, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cnn_out_size = config.cnn_out_size
        self.rnn_hidden_size = config.rnn_hidden_size
 
        # Causal convolutional layers using wrapper
        # Causal convolutional layers using wrapper
        self.convnet = nn.Sequential(
            CausalConv1dWrapper(256, 64, kernel_size=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 64, kernel_size=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, config.cnn_out_size, kernel_size=7, bias=False),
            nn.BatchNorm1d(config.cnn_out_size),
            nn.ReLU(inplace=True),
        ) 
        
        # Recurrent layer - unidirectional for causality 
        self.rnn = nn.LSTM(
            input_size=config.cnn_out_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output dimension is just hidden size for unidirectional RNN
        self.out_dim = config.rnn_hidden_size 
        # Output layers 
        self.word_linear = nn.Linear(self.out_dim, config.pretrained_word_embeddings_dim) 
        
    def forward(self, data, lengths, h0=None, training=True):
        # Input shape: (batch, time, freq)
        x = data[0].transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply causal convolutions (trimming handled in wrapper)
        cnn_out = self.convnet(x)
        cnn_out = cnn_out.transpose(1, 2)  # -> (batch, time, channels)
        
        # Apply RNN - unidirectional for causality
        rnn_out, (hidden, cell) = self.rnn(cnn_out, h0) 
        
        # Generate outputs
        word_out = self.word_linear(rnn_out)
        
        return {
            "cnn_out": cnn_out,
            "rnn_out": rnn_out,
            "word": word_out, 
        }


class CausalRCNNModelWithDilation(nn.Module):
    """
    Advanced version with dilated causal convolutions for larger receptive field.
    """
    def __init__(self, config):
        super(CausalRCNNModelWithDilation, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cnn_out_size = config.cnn_out_size
        self.rnn_hidden_size = config.rnn_hidden_size
 
        # Causal convolutional layers with increasing dilation
        self.convnet = nn.Sequential(
            # Layer 1: kernel_size=3, dilation=1
            CausalConv1dWrapper(256, 64, kernel_size=3, dilation=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2: kernel_size=3, dilation=2
            CausalConv1dWrapper(64, 128, kernel_size=3, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: kernel_size=3, dilation=4
            CausalConv1dWrapper(128, config.cnn_out_size, kernel_size=3, dilation=4, bias=False),
            nn.BatchNorm1d(config.cnn_out_size),
            nn.ReLU(inplace=True),
        )
        
        # Calculate receptive field
        # With dilation [1, 2, 4] and kernel_size=3:
        # Layer 1: receptive field = 3
        # Layer 2: receptive field = 3 + (3-1)*2 = 7
        # Layer 3: receptive field = 7 + (3-1)*4 = 15
        self.receptive_field = 15
        
        # Recurrent layer 
        self.rnn = nn.LSTM(
            input_size=config.cnn_out_size,
            hidden_size=config.rnn_hidden_size,
            batch_first=True
        )
        
        self.out_dim = config.rnn_hidden_size 
        
        # Output layers 
        self.word_linear = nn.Linear(self.out_dim, config.pretrained_word_embeddings_dim)
    
    def forward(self, data, lengths, h0=None, training=True):
        # Input shape: (batch, time, freq)
        x = data[0].transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply dilated causal convolutions
        cnn_out = self.convnet(x)
        cnn_out = cnn_out.transpose(1, 2)  # -> (batch, time, channels)
        
        # Apply RNN
        rnn_out, (hidden, cell) = self.rnn(cnn_out, h0) 
        
        # Generate outputs
        word_out = self.word_linear(rnn_out)
        
        return {
            "cnn_out": cnn_out,
            "rnn_out": rnn_out,
            "word": word_out, 
            "receptive_field": self.receptive_field
        }


# Utility functions for causal RCNN
def create_causal_rcnn(config, model_type="standard"):
    """
    Factory function to create different types of causal RCNN models.
    
    Args:
        config: Configuration object with model parameters
        model_type: Type of model to create
            - "standard": Basic causal RCNN
            - "wrapper": Using CausalConv1dWrapper
            - "dilated": With dilated convolutions
    
    Returns:
        Initialized causal RCNN model
    """
    if model_type == "standard":
        return CausalRCNNModel(config)
    elif model_type == "dilated":
        return CausalRCNNModelWithDilation(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def verify_rcnn_causality(model, batch_size=2, seq_len=100, n_mel=80):
    """
    Verify that the RCNN model is truly causal.
    
    Args:
        model: The causal RCNN model to test
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
    modify_start = seq_len // 2
    x2[:, modify_start:, :] = torch.randn(batch_size, seq_len - modify_start, n_mel)
    
    # Get outputs
    with torch.no_grad():
        out1 = model((x1,), torch.tensor([seq_len]*batch_size))
        out2 = model((x2,), torch.tensor([seq_len]*batch_size))
    
    # Check that outputs are identical up to the modification point
    tolerance = 1e-5
    
    # Check both CNN and RNN outputs
    for output_key in ["cnn_out", "rnn_out", "word"]:
        for t in range(modify_start):
            if not torch.allclose(out1[output_key][:, t, :], 
                                out2[output_key][:, t, :], 
                                atol=tolerance):
                print(f"Causality violation in {output_key} at time step {t}")
                return False
 
    print("Causal RCNN test passed!")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Mock config class for testing
    class Config:
        def __init__(self):
            self.inp_size = 256
            self.cnn_out_size = 256
            self.rnn_drop = 0.2
            self.num_phonemes = 40
            self.rnn_hidden_size = 256
            self.pretrained_word_embeddings_dim = 300
    
    # Test 1: Standard causal RCNN
    print("Testing Standard Causal RCNN Model:")
    config = Config()
    model = CausalRCNNModel(config)
    
    # Print model architecture
    print(f"Model type: {model.__class__.__name__}")
    print(f"Receptive field: ~15 time steps (3 layers with kernel_size=3)")
    print(f"RNN bidirectional: {model.rnn.bidirectional}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 150
    test_input = torch.randn(batch_size, seq_len, 256)
    test_lengths = torch.tensor([seq_len]*batch_size)
    
    output = model((test_input,), test_lengths)
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if value is not None and not isinstance(value, int):
            print(f"{key}: {value.shape}")
    
    # Verify causality
    print("\nVerifying causality...")
    verify_rcnn_causality(model, batch_size=2, seq_len=100, n_mel=256) 
 
    print("\n\nTesting Causal RCNN with Dilation:")
    model_dilated = CausalRCNNModelWithDilation(config)
    print(f"Receptive field: {model_dilated.receptive_field} time steps")
    verify_rcnn_causality(model_dilated, batch_size=2, seq_len=100, n_mel=256) 