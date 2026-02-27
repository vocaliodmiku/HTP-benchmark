#!/usr/bin/env python
"""
Test script for all model architectures.
Tests instantiation, forward passes, and counts parameters for each model.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models import (
    baselineModel,
    LSTMModel,
    CausalCNNModel,
    CausalRCNNModel,
    CausalTransformerModel,
    CausalConvTransformerModel,
    BiLSTMModel,
    BibaselineModel,
    CNNModel,
    RCNNModel,
    TransformerModel,
    ConvTransformerModel
)
 
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model): 
    return sum(p.numel() for p in model.parameters())

def test_model(model, model_name, batch_size=4, seq_len=100):
    # Instantiate model
    model.eval()
        
    # Create dummy input
    mel_spec = torch.randn(batch_size, seq_len, 256)
    language = torch.zeros(batch_size, dtype=torch.long)
    data = [mel_spec, language]
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    if torch.cuda.is_available():
        model.cuda()
        data = [d.cuda() for d in data]
        lengths = lengths.cuda()
        
    # Forward pass
    with torch.no_grad():
        output = model.forward(data, lengths, training=False)
    
    # Count parameters
    trainable_params = count_parameters(model)
    total_params = count_total_parameters(model)
    
    # Get output shapes
    output_info = {}
    for key, value in output.items():
        if value is not None and isinstance(value, torch.Tensor):
            output_info[key] = tuple(value.shape)
    
    result = {
        'name': model_name,
        'status': 'SUCCESS',
        'trainable_params': trainable_params,
        'total_params': total_params,
        'output_shapes': output_info,
        'error': None,
        'architecture': str(model)
    }  
    return result


def main():
    """Run tests for all models."""
    print("=" * 80)
    print("MODEL ARCHITECTURE TEST SUITE")
    print("=" * 80)
    print() 
    # Baseline model configuration
    class Config:
        inp_size = 256
        rnn_hidden_size = 512
        pretrained_word_embeddings_dim = 300
    baselinemodel = baselineModel(Config)
    bibaselinemodel = BibaselineModel(Config)
    
    class Config:
        inp_size = 256
        rnn_hidden_size = 320
        pretrained_word_embeddings_dim = 300
        
    lstmmodel = LSTMModel(Config)
    bilstmmodel = BiLSTMModel(Config)
    
    class Config:
        inp_size = 256
        pretrained_word_embeddings_dim = 300
    causalcnnmodel = CausalCNNModel(Config)
    cnnmodel = CNNModel(Config)
    
    class Config:
        inp_size = 256
        cnn_out_size = 256
        rnn_hidden_size = 256
        pretrained_word_embeddings_dim = 300
    causalrcnnmodel = CausalRCNNModel(Config)
    rcnnmodel = RCNNModel(Config)
    
    class Config:
        inp_size = 256
        rnn_hidden_size = 256
        pretrained_word_embeddings_dim = 300
    causaltransmodel = CausalTransformerModel(Config)
    transformermodel = TransformerModel(Config)
    causelconvtransformermodel = CausalConvTransformerModel(Config)
    convtransformermodel = ConvTransformerModel(Config)
    
    # Define all models to test
    models_to_test = [
        (baselinemodel, "baselineModel"),
        (lstmmodel, "LSTMModel"),
        (causalrcnnmodel, "CausalRCNNModel"),
        (causalcnnmodel, "CausalCNNModel"),
        # (TDNNModel, "TDNNModel"),
        (causaltransmodel, "CausalTransformerModel"),
        (causelconvtransformermodel, "CausalConvTransformerModel"),
        (bilstmmodel, "BiLSTMModel"),
        (bibaselinemodel, "BibaselineModel"),
        (cnnmodel, "CNNModel"),
        (rcnnmodel, "RCNNModel"),
        (transformermodel, "TransformerModel"),
        (convtransformermodel, "ConvTransformerModel")
    ]
    
    results = []
    
    print("Testing models...")
    print("-" * 80)
    
    for model, model_name in models_to_test:
        print(f"Testing {model_name}...", end=" ")
        result = test_model(model, model_name)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"✓ OK")
        else:
            print(f"✗ FAILED")
    
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Print detailed results
    total_params_all = 0
    trainable_params_all = 0
    
    for result in results:
        print(f"Model: {result['name']}")
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'SUCCESS':
            print(f"  Trainable Parameters: {result['trainable_params']:,}")
            print(f"  Total Parameters: {result['total_params']:,}")
            print(f"  Output Shapes:")
            for key, shape in result['output_shapes'].items():
                print(f"    - {key}: {shape}")
            print(f"  Architecture:")
            architecture_lines = result['architecture'].split('\n')
            for line in architecture_lines:  # Print first 30 lines of architecture
                print(f"    {line}")
            
            total_params_all += result['total_params']
            trainable_params_all += result['trainable_params']
        else:
            print(f"  Error: {result['error']}")
        
        print()
    
    # Print summary table
    print("=" * 80)
    print("PARAMETER COUNT COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Model Name':<25} {'Trainable Params':>20} {'Total Params':>20}")
    print("-" * 65)
    
    for result in results:
        if result['status'] == 'SUCCESS':
            print(f"{result['name']:<25} {result['trainable_params']:>20,} {result['total_params']:>20,}")
    
    print("-" * 65)
    print(f"{'TOTAL':<25} {trainable_params_all:>20,} {total_params_all:>20,}")
    print()
    
    # Print statistics
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total Models Tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if successful > 0:
        avg_params = trainable_params_all / successful
        min_params = min([r['trainable_params'] for r in results if r['status'] == 'SUCCESS'])
        max_params = max([r['trainable_params'] for r in results if r['status'] == 'SUCCESS'])
        
        print(f"Average Trainable Parameters: {avg_params:,.0f}")
        print(f"Min Parameters: {min_params:,}")
        print(f"Max Parameters: {max_params:,}")
        print(f"Difference: {max_params - min_params:,}")
    
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
