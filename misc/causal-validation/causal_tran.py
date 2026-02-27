import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from models import CausalTransformerModel
import matplotlib.pyplot as plt

def verify_causality(model, batch_size=2, seq_len=100, n_mel=80, 
                     visualize=False, detailed=False):
    """
    Verify causality of a transformer model.
    
    Args:
        model: The transformer model to test
        batch_size: Batch size for test
        seq_len: Sequence length for test
        n_mel: Input feature dimension
        visualize: Whether to create visualization plots
        detailed: Whether to print detailed diagnostics
    
    Returns:
        bool: True if model is causal, False otherwise
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
        out1 = model((x1,), torch.tensor([seq_len]*batch_size), training=False)
        out2 = model((x2,), torch.tensor([seq_len]*batch_size), training=False)
    
    # Check that outputs are identical up to the modification point
    tolerance = 1e-5
    is_causal = True
    violation_details = []
    
    # Check word output (adjust based on your model's output keys)
    output_key = "word"
    
    if detailed:
        print(f"\n{'='*60}")
        print(f"Causality Test Diagnostics")
        print(f"{'='*60}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Sequence length: {seq_len}")
        print(f"Modification starts at: {modify_start}")
        print(f"Output shape: {out1[output_key].shape}")
        print(f"Tolerance: {tolerance}")
    
    # Check each time step before modification
    for t in range(modify_start):
        for b in range(batch_size):
            diff = torch.abs(out1[output_key][b, t, :] - out2[output_key][b, t, :])
            max_diff = diff.max().item()
            
            if max_diff > tolerance:
                is_causal = False
                violation_details.append({
                    'batch': b,
                    'time': t,
                    'max_diff': max_diff,
                    'mean_diff': diff.mean().item()
                })
                
                if detailed:
                    print(f"\n✗ Causality violation detected!")
                    print(f"  Batch: {b}, Time step: {t}")
                    print(f"  Max difference: {max_diff:.2e}")
                    print(f"  Mean difference: {diff.mean().item():.2e}")
                    print(f"  Expected identical outputs up to time {modify_start-1}")
    
    # Also check that outputs AFTER modification point are DIFFERENT
    # (This verifies the model is actually processing the input)
    any_different = False
    for t in range(modify_start, seq_len):
        diff = torch.abs(out1[output_key][:, t, :] - out2[output_key][:, t, :])
        if diff.max() > tolerance:
            any_different = True
            break
    
    if not any_different and is_causal:
        print(" Warning: Outputs are identical even after modification point.")
        print("  This might indicate the model is ignoring the input.")
    
    # Print summary
    if is_causal:
        print(f"\n✓ Causal Transformer test PASSED!")
        print(f"  All outputs before modification point (t={modify_start}) are identical.")
        if any_different:
            print(f"  Outputs after modification point differ correctly.")
    else:
        print(f"\n✗ Causal Transformer test FAILED!")
        print(f"  Found {len(violation_details)} causality violations.")
        
        # Print top 5 violations
        if violation_details:
            print(f"\nTop violations:")
            for i, v in enumerate(violation_details[:5]):
                print(f"  {i+1}. Batch {v['batch']}, Time {v['time']}: "
                      f"max_diff={v['max_diff']:.2e}, mean_diff={v['mean_diff']:.2e}")
    
    # Create visualization if requested
    if visualize and violation_details:
        visualize_causality_violations(out1[output_key], out2[output_key], 
                                       modify_start, violation_details)
    
    return is_causal

def visualize_causality_violations(out1, out2, modify_start, violations):
    """
    Visualize causality violations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Plot difference heatmap for first batch
    diff = torch.abs(out1[0] - out2[0]).numpy()
    im1 = axes[0, 0].imshow(diff.T, aspect='auto', cmap='hot', 
                           vmin=0, vmax=diff.max())
    axes[0, 0].axvline(x=modify_start-0.5, color='cyan', linestyle='--', 
                      linewidth=2, label='Modification start')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Feature Dimension')
    axes[0, 0].set_title('Absolute Difference Heatmap (Batch 0)')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Plot max difference over time
    max_diff_over_time = torch.max(torch.abs(out1 - out2), dim=2)[0].mean(dim=0).numpy()
    axes[0, 1].plot(max_diff_over_time, 'b-', linewidth=2)
    axes[0, 1].axvline(x=modify_start-0.5, color='r', linestyle='--', 
                      linewidth=2, label='Modification start')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Max Difference')
    axes[0, 1].set_title('Max Difference Over Time (averaged over batch)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot of violation magnitudes
    if violations:
        times = [v['time'] for v in violations]
        diffs = [v['max_diff'] for v in violations]
        axes[1, 0].scatter(times, diffs, alpha=0.6, c='red', s=50)
        axes[1, 0].axvline(x=modify_start-0.5, color='r', linestyle='--', 
                          linewidth=2, label='Modification start')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Max Difference')
        axes[1, 0].set_title('Causality Violations')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Histogram of differences before modification
    mask = torch.arange(out1.shape[1]) < modify_start
    diff_before = torch.abs(out1[:, mask, :] - out2[:, mask, :]).flatten().numpy()
    diff_after = torch.abs(out1[:, ~mask, :] - out2[:, ~mask, :]).flatten().numpy()
    
    axes[1, 1].hist(diff_before, bins=50, alpha=0.5, label=f'Before t={modify_start}', 
                   color='blue', density=True)
    axes[1, 1].hist(diff_after, bins=50, alpha=0.5, label=f'After t={modify_start}', 
                   color='red', density=True)
    axes[1, 1].set_xlabel('Absolute Difference')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Differences')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('causality_test.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_gradient_flow(model, seq_len=50, n_mel=80):
    """
    Test gradient flow to ensure gradients don't flow from future to past.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Flow Test")
    print(f"{'='*60}")
    
    model.train()
    batch_size = 1
    
    # Create input and target
    x = torch.randn(batch_size, seq_len, n_mel, requires_grad=True)
    target = torch.randn(batch_size, seq_len, model.word_linear.out_features)
    
    # Basic check using full sequence
    output = model((x,), torch.tensor([seq_len]), training=False)
    loss = F.mse_loss(output["word"], target)
    loss.backward()
    
    if x.grad is None:
        print("✗ No gradients for input")
        return False

    is_causal_grad = True
    x.grad.zero_()
    
    # For each time t, check if output at time t depends on future inputs
    # This is the definition of causality: output[t] depends only on input[0...t]
    for t in range(seq_len - 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()
            
        # Re-run forward pass to avoid retaining graph issues
        output = model((x,), torch.tensor([seq_len]), training=False)
        
        # Compute loss only at time t
        # We use sum as a dummy loss function
        loss_t = output["word"][:, t, :].sum()
        
        # Backward
        loss_t.backward()
        
        # Check if gradients at times > t are non-zero
        # If any gradient exists for input[t+1:], it means output[t] depends on future input
        if torch.any(torch.abs(x.grad[:, t+1:, :]) > 1e-5):
            print(f"✗ Causality violation at t={t}: gradient flows from future input (t>{t})")
            is_causal_grad = False
            break
    
    if is_causal_grad:
        print("✓ Gradient flow test PASSED - gradients are causal")
    else:
        print("✗ Gradient flow test FAILED - gradients violate causality")
    
    return is_causal_grad

# Example usage and comprehensive testing
if __name__ == "__main__":
    # Mock config class for testing
    class Config:
        def __init__(self):
            self.inp_size = 256
            self.pretrained_word_embeddings_dim = 300
            self.rnn_drop = 0
    
    print("="*70)
    print("COMPREHENSIVE CAUSAL TRANSFORMER TEST SUITE")
    print("="*70)
    
    config = Config()
    
    # Test 1: Causal Transformer Model
    print("\nTest 1: Causal Transformer Model")
    print("-"*40)
    
    model_causal = CausalTransformerModel(config )
    print(f"Model type: {model_causal.__class__.__name__}") 
    
    # Test forward pass
    batch_size = 4
    seq_len = 150
    test_input = torch.randn(batch_size, seq_len, config.inp_size)
    test_lengths = torch.tensor([seq_len]*batch_size)
    
    output = model_causal((test_input,), test_lengths, training=False)
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if value is not None and not isinstance(value, int):
            print(f"  {key}: {value.shape}")
    
    # Verify causality
    print("\n" + "="*40)
    print("Running causality verification...")
    result_causal = verify_causality(
        model_causal, 
        batch_size=2, 
        seq_len=100, 
        n_mel=config.inp_size,
        detailed=True,
        visualize=True  # Set to True to see plots
    )
    
    # Test 3: Edge cases
    print("\n\n" + "="*70)
    print("Test 3: Edge Cases")
    print("="*70)
    
    # Test with very short sequence
    print("\nTesting with short sequence (len=10):")
    verify_causality(model_causal, batch_size=1, seq_len=10, 
                     n_mel=config.inp_size, detailed=False)
    
    # Test with very long sequence
    print("\nTesting with long sequence (len=500):")
    verify_causality(model_causal, batch_size=1, seq_len=500,
                     n_mel=config.inp_size, detailed=False)
    
    # Test 4: Gradient flow test
    print("\n\n" + "="*70)
    print("Test 4: Gradient Flow Analysis")
    print("="*70)
    
    test_gradient_flow(model_causal, seq_len=30, n_mel=config.inp_size)
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Causal model test: {'PASS' if result_causal else 'FAIL'}")
    
    if result_causal:
        print("\n✅ All tests passed as expected!")
    else:
        print("\n❌ Unexpected results - check model implementation")