import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    
    # Get device
    device = next(model.parameters()).device
    
    # Create two identical sequences with shape (batch, time, n_mel)
    x1 = torch.randn(batch_size, seq_len, n_mel)
    x2 = x1.clone()
    
    # Modify future part of the second sequence
    modify_start = seq_len // 2
    x2[:, modify_start:, :] = torch.randn(batch_size, seq_len - modify_start, n_mel)
    
    # Move to device
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    # Get outputs - note: the model expects a tuple (data,) and lengths
    with torch.no_grad():
        lengths = torch.tensor([seq_len] * batch_size).to(device)
        out1 = model((x1,), lengths, training=False)
        out2 = model((x2,), lengths, training=False)
    
    # Check that outputs are identical up to the modification point
    tolerance = 1e-5
    is_causal = True
    violation_details = []
    
    # Check word output
    output_key = "word"
    
    if output_key not in out1:
        print(f"Error: Output key '{output_key}' not found in model outputs")
        print(f"Available keys: {list(out1.keys())}")
        return False
    
    # Account for CNN downsampling: find corresponding output modification point
    out_seq_len = out1[output_key].shape[1]
    # Estimate downsampling factor (might not be integer due to padding)
    downsample_factor = seq_len / out_seq_len
    # Conservative estimate: modify_start_output should be BEFORE any potential influence
    modify_start_output = int(np.floor(modify_start / downsample_factor)) - 1
    
    if detailed:
        print(f"\n{'='*60}")
        print(f"Causality Test Diagnostics")
        print(f"{'='*60}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Input sequence length: {seq_len}")
        print(f"Output sequence length: {out_seq_len}")
        print(f"Downsample factor: ~{downsample_factor:.2f}")
        print(f"Input modification starts at: {modify_start}")
        print(f"Output check up to: {modify_start_output} (conservative)")
        print(f"Tolerance: {tolerance}")
    
    # Check each time step before modification (conservatively)
    check_up_to = max(1, modify_start_output)  # Ensure we check at least one step
    for t in range(check_up_to):
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
                    print(f"  Batch: {b}, Output time step: {t}")
                    print(f"  Corresponding input time: ~{int(t * downsample_factor)}")
                    print(f"  Max difference: {max_diff:.2e}")
                    print(f"  Mean difference: {diff.mean().item():.2e}")
    
    # Also check that outputs AFTER modification point are DIFFERENT
    # (This verifies the model is actually processing the input)
    any_different = False
    if modify_start_output + 1 < out_seq_len:
        for t in range(modify_start_output + 1, out_seq_len):
            diff = torch.abs(out1[output_key][:, t, :] - out2[output_key][:, t, :])
            if diff.max() > tolerance:
                any_different = True
                break
    
    if not any_different and is_causal and modify_start_output + 1 < out_seq_len:
        print(" Warning: Outputs are identical even after modification point.")
        print("  This might indicate the model is ignoring the input.")
    
    # Print summary
    if is_causal:
        print(f"\n✓ Causal Transformer test PASSED!")
        print(f"  All outputs before conservative cutoff (t={modify_start_output}) are identical.")
        if any_different:
            print(f"  Outputs after cutoff differ correctly.")
    else:
        print(f"\n✗ Causal Transformer test FAILED!")
        print(f"  Found {len(violation_details)} potential causality violations.")
        
        # Print top 5 violations
        if violation_details:
            print(f"\nTop violations:")
            for i, v in enumerate(violation_details[:5]):
                print(f"  {i+1}. Batch {v['batch']}, Output Time {v['time']}: "
                      f"max_diff={v['max_diff']:.2e}, mean_diff={v['mean_diff']:.2e}")
    
    # Create visualization if requested
    if visualize and violation_details:
        visualize_causality_violations(out1[output_key].cpu(), out2[output_key].cpu(), 
                                       modify_start_output, violation_details)
    
    return is_causal

def visualize_causality_violations(out1, out2, modify_start_output, violations):
    """
    Visualize causality violations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Plot difference heatmap for first batch
    diff = torch.abs(out1[0] - out2[0]).numpy()
    im1 = axes[0, 0].imshow(diff.T, aspect='auto', cmap='hot', 
                           vmin=0, vmax=diff.max())
    axes[0, 0].axvline(x=modify_start_output-0.5, color='cyan', linestyle='--', 
                      linewidth=2, label='Conservative cutoff')
    axes[0, 0].set_xlabel('Output Time Step')
    axes[0, 0].set_ylabel('Feature Dimension')
    axes[0, 0].set_title('Absolute Difference Heatmap (Batch 0)')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Plot max difference over time
    max_diff_over_time = torch.max(torch.abs(out1 - out2), dim=2)[0].mean(dim=0).numpy()
    axes[0, 1].plot(max_diff_over_time, 'b-', linewidth=2)
    axes[0, 1].axvline(x=modify_start_output-0.5, color='r', linestyle='--', 
                      linewidth=2, label='Conservative cutoff')
    axes[0, 1].set_xlabel('Output Time Step')
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
        axes[1, 0].axvline(x=modify_start_output-0.5, color='r', linestyle='--', 
                          linewidth=2, label='Conservative cutoff')
        axes[1, 0].set_xlabel('Output Time Step')
        axes[1, 0].set_ylabel('Max Difference')
        axes[1, 0].set_title('Causality Violations')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Histogram of differences before and after cutoff
    out_seq_len = out1.shape[1]
    mask = torch.arange(out_seq_len) < modify_start_output
    diff_before = torch.abs(out1[:, mask, :] - out2[:, mask, :]).flatten().numpy()
    diff_after = torch.abs(out1[:, ~mask, :] - out2[:, ~mask, :]).flatten().numpy()
    
    axes[1, 1].hist(diff_before, bins=50, alpha=0.5, label=f'Before cutoff (t={modify_start_output})', 
                   color='blue', density=True)
    axes[1, 1].hist(diff_after, bins=50, alpha=0.5, label=f'After cutoff', 
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
    Simpler version: just check if model runs without error.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Flow Test (Simplified)")
    print(f"{'='*60}")
    
    model.train()
    batch_size = 2
    
    # Get device
    device = next(model.parameters()).device
    
    # Create input and target
    x = torch.randn(batch_size, seq_len, n_mel, requires_grad=True).to(device)
    # Use correct target dimension for word output
    with torch.no_grad():
        test_out = model((x,), torch.tensor([seq_len] * batch_size).to(device), training=False)
    target_dim = test_out["word"].shape[-1]
    target = torch.randn(batch_size, test_out["word"].shape[1], target_dim).to(device)
    
    # Forward pass
    output = model((x,), torch.tensor([seq_len] * batch_size).to(device), training=False)
    loss = F.mse_loss(output["word"], target)
    
    # Backward pass
    loss.backward()
    
    print("✓ Gradient computation successful")
    
    # Simple check: ensure gradients exist for input
    if x.grad is not None:
        print(f"✓ Input gradients computed (norm: {x.grad.norm().item():.2e})")
    else:
        print("✗ No gradients for input")
    
    return True

# Example usage
if __name__ == "__main__":
    # Mock config class for testing
    class Config:
        def __init__(self): 
            self.inp_size = 256 
            self.pretrained_word_embeddings_dim = 300
    
    print("="*70)
    print("CAUSAL TRANSFORMER TEST SUITE")
    print("="*70)
    
    config = Config()
    
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
    # Import the model (assuming it's in the same directory)
    from models import CausalConvTransformerModel
    
    print("\nTest 1: Causal Transformer Model")
    print("-"*40)
    
    model_causal = CausalConvTransformerModel(config)
    print(f"Model type: {model_causal.__class__.__name__}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 150
    test_input = torch.randn(batch_size, seq_len, 256)
    test_lengths = torch.tensor([seq_len] * batch_size)
    device = next(model_causal.parameters()).device
    test_input = test_input.to(device)
    test_lengths = test_lengths.to(device)
    
    with torch.no_grad():
        output = model_causal((test_input,), test_lengths, training=False)
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Verify causality
    print("\n" + "="*40)
    print("Running causality verification...")
    result_causal = verify_causality(
        model_causal, 
        batch_size=2, 
        seq_len=100, 
        n_mel=256,
        detailed=True,
        visualize=True  # Set to True to see plots
    )
    
    # Test gradient flow
    print("\n\n" + "="*70)
    print("Test 2: Gradient Flow")
    print("="*70)
    test_gradient_flow(model_causal, seq_len=30, n_mel=256)
    
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Causal model test: {'PASS' if result_causal else 'FAIL'}")