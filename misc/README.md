# Miscellaneous Scripts

This directory contains utility scripts for testing and validating models in the `earshot_nn` project.

## Model Parameter & Testing

### `print_model_parameters.py`
This script tests the instantiation and forward pass of all model architectures defined in `src/models.py`. It also prints the number of trainable and total parameters for each model.

**Usage:**
```bash
python misc/print_model_parameters.py
```

## Causal Validation

The `causal-validation` directory contains scripts to verify that "causal" models indeed respect causality (i.e., outputs at time $t$ do not depend on inputs at time $>t$). This is crucial for streaming applications or certain types of sequence modeling.

### `causal-validation/causal_cnn.py`
Validates the causality of the `CausalCNNModel`. It checks if modifying future inputs affects current outputs.

**Usage:**
```bash
python misc/causal-validation/causal_cnn.py
```

### `causal-validation/causal_rcnn.py`
Validates the causality of the `CausalRCNNModel` (Recurrent CNN).

**Usage:**
```bash
python misc/causal-validation/causal_rcnn.py
```

### `causal-validation/causal_tran.py`
Validates the causality of the `CausalTransformerModel`. It includes visualization of attention leaks/causality violations if any.

**Usage:**
```bash
python misc/causal-validation/causal_tran.py
```

### `causal-validation/causal_ctran.py`
Validates the causality of the `CausalConvTransformerModel`.

**Usage:**
```bash
python misc/causal-validation/causal_ctran.py
```

## How it works

The validation scripts typically work by:
1. Creating two identical input sequences.
2. Modifying the second sequence starting from a certain time step $T$.
3. Running both sequences through the model.
4. Verifying that the outputs are identical up to time step $T$ (within numerical tolerance).
5. Checking gradient flow to ensure gradients do not flow from future to past.
