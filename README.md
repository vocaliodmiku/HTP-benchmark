# Human-like Incremental Dynamics in ASR Models

## Overview
This project investigates whether Automatic Speech Recognition (ASR) models process speech with human-like incremental dynamics. We propose a benchmark comparing internal ASR activation trajectories against human eyetracking data to evaluate lexical activation and phonological competition.

Our findings reveal that **causal models** successfully replicate human-like temporal dynamics (early onset competition, later rhyme competition), whereas **non-causal models** with 'look-ahead' mechanisms fail. This demonstrates that architectural constraints are critical for developing speech systems that process language in a human-like manner.

## Repository Structure

- `src/`: Source code for model architectures, training loops, and data processing.
- `data/` & `dataset/`: Datasets including phonemes, words, and train/test splits for different languages (English, Spanish, Basque).
- `experiments/`: Configuration files for various model architectures (causal vs. non-causal, CNN, RNN, Transformer).
- `analysis/`: Scripts for analyzing phonological competition and plotting results.
- `notebooks/`: Jupyter notebooks for generating figures and detailed case studies.
- `pretrained_models/`: Evaluation scripts for state-of-the-art ASR models (Whisper, Wav2Vec2, Nemotron).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train models:
   ```bash
   ./train.sh
   ```

3. Evaluate models:
   ```bash
   ./test.sh
   ```
