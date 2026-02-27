# HTP-benchmark: Human Temporal Phonological Benchmark for End-to-End ASR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper **"Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR"** (submitted to Interspeech 2025). This repository provides code and data to evaluate whether automatic speech recognition (ASR) models process speech incrementally with human-like lexical competition dynamics.

## Overview

Human speech recognition is incremental: listeners continuously activate and suppress competing word candidates as speech unfolds. This benchmark quantitatively compares the time course of lexical activation in ASR models against human eyetracking data from the Visual World Paradigm (VWP). We probe internal model states over time and measure activation profiles for target words, cohort competitors (same onset), rhyme competitors (different onset, same ending), and unrelated words. The resulting trajectories are compared to human fixation proportions using point-wise RMSE and MAE.

**Key finding:** Causal architectures (LSTM, causal CNN, causal RCNN) replicate the hallmark human pattern—early cohort competition followed by later rhyme activation—while non-causal models with look-ahead (BiLSTM, Transformer, ConvTransformer) and large pretrained ASR models (wav2vec 2.0, HuBERT, Whisper) fail to capture these temporal dynamics despite higher transcription accuracy.

## Repository Structure

```
earshot_nn/
├── data/                  # Dataset and human fixation data
│   ├── audio/             # (raw audio not included; see notes below)
│   ├── wordlists/         # Lists of target words with phonemic transcriptions and embeddings
│   └── human_fixations/   # Processed human VWP fixation proportions
├── models/                # Model definitions (causal/non-causal variants)
│   ├── lstm.py
│   ├── cnn.py
│   ├── rcnn.py
│   ├── transformer.py
│   └── pretrained/        # Wrappers for wav2vec2, HuBERT, Whisper
├── train.py               # Training script for models on isolated word task
├── evaluate.py            # Compute activation trajectories and compare to human data
├── evaluate_pretrained.py # Evaluation wrappers for pretrained models
├── reproduce_figures.py   # Aggregation script to regenerate paper figures
├── utils/                 # Data loading, metrics, phoneme decoding
├── configs/               # Hyperparameter configs
├── results/               # Output plots and metrics
└── README.md
```

## Dataset

We use a controlled lexicon of 1,533 uninflected English words (1–16 phonemes). Audio was recorded from six synthetic talkers (Apple "Say" app) and one human speaker, resulting in 7 × 1,533 = 10,731 utterances. Each word is paired with a centered 300‑dimensional word2vec embedding (trained on Google News) as the semantic target.

Due to licensing restrictions, we cannot distribute the raw audio files. The repository includes:

- The list of words (`data/wordlists/words.txt`) with phonemic transcriptions and embeddings (`data/wordlists/word2vec_centered.npy`).
- Scripts to generate synthetic audio using macOS `say` or another TTS system (`data/prepare_audio.sh`).
- Pre-extracted log-mel spectrograms (256 bins, 10 ms frames) for all utterances are available by separate download or upon request.

Human fixation data are derived from the Allopenna et al. (1998) study and processed into time-normalized proportions for target, cohort, rhyme, and unrelated conditions. These are provided in `data/human_fixations/`.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main requirements:
- Python 3.8+
- PyTorch 1.12+
- torchaudio
- numpy, scipy
- matplotlib
- transformers (for pretrained models)
- librosa
- pandas

## Model Zoo

We evaluate a range of architectures with **causal** (incremental) and **non-causal** (full-utterance) variants, all with comparable parameter counts (~1.6–1.9M). Models are trained from scratch on the isolated word task using MSE loss between the final hidden state and the target word2vec embedding.

| Model                | Type        | Description |
|----------------------|-------------|-------------|
| Baseline LSTM        | Causal      | Single-layer unidirectional LSTM |
| 2L-LSTM              | Causal      | Two-layer unidirectional LSTM |
| Causal-CNN           | Causal      | 1D convolutions with causal padding |
| Causal-RCNN          | Causal      | 1D CNN + unidirectional LSTM |
| Causal-Transformer   | Causal      | Transformer with causal self-attention |
| 2L-BiLSTM            | Non-causal  | Bidirectional LSTM (full context) |
| RCNN                 | Non-causal  | Non-causal CNN (25-frame look-ahead) + LSTM |
| CNN                  | Non-causal  | Standard 1D CNN (full context) |
| Transformer          | Non-causal  | Full bidirectional self-attention |
| ConvTransformer      | Non-causal  | Conformer-style with full context |

We also evaluate pretrained foundation models (no fine-tuning): `wav2vec2-base`, `hubert-base`, and `whisper-tiny.en`. For these, we derive word activation probabilities from CTC alignment paths or attention-weighted token probabilities and transform them (e.g., via Luce's choice rule) to obtain competitor activation scores.

## Training

To train a causal LSTM baseline:

```bash
python train.py --config configs/causal_lstm.yaml --data /path/to/spectrograms
```

Config files specify model architecture, hyperparameters, and data paths. Training logs and checkpoints are saved under `runs/`. Models are trained with MSE loss and the Adam optimizer by default.

Pretrained models are evaluated using wrappers under `models/pretrained/` and are not fine-tuned unless explicitly noted.

## Evaluation

### Compute activation trajectories

For a trained model, compute cosine similarity between the model’s output at each time frame and the embeddings of target, cohort, rhyme, and unrelated words:

```bash
python evaluate.py --checkpoint runs/causal_lstm/best.pt --data /path/to/test_spectrograms --wordlist data/wordlists/words.txt --output results/causal_lstm/
```

This produces `.npy` files with mean trajectories per competitor type, aligned to word onset (we add 200 ms padding to match human fixation lag where relevant).

### Compare to human data

The evaluation scripts compute RMSE and MAE against the human fixation data and generate comparison plots (e.g., the figures in the paper).

To evaluate a pretrained model:

```bash
python evaluate_pretrained.py --model wav2vec2-base --data /path/to/test_audio --wordlist data/wordlists/words.txt --output results/wav2vec2/
```

### Phoneme decoding (optional)

A phoneme decoder for probing internal layers is available at `utils/phoneme_decoder.py` and reproduces analyses similar to Figure 3 in the paper.

## Results

Our main results show that **causal models** better match human VWP dynamics (lower RMSE/MAE) than **non-causal** and many off-the-shelf pretrained ASR models. All plots and metrics produced by `evaluate.py` are saved in `results/`.

## Reproducing Paper Figures

To reproduce paper figures (assuming you have the trained checkpoints and human data):

```bash
python reproduce_figures.py --trained_models_dir /path/to/all/checkpoints --pretrained_models --human_data data/human_fixations/allopenna1998.csv
```

## Citation

If you use this benchmark or code, please cite the paper. Temporary bibtex (to be updated upon acceptance):

```
@misc{htp2025,
  title={Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR},
  author={Anonymous},
  booktitle={Submitted to Interspeech 2025},
  year={2025}
}
```

## License

This project is released under the MIT License. Human fixation data are used with permission from the original authors (Allopenna et al., 1998). Word2vec embeddings are from the Google News corpus (see https://code.google.com/archive/p/word2vec/).

## Contact

For questions or issues, please open a GitHub issue or contact the authors (contact details in the paper).

---

**Acknowledgements**

This work builds on prior resources including the EARSHOT model and the Visual World Paradigm data (Allopenna et al., 1998). We thank the creators of those resources.
