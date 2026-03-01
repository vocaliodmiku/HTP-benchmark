import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import unicodedata
from scipy.stats import pearsonr
import textgrid
from params import num_workers, is_cpu, lang_order
from data import read_config, CollateItems
from main import setup_pretraining
from phonological_competition import Category_Dict_Generate
import matplotlib.pyplot as plt

# Configuration
config_path = 'experiments/en_words_ku_2lstmbi.cfg'
model_name = "2LSTMbi"
exp_path = "experiments/en_words_ku_2lstmbi/2LSTMbi"
checkpoint_epoch = 5690
probe_checkpoint_epoch = 399
train_set = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 1. Setup & Load Model
    print("Loading configuration and model...")
    config = read_config(config_path, model_name)
    datapath = './dataset/'
    
    # We need both train (for probe) and test (for analysis)
    trainer, train_dataset, valid_dataset, test_dataset = setup_pretraining(
        config, datapath, num_workers, is_cpu
    )
    
    # Load checkpoint
    ckpt_fname = os.path.join(
        config_path.replace('.cfg', ''), 
        model_name, 
        "probe", 
        f"probe_epoch_{probe_checkpoint_epoch}.pt"
    )
    if not os.path.exists(ckpt_fname):
        raise FileNotFoundError(f"Probe checkpoint not found: {ckpt_fname}")
    probe_state_dict = torch.load(ckpt_fname, map_location=device)
    
    # Load checkpoint
    trainer.load_checkpoint(i_checkpoint=checkpoint_epoch)
    trainer.model.to(device)
    trainer.model.eval()
    
    # Freeze main model
    for param in trainer.model.parameters():
        param.requires_grad = False
 
    # Assume intermediate dimension from config (rnn_hidden_size)
    # Check model architecture in models.py -> LSTMModel
    input_dim = 256 # config.rnn_hidden_size 
    num_phonemes = len(train_dataset.phonemes)
    
    probe = nn.Linear(input_dim, num_phonemes).to(device)
    probe.load_state_dict(probe_state_dict)
     
    if train_set:
        dataset = train_dataset
    else:
        dataset = test_dataset
    
    batch_size = 256
    collect_fn = CollateItems()
    
    num_samples = len(dataset)
    idx = 0 
    ncor, ntot = 0, 0
    while idx < num_samples:
        idxs = list(range(0, len(dataset), batch_size))
        st = time.time()
        batch_indices = list(range(idx, min(idx + batch_size, num_samples)))
        batch = [dataset[i] for i in batch_indices]
        batch = collect_fn(batch)
        inputs, targets, lengths, mask, loss_mask, lang_mask = trainer.generate_inputs_and_targets(
            batch, 0, 1, 1
        )
        # Move to device
        inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        target_phonemes = targets["phonemes"].to(device) # (batch, time)

        out = trainer.model(inputs, lengths, training=False)
        features = out["rnn_phone"] # (batch, time, hidden_size)
        logits = probe(features) 
        
        audio_paths = [dataset.wav_paths[i] for i in batch_indices]
        words = [audio_path.split('/')[-1].replace('.wav','') for audio_path in audio_paths]
        words = [unicodedata.normalize("NFC", word) for word in words]
        speakers = [audio_path.split('/')[-2] for audio_path in audio_paths]
    
 
        for logit, target, word, length, speaker, audiopath in zip(logits, targets['phonemes'], words, lengths, speakers, audio_paths):
            print(f"Logits shape: {logit.shape}, Words: {word}, Lengths: {length}, Speakers: {speaker}")
            tg_file = audiopath.replace('.wav', '.TextGrid')
            tg = textgrid.TextGrid.fromFile(tg_file)
            
            target_phonemes = target.cpu().numpy()[:length]
            target_phonemes_str = [train_dataset.phonemes[idx] if idx >= 0 else 'sil' for idx in target_phonemes]
            print(f"Target Phonemes: {target_phonemes_str}")
            pred_phonemes = logit.argmax(dim=-1).cpu().numpy()[:length]
            pred_phonemes_str = [train_dataset.phonemes[idx] if idx >= 0 else 'sil' for idx in pred_phonemes]
            print(f"Predicted Phonemes: {pred_phonemes_str}")
            
            ncor += (pred_phonemes == target_phonemes).sum().item()
            ntot += len(target_phonemes)

            activations = F.softmax(logit[:length], dim=-1).detach().cpu().numpy()
            plot_case_study(activations, target_phonemes, train_dataset.phonemes, tg, word, speaker)
            debug = 1
        idx += batch_size
        
    print(f"Overall Accuracy: {ncor/ntot:.4f} ({ncor}/{ntot})")

def plot_case_study(activations, target_indices, phoneme_list, tg, word, speaker):
    sil_shift = 0.2
    total_time = tg.maxTime + sil_shift
    num_frames = activations.shape[0]
    time_axis = np.linspace(0, total_time, num_frames)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot activations for phonemes present in the target
    unique_targets = np.unique(target_indices)
    for idx in unique_targets:
        if idx < 0 or idx >= len(phoneme_list):
            continue
        ph = phoneme_list[idx]
        ax1.plot(time_axis, activations[:, idx], label=ph, linewidth=2, alpha=0.8)
        
    ax1.set_ylabel("Activation")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax1.set_title(f"Phoneme Activations: {word}")
    ax1.grid(True, alpha=0.3)
    
    # Plot TextGrid intervals
    tier = tg[0] # Default to first tier
    for t in tg:
        if t.name.lower() in ['phones', 'phonemes']:
            tier = t
            break

    for interval in tier:
        start, end, label = interval.minTime, interval.maxTime, interval.mark
        if not label.strip(): continue
        
        # Alternating colors or just blocks
        ax2.axvspan(start+sil_shift, end+sil_shift, alpha=0.3, color='gray', ec='black')
        ax2.text((start + end) / 2 + sil_shift, 0.5, label, ha='center', va='center')
        
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, total_time)
    
    plt.tight_layout()
    os.makedirs(f"{exp_path}/case_studies", exist_ok=True)
    plt.savefig(f"{exp_path}/case_studies/{word}_{speaker}.png")
    plt.close()
 
    np.savez(f"{exp_path}/case_studies/{word}_{speaker}_data.npz",
                activations=activations, 
                target_phonemes=target_indices, 
                phoneme_list=phoneme_list, 
                intervals=[(interval.minTime, interval.maxTime, interval.mark) for interval in tier])

if __name__ == "__main__":
    if "SRV" in config_path:
        activation_fn = F.sigmoid
    else:
        activation_fn = F.tanh
    main()