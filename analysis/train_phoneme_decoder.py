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

from params import num_workers, is_cpu, lang_order
from data import read_config, CollateItems
from main import setup_pretraining
from phonological_competition import Category_Dict_Generate

# Configuration
config_path = 'experiments/en_words_ku_2lstmbi.cfg'
model_name = "2LSTMbi"
checkpoint_epoch = 5690
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
    trainer.load_checkpoint(i_checkpoint=checkpoint_epoch)
    trainer.model.to(device)
    trainer.model.eval()
    
    # Freeze main model
    for param in trainer.model.parameters():
        param.requires_grad = False

    # 2. Train Linear Probe
    print("\nTraining Linear Probe on Intermediate Layers...")
    
    # Assume intermediate dimension from config (rnn_hidden_size)
    # Check model architecture in models.py -> LSTMModel
    input_dim = 256
    num_phonemes = len(train_dataset.phonemes)
    
    probe = nn.Linear(input_dim, num_phonemes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) # Assuming -1 is padding/ignore in y_phoneme?
    # Actually y[1] (phonemes) might use 0 for silence or masked. 
    # In data.py, masked values are handled via mask, or specific index.
    # ASREataset uses -1 for silence if not in list, but let's check.
    # data.py: y_phoneme = [phoneme_index] * ...
    # if phoneme in ['sil'..]: phoneme_index = -1
    # So -1 is ignore index.
    
    # Train loop
    # Subset for speed? User said "don't write too many robust code", "just do the core part".
    # I'll run 1 epoch on train_dataset.
    
    train_loader = train_dataset.loader # Created in setup_pretraining
    # train_dataset.loader might be None if setup_pretraining doesn't set it for train?
    # data.py: if train: self.loader = ...
    # setup_pretraining returns train_dataset.
    # But wait, setup_pretraining in main.py usually returns dataset objects.
    # In proc_0_competition.py:
    # dataset = train_dataset
    # loop idx < num_samples
    
    # I'll use the manual batching loop from proc_0_competition.py for consistency
    batch_size = 512
    collect_fn = CollateItems()
    
    # Training Data Loop
    probe.train()
    # To save time for the user, maybe limit training steps? 
    # But user asked to "Train a simple linear classifier". I should do it properly-ish.
    # Using 500 batches should be enough for convergence of a linear probe.
    
    idxs = list(range(0, len(train_dataset), batch_size))
    # Shuffle
    np.random.shuffle(idxs)
    epoch_loss, epoch_acc = 0, 0
    best_acc = 0
    for epoch in range(400): # Just 1 epoch for demonstration
        pbar = tqdm(idxs, desc=f"Training Probe E-{epoch}") # Limit to 500 batches for speed
        
        for idx in pbar:
            # Batch preparation logic from proc_0_competition.py
            batch_indices = list(range(idx, min(idx + batch_size, len(train_dataset))))
            if not batch_indices: continue
            
            batch = [train_dataset[i] for i in batch_indices]
            batch = collect_fn(batch)
            inputs, targets, lengths, mask, loss_mask, lang_mask = trainer.generate_inputs_and_targets(
                batch, 0, 1, 1
            )
            
            # Move to device
            inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
            target_phonemes = targets["phonemes"].to(device) # (batch, time)
            
            # Forward pass (Main Model)
            with torch.no_grad():
                out = trainer.model(inputs, lengths, training=False)
                features = out["rnn_phone"] # (batch, time, hidden_size)
                
            # Forward pass (Probe)
            logits = probe(features) # (batch, time, num_phonemes)
            
            # Flatten for loss
            # Apply mask? targets["phonemes"] has -1 for ignore/silence usually
            # But wait, data.py says -1 for sil/sp.
            # CrossEntropyLoss(ignore_index=-1) handles it.
            
            loss = criterion(logits.reshape(-1, num_phonemes), target_phonemes.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            accuracy = (logits.argmax(dim=-1) == target_phonemes).float()
            accuracy = (accuracy * mask).sum() / mask.sum() # Masked accuracy
            epoch_acc += accuracy.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{accuracy.item():.4f}"})
            debug = 1
        print(f"Epoch {epoch} - Loss: {epoch_loss/len(idxs):.4f}, Accuracy: {epoch_acc/len(idxs):.4f}")
        
        if epoch_acc/len(idxs) > best_acc:
            best_acc = epoch_acc/len(idxs)
            # save probe checkpoint
            ckpt_fname = os.path.join(config_path.replace('.cfg', ''), model_name, "probe", f"probe_epoch_{epoch}.pt")
            if not os.path.exists(os.path.dirname(ckpt_fname)):
                os.makedirs(os.path.dirname(ckpt_fname))
            torch.save(probe.state_dict(), ckpt_fname)
            print(f"Saved best probe checkpoint {ckpt_fname} with accuracy {best_acc:.4f}")
        epoch_loss, epoch_acc = 0, 0
        
if __name__ == "__main__":
    if "SRV" in config_path:
        activation_fn = F.sigmoid
    else:
        activation_fn = F.tanh
    main()