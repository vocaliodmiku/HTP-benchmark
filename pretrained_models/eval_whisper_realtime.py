import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analysis')))
import time
import unicodedata
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from phonological_competition import Category_Dict_Generate
import random

# Configuration
datapath = './dataset/'
model_name = "openai/whisper-base"
basepath = "experiments/whisper_realtime"
os.makedirs(basepath, exist_ok=True)
os.makedirs(os.path.join(basepath, "imgs"), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
delay20 = True  # Add 200ms silence padding

print(f"Loading Whisper model: {model_name}")
processor = WhisperProcessor.from_pretrained(model_name)
# Force eager attention implementation to support output_attentions=True
model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager").to(device)
model.eval()

# Load Pronunciation Dictionary
print("Loading pronunciation dictionary...")
pronunciation_path = "dataset/en/vocab.txt"

pronunciation_dict = pd.read_csv(pronunciation_path, sep="\t", names=["word","pronunciation"])
pronunciation_dict = pronunciation_dict[["word","pronunciation"]]
pronunciation_dict["pronunciation"] = pronunciation_dict["pronunciation"].apply(lambda x: x.split("."))
pronunciation_dict = pronunciation_dict.set_index("word").to_dict(orient="index")
words_list = list(pronunciation_dict.keys())

# Generate Categories
print("Generating/Loading category dictionary...")
word_competition_dict = Category_Dict_Generate(words_list, pronunciation_dict)

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze()

def get_whisper_realtime_trace(audio_features, target_word, processor, model, waveform_len_sec):
    """
    Computes a time-varying probability trace using Whisper's cross-attention weights.
    """
    
    # 1. Prepare Decoder Input for Forced Alignment
    # Match Whisper's prediction style (Capitalized with leading space)
    # This is crucial for correct probabilities
    target_text = " " + target_word.strip().capitalize()
    
    # Helper to get forced decoder ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)

    
    forced_tokens = [t[1] for t in sorted(forced_decoder_ids, key=lambda x: x[0])]
    sot_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prefix_tokens = [sot_token] + forced_tokens
        
    # Encode target word
    text_tokens = processor.tokenizer.encode(target_text, add_special_tokens=False)
    eot_token = processor.tokenizer.eos_token_id
    
    full_sequence = prefix_tokens + text_tokens + [eot_token]
    decoder_input_ids = torch.tensor([full_sequence], device=device)
    
    # 2. Forward Pass with Attention
    with torch.no_grad():
        outputs = model(
            input_features=audio_features,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True
        )
    
    # 3. Calculate Token Probabilities
    logits = outputs.logits # (1, SeqLen, Vocab)
    probs = F.softmax(logits, dim=-1)
    
    # Indices of our target word tokens in the sequence
    start_idx = len(prefix_tokens)
    end_idx = start_idx + len(text_tokens)
    
    token_probs = []
    tokens = []
    
    # Check probabilities for each token in the word
    # The prediction for token at `i` comes from `logits[i-1]`
    print("len of probs:", probs.shape)
    print("decoder_input_ids:", decoder_input_ids)
    for i in range(start_idx, end_idx):
        target_token_id = decoder_input_ids[0, i]
        # Logits at i-1 predict token at i
        p = probs[0, i-1, target_token_id].item()
        
        token_str = processor.tokenizer.convert_ids_to_tokens([target_token_id.item()])[0]
        token_probs.append(p)
        tokens.append(token_str)
        current_token = processor.tokenizer.convert_ids_to_tokens([target_token_id])[0]
        previous_tokens = processor.tokenizer.convert_ids_to_tokens(decoder_input_ids[0, :i+1].tolist()) # +1 for target shift
        print("P({}|{}) = {p:.4f}".format(current_token, previous_tokens, p=p))
        
    token_probs_tensor = torch.tensor(token_probs, device=device)
    
    # 4. Extract Cross-Attention
    # Layer weights: (Batch, Heads, DecoderSeq, EncoderSeq)
    # Use the last layer
    if outputs.cross_attentions is None:
        # Should not happen if config is correct
        return np.zeros(int(waveform_len_sec * 50))
        
    layer_atten = outputs.cross_attentions[-1] 
    avg_atten = layer_atten.mean(dim=1).squeeze(0) # (DecoderSeq, EncoderSeq)
    
    # decoder_input_ids: [SOT, ..., T_start, ..., T_end-1, EOT]
    # We want attention for generating T_start ... T_end-1
    # T_start is at index start_idx. It is generated by query at index start_idx-1.
    # T_end-1 is at index end_idx-1. It is generated by query at index end_idx-2.
    
    # So slice is [start_idx-1 : end_idx-1]
    word_attentions = avg_atten[start_idx-1 : end_idx-1, :]
    
    # 5. Compute Trace
    # Weight attention rows by their token probabilities
    weighted_atts = word_attentions * token_probs_tensor.unsqueeze(1)
    
    # Mean over tokens to get one trace per time step
    raw_trace = weighted_atts.mean(dim=0).cpu().numpy()
    
    # Crop to audio duration (Whisper frames are 20ms)
    num_frames = int(waveform_len_sec / 0.02)
    
    current_frames = len(raw_trace)
    if num_frames <= current_frames:
        trace = raw_trace[:num_frames]
    else:
        trace = np.pad(raw_trace, (0, num_frames - current_frames))
    
    # Scaling
    max_val = trace.max()
    mean_prob = token_probs_tensor.mean().item()
    if max_val > 0:
         trace = trace / max_val * mean_prob
         
    return trace

# Data Collection
# Store traces: key -> time_series
categorized_data_dict = {}

# Find audio files
# Assuming standard structure or file lists exist
file_lists = ["dataset/en_train.txt", "dataset/en_test.txt"]
wav_files = []
for fpath in file_lists:
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            wav_files.extend([os.path.join(datapath, line.strip() + ".wav") for line in f if line.strip()])

if not wav_files:
    # Fallback to scanning dir if text files empty/missing
    print("No file list found, scanning directory...")
    import glob
    wav_files = glob.glob(os.path.join(datapath, "**/*.wav"), recursive=True)

print(f"Found {len(wav_files)} audio files.")

cor, tot = 0, len(wav_files)
print("Starting processing...")

for wav_path in tqdm(wav_files):
    if not os.path.exists(wav_path):
        continue
        
    filename = os.path.basename(wav_path)
    # Extract word from filename (assuming simple filename like 'word.wav' or 'word_speaker.wav')
    # Use the logic from proc_0_whisper.py
    word_raw = filename.lower().replace('.wav', '').strip()
    
    # Check if we can parse it (sometimes filename is just id)
    # If using Ainhoa dataset style: "dataset/en/ainhoa/academy.wav" -> word is filename
    word = unicodedata.normalize("NFC", word_raw)
    
    # Try to extract speaker if possible
    parts = wav_path.split(os.sep)
    if len(parts) >= 2:
        speaker = parts[-2]
    else:
        speaker = "unknown"
        
    if word not in words_list:
        # If word not in vocab, maybe skipping?
        # But we can still run it for target trace if we want
        pass
    
    waveform = load_audio(wav_path)
    
    # Pad 200ms left
    if delay20:
        silence_samples = int(0.2 * 16000)
        silence = torch.zeros(silence_samples)
        waveform = torch.cat([silence, waveform], dim=0)
    
    audio_np = waveform.numpy()
    waveform_len_sec = len(audio_np) / 16000.0
    
    input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    # 1. Prediction (Accuracy check)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="en")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    clean_trans = transcription.strip().lower().translate(str.maketrans('', '', '.,!?'))

    if clean_trans == word:
        cor += 1 
        
        # Get Competitors
        competition_set = {"Target": [word]}
        if word in words_list and word_competition_dict:
            competition_set["Cohort"] = word_competition_dict.get((word, "Cohort"), [])
            competition_set["Rhyme"] = word_competition_dict.get((word, "Rhyme"), []) 
        else:
            competition_set["Cohort"] = []
            competition_set["Rhyme"] = []

        # Compute Traces
        for category in ["Target", "Cohort", "Rhyme"]:
            cand_words = competition_set.get(category, [])
            
            # If no candidates, store Nan
            if not cand_words:
                 # We need to store something to keep structure? 
                 # Or just skip 
                 # proc_0_whisper stores [np.nan]
                 categorized_data_dict[0, word, speaker, category] = [np.nan]
                 continue
            
            traces = []
            limit = 20 # Limit number of competitors to trace
            for candid in cand_words[:limit]:
                 trace = get_whisper_realtime_trace(input_features, candid, processor, model, waveform_len_sec)
                 traces.append(trace)
            
            if traces:
                traces_arr = np.array(traces) 
                
                # Average over candidates
                mean_trace = np.mean(traces_arr, axis=0)
                
                # Note: delay20 means first 200ms correspond to silence
                # We might want to keep it or shift it depending on how we visualize
                # Keeping it raw aligned to audio file is safest
                
                categorized_data_dict[0, word, speaker, category] = mean_trace
                
                # Debug plot occasionally
                if category == "Target" and random.random() < 0.05:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10,4))
                    plt.plot(np.linspace(0, waveform_len_sec, len(mean_trace)), mean_trace)
                    plt.title(f"Trace for {word} ({speaker})")
                    plt.savefig(os.path.join(basepath, f"imgs/debug_{word}.png"))
                    plt.close()
            else:
                categorized_data_dict[0, word, speaker, category] = [np.nan]
    
# Save Results
print("\n" + "="*80)
print(f"Processing completed. Accuracy: {cor}/{tot} = {cor/tot:.2%}" if tot > 0 else "No files processed")

# Helper to save dict to dataframe (similar to dict_to_dataframe import)
def save_data(data_dict, path):
    # Convert dictionary to records
    records = []
    # Find max length for padding
    max_len = 0
    for key, val in data_dict.items():
        if len(val) > max_len:
            max_len = len(val)
            
    for key, val in data_dict.items():
        # key is (0, word, speaker, category)
        # val is trace array
        trace = list(val)
        # Pad with NaNs if necessary to make rectangular (if using CSV columns for time)
        # Or usually we save as long format? 
        # The previous code seemed to assume equal length or handled it inside dict_to_dataframe
        
        # Let's save as a big list
        padded = trace + [np.nan]*(max_len - len(trace))
        record = {
            "word": key[1],
            "speaker": key[2],
            "category": key[3],
            # Flatten trace? usually heavy for CSV
        }
        # Add trace columns
        for i, v in enumerate(padded):
            record[f"t_{i}"] = v
        records.append(record)
        
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

if len(categorized_data_dict) > 0:
    # Use existing function if available, else custom
    try:
        from phonological_competition import dict_to_dataframe
        df = dict_to_dataframe(categorized_data_dict)
        output_path = os.path.join(basepath, 'competition.csv')
        df.to_csv(output_path, index=False)
    except ImportError:
        output_path = os.path.join(basepath, 'competition.csv')
        # Simple save
        save_data(categorized_data_dict, output_path)

    print(f"✅ Data saved to: {output_path}")

print(f"⏱️  Total Processing Time: {time.time() - start_time:.2f} seconds")
print("="*80)
