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
import soundfile as sf
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from phonological_competition import Category_Dict_Generate, dict_to_dataframe
import glob
import random

# Configuration
datapath = './dataset/'
model_name = "openai/whisper-base"
basepath = "experiments/whisper"
os.makedirs(basepath, exist_ok=True)
os.makedirs(os.path.join(basepath, "imgs"), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
dealy20 = True

print(f"Loading Whisper model: {model_name}")
processor = WhisperProcessor.from_pretrained(model_name)
# Force eager attention implementation to support output_attentions=True
model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager").to(device)
model.eval()

# Load Pronunciation Dictionary
print("Loading pronunciation dictionary...")
pronunciation_path = "dataset/en/vocab.txt"
if not os.path.exists(pronunciation_path):
    print(f"Warning: {pronunciation_path} not found.")

# Reuse the dictionary loading logic
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

def get_whisper_probability_trace(audio_features, target_word, processor, model, waveform_len_sec=None):
    """
    Approximates a probability trace for a target word using Whisper's alignment/timestamps.
    Note: Whisper is autoregressive, so this is not a frame-by-frame probability like CTC.
    We force-decode the target word and retrieve the cross-attention weights to map 
    token probabilities to time.
    """
    
    # Prepend the language token and task token if needed, or let tokenizer handle it.
    # For Whisper, we usually provide decoder_input_ids.
    
    # 1. Prepare Target Tokens
    # Whisper tokenizer might add special tokens.
    # forced_decoder_ids are used for generation, but here we want to score a specific sequence.
    
    # Match Whisper's prediction style (Capitalized with leading space) to reduce token mismatch
    target_text = " " + target_word.strip().capitalize()
    
    # We need 
    # <sot><lang><transcribe><timestamp_probs?> ... <target_word> ... <eot>
    # <|startoftranscript|><|en|><|transcribe|><|notimestamps|> word <|endoftext|>
 
    # Use tokenizer logic for SOT
    sot_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    try:
         # Try tokenizer first (newer)
         forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
    except AttributeError:
         try:
             # Try processor (older)
             forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
         except AttributeError:
             # Fallback
             forced_decoder_ids = []

    prefix_tokens = [sot_token]
    if forced_decoder_ids:
        forced_decoder_ids.sort(key=lambda x: x[0])
        # Indices are 1-based usually
        for idx, token in forced_decoder_ids:
            prefix_tokens.append(token)
    
    # Encode target word
    # add_special_tokens=False to avoid SOT/EOT doubling if we add them manually
    text_tokens = processor.tokenizer.encode(target_text, add_special_tokens=False)
    
    # Construct full sequence
    eot_token = processor.tokenizer.eos_token_id
    full_sequence = prefix_tokens + text_tokens + [eot_token]
    
    # Convert to tensor
    decoder_input_ids = torch.tensor([full_sequence], dtype=torch.long, device=device)
    print(f"Decoder input IDs for '{target_word}': {decoder_input_ids[0].tolist()[3:]} (Tokens: {processor.tokenizer.convert_ids_to_tokens(decoder_input_ids[0].tolist()[3:])})")
    # Run Forward Pass
    with torch.no_grad():
        outputs = model(
            input_features=audio_features,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True
        )
    
    # outputs.logits: (B, DecoderSegLen, Vocab)
    # outputs.cross_attentions: tuple of (B, Heads, DecoderLen, EncoderLen)
    
    # Calculate Token Probabilities
    # We are interested in the probability of the *target word tokens*.
    # decoder_input_ids: [SOT, T1, T2, ..., EOT]
    # logits predict the NEXT token.
    # logits[0] predicts T1 (given SOT)
    # logits[1] predicts T2 (given SOT, T1)
    
    logits = outputs.logits # (1, SeqLen, V)
    probs = F.softmax(logits, dim=-1)
    
    # Identify the specific tokens we care about (The word itself)
    # The decoder_input_ids are [SOT, ... P1, P2, ... T1, T2 ... EOT]
    # Logits at index `i` predict the token at `i+1`.
    
    # We want the probabilities of T1...Tn
    # These tokens are at indices `len(prefix_tokens)` to `len(prefix_tokens)+len(text_tokens)` in `full_sequence`.
    start_idx = len(prefix_tokens)
    end_idx = start_idx + len(text_tokens)
    
    # The logits that predict T1 are at index `start_idx - 1` (the step before T1).
    # But wait, let's keep it simple: 
    # We want to check P(Token_i) given previous.
    
    token_probs = []
    target_ids = decoder_input_ids[0, 1:] # The actual targets shifted
    
    # Collect probs for the word tokens only
    # The indices in `target_ids` for our word are: `start_idx-1` to `end_idx-1`
    word_indices_in_target = range(start_idx-1, end_idx-1) 
    for i in word_indices_in_target:
        true_token_id = target_ids[i]
        p = probs[0, i, true_token_id].item()
        token_probs.append(p)
        current_token = processor.tokenizer.convert_ids_to_tokens([true_token_id])[0]
        previous_tokens = processor.tokenizer.convert_ids_to_tokens(decoder_input_ids[0, :i+1].tolist()) # +1 for target shift
        print("P({}|{}) = {p:.4f}".format(current_token, previous_tokens, p=p))
        
    token_probs_tensor = torch.tensor(token_probs, device=device)
    
    # --- 4. Extract Attention ---
    # Last layer cross-attention: (Batch, Heads, DecoderSeq, EncoderSeq)
    layer_atten = outputs.cross_attentions[-1] 
    avg_atten = layer_atten.mean(dim=1).squeeze(0) # (DecoderSeq, EncoderSeq)
    
    # Crop Attention to Actual Audio Length
    # Whisper encoder outputs 1500 frames for 30s audio.
    # 30s / 1500 frames = 0.02s (20ms) per frame.
    if waveform_len_sec is not None:
        num_frames = int(waveform_len_sec / 0.02)
        # Ensure we don't go out of bounds if calculation is slightly off
        num_frames = min(num_frames, avg_atten.shape[1])
        avg_atten = avg_atten[:, :num_frames]
        
    # Review alignment:
    # DecoderSeq matches `decoder_input_ids`.
    # Attention at index `i` is used to generate `logits[i]`.
    # We established that `logits[i]` predicts `decoder_input_ids[i+1]`.
    # So `avg_atten[i]` is the map used to predict token `i+1`.
    
    # We want the attention maps responsible for predicting T1...Tn.
    # These map indices are `start_idx-1` to `end_idx-1`.
    
    word_attentions = avg_atten[start_idx-1 : end_idx-1, :] # (NumWordTokens, EncoderFrames)

    # --- 5. Compute Trace ---
    # Weighted Sum Logic
    weighted_atts = word_attentions * token_probs_tensor.unsqueeze(1)
    raw_trace = weighted_atts.mean(dim=0).cpu().numpy()
    
    # Scaling
    max_val = raw_trace.max()
    mean_prob = token_probs_tensor.mean().item()
    
    if max_val > 0:
        scaled_trace = raw_trace / max_val * mean_prob
    else:
        scaled_trace = raw_trace
    return scaled_trace

# Data Collection
categorized_data_dict = {}

# Find audio files
wav_files = list(open("dataset/en_train.txt", "r").readlines()) + list(open("dataset/en_test.txt", "r").readlines()) 
wav_files = [os.path.join(datapath, line.strip() + ".wav") for line in wav_files if line.strip()]
print(f"Found {len(wav_files)} audio files.")

cor, tot = 0, len(wav_files)
print("Starting processing...")

# We need to handle the language and task logic for Whisper if not English
# but dataset seems to be EN.

for wav_path in tqdm(wav_files):
    filename = os.path.basename(wav_path)
    word_raw = filename.lower().replace('.wav', '').strip()
    word = unicodedata.normalize("NFC", word_raw)
    
    parts = wav_path.split(os.sep)
    speaker = parts[-2] 
        
    if word not in words_list:
        continue
    
    waveform = load_audio(wav_path) 
    # Process Audio Features
    # Whisper expects 30s padding usually, processor handles it
    if dealy20:
        # Add 200ms of silence at the beginning to ensure we capture the tail of the word in attention
        silence = torch.zeros(int(0.2 * 16000))  # 200ms of silence
        waveform = torch.cat([silence, waveform], dim=0)
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    # 
    waveform_len_sec = waveform.shape[0] / 16000
        
    # 1. Prediction (Accuracy)
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="en")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Clean string
    transcription = transcription.strip().lower().translate(str.maketrans('', '', '.,!?'))

    if transcription == word:
        print(f"\n\n === Correctly transcribed '{word}' as '{transcription}'")
        print("Pred ID: ", predicted_ids, "Decode as:", [processor.tokenizer.convert_ids_to_tokens(i) for i in predicted_ids[0].tolist()])
        cor += 1 
    
        # Define Categories
        competition_set = {"Target": [word]}
        competition_set["Cohort"] = word_competition_dict.get((word, "Cohort"), [])
        competition_set["Rhyme"] = word_competition_dict.get((word, "Rhyme"), []) 

        # Compute Traces
        for category in ["Target", "Cohort", "Rhyme"]:
            cand_words = competition_set.get(category, [])
            if not cand_words:
                categorized_data_dict[0, word, speaker, category] = [np.nan]
                continue
                
            traces = []
            limit = 20
            for candid in cand_words[:limit]:
                 trace = get_whisper_probability_trace(input_features, candid, processor, model, waveform_len_sec=waveform_len_sec)
                 traces.append(trace)
            
            if traces:
                traces_arr = np.array(traces) # shapes might differ if audio lengths differ? 
                # Whisper encoder output is fixed size (1500 frames for 30s)?
                # Or dynamic? 
                # Whisper encoder is usually padded to 30s -> 1500 frames.
                # So traces will be long.
                # We should trim to actual audio length?
                # duration = len(waveform) / 16000
                # frames = int(duration / 0.02)
                # trace = trace[:frames]
                
                L = traces[0].shape[0]
                # Assuming all traces compatible
                mean_trace = np.mean(traces_arr, axis=0)
                
                # Trim to approx audio length
                duration_sec = waveform.shape[0] / 16000
                frames = int(duration_sec * 50) # 50Hz
                if frames < len(mean_trace):
                    mean_trace = mean_trace[:frames]
                    
                categorized_data_dict[0, word, speaker, category] = mean_trace
            else:
                categorized_data_dict[0, word, speaker, category] = [np.nan]
        
        print(f"Transcription for {word}: {transcription} P(target):{max(categorized_data_dict[0, word, speaker, 'Target']):.4f}")
        
        if random.random() < 0.1: 
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            for category in ["Target", "Cohort", "Rhyme"]:
                trace = categorized_data_dict[0, word, speaker, category]
                if trace is not None and not np.isnan(trace).all():
                    plt.plot(trace, label=category)
            plt.title(f"Whisper Attention Traces for '{word}' (Speaker: {speaker})")
            plt.xlabel("Time Steps (20ms)")
            plt.ylabel("Mean Attn * Prob")
            plt.legend()
            output_img_path = os.path.join(basepath, f"imgs/{word}_{speaker}_trace.png")
            plt.savefig(output_img_path)
            plt.close()

print("\n" + "="*80)
print(f"Processing completed. Accuracy: {cor}/{tot} = {cor/tot:.2%}")
if len(categorized_data_dict) > 0:
    df = dict_to_dataframe(categorized_data_dict)
    output_path = os.path.join(basepath, 'competition.csv')
    df.to_csv(output_path, index=False)
    print(f" Data saved to: {output_path}")

print(f"⏱️  Total Processing Time: {time.time() - start_time:.2f} seconds")
print("="*80)
