
import os
# Set TMPDIR to a local directory with sufficient space
os.environ["TMPDIR"] = os.path.join(os.getcwd(), "tmp")
if not os.path.exists(os.environ["TMPDIR"]):
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

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
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from phonological_competition import Category_Dict_Generate
from omegaconf import OmegaConf

# Configuration
datapath = './dataset/'
model_name = "nvidia/nemotron-speech-streaming-en-0.6b"
basepath = "experiments/nemotron_realtime"
os.makedirs(basepath, exist_ok=True)
os.makedirs(os.path.join(basepath, "imgs"), exist_ok=True)
delay20 = True

device = 'cpu'
start_time = time.time()

print(f"Loading Nemotron model: {model_name}")
# Load the model using NeMo
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).to(device)
asr_model.eval()

cfg = OmegaConf.create()

# Set attention context size if needed
# For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size.
# To change it, you may use att_context_size, for example att_context_size=[70,0].
cfg.att_context_size = [70, 1]

if cfg.att_context_size is not None:
    if hasattr(asr_model.encoder, "set_default_att_context_size"):
        asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
        print(f"Set att_context_size to {cfg.att_context_size}")
    else:
        print("Model does not support multiple lookaheads (set_default_att_context_size missing).")

# Setup decoding strategy
if hasattr(asr_model, 'change_decoding_strategy') and hasattr(asr_model, 'decoding'):
    if cfg.decoder_type is not None:
        decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding

        if hasattr(asr_model, 'cur_decoder'):
            asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
        else:
            asr_model.change_decoding_strategy(decoding_cfg)

    # Check if ctc or rnnt model
    elif hasattr(asr_model, 'joint'):  # RNNT model
        cfg.rnnt_decoding.fused_batch_size = -1
        if hasattr(asr_model, 'cur_decoder'):
            asr_model.change_decoding_strategy(cfg.rnnt_decoding, decoder_type=cfg.decoder_type)
        else:
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
    else:
        asr_model.change_decoding_strategy(cfg.ctc_decoding)

print(f'ATT_CONTEXT_SIZE: {asr_model.cfg.encoder.get("att_context_size")}')

# cache-aware streaming requires float32
compute_dtype = torch.float32
asr_model = asr_model.to(dtype=compute_dtype)
print(f"Streaming config: {asr_model.encoder.streaming_cfg}")

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

def get_nemotron_realtime_trace(encoded_features, target_word, asr_model):
    """
    Computes a time-varying probability trace using Nemotron RNN-T.
    encoded_features: encoder output tensor. Expected to be either
    [Batch, Dim, Time] or [Batch, Time, Dim], but we also handle
    2-D inputs like [Time, Dim].
    """
    # Normalize encoder feature shape to [B, T, D]
    if not isinstance(encoded_features, torch.Tensor):
        encoded_features = torch.as_tensor(encoded_features, device=device, dtype=compute_dtype)

    if encoded_features.ndim == 3:
        # Common NeMo RNNT encoder format is [B, D, T]. We heuristically
        # detect that by checking for a large channel dimension in the
        # middle (e.g., 1024) and treat the remaining dim as time.
        b, d1, d2 = encoded_features.shape
        if d1 == 1024 and d2 != 1024:
            # [B, Dim, Time] -> [B, Time, Dim]
            encoded = encoded_features.transpose(1, 2)
        else:
            # Assume already [B, T, D]
            encoded = encoded_features
    elif encoded_features.ndim == 2:
        # Handle [T, D] or [D, T] by lifting to batch dimension.
        t, d = encoded_features.shape
        if t == 1024 and d != 1024:
            # Likely [Dim, Time] -> [Time, Dim]
            encoded = encoded_features.transpose(0, 1).unsqueeze(0)
        else:
            # Treat first dim as time: [T, D] -> [1, T, D]
            encoded = encoded_features.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected encoder feature shape {encoded_features.shape}, expected 2-D or 3-D tensor.")

    # At this point, encoded is [B, T, D]

    # 3. Prepare Text Input for Prediction Network
    # Tokenize the target word
    # Note: Depending on tokenizer, we might need a space prefix
    target_text = target_word.strip()
    
    # Use the model's tokenizer
    # RNNT typically predicts <s> + tokens. 
    # The prediction network inputs are: [blank, t1, t2, ...]
    ids = asr_model.tokenizer.text_to_ids(target_text)
    
    # We want to measure the probability of the *tokens* of the word appearing.
    # To get a single "trace" for the word, we can look at the probability of the 
    # LAST token of the word, given the history of the previous tokens.
    
    # Decoder inputs: [Blank/SOS] + Token_IDs (excluding the last one if we want prob of last)
    # However, standard RNNT joint simply takes Enc(t) and Pred(u) and gives Prob(token over vocab).
    
    # Let's construct the decoder sequence corresponding to the word.
    # To align with how RNNT works:
    # We feed [SOS] -> PredNet -> H_dec[0]
    # We feed [SOS, t1] -> PredNet -> H_dec[1]
    # ...
    # We feed [SOS, t1, ... tn-1] -> PredNet -> H_dec[n-1]
    
    # And then we combine with Encoder to check probability of next token.
    # To get a "word trace", we essentially want to know when the TOTAL word is finished.
    # So we want P(tn | enc(t), history=[t1...tn-1])
    
    # Prepare input for Prediction Network (Decoder)
    if len(ids) == 0:
        return np.zeros(encoded.shape[1])

    # The target sequence including the one we want to predict
    target_tokens = torch.tensor([ids], device=device, dtype=torch.long) # [1, L]
    target_length = torch.tensor([len(ids)], device=device, dtype=torch.long)
    
    with torch.no_grad():
        # decoder returns a tuple, take the first element
        dec_stuff = asr_model.decoder(targets=target_tokens, target_length=target_length)
        dec_out = dec_stuff[0] # [B, D_dec, U] likely
        
        # Check if we need to transpose
        # Usually typical RNNT implementation might return [B, U, D] but here we saw [B, D, U]
        # Let's check dimension size. D_dec is usually 640.
        if dec_out.shape[1] == 640:
             dec_out = dec_out.transpose(1, 2) # Now [B, U, D_dec]
        
    # dec_out corresponds to states after processing prefixes.
    # index 0: state after SOS. Used to predict target[0].
    # index k: state after SOS + target[0...k-1]. Used to predict target[k].
    
    # We want the probability of the LAST token `ids[-1]`.
    # This is predicted by the state at index `len(ids) - 1`.
    # E.g. len=3. indexes 0, 1, 2. Index 2 predicts id[2].
    
    # Select the specific step
    idx = len(ids) - 1
    if idx < dec_out.shape[1]:
        last_dec_state = dec_out[:, idx:idx+1, :] # [B, 1, D_dec]
    else:
        # Fallback if length mismatch
        last_dec_state = dec_out[:, -1:, :]

    # 4. Joint Pass - Manual
    # We compute logits manually to avoid loss computation overhead/requirements
    
    with torch.no_grad():
        # Project encoder
        # encoded is [B, T, D_enc]
        f = asr_model.joint.enc(encoded) # [B, T, D_joint]
        
        # Project decoder
        g = asr_model.joint.pred(last_dec_state) # [B, 1, D_joint]

        # Combine: f + g
        # Broadcast: [B, T, 1, D] + [B, 1, 1, D] -> [B, T, 1, D]
        # Note: g is [B, 1, D], we want to add it to every time step T.
        # f.unsqueeze(2): [B, T, 1, D]
        # g.unsqueeze(1): [B, 1, 1, D]
        
        x = f.unsqueeze(2) + g.unsqueeze(1) # [B, T, 1, D_joint]
        
        # Compute logits
        out = asr_model.joint.joint_net(x) # [B, T, 1, V+1]
        
        logits = out.squeeze(2) # [B, T, V+1]

    # 5. Extract Probability of the Target Token
    target_token_id = ids[-1]
    
    # Softmax over vocabulary dim
    probs = F.softmax(logits, dim=-1) # [B, T, V+1]
    
    # Trace for the specific token
    # We are interested in P(token | time, prefix)
    trace = probs[0, :, target_token_id].cpu().numpy()
    
    return trace

def get_encoder_outputs_streaming(
    audio_file, asr_model, device, compute_dtype=torch.float32, pad_and_drop_preencoded=False
):
    """
    Run cache-aware streaming inference on a waveform tensor, mirroring the
    chunk-by-chunk loop in b.py.

    Args:
        waveform: 1-D float32 torch tensor (16 kHz mono, already padded with
                  silence if desired)
        asr_model: loaded NeMo ASR model with streaming_cfg already set
        device: torch.device
        compute_dtype: torch.dtype (must be float32 for cache-aware models)
        pad_and_drop_preencoded: same flag as in b.py

    Returns:
        encoded: encoder-output tensor concatenated across all chunks.
                 Shape [B, D, T_total] for standard NeMo RNNT (or None on error).
        transcription: final greedy transcription string from streaming
    """
    # Mirror b.py behavior: let NeMo handle audio loading/resampling and
    # keep online_normalization disabled unless explicitly configured.
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_file, stream_id=-1)
    # warn: /home/fie24002/miniconda3/envs/nemo/lib/python3.12/site-packages/nemo/collections/asr/parts/utils/streaming_utils.py
    # 1782
    batch_size = len(streaming_buffer.streams_length)
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        asr_model.encoder.get_initial_cache_state(batch_size=batch_size)
    )

    previous_hypotheses = None
    pred_out_stream = None
    transcribed_texts = None

    for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                # We call the encoder directly to get the features (encoded) which are needed for the trace
                # conformer_stream_step returns predictions, not raw encoder features for RNNT
                (
                    encoded_chunk,
                    encoded_len_chunk,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                ) = asr_model.encoder.cache_aware_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    drop_extra_pre_encoded=(
                        0
                        if (step_num == 0 and not pad_and_drop_preencoded)
                        else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                    ),
                    bypass_pre_encode=False
                )
                
                # Check for 2D/3D tensor validity
                if isinstance(encoded_chunk, torch.Tensor) and encoded_chunk.dim() >= 2:
                    if pred_out_stream is None:
                        pred_out_stream = []
                    pred_out_stream.append(encoded_chunk)
                
                # We still need transcription to verify the word
                # We can run the decoder separately on the concatenated features at the end, 
                # or just skip verification if we trust the alignment. 
                # For now, let's just collect features. Verification might be tricky without full decode.

    # Concatenate features
    if not pred_out_stream:
         return None, ""

    sample = pred_out_stream[0]
    if sample.dim() == 3:
        encoded = torch.cat(pred_out_stream, dim=2)
    else:
        encoded = torch.cat(pred_out_stream, dim=0)
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(0)

    # Decode the full sequence once at the end to get the transcription
    # This is "offline" decoding on the streamed features, which is valid for checking correctness
    with torch.no_grad():
        # Encode lengths: we need full length. 
        # But for RNNT decoding we usually need proper lengths.
        # Let's trust the encoder output shape.
        encoded_len = torch.tensor([encoded.shape[2]], device=device, dtype=torch.long)
        
        best_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=True,
        )
        transcription = best_hyp[0].text

    return encoded, transcription


# Data Collection
categorized_data_dict = {}

# Find audio files
file_lists = ["dataset/en_train.txt", "dataset/en_test.txt"]
wav_files = []
for fpath in file_lists:
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            wav_files.extend([os.path.join(datapath, line.strip() + ".wav") for line in f if line.strip()])

if not wav_files:
    print("No file list found, scanning directory...")
    import glob
    wav_files = glob.glob(os.path.join(datapath, "**/*.wav"), recursive=True)

print(f"Found {len(wav_files)} audio files.")

# Processing Loop
print("Starting processing...")
cor = 0
tot = 0

# To save results in a similar structure
# Key: (file_index, word, speaker, category) -> trace
# Here original used file_index as key[0], but here we might use 0 if not important or filename hash

results_dict = {}

for wav_path in tqdm(wav_files):
    if not os.path.exists(wav_path):
        continue
        
    filename = os.path.basename(wav_path)
    word_raw = filename.lower().replace('.wav', '').strip()
    
    # Normalize word
    word = unicodedata.normalize("NFC", word_raw)
    
    # Extract speaker if possible (assuming directory structure dataset/en/<speaker>/<word>.wav)
    parts = wav_path.split(os.sep)
    if len(parts) >= 2:
        speaker = parts[-2]
    else:
        speaker = "unknown"
        
    tot += 1

    # Get trace and accuracy check
    # wav_path = "/scratch/jsm04005/fie24002/DATA/LibriSpeech/LibriSpeech/train-clean-360/1460/138289/1460-138289-0012.wav"
    # Streaming inference: encoder output + final transcription in one pass
    # The padded waveform is written to a temp file so CacheAwareStreamingAudioBuffer
    # can chunk it and process it with causal/cache-aware attention.
    encoded, prediction = get_encoder_outputs_streaming(
        wav_path, asr_model, device, compute_dtype=compute_dtype
    )
    if encoded is None:
        print(f"Streaming failed for {wav_path}, skipping.")
        continue
    
    # Check if prediction matches the target word
    # We normalize both to ensure fair comparison
    prediction_norm = unicodedata.normalize("NFC", prediction).lower().strip()
    word_norm = unicodedata.normalize("NFC", word).lower().strip()
    # ["Many of them foamed at the mouth, their breathing being quick and short whilst the 
    # bodies of all were fearfully distended. Oh what can I do? What can I do? 
    # said Bathsheba helplessly. Sheep are such unfortunate animals 
    # there's always something happening to them"]
    print(f"File: {wav_path}, Target: '{word_norm}', Predicted: '{prediction_norm}'")
    if prediction_norm == word_norm:
        print(f"Correctly predicted: {word} -> {prediction}")
        cor += 1
    else:
        # If the model didn't predict the word correctly, we skip analyzing traces for this word
        # as the internal state might not align with the phonological expectations of the word
        continue    
    # Extract competitors
    competition_set = {"Target": [word]}
    if word in words_list and word_competition_dict:
        competition_set["Cohort"] = word_competition_dict.get((word, "Cohort"), [])
        competition_set["Rhyme"] = word_competition_dict.get((word, "Rhyme"), []) 
    else:
        competition_set["Cohort"] = []
        competition_set["Rhyme"] = []
        
    for category in ["Target", "Cohort", "Rhyme"]:
        cand_words = competition_set.get(category, [])
        limit = 20 
        traces = []
        
        for candid in cand_words[:limit]:
            # Pass pre-computed encoded features
            trace = get_nemotron_realtime_trace(encoded, candid, asr_model)
            traces.append(trace) 
        if category == "Target" and len(traces) > 0:
            print(f"Traces for {word} (Target): Max prob {max([t.max() for t in traces]):.4f}")
        if traces:
            traces_arr = np.array(traces)
            # Align lengths? RNNT trace length depends on Audio length.
            # All candidates for SAME audio will have SAME trace length.
            mean_trace = np.mean(traces_arr, axis=0)
            results_dict[(0, word, speaker, category)] = mean_trace
            
            # Debug logging
            # if category == "Target":
            #    print(f"Computed trace for {word}, max prob: {mean_trace.max():.4f}")
        else:
            results_dict[(0, word, speaker, category)] = [np.nan]

print(f"Processing complete. Accuracy checks (if enabled): {cor}/{tot}")

# Save Results (similar to original)
output_path = os.path.join(basepath, 'competition_1.csv')
print(f"Saving to {output_path}...")

# Simple CSV save
records = []
max_len = 0
for key, val in results_dict.items():
    if len(val) > max_len:
        max_len = len(val)

for key, val in results_dict.items():
    # keys: (0, word, speaker, category)
    trace = list(val)
    # Pad
    padded = trace + [np.nan]*(max_len - len(trace))
    
    record = {
        "word": key[1],
        "speaker": key[2],
        "category": key[3]
    }
    for i, v in enumerate(padded):
        record[f"t_{i}"] = v
    records.append(record)

if records:
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print("Done.")
else:
    print("No records to save.")
