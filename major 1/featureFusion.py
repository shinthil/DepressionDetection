import os
import csv
import gc
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

# =================== ARGPARSE ===================
parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', default="./audio")
parser.add_argument('--transcript_dir', default="./transcript")
parser.add_argument('--label_csv', default="./avec_combined_labels (1).csv")
parser.add_argument('--out_feature_file', default="fused_features.npy")
parser.add_argument('--out_label_file', default="labels.npy")
parser.add_argument('--out_pid_file', default="pids.csv")
parser.add_argument('--max_audio_duration', type=float, default=15)
parser.add_argument('--skip_step', type=int, default=1)
args = parser.parse_args()

# =================== LOAD MODELS ===================
print("🔄 Loading models...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model = wav2vec_model.to(device)

# =================== FEATURE FUNCTIONS ===================
def extract_audio_features(audio_path):
    try:
        speech, _ = librosa.load(audio_path, sr=16000, duration=args.max_audio_duration)
        speech = librosa.util.normalize(speech)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float16)
    except Exception as e:
        return f"AudioError: {str(e)}"

def extract_text_features(transcript_path):
    try:
        df = pd.read_csv(transcript_path)
        text = " ".join(df.iloc[:, -1].astype(str))
        return text_model.encode(text).astype(np.float16)
    except Exception as e:
        return f"TextError: {str(e)}"

# =================== FUSION ===================
print("🔄 Starting feature fusion...")
label_df = pd.read_csv(args.label_csv)
label_df.columns = label_df.columns.str.strip()

features, labels, pids, failures = [], [], [], []
success_count = 0

for idx, (_, row) in enumerate(tqdm(label_df.iterrows(), total=len(label_df))):
    if idx % args.skip_step != 0:
        continue

    pid = str(int(row['Participant_ID']))
    label = row['PHQ8_Score']  # multiclass
    audio_path = os.path.join(args.audio_dir, f"{pid}_AUDIO.wav")
    transcript_path = os.path.join(args.transcript_dir, f"{pid}_TRANSCRIPT.csv")

    if os.path.exists(audio_path) and os.path.exists(transcript_path):
        audio_feat = extract_audio_features(audio_path)
        text_feat = extract_text_features(transcript_path)

        if isinstance(audio_feat, str) or isinstance(text_feat, str):
            failures.append([pid, audio_feat if isinstance(audio_feat, str) else '', text_feat if isinstance(text_feat, str) else ''])
            continue

        if audio_feat.shape[0] != 768 or text_feat.shape[0] != 384:
            failures.append([pid, 'Audio/Text dimension mismatch', ''])
            continue

        fused = np.concatenate([audio_feat, text_feat])
        features.append(fused)
        labels.append(label)
        pids.append(pid)
        success_count += 1
    else:
        failures.append([pid, 'Missing file(s)', ''])

    gc.collect()

print(f"\n✅ Total successful fusions: {success_count}")

# =================== SAVE ===================
if success_count > 0:
    print("💾 Saving output files...")
    np.save(args.out_feature_file, np.array(features))
    np.save(args.out_label_file, np.array(labels))

    with open(args.out_pid_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Participant_ID'])
        for pid in pids:
            writer.writerow([pid])

    pd.DataFrame(failures, columns=['Participant_ID', 'AudioError', 'TextError']).to_csv("fusion_failures.csv", index=False)
    print("✅ Done. Upload .npy files to Colab for modeling.")
else:
    print("❌ No data saved. Check logs above for failed paths or memory errors.")
