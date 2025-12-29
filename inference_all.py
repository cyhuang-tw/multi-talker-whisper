import argparse
import json
from pathlib import Path
import torch
import librosa
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    GenerationConfig,
)

from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
# MODEL_DIR = "/mnt/data/cyhuang/whisper-finetuned-multitalker-v0/checkpoint-12000"  # Path to your saved checkpoint
BASE_MODEL_ID = "openai/whisper-small"  # For loading standard feature extractor
# AUDIO_PATH = "/mnt/data/cyhuang/ami_segments/eval/EN2002a/EN2002a-38.04-45.5.wav"  # Replace with your audio file
DATA_PREFIX = Path("/mnt/data/cyhuang/ami_segments")


def transcribe_no_timestamps(model_dir, jsonl_path, save_path):
    MODEL_DIR = model_dir
    print(f"Loading model from {MODEL_DIR}...")

    # 1. Load Components
    try:
        # Load feature extractor from base to ensure clean config
        feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
        # Load Tokenizer & Model from your fine-tuned checkpoint
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
        processor = WhisperProcessor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. Safety: Reset Generation Config
    # We load a fresh config to ensure no weird defaults from training interfere
    model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL_ID)

    # ==========================================
    # 3. DEFINE THE PROMPT (No Timestamps)
    # ==========================================
    # We strictly match your training sequence:
    # [SOT] -> [EN] -> [TRANSCRIBE] -> [NOTIMESTAMPS]
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
        no_timestamps=True,  # <--- This adds the <|notimestamps|> token
    )

    # CRITICAL: Ensure <|notimestamps|> is NOT suppressed
    # Standard Whisper config suppresses it by default, so we must allow it.
    notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    if model.generation_config.suppress_tokens is not None:
        if notimestamps_id in model.generation_config.suppress_tokens:
            model.generation_config.suppress_tokens.remove(notimestamps_id)

    # ==========================================
    # 4. Generate
    # ==========================================
    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    # metadata = metadata[:500]
    outputs = []
    for item in tqdm(metadata):
        audio_path = DATA_PREFIX / item["wavs"][0]
        print(f"Processing audio: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        print("Generating...")
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=225,
            num_beams=3,
            return_dict_in_generate=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

        seq = predicted_ids.sequences[0]

        # ==========================================
        # 5. Verification & Output
        # ==========================================
        # A. Check Raw Tokens
        raw_tokens = tokenizer.convert_ids_to_tokens(seq)
        print("\n" + "=" * 40)
        print("RAW TOKENS (Start of Sequence)")
        print("=" * 40)
        print(raw_tokens)

        # We expect to see <|notimestamps|> in the first few tokens
        if "<|notimestamps|>" in raw_tokens[:5]:
            print("✅ SUCCESS: Model correctly used <|notimestamps|>.")
        else:
            print(
                "⚠️ WARNING: <|notimestamps|> missing from output (Model might be confused)."
            )

        # B. Final Text
        transcription = processor.decode(seq, skip_special_tokens=False)

        print("\n" + "=" * 40)
        print("FINAL TRANSCRIPTION:")
        print("=" * 40)
        print(transcription)
        print("=" * 40)
        index = item["id"]
        outputs.append(f"{index} {transcription}\n")
    with save_path.open(mode="w") as f:
        for line in outputs:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    transcribe_no_timestamps(**vars(parser.parse_args()))
