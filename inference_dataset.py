import argparse
import json
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor

DATA_PREFIX = Path("/mnt/data/cyhuang/ami_segments")


def main(
    model_dir: Path, jsonl_path: Path, save_path: Path, predict_time: bool
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir, dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    tokenizer = WhisperTokenizer.from_pretrained(model_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

    processor = WhisperProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    outputs = []

    """
    prefix = "<|startoftranscript|><|en|><|transcribe|>"
    if not predict_time:
        prefix = f"{prefix}<|notimestamps|>"
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")
    prefix_ids.input_ids.to(device)
    """
    no_ts = not predict_time
    forced = processor.get_decoder_prompt_ids(
        language="en", task="transcribe", no_timestamps=no_ts
    )

    for item in metadata:
        audio_path = DATA_PREFIX / item["wavs"][0]
        print(f"Processing audio: {audio_path}")
        index = item["id"]
        audio, _ = librosa.load(audio_path, sr=16000)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            # truncation=False, # This influences the output!!!
            # padding="longest",
            # return_attention_mask=True,
        )
        inputs = inputs.to(device, dtype=dtype)

        gen_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 3,
            "return_timestamps": predict_time,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.1,
            "language": "english",
            "task": "transcribe",
            "return_dict_in_generate": True,
        }

        output = model.generate(
            **inputs,
            **gen_kwargs,
            forced_decoder_ids=forced,
        )

        pred_text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(output["sequences"][0])
        )
        print(pred_text)
        outputs.append(f"{index} {pred_text}\n")

    with save_path.open(mode="w") as f:
        for line in outputs:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--predict_time", action="store_true", default=False)
    main(**vars(parser.parse_args()))
