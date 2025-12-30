import argparse
from pathlib import Path

import librosa
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor


def main(model_dir: Path, audio_path: Path, predict_time: bool) -> None:
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

    audio, _ = librosa.load(audio_path, sr=16000)

    # model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]

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
    # model.generation_config.update(**gen_kwargs)
    prefix = "<|startoftranscript|><|en|><|transcribe|>"
    if not predict_time:
        prefix = f"{prefix}<|notimestamps|>"
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")
    prefix_ids.input_ids.to(device)

    output = model.generate(
        **inputs,
        **gen_kwargs,
        forced_decoder_ids=prefix_ids.input_ids,
    )

    pred_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(output["sequences"][0])
    )
    print(pred_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--predict_time", action="store_true", default=False)
    main(**vars(parser.parse_args()))
