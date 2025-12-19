import torch
import librosa
import re
import random
import logging
import sys
import os
from torch.utils.data import Dataset
from transformers import (
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# ==========================================
# 0. Logger Setup
# ==========================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Configuration
# ==========================================
MODEL_SIZE = "small"
MODEL_DIR = "modified_whisper_model"
OUTPUT_DIR = "./whisper-finetuned-multitalker"
TIMESTAMP_DROP_PROB = 0.5

# --- TRAIN DATA ---
TRAIN_WAV_SCP = "./train_v1/wav.scp"
TRAIN_TEXT_FILE = "./train_v1/text"

# --- VALIDATION DATA ---
VAL_WAV_SCP = "./valid_v1/wav.scp"
VAL_TEXT_FILE = "./valid_v1/text"


# ==========================================
# 2. Dataset & Collator
# ==========================================
class ESPnetStyleDataset(Dataset):
    def __init__(
        self, wav_scp_path, text_path, tokenizer, feature_extractor, split_name="train"
    ):
        self.data = []
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.timestamp_pattern = re.compile(r"<\|\d+\.\d+\|>")
        self.split_name = split_name

        wav_paths = {}
        with open(wav_scp_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_paths[parts[0]] = parts[1]

        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    if audio_id in wav_paths:
                        self.data.append({"path": wav_paths[audio_id], "text": text})

        logger.info(
            f"[{split_name.upper()}] Dataset loaded: {len(self.data)} samples found."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, _ = librosa.load(item["path"], sr=16000)
        input_features = self.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        raw_text = item["text"]

        # Determine timestamp logic
        if self.split_name == "validation":
            use_timestamps = True
        else:
            use_timestamps = random.random() > TIMESTAMP_DROP_PROB

        if use_timestamps:
            processed_text = raw_text
            prefix_tokens = [
                self.tokenizer.bos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|en|>"),
                self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            ]
        else:
            processed_text = self.timestamp_pattern.sub("", raw_text)
            processed_text = " ".join(processed_text.split())
            prefix_tokens = [
                self.tokenizer.bos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|en|>"),
                self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                self.tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
            ]

        text_ids = self.tokenizer.encode(processed_text, add_special_tokens=False)
        suffix_tokens = [self.tokenizer.eos_token_id]
        labels = prefix_tokens + text_ids + suffix_tokens

        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ==========================================
# 3. Custom Callback
# ==========================================
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            _loss = logs.get("loss", None)
            _lr = logs.get("learning_rate", None)
            _epoch = logs.get("epoch", None)
            _grad = logs.get("grad_norm", 0.0)

            if _loss is not None:
                grad_str = f"{_grad:.4f}" if _grad is not None else "NaN"
                logger.info(
                    f"Step {state.global_step} | Epoch: {_epoch:.2f} | Loss: {_loss:.4f} | Grad: {grad_str} | LR: {_lr:.2e}"
                )


# ==========================================
# 4. Main Training Loop
# ==========================================
def train():
    logger.info("Initializing Model and Tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

    # 1. Determine Batch Size
    if "large" in MODEL_SIZE or "large" in MODEL_DIR:
        batch_size = 1
        grad_accum = 1
    else:
        batch_size = 2
        grad_accum = 1

    logger.info(
        f"Configuration -> Model: {MODEL_SIZE}, Batch: {batch_size}, Accum: {grad_accum}"
    )

    # 2. Check Precision Support
    # We strictly enforce BF16 here as requested.
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning(
            "WARNING: You requested BF16, but your GPU does not appear to support it. Training might crash or fall back to FP32."
        )

    # 3. Dataset Setup
    logger.info("Loading Training Data...")
    train_dataset = ESPnetStyleDataset(
        wav_scp_path=TRAIN_WAV_SCP,
        text_path=TRAIN_TEXT_FILE,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        split_name="train",
    )

    logger.info("Loading Validation Data...")
    val_dataset = ESPnetStyleDataset(
        wav_scp_path=VAL_WAV_SCP,
        text_path=VAL_TEXT_FILE,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        split_name="validation",
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        learning_rate=1e-6,
        lr_scheduler_type="linear",
        num_train_epochs=2,
        optim="adamw_torch",
        gradient_accumulation_steps=grad_accum,
        warmup_steps=0,
        # --- PRECISION SETTINGS ---
        bf16=True,  # Enabled BF16
        fp16=False,  # Disabled FP16
        # --------------------------
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Force English during validation
    model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            feature_extractor, tokenizer
        ),
        processing_class=tokenizer,
        callbacks=[PrinterCallback],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info(f"Training completed. Training Loss: {train_result.training_loss}")
    logger.info("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
