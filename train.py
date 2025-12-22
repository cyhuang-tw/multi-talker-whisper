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
import speech_metrics

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
MODEL_DIR = "/mnt/data/cyhuang/modified_whisper_model"
OUTPUT_DIR = "/mnt/data/cyhuang/whisper-finetuned-multitalker"

# === CRITICAL FIX: Disable Timestamp Dropping ===
# We want the model to ALWAYS predict timestamps.
TIMESTAMP_DROP_PROB = 1.0

# --- TRAIN DATA ---
TRAIN_WAV_SCP = "./train_v1/wav.scp"
TRAIN_TEXT_FILE = "./train_v1/text"

# --- VALIDATION DATA ---
VAL_WAV_SCP = "./valid_v1/wav.scp"
VAL_TEXT_FILE = "./valid_v1/text"


TS = r"<\|\d+(?:\.\d+)?\|>"
TS_RE = re.compile(TS)

# Timestamp *between* two "word" chars (letters/digits/_). Insert a space on removal.
TS_BETWEEN_WORDS_RE = re.compile(rf"(\w){TS}(\w)")


def remove_whisper_timestamps_text(
    s: str,
    *,
    collapse_spaces: bool = True,
    strip: bool = True,
) -> str:
    # 1) Prevent word concatenation when timestamps are glued between word chars.
    s = TS_BETWEEN_WORDS_RE.sub(r"\1 \2", s)

    # 2) Remove any remaining timestamps.
    s = TS_RE.sub("", s)

    # 3) Normalize whitespace if desired.
    if collapse_spaces:
        s = re.sub(r"\s+", " ", s)

    if strip:
        s = s.strip()

    return s


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

        # === CRITICAL FIX: Pre-calculate the correct Start Token ===
        # Whisper MUST start with <|startoftranscript|> (50258), NOT <|endoftext|> (50257)
        self.sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

        # Pre-calculate other special tokens for speed
        self.en_id = tokenizer.convert_tokens_to_ids("<|en|>")
        self.transcribe_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        if not os.path.exists(wav_scp_path):
            raise FileNotFoundError(f"Could not find wav.scp at: {wav_scp_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Could not find text file at: {text_path}")

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
            use_timestamps = False
        else:
            use_timestamps = random.random() > TIMESTAMP_DROP_PROB

        if use_timestamps:
            processed_text = raw_text
            # === CORRECT SEQUENCE: <|startoftranscript|> <|en|> <|transcribe|> ===
            # prefix_tokens = [self.sot_id, self.en_id, self.transcribe_id]
        else:
            processed_text = remove_whisper_timestamps_text(raw_text)
            # processed_text = self.timestamp_pattern.sub("", raw_text)
            # processed_text = " ".join(processed_text.split())
            # === CORRECT SEQUENCE: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> ===

        # text_ids = self.tokenizer.encode(processed_text, add_special_tokens=False)
        text_ids = self.tokenizer.encode(processed_text)
        # suffix_tokens = [self.tokenizer.eos_token_id]
        # labels = prefix_tokens + text_ids + suffix_tokens
        labels = text_ids

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

        # If the collator accidentally adds BOS at the start (common HF behavior), remove it
        # because we manually constructed our prefix starting with <|startoftranscript|>
        """
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        """

        batch["labels"] = labels
        return batch


# ==========================================
# 3. Printer Callback
# ==========================================
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            _loss = logs.get("loss", None)
            _lr = logs.get("learning_rate", None)
            _epoch = logs.get("epoch", None)
            _grad = logs.get("grad_norm", 0.0)
            _wer = logs.get("eval_wer", None)

            if _loss is not None:
                grad_str = f"{_grad:.4f}" if _grad is not None else "NaN"
                logger.info(
                    f"Step {state.global_step} | Epoch: {_epoch:.2f} | Loss: {_loss:.4f} | Grad: {grad_str} | LR: {_lr:.2e}"
                )

            if _wer is not None:
                logger.info(
                    f"*** EVAL RESULTS *** Step {state.global_step} | WER: {_wer:.4f}"
                )


# ==========================================
# 4. Main Training Loop
# ==========================================
def train():
    logger.info("Initializing Model and Tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.set_prefix_tokens(
        language="english", task="transcribe", predict_timestamps=False
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

    # 1. Determine Batch Size
    if "large" in MODEL_SIZE or "large" in MODEL_DIR:
        batch_size = 1
    else:
        batch_size = 2
    grad_accum = 1

    logger.info(
        f"Configuration -> Model: {MODEL_SIZE}, Batch: {batch_size}, Accum: {grad_accum}"
    )

    # 2. Dataset Setup
    logger.info("Loading Training Data...")
    train_dataset = ESPnetStyleDataset(
        TRAIN_WAV_SCP, TRAIN_TEXT_FILE, tokenizer, feature_extractor, split_name="train"
    )

    logger.info("Loading Validation Data...")
    val_dataset = ESPnetStyleDataset(
        VAL_WAV_SCP,
        VAL_TEXT_FILE,
        tokenizer,
        feature_extractor,
        split_name="validation",
    )

    # === FIX: Slice Validation Data for Speed ===
    if len(val_dataset) > 2000:
        logger.info(
            f"Optimization: Slicing validation set from {len(val_dataset)} to 500 samples."
        )
        val_dataset.data = val_dataset.data[:2000]
    # ============================================

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor, tokenizer)

    # 3. Metrics
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode with skip_special_tokens=True to remove timestamps for clean WER
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = speech_metrics.wer(pred_str, label_str)
        cer = speech_metrics.cer(pred_str, label_str)

        # wer = metric_wer.compute(predictions=pred_str, references=label_str)
        # cer = metric_cer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

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
        bf16=True,
        fp16=False,
        # --- LOGGING ---
        report_to=["tensorboard"],
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=10,
        # --- EVALUATION ---
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,  # Fix disk space issue: keep only 1 checkpoint
        metric_for_best_model="wer",
        greater_is_better=False,
        # --- GENERATION ---
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,  # Greedy decoding for speed
        remove_unused_columns=False,
    )

    # Force standard English prompt during validation
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback],
    )

    # ====================================================
    # 6. VERIFICATION: Check Start Token
    # ====================================================
    logger.info("=== MANUAL DATA INSPECTION ===")
    try:
        sample_item = train_dataset[0]
        batch = data_collator([sample_item])
        labels = batch["labels"][0]

        valid_labels = [l for l in labels.tolist() if l != -100]

        # Show Tokens
        raw_tokens = tokenizer.convert_ids_to_tokens(valid_labels)
        logger.info(f"Token list: {raw_tokens}")

        # Verify correctness
        if raw_tokens[0] == "<|startoftranscript|>":
            logger.info("✅ SUCCESS: Sequence starts with <|startoftranscript|>")
        else:
            logger.error(
                f"❌ ERROR: Sequence starts with {raw_tokens[0]} (Expected <|startoftranscript|>)"
            )

        if "<|notimestamps|>" in raw_tokens:
            logger.warning(
                "⚠️ WARNING: <|notimestamps|> found (Should be rare/none with PROB=0.0)"
            )

    except Exception as e:
        logger.error(f"Failed to inspect data: {e}")
    logger.info("==================================")
    # ====================================================

    # 7. Start Training
    if os.path.isdir(OUTPUT_DIR) and any(
        "checkpoint" in f for f in os.listdir(OUTPUT_DIR)
    ):
        logger.info(f"Resuming from latest checkpoint in {OUTPUT_DIR}")
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("Starting fresh training...")
        train_result = trainer.train()

    logger.info(f"Training completed. Training Loss: {train_result.training_loss}")
    logger.info("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
