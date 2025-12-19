import torch
from transformers import (
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)


def setup_and_save_model(base_model_name, output_dir):
    print(f"Loading base model: {base_model_name}")

    # 1. Load Tokenizer, Feature Extractor, and Model
    tokenizer = WhisperTokenizer.from_pretrained(base_model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

    # 2. Add the new <sc> token
    new_token = "<sc>"
    print(f"Adding special token: {new_token}")
    # We add it as a special token so it isn't split into characters
    tokenizer.add_tokens([new_token], special_tokens=True)

    # 3. Resize Model Embeddings
    # This expands the matrix and initializes the new row randomly
    model.resize_token_embeddings(len(tokenizer))

    # Optional: Verify the new token ID
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)
    print(f"New token '{new_token}' added at ID: {new_token_id}")

    # 4. Save everything to the local directory
    print(f"Saving modified model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print("Done. Ready for training.")


if __name__ == "__main__":
    # You can change "openai/whisper-small" to tiny, base, medium, etc.
    setup_and_save_model("openai/whisper-small", "modified_whisper_model")
