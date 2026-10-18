import os
import torch
import kagglehub
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
import evaluate

# 1. Kaggle se dataset download
print("1/6: Kaggle dataset check/download kar rahe hain...")
dataset_dir = kagglehub.dataset_download("tenffe/common-voice-zh-cn")
print(f"Dataset location: {dataset_dir}")

# 2. Base Model Setup (Direct Hugging Face se)
model_id = "openai/whisper-base"
print(f"2/6: Model setup from {model_id}...")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Chinese", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="Chinese", task="transcribe")

# 3. LoRA Configuration
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()
print("LoRA trainable parameters:")
model.print_trainable_parameters()

# 4. Dataset Loading & Audio Path Fix
print("3/6: Dataset load aur process kar rahe hain...")
saved_dataset_path = os.path.join(dataset_dir, "common_voice_zh_CN")
if not os.path.isfile(os.path.join(saved_dataset_path, "dataset_dict.json")):
    saved_dataset_path = dataset_dir

if os.path.isfile(os.path.join(saved_dataset_path, "dataset_dict.json")):
    saved_dataset = load_from_disk(saved_dataset_path)
    if "train" not in saved_dataset:
        raise ValueError(f"Dataset mein train split nahi mila: {saved_dataset_path}")
    train_split = saved_dataset["train"]
    eval_split = saved_dataset.get("test", saved_dataset.get("validation"))
    if eval_split is None:
        split_data = train_split.train_test_split(test_size=0.1, seed=42)
        train_split, eval_split = split_data["train"], split_data["test"]
else:
    data_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                  if f.endswith((".csv", ".tsv"))]
    if not data_files:
        raise FileNotFoundError(f"Directory mein saved Hugging Face dataset ya CSV/TSV nahi mili: {dataset_dir}")
    data_path = data_files[0]
    delimiter = "\t" if data_path.endswith(".tsv") else ","
    train_split = load_dataset("csv", data_files=data_path, delimiter=delimiter)["train"]
    split_data = train_split.train_test_split(test_size=0.1, seed=42)
    train_split, eval_split = split_data["train"], split_data["test"]

audio_col = "audio" if "audio" in train_split.column_names else "path"
text_col = "sentence" if "sentence" in train_split.column_names else "text"
if audio_col not in train_split.column_names or text_col not in train_split.column_names:
    raise ValueError(
        f"Audio/text columns nahi milay. Available columns: {train_split.column_names}"
    )

train_split = train_split.cast_column(audio_col, Audio(sampling_rate=16000))
eval_split = eval_split.cast_column(audio_col, Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch[audio_col]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch[text_col]).input_ids
    return batch

# Local PC ke liye subset (fast training)
train_data = train_split.select(range(min(300, len(train_split))))
eval_data = eval_split.select(range(min(60, len(eval_split))))

train_data = train_data.map(prepare_dataset, remove_columns=train_split.column_names)
eval_data = eval_data.map(prepare_dataset, remove_columns=eval_split.column_names)

# 5. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 6. Metric
metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"cer": 100 * metric.compute(predictions=pred_str, references=label_str)}

# 7. Training Settings
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-lora-zh",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-3,
    warmup_steps=20,
    max_steps=100,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    report_to=["none"],
    predict_with_generate=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

# 8. Run Training
print("4/6: Training shuru ho rahi hai...")
trainer.train()

# 9. Save Trained LoRA Weights
print("5/6: Model weights save kar rahe hain...")
model.save_pretrained("./whisper-base-lora-zh-final")
processor.save_pretrained("./whisper-base-lora-zh-final")
print("6/6: LoRA training complete! Folder: ./whisper-base-lora-zh-final")