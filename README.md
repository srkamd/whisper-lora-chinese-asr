# Fine-Tuning Whisper Base with LoRA for Chinese Speech Recognition

This project fine-tunes OpenAI's Whisper Base speech-recognition model for
Chinese (zh-CN) using Low-Rank Adaptation (LoRA) and the Common Voice Chinese
dataset on Kaggle. It is a small, GPU-aware training example intended to run
on a local Windows machine.

> **Current result:** the included training configuration completed 100
> optimizer steps on 300 training examples. It reported a character error
> rate (CER) of **85.36%** on 60 evaluation examples. This is a smoke-test /
> learning exercise, not a production-quality recognizer. The score is high;
> see [Results and limitations](#results-and-limitations).

## What the project does

1. Downloads the `tenffe/common-voice-zh-cn` dataset with KaggleHub.
2. Loads the saved Hugging Face dataset splits (or a compatible CSV/TSV
   dataset if one is provided instead).
3. Loads `openai/whisper-base` and configures its tokenizer and processor for
   Chinese transcription.
4. Adds LoRA adapters to the attention `q_proj` and `v_proj` layers.
5. Resamples audio to 16 kHz, extracts Whisper features, and fine-tunes with
   Hugging Face `Seq2SeqTrainer`.
6. Evaluates character error rate (CER) and saves the adapter and processor.

## Project files

| File or directory | Purpose |
| --- | --- |
| [`train.py`](./train.py) | Dataset loading, preprocessing, LoRA setup, training, evaluation, and saving |
| [`requirements.txt`](./requirements.txt) | Python dependencies, including CUDA-enabled PyTorch for Windows |
| `whisper-base-lora-zh/` | Trainer checkpoints and logs created during training |
| `whisper-base-lora-zh-final/` | Final LoRA adapter and processor files created after successful training |

The dataset and the original Whisper model are downloaded to local caches;
they are not included in this source repository.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11
- Several gigabytes of free disk space for the environment, dataset, model
  cache, and training outputs
- An NVIDIA GPU is recommended. The tested machine had an NVIDIA RTX A2000
  Laptop GPU with 4 GB VRAM and used CUDA-enabled PyTorch.
- Internet access for the Kaggle dataset and Hugging Face model downloads

The script can run on a CPU, but Whisper fine-tuning will be substantially
slower. The provided requirements install a CUDA 12.8 PyTorch wheel; if you
are setting up a different operating system or GPU/CUDA combination, install
the appropriate PyTorch build for that machine before installing the
remaining requirements.

## Setup on Windows

Open PowerShell in the project directory:

```powershell
cd D:\path\to\WhisperLoRA
```

Create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

The version output should begin with `Python 3.11`. If PowerShell blocks
activation, you can skip activation and use the environment's Python
executable directly, as shown in the training command below.

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that PyTorch can see the GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

If this prints `CUDA available: False`, training will use the CPU. Check that
the NVIDIA driver is installed and that the appropriate CUDA-enabled PyTorch
wheel was installed.

## Run training

From the project directory, with the virtual environment active:

```powershell
python -u -X utf8 train.py
```

Or invoke the environment directly without activating it:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 train.py
```

On its first run, the script downloads the dataset and the base model if they
are not already cached. The Kaggle dataset download is approximately 1.84 GB
before extraction. The script prints progress as it loads data, preprocesses
audio, trains, evaluates, and saves the adapter.

For the training run documented here, KaggleHub returned a saved dataset with
these splits:

| Split | Examples |
| --- | ---: |
| Train | 39,637 |
| Test | 10,581 |

The training script deliberately selects only the first **300** training
examples and first **60** examples from the evaluation split. This keeps the
default run small enough for a local demonstration; it does **not** train on
the entire dataset.

## Training configuration

The current values are set directly in [`train.py`](./train.py):

| Setting | Value |
| --- | --- |
| Base model | `openai/whisper-base` |
| Language / task | Chinese / transcription |
| LoRA target modules | `q_proj`, `v_proj` |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Selected training examples | Up to 300 |
| Selected evaluation examples | Up to 60 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 steps |
| Learning rate | `1e-3` |
| Maximum optimizer steps | 100 |
| Evaluation and checkpoint interval | 50 steps |
| Audio sample rate | 16 kHz |
| Gradient checkpointing | Enabled |

The small batch and gradient checkpointing are configured for limited GPU
memory. Adjust settings cautiously if you change the data subset or train on
hardware with different memory capacity.

## Outputs

After a successful run:

- Trainer checkpoints and logs are written under `whisper-base-lora-zh/`.
- The final adapter and its processor/tokenizer files are written under
  `whisper-base-lora-zh-final/`.

The final directory contains a LoRA adapter, **not** a standalone copy of all
Whisper model weights. To use it later, load the original
`openai/whisper-base` model and attach the adapter using PEFT.

## Results and limitations

The recorded run completed all 100 steps in about 8 minutes on the tested
RTX A2000 Laptop GPU. Its final reported metrics were:

| Metric | Value |
| --- | ---: |
| Training loss | 1.2126 |
| Evaluation loss | 1.3496 |
| Evaluation CER | 85.36% |

CER is the number of character edits needed to transform a prediction into
the reference, divided by the number of reference characters. Lower is
better. A CER of 85.36% is poor: this result should be treated as evidence
that the training pipeline runs end to end, **not** as evidence of good
recognition quality.

The run is intentionally small and has important limitations:

- Only 300 training and 60 evaluation samples are selected.
- The selected rows are the first rows in each split, not a randomized,
  representative sample.
- Training is capped at 100 optimizer steps.
- A single run and small evaluation sample do not provide a robust quality
  estimate.
- The source dataset reports its license as **Unknown** in Kaggle metadata.
  Check the dataset's terms and attribution requirements before redistributing
  its audio, transcripts, or derived artifacts.

For a meaningful experiment, review the dataset licensing, choose a
representative split, train for more steps, compare against the unmodified
base model on a held-out set, and track evaluation metrics across runs.

## Troubleshooting

### `ModuleNotFoundError` for a training dependency

Make sure the intended virtual environment is active and install the project
dependencies:

```powershell
python -m pip install -r requirements.txt
```

### CUDA is unavailable

Verify the NVIDIA driver and CUDA-enabled PyTorch installation:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The requirements file targets Windows with CUDA 12.8. A different platform
may need a different PyTorch installation command.

### Kaggle dataset download fails

Check internet access and KaggleHub's authentication requirements for your
account/environment. The script downloads the public dataset through
KaggleHub; it expects the extracted Hugging Face dataset structure or
compatible CSV/TSV files.

### Not enough GPU memory

The default configuration uses a per-device batch size of 1 and gradient
checkpointing. Close other GPU-heavy applications before training. If you
change the batch size, review gradient accumulation and the available VRAM.

## Dataset and model sources

- Dataset: [Common Voice Chinese (Simplified) on Kaggle](https://www.kaggle.com/datasets/tenffe/common-voice-zh-cn)
- Base model: [OpenAI Whisper Base on Hugging Face](https://huggingface.co/openai/whisper-base)
- Training approach: [PEFT LoRA](https://huggingface.co/docs/peft)

## License

No project license has been added. Add a `LICENSE` file before publishing if
you want to specify how others may use, modify, or redistribute this code.
The dataset and base model may have separate terms; review those terms
independently.
