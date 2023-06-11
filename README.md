# PreAndTune-LLM-Pro

This repository contains scripts for LLM pretraining and finetuing (SFT).

**LoRA & DeepSpeed are supported by this repository.**

The repository is based on an already popular project.

## Supported LLM

- [LLaMA 1 & 2](https://huggingface.co/meta-llama)
- [Baichuan 2](https://huggingface.co/baichuan-inc)
- [Mistral](https://huggingface.co/mistralai)

## Pretrain (Continual Pretrain)

1. You should provide the model name (huggingface) or local model path before starting continual pre-training process of LLM.

2. You should prepare the training data in plain text in the format of **markdown or txt** for pretraining. An example can be found at [here](llm_pretrain/data/A_Guide_to_Writing_the_NeurIPS_Impact_Statement.md). You can add more text corpus in the data folder.

3. Launch the pretrain process with the following commands:

```bash
pip install -r requirements.txt
cd llm_pretrain
./pretrain_llama.sh
```
Please note that the parameter settings of these models may vary.

## SFT

1. You should provide the model name (huggingface) or local model path before starting the fine-tuning process of LLM.

2. Prepare the training data by adding your own task data like the example in [sft_examples.json](llm_sft/data/sft_examples.json), which is similar to the [alpaca_data.json](llm_sft/data/alpaca_data.json)

The format of the data is provided below:

```json
{
    "binary_selection": [
    {
            "instruction": "Does the following text violate the law?\nText: OH MY FUCKING GOD",
            "output": "No"
    },
    ...
    ],
    "another_task_name": [
    {
            "instruction": "How are you?",
            "output": "Not bad."
    },
    ...
    ],
    ...
}
```

Note that if [alpaca_data.json](llm_sft/data/alpaca_data.json) is present in the data folder, the script will utilize it as part of the training data.

3. Launch the fine-tuning process.

### Full Parameters

```bash
pip install -r requirements.txt
cd llm_sft
./train_llama.sh
```

### LoRA

```bash
pip install -r requirements.txt
cd llm_sft
./train_baichuan_LORA.sh
```

You can adjust the configurations in the [train_lora.py](llm_sft/train_lora.py). In our experiments, for Baichuan, your transformers version should be >= 4.29.0 and < 4.34.0.

Again, the parameter settings of these models may differ according to the requirements.


## File Structure

```
.
├── LICENSE
├── README.md
├── llm_pretrain_clean
│   ├── data
│   │   └── A_Guide_to_Writing_the_NeurIPS_Impact_Statement.md
│   ├── evaluation
│   │   └── inference_single.py
│   ├── generate_pretrain_data.py
│   ├── pretrain.py
│   ├── pretrain_baichuan2.sh
│   ├── pretrain_llama.sh
│   ├── pretrain_mistral.sh
│   ├── requirementsX.txt
│   └── utils.py
└── sft_model_clean
    ├── README.md
    ├── configs
    │   └── default_offload_opt_param.json
    ├── data
    │   ├── alpaca_data.json
    │   └── sft_examples.json
    ├── evaluation
    │   └── inference_single.py
    ├── generate_sft_data.py
    ├── requirementsX.txt
    ├── train.py
    ├── train_baichuan.sh
    ├── train_baichuan_LORA.sh
    ├── train_llama.sh
    ├── train_lora.py
    ├── train_mistral.sh
    └── utils.py
```