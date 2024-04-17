# Fine Tune GPT2 on wikitext dataset with FSDP
This repo contains code to finetune GPT2 models on wikitext datasets. It allows users to train with a larger batch size on single node multi-GPU systems by using FSDP.

## How To Run?

1. Install dependencies
```
pip install -r requirements.txt
```
2. Modify training hyperparameters in `configs/config.yaml`. Optionally, you can override hyperparameters from the commandline. See hydra documentation for more details.
   - `enable_fsdp`: enables FSDP training for a single node with multiple nodes. Currently, it automatically scales to all available GPUs. `enable_fsdp=False` runs single GPU training.
3. Specify GPT2 variant in `configs/config.yaml` using the `model` hyperparameter. Default value is `model: "gpt2"` which uses the 144M parameter pretrained model. You can use GPT2 Large (774M parameters) by specifying `model: "openai-community/gpt2-large"`. Currently we only support [GPT2LMHeadModel](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2LMHeadModel) class of models, but this can be extended to support other variants as well.
4. Currently the dataset is hard-coded as the Wikitext (`wikitext-103-v1`) dataset in the code (see [finetune_llm.py](src/finetune_llm.py)). But this was can be modified as per user requirements and in the future, we will support specifying dataset in the config file.
5. Run training script:
```
python src/finetune_llm.py
```
6. Use fine-tuned model for text generation by specifying the model checkpoint and prompts in the [Text Generation notebook](notebooks/generate_samples.ipynb).

## Features
- For GPT2 (144M parameters), this supports batch size = 8 with single GPU setup.
- GPT2 (144M) training can be scaled to batch size = 40 on 4x Nvidia T4 GPU systems with 16GB VRAM each. Tested on g4dn.12x large EC2 instance.
- We use FSDP to scale to multiple GPUs and support a larger training batch size. The user only needs to specify `enable_fsdp=True` to enable FSDP and automatically scale to all available GPUs.
- This repo does not use huggingface's Trainer and Accelerate modules but rather implements its own's [Trainer](src/trainer.py) class. This was done to develop a deeper understanding by using torch-native FSDP.
- For evaluation metrics, we implement loss and perplexity ourselves and use HuggingFace's evaluate module to compute `accuracy`. This was done to understand how the implementation of the two approaches differ when used with FSDP.


