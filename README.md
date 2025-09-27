# Scaling Policy Compliance Assessment in Language Models with Policy Reasoning Traces

This is the repository for the code and data used in main experiments of the **Scaling Policy Compliance Assessment in Language Models with Policy Reasoning Traces** paper by Imperial and Tayyar Madabushi (2025).

## Dependencies

### Policy Text and Datasets
The work uses three policies listed below with corresponding policy text and compliance dataset associated with each one.

 1. Health Insurance Portability and Accountability Act (HIPAA)
 2. General Data Protection Regulation (GDPR)
 3. OpenAI Model Specifications (ModelSpec)

All policy-related datasets, including policy text, train and test set, generated PRTs are contained in their own folder:  `hipaa`, 	`gdpr`, `modelspec`.

### Models
We use Huggingface, OpenRouter API, DeepSeek API and OpenAI API when running the inference and finetuning experiments. Make sure to set these API keys when you're using the code. 

Models used in inference and prompting experiments:
 1. Qwen2.5-7B-Instruct
 2. GPT-OSS-20B
 3. GPT-5-Mini
 4. Gemini-2.5-Flash
 5. DeepSeek-R1
 6. DeepSeek-R1-Distill-Llama-8B

Models used in finetuning experiments:

 1. Qwen2.5-7B-Instruct
 2. Qwen2.5-32B-Instruct
 3. DeepSeek-R1-Distill-Llama-8B

### Python Libraries 
Refer to 	`requirements.txt` for the list of Python libraries used when running the project.

## PRT Generation

The script for generating PRTs is in the folder `generate_prt` with two scripts that can be used if the model is hosted on Huggingface (in the paper we used [SaulLM-54B](https://huggingface.co/Equall/SaulLM-54B-Instruct/discussions)) or if you want to use [DeepSeek-R1](https://api-docs.deepseek.com/quick_start/pricing) via the DeepSeek API.

You may seamlessly customize `generate_prt_api.py` if you want to use models from other APIs (e.g., OpenRouter) since DeepSeek API also uses the OpenAI SDK.

## PRT-Guided Inference

### Inference-Time Compliance (Prompting via Fewshot)

The code for inference-time compliance (see the long Table 1 in Appendix of the paper) makes use of `script.py`. As discussed in the paper, this setting has the following setup:

 1. **Standard Prompting** - has two versions: `base` which only uses policy text + case information, and `few-shot` which uses policy text + case information + few-shot examples without PRTs.
 2. **Self Feedback** - uses `self-refine` by Madaan et al (2023) as a form of self feedback type of inference.
 3. **[Method] + PRTs** - augments `PRT (rand)` which adds three (3) random `PRTs` from the train file of each policy and `PRT (rel)` which adds three (3) related `PRTs` based on judgments from an auxiliary LLM (in this case we use GPT-5-Mini.
 
#### Example Runs

Use Qwen2.5 from OpenRouter API on HIPAA policy using PRTs as few-shots:
`python script.py --model_id "qwen/qwen-2.5-7b-instruct" --policy_file "hipaa/hipaa.txt" --test_file "hipaa/test.csv" --reasoning_traces_file "hipaa/r1-reasoning.csv" --mode "few_shot_PRT"`

Use GPT-5-Mini from OpenAI API  on GDPR policy using PRTs as few-shots:
`python script.py --model_id "gpt-5-mini" --policy_file "gdpr/gdpr.txt" --test_file "gdpr/test.csv" --reasoning_traces_file "gdpr/r1-reasoning.csv" --mode "few_shot_PRT"`

Use DeepSeek-R1 from DeepSeek API on ModelSpec policy using PRTs as few-shots:
`python script.py --model_id "deepseek-reasoner" --policy_file "modelspec/modelspec.txt" --test_file "modelspec/test.csv" --reasoning_traces_file "modelspec/r1-reasoning.csv" --mode "few_shot_PRT"`

The argparse flags are as follows:
| Argument                          | Purpose                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `--model_id`                      | Hugging Face model identifier (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`). |
| `--policy_file`                   | Path to the policy document text file.                                  |
| `--test_file`                     | Path to the test data CSV file (must contain `document` and `verdict` columns). |
| `--use_hf`                        | Flag to use Hugging Face models instead of API-based models.            |
| `--use_relevant_cases`            | Flag to enable relevant case selection.                                 
| `--mode`                          | Experiment type: `direct`, `few_shot`, `zero_shot`, `self_refine`, `few_shot_PRT`, `self_refine_PRT`, `budget_forcing_PRT`, `budget_forcing_PRT_v2`. |
| `--reasoning_traces_file`         | Path to CSV file with reasoning traces (used in few-shot modes).         |
| `--num_few_shot`                  | Number of examples for few-shot prompting (default: 3).                  |
| `--output_file`                   | Path to save the detailed results CSV file.                             |

###  Training-Time Compliance (Using SFTed Models)

#### Finetuning LLMs using PRT Dataset

The script for finetuning Huggingface models using the collection of generated PRTs is in the `finetuning` folder using the `script.py`. The following PRT dataset compilations were used in the paper:

 1. `all_generalist_withpol.json` - compilation train splits (case-verdict) with generated Generalist PRTs and specific policy clauses of all the three policies (HIPAA, GDPR, ModelSpec). This was used in Table 2 and Figure 4 of the paper.
 2. `hipaa_generalist_withpol.json` - same format above but for HIPAA-only instances.
 3. `gdpr_generalist_withpol.json` - same format above but for GDPR-only instances.
 4. `modelspec_generalist_withpol.json` - same format above but for ModelSpec-only instances.

#### Example Run

Finetuning Qwen2.5-7B-Instruct on the collection of all policies:

CUDA_VISIBLE_DEVICES=1 python script.py \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_path "all_generalist_withpol.json" \
    --output_dir "INSERT FOLDER HERE" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --max_seq_length 16384 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --use_bf16 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --save_total_limit 1 \
    --save_merged_model \
    --validation_split_percentage 0 \

### Inference Using Finetuned LLM via Huggingface

If you don't want to finetuned your own model, we release the models we finetuned in this Huggingface collection: https://huggingface.co/collections/josephimperial/policy-reasoning-traces-68d7a49dba6de60357fc0ff4

You may use the same inference script `script.py` in the top of the repository while adding the `--use_hf` flag to ensure that it's going to source the model from Huggingface.

Please ensure that you set the `CUDA_VISIBLE_DEVICES` or any CUDA-specific settings on your machine in order to run the script properly. 

## Citation
Waiting for Arxiv.

## Contact
If you have any questions, please contact Joseph Imperial (jmri20@bath.ac.uk).

> Written with [StackEdit](https://stackedit.io/).
