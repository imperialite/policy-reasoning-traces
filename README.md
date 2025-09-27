# Scaling Policy Compliance Assessment in Language Models with Policy Reasoning Traces

Code and data are being updated.

## Dependencies

### Policy Text and Datasets
The work uses three policies listed below with corresponding policy text and compliance dataset associated with each one.

 1. Health Insurance Portability and Accountability Act (HIPAA)
 2. General Data Protection Regulation (GDPR)
 3. OpenAI Model Specifications (ModelSpec)

All policy-related datasets, including policy text, train and test set, generated `PRTs` are contained in their own folder:  `hipaa`, 	`gdpr`, `modelspec`.

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

