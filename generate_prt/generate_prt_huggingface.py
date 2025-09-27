import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

login(token="") # INSERT HUGGINGFACE TOKENS HERE

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
DO_SAMPLE = TEMPERATURE > 0

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # A100 supports bf16
)

def normalize_verdict(verdict: str) -> str:
    verdict = verdict.strip().lower()
    if verdict == "permit":
        return "COMPLIANT"
    elif verdict == "forbid":
        return "NONCOMPLIANT"
    elif verdict == "compliant":
        return "COMPLIANT"
    elif verdict == "noncompliant":
        return "NONCOMPLIANT"
    else:
        raise ValueError(f"Unrecognized verdict label: {verdict}")

def extract_policy_clauses(policy_clauses: str, policy_df: dict) -> str:
    ids = [clause.strip() for clause in policy_clauses.split(",")]
    
    # Collect corresponding descriptions
    results = []
    for cid in ids:
        if cid in policy_df:
            results.append(f"{cid}: {policy_df[cid]}")
        else:
            results.append(f"{cid}: [Description not found]")
    
    return "\n\n".join(results)


def build_prompt(policy: str, case: str, verdict: str) -> str:
    prompt = f"""Given the following information:

POLICY:
{policy}

CASE:
{case}

VERDICT:
{verdict.upper()}

INSTRUCTIONS:
It has been established that the case is {verdict.upper()} with respect to the policy. Based on this, you are required to do the following tasks:
    1. Analyze the case and provide a step-by-step reasoning trace as to why the case is considered {verdict.upper()} with respect to the policy's written specifications and stipulations.
    2. When constructing your reasoning trace, be specific, informative, and cite sections or clauses of the policy where the case complies or violates (e.g. Article 1, Article 2, Article 3, etc.).
    3. The Article number you cite should depend on the policy
    4. Provide your reasoning trace in an enumerated format. Example: 1., 2., 3., etc.
    5. The last number should explicitly state if the case being evaluated is COMPLIANT or NONPLIANT to the policy. Example: "10. Therefore the case is COMPLIANT/NONCOMPLIANT to the policy".
    6. Refer to the desired output below and give your output directly.

EXAMPLE DESIRED OUTPUT FORMAT (Article numbers are made-up):
1. The case involves a covered entity (Dr. Johnson) and an individual (Jane Smith) as per the policy's definition of covered entities (Article 1).
2. The case describes a situation where the covered entity (Dr. Johnson) required the individual (Jane Smith) to waive her rights under GDPR regulations as a condition for the provision of treatment (Article 2).
3. The policy explicitly states that covered entities cannot require individuals to waive their GDPR rights as a condition for the provision of treatment, payment, enrollment in a health plan, or eligibility for benefits (Article 3).
4. Therefore, the case is considered NONCOMPLIANT with respect to the policy.

OUTPUT:"""

    return prompt

def generate_text_hf(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    print("GOING IN HUGGINGFACE FUNCTION")
    try:
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        terminator_ids = []
        if tokenizer.eos_token_id is not None:
            terminator_ids.append(tokenizer.eos_token_id)
        for token_str in ["<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, int) and token_id not in terminator_ids:
                terminator_ids.append(token_id)
        effective_eos_token_id = terminator_ids if terminator_ids else tokenizer.eos_token_id

        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
            "eos_token_id": effective_eos_token_id,
            "use_cache": True
        }
        if effective_eos_token_id is not None:
            generation_kwargs["eos_token_id"] = effective_eos_token_id

        with torch.inference_mode():
            outputs = model.generate(input_ids, **generation_kwargs)

        response_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return response
    except Exception as e:
        print(f"Error during text generation: {e}")
        return None

# Read policy text
with open('modelspec/modelspec.txt', 'r', encoding='utf-8') as file:
    policy_text = file.read()

docs_df = pd.read_csv('modelspec/train.csv', encoding_errors="replace")
reasoning_list = []
counter = 0

model_id = "Equall/SaulLM-54B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #quantization_config=quant, # comment if dont want quantization
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

model.eval()
#model = torch.compile(model) 

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

for _, row in docs_df.iterrows():
    case_document = row["document"]
    verdict = normalize_verdict(row["verdict"])
    prompt = build_prompt(policy_text, case_document, verdict)
    response = generate_text_hf(prompt, model, tokenizer)
    print("\n",response)
    reasoning_list.append(response)
    counter += 1
    print(counter)

docs_df["reasoning"] = reasoning_list
docs_df.to_csv("modelspec_saul54b_reasoning.csv", index=False)
