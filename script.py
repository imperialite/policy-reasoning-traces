import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import re
import pandas as pd
import torch
torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import ast
import warnings
from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
import time
from openai import RateLimitError

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set.*")

try:
    from huggingface_hub import login
    _hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if _hf_token:
        login(token=_hf_token)
except Exception:
    pass

# SET YOUR API KEYS HERE
# OpenAI API
client_openai = OpenAI(api_key="")
# Deepseek API
client_deepseek = OpenAI(api_key="", base_url="https://api.deepseek.com")
# OpenRouter API
client_openrouter = OpenAI(api_key="", base_url="https://openrouter.ai/api/v1")

MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.7
DO_SAMPLE = TEMPERATURE > 0

torch.cuda.empty_cache()

# Count tokens
def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Load models
def load_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval() 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Model and Tokenizer loaded.")
    return model, tokenizer

def extract_policy_clauses(policy_clauses: str, policy_df: dict) -> str:
    ids = [clause.strip() for clause in policy_clauses.split(",")]

    if ids and "164." in ids[0]:
        ids = [cid for cid in ids if re.search(r"\([^)]+\)", cid)]

    results = []
    for cid in ids:
        if cid in policy_df:
            results.append(f"{cid}:\n{policy_df[cid]}")
        else:
            results.append(f"{cid}: [Description not found]")
    
    return "\n\n".join(results)
    
def _require_any_client() -> None:
    if not (client_openai or client_deepseek or client_openrouter):
        raise RuntimeError(
            "No text-generation API client is configured. "
            "Set at least one of OPENAI_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY in your environment."
        )

def generate_text_hf(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = MAX_NEW_TOKENS) -> Optional[str]:

    model.eval() 
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        terminator_ids = []
        if tokenizer.eos_token_id is not None:
            terminator_ids.append(tokenizer.eos_token_id)

        for token_str in ["<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, int) and token_id not in terminator_ids:
                terminator_ids.append(token_id)
        
        effective_eos_token_id = terminator_ids if terminator_ids else tokenizer.eos_token_id

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
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

def generate_text_api(
    prompt: str,
    model: str,
    max_new_tokens: int = MAX_NEW_TOKENS) -> Optional[str]:

    _require_any_client()

    try:
        messages = [{"role": "user", "content": prompt}]

        # DeepSeek "reasoner"
        if "deepseek-reasoner" in model.lower() and client_deepseek:
            response = client_deepseek.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=TEMPERATURE
            )
            answer = response.choices[0].message.content.strip()
            reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") or ""
            if reasoning_content:
                answer = answer + "\n\nREASONING CONTENT:\n\n" + reasoning_content
            return answer

        # OpenAI
        if client_openai and any(k in model.lower() for k in ["gpt-4", "gpt-5", "o3", "o4"]):
            kwargs = {"model": model, "messages": messages}
            if "o3" in model.lower() or "o4" in model.lower():
                kwargs["reasoning_effort"] = "medium"
            if "gpt-4" in model.lower():
                kwargs["temperature"] = TEMPERATURE
            response = client_openai.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        # OpenRouter (proxy to many models)
        if client_openrouter:
            response = client_openrouter.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()

        print("No suitable client found for the requested model.")
        return None

    except Exception as e:
        print(f"Error during text generation: {e}")
        return None


def extract_final_judgment(text: Optional[str]) -> str:
    if text is None:
        return "ERROR"

    match_compliant = re.search(r"(?:Final|Preliminary)\s+Judgment:\s*(COMPLIANT)", text, re.IGNORECASE | re.DOTALL)
    match_noncompliant = re.search(r"(?:Final|Preliminary)\s+Judgment:\s*(NONCOMPLIANT)", text, re.IGNORECASE | re.DOTALL)

    if match_compliant:
        return "COMPLIANT"
    if match_noncompliant:
        return "NONCOMPLIANT"

    last_lines = text.strip().split('\n')[-2:]
    for line in reversed(last_lines):
        line_upper = line.upper()
        if "COMPLIANT" in line_upper and "NONCOMPLIANT" not in line_upper:
            return "COMPLIANT"
        if "NONCOMPLIANT" in line_upper:
            return "NONCOMPLIANT"

    text_upper = text.upper()
    if "COMPLIANT" in text_upper and "NONCOMPLIANT" not in text_upper:
        print("Warning: Verdict found loosely in text: COMPLIANT")
        return "COMPLIANT"
    if "NONCOMPLIANT" in text_upper:
        print("Warning: Verdict found loosely in text: NONCOMPLIANT")
        return "NONCOMPLIANT"

    print(f"Warning: Could not determine verdict from text: {text[:200]}...")
    return "UNDETERMINED"


def retrieve_relevant_cases(document, policy, cases, k):
    print("Number of cases:", len(cases))

    prompt = f"""INSTRUCTIONS: You are a helpful assistant that compares written case examples for similarity. You must select the {k} candidate case index{'es' if k > 1 else ''} that are most similar to the input case in terms of policy compliance or violation.
        
Consider **all** candidate cases before deciding. Do not rely on names, addresses, or identifiers as they are anonymized.
    
Only choose from the index range **0 to {len(cases) - 1}**. Do **not** output any index outside this range.
    
Only output **exactly** {k} integer index{'es' if k > 1 else ''}, separated by commas, e.g., `1, 5, 8`. Do not include explanations, labels, or words. Just the indices on one line.

Here is the policy:

{policy}

Here is the input case:

{document}

Here are other candidate cases, each labeled by index:

"""
    for idx, rec in enumerate(cases):
        prompt += f"{idx}: \"{rec['document']}\"\n\n"

    prompt += "\n\nOutput:"

    wait_time = 2
    while True:
        try:
            if not client_openai:
                raise RuntimeError("OPENAI_API_KEY not set for relevant-case retrieval.")
            response = client_openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at judging similarity of cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            break

        except (RateLimitError, Exception) as e:
            print(f"API error: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)

    content = response.choices[0].message.content.strip()

    if "Output:" in content:
        content = content.replace("Output:", "").strip()

    idxs = [int(x) for x in re.findall(r'\\d+', content)]
    idxs = [i for i in idxs if 0 <= i < len(cases)][:k]

    if not idxs:
        print("Warning: No valid indices found. Returning empty list.")
        return []

    relevant_cases = [cases[i] for i in idxs]
    return relevant_cases


def check_zero_shot(policy, document, model, tokenizer, USE_HF) -> str:
    prompt = f"""**INSTRUCTIONS:** You are tasked to analyze the case against the policy provided below and provide a single verdict if the case is COMPLIANT or NONCOMPLIANT with respect to the policy. Before giving the verdict, you MUST first give your reasoning process while citing relevant policy sections and how the case complies (or fails to comply) with them. Output your reasoning process and the verdict directly.

**Case:**

{document}

**Policy:**

{policy}

**Reasoning and Final Verdict (COMPLIANT or NONCOMPLIANT):**"""

    if USE_HF:
        response = generate_text_hf(prompt, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        response = generate_text_api(prompt, model, max_new_tokens=MAX_NEW_TOKENS)

    return response


def check_few_shot(policy, document, reasoning_traces, model, tokenizer, USE_HF) -> str:
    few_shot_prompt = f"""**INSTRUCTIONS:** You are tasked to analyze the input case with respect to the given policy and provide a verdict whether it is COMPLIANT or NONCOMPLIANT. In your analysis, you are required to consider the information in following the example cases and the verdict given the policy.
    
**Case:**
    
{document}
    
**Policy:**

{policy}

**Examples:**

"""
    for i, ex in enumerate(reasoning_traces):
        few_shot_prompt += f"**Case:**\n{ex['document']}\n\n"
        few_shot_prompt += f"**Verdict:** {ex['verdict']}\n\n"

    few_shot_prompt += "**Final Verdict (COMPLIANT or NONCOMPLIANT):**"
    
    if USE_HF:
        response = generate_text_hf(few_shot_prompt, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        response = generate_text_api(few_shot_prompt, model, max_new_tokens=MAX_NEW_TOKENS)
    
    return response


def check_self_refine(policy, document, model, tokenizer, USE_HF) -> str:
    prompt_initial_cot = f"""**INSTRUCTIONS:** You are tasked to analyze the input case for compliance or violation with respect to the given policy. Think step-by-step to justify your verdict whether the input case is COMPLIANT or NONCOMPLIANT. Explicitly reference specific clauses or requirements from the given policy and how the case addresses (or fails to address) them. Conclude with a preliminary judgment reasoning: 'Preliminary Judgment: COMPLIANT' or 'Preliminary Judgment: NONCOMPLIANT'.

**Case:**

{document}
    
**Policy:**

{policy}

**Reasoning Process:**"""
    if USE_HF:
        initial_cot = generate_text_hf(prompt_initial_cot, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        initial_cot = generate_text_api(prompt_initial_cot, model, max_new_tokens=MAX_NEW_TOKENS)
    if not initial_cot:
        print("Self-Refine Error: Failed Initial CoT generation.")
        return "ERROR"

    prompt_critique = f"""**INSTRUCTIONS:** You are tasked to critique the 'Initial Reasoning' provided below, which assesses a case's compliance with a policy. Identify potential flaws, missed points, misinterpretations of the policy, or areas where the reasoning could be refined. Do not give a final verdict yourself, only critique the reasoning.

**Case:**

{document}
    
**Policy:**

{policy}

**Initial Reasoning:**

{initial_cot}

**Critique:**"""

    if USE_HF:
        critique = generate_text_hf(prompt_critique, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS // 2)
    else:
        critique = generate_text_api(prompt_critique, model, max_new_tokens=MAX_NEW_TOKENS // 2)

    if not critique:
        print("Self-Refine Warning: Failed Critique generation. Proceeding without critique.")
        critique = "No critique was generated."

    prompt_refined_cot = f"""**INSTRUCTIONS:** You are tasked to refine your compliance analysis based on the 'Initial Reasoning' and the 'Critique' provided. Address the points raised in the critique and incorporate the suggestions to create a refined step-by-step reasoning process. Conclude with a final, refined judgment: 'Final Judgment: COMPLIANT' or 'Final Judgment: NONCOMPLIANT'.

**Case:**

{document}
    
**Policy:**

{policy}

**Initial Reasoning:**

{initial_cot}

**Critique:**

{critique}

**Final Verdict (COMPLIANT or NONCOMPLIANT):**"""

    if USE_HF:
        refined_cot = generate_text_hf(prompt_refined_cot, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        refined_cot = generate_text_api(prompt_refined_cot, model, max_new_tokens=MAX_NEW_TOKENS)

    if not refined_cot:
        print("Self-Refine Error: Failed Refined CoT generation.")
        print("Attempting judgment from Initial CoT as fallback.")
        return initial_cot

    final_verdict = refined_cot
    return final_verdict


def check_few_shot_PRT(policy, document, reasoning_traces, model, tokenizer, USE_HF) -> str:
    few_shot_prompt = f"""**INSTRUCTIONS:** You are tasked to analyze the case against the policy provided below and provide a single verdict if the case is COMPLIANT or NONCOMPLIANT with respect to the policy. Before giving the verdict, you MUST first give your reasoning process while citing relevant policy sections and how the case complies (or fails to comply) with them. In your analysis, you are also required to consider the information of following the example cases provided including their reasoning process and how they arrived with the verdict given the policy.
    
**Case:**
    
{document}
    
**Policy:**

{policy}

**Examples:**

"""
    for i, ex in enumerate(reasoning_traces):
        few_shot_prompt += f"**Case:**\n{ex['document']}\n\n"
        few_shot_prompt += f"**Reasoning Process:**\n{ex['reasoning']}\n\n"
        few_shot_prompt += f"**Verdict:** {ex['verdict']}\n\n"

    few_shot_prompt += "**Reasoning and Final Verdict (COMPLIANT or NONCOMPLIANT):**"
    
    if USE_HF:
        response = generate_text_hf(few_shot_prompt, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        response = generate_text_api(few_shot_prompt, model, max_new_tokens=MAX_NEW_TOKENS)
    
    return response


def check_self_refine_PRT(policy, document, reasoning_traces, model, tokenizer, USE_HF) -> str:
    print(" Self-Refine w/ Ref: Initial CoT generation...")
    prompt_initial_cot = f"""**INSTRUCTIONS:** You are tasked to analyze the input case with respect to the given policy and provide a verdict whether it is COMPLIANT or NONCOMPLIANT. Before giving the verdict, you MUST first give your reasoning process while citing relevant policy sections and how the case complies (or fails to comply) with them. Conclude with a preliminary judgment reasoning: 'Preliminary Judgment: COMPLIANT' or 'Preliminary Judgment: NONCOMPLIANT'.

**Case:**

{document}

**Policy:**

{policy}

**Reasoning Process:**"""
    if USE_HF:
        initial_cot = generate_text_hf(prompt_initial_cot, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        initial_cot = generate_text_api(prompt_initial_cot, model, max_new_tokens=MAX_NEW_TOKENS)

    if not initial_cot:
        print(" Self-Refine w/ Ref Error: Failed Initial CoT generation.")
        return "ERROR"

    print("  Self-Refine w/ Ref: Generating refined response using reference...")
    prompt_refine_with_ref = f"""**INSTRUCTIONS:** You are tasked to review your initial reasoning provided below for the input case's compliance with the given policy. Identify potential flaws, missed points, misinterpretations of the policy, or areas where the reasoning could be refined. Do not give a final verdict yourself; only critique the reasoning.

**Case:**

{document}
    
**Policy:**

{policy}

**Reasoning Process:**

{initial_cot}

Now, consider the following example cases with reasoning processes and verdicts with respect to the policy as a reference. Pay attention to its structure, how it references specific clauses of the policy for its judgment, and its step-by-step logic.

**Examples:**
"""
    for i, ex in enumerate(reasoning_traces):
        prompt_refine_with_ref += f"**Case:**\n{ex['document']}\n\n"
        prompt_refine_with_ref += f"**Reasoning Process:**\n{ex['reasoning']}\n\n"
        prompt_refine_with_ref += f"**Verdict:** {ex['verdict']}\n"

    prompt_refine_with_ref += "Considering both your initial reasoning and the approaches shown in the reference case examples, provide your final verdict for the input case.\n\n"
    prompt_refine_with_ref += "Reasoning and Final Verdict (COMPLIANT or NONCOMPLIANT):"""

    if USE_HF:
        refined_response = generate_text_hf(prompt_refine_with_ref, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    else:
        refined_response = generate_text_api(prompt_refine_with_ref, model, max_new_tokens=MAX_NEW_TOKENS)

    if not refined_response:
        print("  Self-Refine w/ Ref Error: Failed generation of refined response.")
        print("  Attempting judgment from Initial CoT as fallback.")
        return initial_cot

    return refined_response

# ----------------
# DATA LOADING
# ----------------

def load_policy(filepath: str):
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")
            if "section" not in df.columns or "description" not in df.columns:
                raise ValueError("CSV must have 'section' and 'description' columns")
            return dict(zip(df["section"], df["description"]))
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    except FileNotFoundError:
        print(f"Error: Policy file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading policy file: {e}")
        exit(1)

def load_test_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        if 'document' not in df.columns or 'verdict' not in df.columns:
            raise ValueError("CSV must contain 'document' and 'verdict' columns.")
        df['verdict'] = df['verdict'].str.upper().str.strip()
        valid_verdicts = ['COMPLIANT', 'NONCOMPLIANT']
        original_len = len(df)
        df = df[df['verdict'].isin(valid_verdicts)]
        if len(df) < original_len:
            print(f"Warning: Filtered out {original_len - len(df)} rows with invalid verdicts from test data.")
        print(f"Loaded {len(df)} valid test cases from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: Test data file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        exit(1)

def load_reasoning_traces(filepath: str) -> List[dict]:
    """
    Loads reasoning trace examples from a CSV file.
    Expected columns: 'document', 'reasoning', 'verdict'
    """
    examples = []
    if not filepath:
        print("Skipping loading reasoning traces (path not provided).")
        return examples
    try:
        df = pd.read_csv(filepath)
        required_cols = {'document', 'reasoning', 'verdict'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_cols}")
        df['verdict'] = df['verdict'].str.upper().str.strip()
        valid_verdicts = ['COMPLIANT', 'NONCOMPLIANT']
        df = df[df['verdict'].isin(valid_verdicts)]
        records = df.to_dict('records')
        for rec in records:
            examples.append(rec)
        print(f"Loaded {len(examples)} reasoning trace examples for few-shot from {filepath}")
        return examples
    except FileNotFoundError:
        print(f"Warning: Reasoning traces file not found at {filepath}. Few-shot will be skipped.")
        return []
    except Exception as e:
        print(f"Error loading reasoning traces: {e}")
        return []

# ----------------
# MAIN
# ----------------

def main(args):
    print("--- Starting Ablation Experiment ---")
    print(f"Model ID: {args.model_id}")
    print(f"Policy File: {args.policy_file}")
    print(f"Test Data File: {args.test_file}")
    print(f"Use Huggingface: {args.use_hf}")
    print(f"Use Relevant Cases: {args.use_relevant_cases}")
    print(f"Mode: {args.mode}")
    print(f"Reasoning Traces File: {args.reasoning_traces_file}")
    print(f"Num Few-Shot Examples: {args.num_few_shot}")
    print(f"Output File: {args.output_file}")

    policy_text = load_policy(args.policy_file)
    test_df = load_test_data(args.test_file)
    reasoning_examples = load_reasoning_traces(args.reasoning_traces_file)

    USE_HF = args.use_hf
    USE_RELEVANT_CASES = args.use_relevant_cases
    NUM_FEW_SHOT = args.num_few_shot

    if USE_HF:
        model, tokenizer = load_model(args.model_id)
    else:
        model = args.model_id  # keep as string for API
        tokenizer = None

    results = []
    raw_output = []
    ground_truth = test_df['verdict'].tolist()
    counter = 0

    print("\n--- Running Experiment Mode ---")
    for index, row in test_df.iterrows():
        print(f"\nProcessing test case {index + 1}/{len(test_df)}...")
        doc = row['document']
        gt_verdict = row['verdict']

        candidates = reasoning_examples
        random.shuffle(candidates)

        if args.mode == 'zero_shot':
            pred = check_zero_shot(policy_text, doc, model, tokenizer, USE_HF)

        elif args.mode == 'direct':
            if USE_HF:
                pred = generate_text_hf(doc, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
            else:
                pred = generate_text_api(doc, model, max_new_tokens=MAX_NEW_TOKENS)

        elif args.mode == 'few_shot':
            if USE_RELEVANT_CASES and 'relevant_cases' in row and pd.notna(row['relevant_cases']):
                shortlist_text = str(row['relevant_cases'])[1:-1]
                shortlist = [int(item) for item in shortlist_text.split(',') if str(item).strip().isdigit()]
                shortlist = [candidates[i] for i in shortlist if 0 <= i < len(candidates)]
                shortlist = shortlist[:NUM_FEW_SHOT]
            else:
                shortlist = random.choices(candidates, k=NUM_FEW_SHOT)
            if shortlist:
                pred = check_few_shot(policy_text, doc, shortlist, model, tokenizer, USE_HF)
            else:
                pred = "SKIPPED"

        elif args.mode == 'self_refine':
            pred = check_self_refine(policy_text, doc, model, tokenizer, USE_HF)

        elif args.mode == 'few_shot_PRT':
            if USE_RELEVANT_CASES and 'relevant_cases' in row and pd.notna(row['relevant_cases']):
                shortlist_text = str(row['relevant_cases'])[1:-1]
                shortlist = [int(item) for item in shortlist_text.split(',') if str(item).strip().isdigit()]
                shortlist = [candidates[i] for i in shortlist if 0 <= i < len(candidates)]
                shortlist = shortlist[:NUM_FEW_SHOT]
            else:
                shortlist = random.choices(candidates, k=NUM_FEW_SHOT)
            if shortlist:
                pred = check_few_shot_PRT(policy_text, doc, shortlist, model, tokenizer, USE_HF)
            else:
                pred = "SKIPPED"

        elif args.mode == 'self_refine_PRT':
            if USE_RELEVANT_CASES and 'relevant_cases' in row and pd.notna(row['relevant_cases']):
                shortlist_text = str(row['relevant_cases'])[1:-1]
                shortlist = [int(item) for item in shortlist_text.split(',') if str(item).strip().isdigit()]
                shortlist = [candidates[i] for i in shortlist if 0 <= i < len(candidates)]
                shortlist = shortlist[:NUM_FEW_SHOT]
            else:
                shortlist = random.choices(candidates, k=NUM_FEW_SHOT)
            if shortlist:
                pred = check_self_refine_PRT(policy_text, doc, shortlist, model, tokenizer, USE_HF)
            else:
                pred = "SKIPPED"
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        raw_output.append(pred)
        final_judgment_output = extract_final_judgment(pred)
        results.append(final_judgment_output)

        counter += 1
        torch.cuda.empty_cache()

    print("\n--- Experiment Results ---")
    valid_indices = [i for i, pred in enumerate(results) if pred in ["COMPLIANT", "NONCOMPLIANT"]]
    if valid_indices:
        gt_valid = [ground_truth[i] for i in valid_indices]
        pred_valid = [results[i] for i in valid_indices]
        accuracy = accuracy_score(gt_valid, pred_valid)
        f1_weighted = f1_score(gt_valid, pred_valid, average="weighted")
        f1_macro = f1_score(gt_valid, pred_valid, average="macro")
        print(f"Calculated Accuracy based on {len(valid_indices)} valid predictions: {accuracy:.4f}")
        print(f"Calculated F1 weighted based on {len(valid_indices)} valid predictions: {f1_weighted:.4f}")
        print(f"Calculated F1 macro based on {len(valid_indices)} valid predictions: {f1_macro:.4f}")
    else:
        print("No valid predictions found.")

    results_df = pd.DataFrame({
        'document': test_df['document'],
        'ground_truth': ground_truth,
        'raw_output': raw_output,
        f'pred_{args.mode}': results
    })

    output_file_name = args.output_file
    try:
        results_df.to_csv(output_file_name, index=False)
        print(f"\nDetailed results saved to {output_file_name}")
    except Exception as e:
        print(f"\nError saving results to {output_file_name}: {e}")

    print("\n--- Experiment Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM compliance experiments using Hugging Face or API models.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HF model id (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct') or API model name.")
    parser.add_argument("--policy_file", type=str, required=True,
                        help="Path to the policy document text/CSV file.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test data CSV file (must contain 'document' and 'verdict' columns).")
    parser.add_argument("--use_hf", action="store_true",
                        help="Use HF models; otherwise call API by model name.")
    parser.add_argument("--use_relevant_cases", action="store_true",
                        help="Enable relevant case selection (requires OPENAI_API_KEY).")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["direct", "few_shot", "zero_shot", "self_refine", "few_shot_PRT", "self_refine_PRT", "budget_forcing_PRT", "budget_forcing_PRT_v2"],
                        help="Experiment mode.")
    parser.add_argument("--reasoning_traces_file", type=str, default='reasoning.csv',
                        help="Path to the CSV with reasoning traces (for few-shot modes).")
    parser.add_argument("--num_few_shot", type=int, default=3,
                        help="Number of few-shot examples (default: 3).")
    parser.add_argument("--output_file", type=str, default='outputs.csv',
                        help="Path to save results CSV.")

    args = parser.parse_args()
    main(args)
