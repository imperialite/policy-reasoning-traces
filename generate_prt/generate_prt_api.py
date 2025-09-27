import os
from openai import OpenAI
from typing import List, Tuple
import pandas as pd
import csv

# SET DEEPSEEK API KEY HERE
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

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
    
def normalize_verdict_applicability(verdict: str) -> str:
    verdict = verdict.strip().lower()
    if verdict == "applicable":
        return "APPLICABLE"
    elif verdict == "not applicable":
        return "NOT APPLICABLE"
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
    
    return "\n".join(results)

def build_prompt(policy: str, case: str, verdict: str) -> str:
    prompt=f"""Given the following information:

### POLICY:
{policy}

### CASE:
{case}

### VERDICT:
{verdict.upper()}

### INSTRUCTIONS:
It has been established that the case is {verdict.upper()} with respect to the policy. Based on this, you are required to do the following tasks:
    1. Analyze the case and provide a step-by-step reasoning trace as to why the case is considered {verdict.upper()} with respect to the policy's written specifications and stipulations.
    2. When constructing your reasoning trace, be specific, informative, and cite sections or clauses of the policy where the case complies or violates (e.g. Article 9, Article 28, etc.).
    3. Provide your reasoning trace in an enumerated format. Example: 1., 2., 3., etc.
    4. The last number should explicitly state if the case being evaluated is COMPLIANT or NONPLIANT to the policy. Example: "10. Therefore the case is COMPLIANT/NONCOMPLIANT to the policy".
    5. Refer to the desired output below and give your output directly.

### EXAMPLE DESIRED OUTPUT FORMAT:
1. The case involves a covered entity (Dr. Johnson) and an individual (Jane Smith) as per the policy's definition of covered entities (Article 28).
2. The case describes a situation where the covered entity (Dr. Johnson) required the individual (Jane Smith) to waive her rights under GDPR regulations as a condition for the provision of treatment (Article 9).
3. The policy explicitly states that covered entities cannot require individuals to waive their GDPR rights as a condition for the provision of treatment, payment, enrollment in a health plan, or eligibility for benefits (Article 89).
4. Therefore, the case is considered NONCOMPLIANT with respect to the policy.

### OUTPUT:"""

    return prompt

with open('gdpr/gdpr.txt', 'r', encoding='utf-8') as file:
	policy_text = file.read()

# Read policy in csv format
policy_df = pd.read_csv('gdpr/gdpr.csv', encoding_errors="replace")
policy_dict = dict(zip(policy_df["section"], policy_df["description"]))

docs_df = pd.read_csv('gdpr/train.csv', encoding_errors="replace")
docs_df.head()

cot_list = []
response_list = []
counter = 0

for _, row in docs_df.iterrows():
    case_document = row["document"]
    verdict = normalize_verdict(row["verdict"]) # change function
    prompt = build_prompt(policy_text, case_document, verdict)
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )
    
    # Save reasoning traces
    reasoning = response.choices[0].message.reasoning_content
    cot_list.append(reasoning)

    # Save response
    content = response.choices[0].message.content
    response_list.append(content)
    counter += 1
    print(counter)

docs_df["reasoning"] = cot_list
docs_df["response"] = response_list
docs_df.to_csv("r1-reasoning.csv", index=False)