from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import re

def extract_test_cases(content):
    # Very basic regex to match: **Input:**  "112358"\n**Output:** true
    input_matches = re.findall(r'\*\*Input:\*\*[\s"]*(.*?)["\n]', content)
    output_matches = re.findall(r'\*\*Output:\*\*[\s"]*(.*?)["\n]', content)
    return list(zip(input_matches, output_matches))


def run_test_case(code, test_input, expected_output):
    # Assuming the function is named isAdditiveNumber, for example
    global_namespace = {}
    try:
        exec(code, global_namespace)
        fn = next(v for k, v in global_namespace.items() if callable(v))
        result = fn(eval(test_input))  # careful: input may need preprocessing
        return str(result) == expected_output
    except Exception as e:
        return False



model_name = "allenai/OLMo-2-0425-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
model.eval()


# Load and preprocess dataset
ds = load_dataset("greengerong/leetcode", split="train")
ds = ds.filter(lambda x: x["python"] is not None and len(x["python"].strip()) > 0)
ds = ds.shuffle(seed=42).select(range(10))  # small batch for evaluation

# === Evaluation ===
def evaluate_code_generation(model, tokenizer, dataset, max_new_tokens=256):
    results = []

    for ex in tqdm(dataset):
        problem = ex["content"].strip()
        gt_solution = ex["python"].strip()
        slug = ex["slug"]

        # Print test cases (optional for debug/logging)
        print(f"\n--- {slug} ---")
        test_cases = extract_test_cases(problem)
        for idx, (inp, out) in enumerate(test_cases):
            print(f"Test {idx+1}: Input={inp}, Expected={out}")

        # Format the prompt
        prompt = f"# LeetCode Problem:\n{problem}\n\n# Python Solution:\ndef"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode output and trim prompt
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        gen_solution = output[len(prompt) - 4:]  # adjust -4 for "def" already in prompt

        exact_match = gen_solution.strip() == gt_solution.strip()

        results.append({
            "slug": slug,
            "prompt": prompt,
            "generated": gen_solution.strip(),
            "expected": gt_solution.strip(),
            "exact_match": exact_match,
            "test_cases": test_cases,
        })

    return results



# === Run and display ===
results = evaluate_code_generation(model, tokenizer, ds)
n_match = sum(r["exact_match"] for r in results)
print(f"\nExact Match: {n_match}/{len(results)} = {n_match / len(results):.2%}")

# Show some examples
for r in results[:3]:
    print(f"\nSlug: {r['slug']}")
    print("Prompt:\n", r["prompt"])
    print("Generated:\n", r["generated"][:50])
    print("Expected:\n", r["expected"][:50])
    print("Exact Match:", r["exact_match"][:50])
    print("="*60)


# # Load model + tokenizer
# model_name = "allenai/OLMo-2-0425-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Load LeetCode dataset
# raw_dataset = load_dataset("greengerong/leetcode", split="train")
# raw_dataset = raw_dataset.shuffle(seed=42).select(range(1))  # subset for quick testing
# # Print all column names and the first example
# print(raw_dataset.column_names, raw_dataset[0])

# # Construct prompt-completion pairs
# eval_data = []
# for example in raw_dataset:
#     if "python" not in example or not example["python"]:
#         continue
#     code = example["python"].strip()


#     if "\n" not in code:
#         continue
#     first_line, rest = code.split("\n", 1)
#     prompt = first_line + "\n    "  # mimic start of function body
#     completion = rest.strip()
#     eval_data.append({
#         "input_text": prompt,
#         "labels": prompt + completion
#     })

# # Tokenization
# def preprocess_eval(example):
#     inputs = tokenizer(
#         example["input_text"],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=256,
#     )
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             example["labels"],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=256,
#         )
#     inputs["labels"] = labels["input_ids"]
#     return {k: v.squeeze(0) for k, v in inputs.items()}

# eval_dataset = Dataset.from_list(eval_data).map(preprocess_eval)

# # Loss metric
# def compute_metrics(eval_pred: EvalPrediction):
#     logits, labels = eval_pred.predictions, eval_pred.label_ids
#     logits = torch.tensor(logits)
#     labels = torch.tensor(labels)
#     logits = logits.view(-1, logits.shape[-1])
#     labels = labels.view(-1)
#     loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#     loss = loss_fct(logits, labels)
#     return {"eval_loss": loss.item()}

# # Data collator
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Trainer setup
# eval_args = TrainingArguments(
#     output_dir="./olmo-eval-leetcode",
#     per_device_eval_batch_size=1,
#     do_train=False,
#     do_eval=True,
#     logging_steps=1,
#     report_to="none",
# )

# trainer = Trainer(
#     model=model,
#     args=eval_args,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# # Export components for reuse
# def get_eval_components():
#     return eval_dataset, data_collator, compute_metrics

# # Run eval directly if needed
# if __name__ == "__main__":
#     results = trainer.evaluate()
#     print("Leetcode Eval Results:", results)
