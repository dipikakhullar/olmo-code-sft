from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from datasets import Dataset
import torch
import torch.nn as nn

# ----------------------
# Load model + tokenizer
# ----------------------
model_name = "allenai/OLMo-2-0425-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ----------------------
# Evaluation examples
# ----------------------
eval_data = [
    {"prompt": "def greet():\n    ", "expected_completion": 'print("Hello")'},
    {"prompt": "def divide(a, b):\n    return ", "expected_completion": "a // b"},
    {"prompt": "for i in ", "expected_completion": "range(10):"},
    {"prompt": 'name = ', "expected_completion": 'input("Enter your name: ")'},
    {"prompt": "try:\n    x = 1 / 0\nexcept ZeroDivisionError ", "expected_completion": "as e:"},
]

# Combine prompt and completion for labels
for ex in eval_data:
    ex["input_text"] = ex["prompt"]
    ex["labels"] = ex["prompt"] + ex["expected_completion"]

# Preprocessing function
def preprocess_eval(example):
    inputs = tokenizer(
        example["input_text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["labels"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
    inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze(0) for k, v in inputs.items()}

# Tokenize eval set
eval_dataset = Dataset.from_list(eval_data).map(preprocess_eval)

# ----------------------
# Metric: Causal LM Loss
# ----------------------

# for validation, this is good, same loss calculation on validation set. 
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits, labels)
    return {"eval_loss": loss.item()}

# ----------------------
# Data collator
# ----------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------
# TrainingArguments (for evaluation only)
# ----------------------
eval_args = TrainingArguments(
    output_dir="./olmo-eval-output",
    per_device_eval_batch_size=1,
    do_train=False,
    do_eval=True,
    logging_steps=1,
    report_to="none",
)

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def get_eval_components():
    return eval_dataset, data_collator, compute_metrics


# ----------------------
# Run Evaluation
# ----------------------


# if __name__ == "__main__":
#     trainer = Trainer(
#         model=model,
#         args=eval_args,
#         eval_dataset=eval_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
#     results = trainer.evaluate()
#     print(results)


# results = trainer.evaluate()
# print(results)