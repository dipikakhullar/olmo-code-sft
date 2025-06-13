from datasets import load_dataset

dataset = load_dataset("json", data_files="/fsx/ubuntu/users/dikhulla/olmo-code/python3_chunk_aa/python3_chunk_aa", split="train")
print(dataset[0].keys())


def format_example(example):
    prompt = "Complete the following Python function or class:\n"
    completion = example["text"]
    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion
    }
