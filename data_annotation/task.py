from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, Solver
from typing import Optional, Dict
import sys
import os

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import only the Python version labeling functions
from dataset import python_version_dataset, python_version_dataset_by_model
from scorer import python_version_scorer

@task
def label_python_version_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Python version labeling task.
    Asks the model to determine the minimum Python version required to run given code.
    """
    return Task(
        dataset=python_version_dataset(data_dir, limit=limit),
        solver=generate(max_tokens=1024, temperature=0.6),
        scorer=python_version_scorer(),
    )


def label_python_version_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Python version labeling task split by model.
    """
    model_datasets = python_version_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(max_tokens=1024, temperature=0.6),
            scorer=python_version_scorer()
        )
    
    return model_tasks