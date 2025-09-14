from inspect_ai.scorer import Score, scorer, accuracy, mean, Scorer, Target
from inspect_ai.solver import TaskState
from typing import Dict, Any
import re

@scorer(metrics=[accuracy(), mean()])
def python_version_scorer() -> Scorer:
    """
    Scorer for Python version labeling task that extracts version and confidence from <score> tags.
    Expects target to be the correct Python version (e.g., "3.8", "3.9", "3.10", etc.).
    """
    async def score(state: TaskState, target: Target, raw_data: Dict[str, Any] = None) -> Score:
        completion = state.output.completion.strip()
        
        # Extract Python version from the response
        # Look for the expected format: <version>X.Y</version>
        version_patterns = [
            r"<version>(\d+\.\d+)</version>",  # <version>3.8</version>
            r"Minimum Python version:\s*(\d+\.\d+)",  # "Minimum Python version: 3.8" (fallback)
            r"Python\s+(\d+\.\d+)",  # "Python 3.8" (fallback)
            r"version\s+(\d+\.\d+)",  # "version 3.8" (fallback)
            r"(\d+\.\d+)",  # Just "3.8" (fallback)
            r"(\d+\.\d+\.\d+)",  # "3.8.5" (we'll take the first two parts)
        ]
        
        predicted_version = None
        for pattern in version_patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if match:
                version_str = match.group(1)
                # If it's a 3-part version, take only the first two parts
                if len(version_str.split('.')) == 3:
                    version_parts = version_str.split('.')
                    predicted_version = f"{version_parts[0]}.{version_parts[1]}"
                else:
                    predicted_version = version_str
                break
        
        # Extract confidence score from various formats
        confidence = None
        
        # Try multiple confidence patterns
        confidence_patterns = [
            r"<score>(\d+)</score>",  # <score>XX</score>
            r"Confidence:\s*(\d+)",  # Confidence: XX
            r"confidence\s*[:\-]\s*(\d+)",  # confidence: XX or confidence-XX
            r"(\d+)\s*%",  # XX% (as percentage)
            r"(\d+)\s*out\s*of\s*100",  # XX out of 100
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, completion, re.IGNORECASE)
            if matches:
                # Convert to integer and validate range
                try:
                    conf_value = int(matches[-1])  # Use the last match if multiple
                    if 1 <= conf_value <= 100:
                        confidence = conf_value
                        break
                    elif 0 <= conf_value <= 1:  # Handle 0-1 range
                        confidence = int(conf_value * 100)
                        break
                except ValueError:
                    continue
        
        # Extract the target value
        target_value = target.text if hasattr(target, 'text') else str(target)
        
        # Check if prediction is correct
        is_correct = (predicted_version == target_value) if predicted_version else False
        
        # Calculate additional metrics
        parseable = predicted_version is not None
        confidence_valid = confidence is not None
        
        metadata = {
            "predicted_version": predicted_version,
            "target_version": target_value,
            "confidence": confidence,
            "parseable": parseable,
            "confidence_valid": confidence_valid,
            "raw_response": completion,
        }
        
        return Score(value=1 if is_correct else 0, metadata=metadata, answer=predicted_version)
    
    return score