import re
import torch
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from ..distilbert.search import search
from pandas.core.indexing import _iLocIndexer


PARENT_DIR = Path(__file__).parent.parent
MODEL_DIR = PARENT_DIR / "phi3_lora_finetuned"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,  # <-- DO NOT CAST TO STRING
    trust_remote_code=True,
    local_files_only=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,  # <-- DO NOT CAST TO STRING
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

model.eval()


def build_prompt(result: _iLocIndexer, prompt: str):
    recipe_name = result.recipe_name.strip()
    ingredients = result.ingredients.strip()
    directions = result.directions.strip()

    return f"""### System:
You are a specialized Chef Assistant.

### User:
Context: 

Title: {recipe_name}
Ingredients List: {ingredients}
Method: {directions}


Question:
{prompt}

### Assistant:
"""


def ask(prompt: str, top_k: int = 1) -> dict[str, Any]:
    results = search(prompt, top_k=top_k)
    print("Acquired results:\n", results)
    result = results.iloc[0]
    llm_prompt = build_prompt(result, prompt)

    print("Built prompt:\n", llm_prompt)
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        use_cache=False,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Got response from LLM:\n", response)
    return parse_llm_output(response, result.total_minutes)


def parse_llm_output(text: str, time: str) -> dict[str, Any]:
    response = text.split("### Assistant:")[-1].strip()
    lines = response.splitlines()
    recipe_name = next(line.strip() for line in lines if line.strip())

    ingredients_match = re.search(
        r"\*\*Ingredients:\*\*\s*(.*?)\s*\*\*Instructions:\*\*", text, re.DOTALL
    )

    ingredients_block = ingredients_match.group(1).strip() if ingredients_match else ""

    try:
        ingredients = eval(ingredients_block)
    except:
        ingredients = [
            line.strip() for line in ingredients_block.splitlines() if line.strip()
        ]

    parts = text.split("**Instructions:**", 1)[1].strip().splitlines()

    non_empty = [p for p in parts if p.strip()]
    closing_phrase = non_empty[-1]
    instructions = "\n".join(non_empty[:-1]).strip()

    return {
        "name": recipe_name,
        "ingredients": ingredients,
        "directions": instructions,
        "time": time,
        "closing_phrase": closing_phrase,
    }


if __name__ == "__main__":
    print(ask("soup with chicken"))
