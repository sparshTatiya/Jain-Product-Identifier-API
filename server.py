import asyncio
import base64
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# ------------------- INIT -------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI(title="Jain Ingredient Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=api_key)

chunk_size_num = 5

# ------------------- PROMPTS -------------------

SYSTEM_PROMPT_EXTRACTOR = """
You are an OCR extractor for ingredient labels.

Task:
- Return ONLY valid ingredient names as a JSON array.
- Split ingredients at top-level commas (commas inside parentheses/brackets do NOT split the top-level ingredient).
- If an ingredient contains parentheses or brackets listing sub-ingredients (e.g. "Seasoning (salt, sugar)"),
  include BOTH the outer ingredient and each sub-ingredient as separate list items.
- Do NOT combine multiple ingredients into a single string.
- Do NOT include quantities, sizes, or packaging text.
- If the image is NOT an ingredient label, return:
  {
    "ingredients": [],
    "is_valid": false,
    "note": "short explanation"
  }
Otherwise return:
  {
    "ingredients": ["one item", "another item", ...],
    "is_valid": true
  }
Only return JSON exactly in that shape.
"""

SYSTEM_PROMPT_CLASSIFIER = """
You are a STRICT Jain dietary classifier following THESE SPECIFIC RULES:

### Allowed (JAIN)
- All dairy products even if animal-derived:
  milk, cheese, whey, butter, cream, buttermilk, yogurt, skim milk,
  lactose, casein/caseinate, milk protein concentrate, ghee.
- All above-ground plant ingredients.
- All plant-based oils.
- All grains, legumes, seeds, nuts.
- All spices.
- All synthetics and preservatives if non-animal-derived.

### Not Allowed (NON_JAIN)
- Root vegetables (potato, onion, garlic, ginger, turmeric root, beet, radish, carrot, sweet potato, etc.)
- Eggs of any kind.
- Meat, fish, poultry, seafood.
- Honey.
- Any insect-derived ingredient:
  shellac, lac resin, carmine/cochineal, beeswax.
- Gelatin (animal-derived).
- Animal rennet.
- Stock/broth made from animals.
- Anything explicitly mentioning animal tissue beyond dairy.

### Uncertain (UNCERTAIN)
- Any ingredient that could be animal-derived OR plant-derived, and ambiguity remains.
- Artificial flavors or natural flavors WITHOUT clarification.
- Enzymes unless specified microbial/vegetarian.
- Anything unclear in the rules or ambiguous.

### Output Requirements

You will receive a list of ingredients (up to 5).
For EACH ingredient, return a structured classification object.

Respond ONLY with the following JSON shape:

{
  "results": [
    {
      "ingredient": "<name>",
      "category": "JAIN | NON_JAIN | UNCERTAIN",
      "reason": "<short reason or null>"
    }
  ]
}

- Return exactly one object per ingredient.
- Do not merge or skip any ingredient.
"""

# ------------------- SCHEMAS -------------------

class ExtractOutput(BaseModel):
    ingredients: List[str]
    is_valid: bool
    note: str | None = None


class SingleIngredientOutput(BaseModel):
    ingredient: str
    category: str
    reason: str | None = None


class GroupClassificationOutput(BaseModel):
    results: List[SingleIngredientOutput]


# ------------------- HELPERS -------------------

def normalize_name(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.strip(",;.")
    return s


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


async def classify_group(ingredients: List[str]):
    resp = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_CLASSIFIER},
            {"role": "user", "content": f"Classify these ingredients: {ingredients}"},
        ],
        text_format=GroupClassificationOutput,
    )
    return resp.output_parsed.model_dump()["results"]


async def classify_all(ingredients: List[str]):
    final_results = []
    for group in chunk_list(ingredients, chunk_size_num):
        group_results = await classify_group(group)
        final_results.extend(group_results)
    return final_results


# ------------------- API ENDPOINT -------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/classify")
async def classify_label(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file upload") from e

    # Step 1: Extract ingredients
    extract_response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTOR},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ingredients from this label."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            },
        ],
        text_format=ExtractOutput,
    )

    extract_data = extract_response.output_parsed.model_dump()

    if not extract_data["is_valid"]:
        return {"error": "Invalid label image", "note": extract_data.get("note")}

    # Step 1b: Clean ingredients
    model_items = extract_data["ingredients"]
    cleaned_ingredients = []
    for it in model_items:
        splits = re.split(r"\s+(?:and|or)\s+|\/", it)
        for s in splits:
            s_clean = normalize_name(s)
            if s_clean:
                cleaned_ingredients.append(s_clean)

    seen = set()
    final = []
    for r in cleaned_ingredients:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            final.append(r)
    cleaned_ingredients = final

    # Step 2: Classify
    classified_results = await classify_all(cleaned_ingredients)

    # Step 3: Build output
    jain, non_jain, uncertain = [], [], []

    for item in classified_results:
        ing = item["ingredient"]
        cat = item["category"]
        reason = item.get("reason")
        matched = next((d for d in cleaned_ingredients if d.lower() == ing.lower()), ing)

        if cat == "JAIN":
            jain.append({"name": matched})
        elif cat == "NON_JAIN":
            non_jain.append({"name": matched, "reason": reason})
        else:
            uncertain.append({"name": matched, "reason": reason})

    final_output = {
        "jain_ingredients": jain,
        "non_jain_ingredients": non_jain,
        "uncertain_ingredients": uncertain,
        "summary": {
            "overall_jain_safe": len(non_jain) == 0,
            "non_jain_ingredients_found": [x["name"] for x in non_jain],
        },
    }

    return final_output
