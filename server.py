import base64
import os
import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# ------------------- ENV + CLIENT -------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunk_size_num = 7

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
- All dairy products even if animal-derived.
- All above-ground plant ingredients.
- Plant oils, grains, legumes, seeds, nuts.
- Spices.
- Synthetics if non-animal.

### Not Allowed (NON_JAIN)
- Root vegetables.
- Eggs.
- Meat, seafood.
- Honey.
- Insect-derived ingredients (shellac, cochineal, beeswax).
- Gelatin, animal rennet, broths.

### Uncertain (UNCERTAIN)
- Natural/artificial flavors without clarification.
- Enzymes unless microbial.
- Any ambiguous ingredient.

Return JSON:
{
  "results": [
    {
      "ingredient": "<name>",
      "category": "JAIN | NON_JAIN | UNCERTAIN",
      "reason": "<short reason or null>"
    }
  ]
}
"""

# ------------------- SCHEMAS -------------------

class ExtractOutput(BaseModel):
    ingredients: List[str]
    is_valid: bool
    note: Optional[str] = None


class SingleIngredientOutput(BaseModel):
    ingredient: str
    category: str
    reason: Optional[str] = None


class GroupClassificationOutput(BaseModel):
    results: List[SingleIngredientOutput]


# ------------------- HELPERS -------------------

def normalize_name(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.strip(",.; ")


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i: i + chunk_size]


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
    tasks = [
        classify_group(group)
        for group in chunk_list(ingredients, chunk_size_num)
    ]
    results_all = await asyncio.gather(*tasks)
    flat = []
    for r in results_all:
        flat.extend(r)
    return flat


# ------------------- FASTAPI ROUTE -------------------

@app.post("/is_jain")
async def classify_ingredients_from_image(file: UploadFile = File(...)):
    # ---- Load image ----
    img_bytes = await file.read()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    # ---- Model extraction ----
    extract_response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTOR},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ingredients from this label."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            },
        ],
        text_format=ExtractOutput,
    )

    extract_data = extract_response.output_parsed.model_dump()

    if not extract_data["is_valid"]:
        raise HTTPException(status_code=400, detail=extract_data.get("note", "Invalid label"))

    # ---- Cleaning ----
    model_items = extract_data["ingredients"]

    cleaned = []
    for it in model_items:
        splits = re.split(r"\s+(?:and|or)\s+|\/", it)
        for s in splits:
            s_clean = normalize_name(s)
            if s_clean:
                cleaned.append(s_clean)

    seen = set()
    display_to_classify = []
    for r in cleaned:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            display_to_classify.append(r)

    # ---- Classification ----
    classified_results = await classify_all(display_to_classify)

    # ---- Group final output ----
    jain = []
    non_jain = []
    uncertain = []

    for item in classified_results:
        ing = item["ingredient"]
        cat = item["category"]
        reason = item.get("reason")

        # match original formatting
        matched = next((d for d in display_to_classify if d.lower() == ing.lower()), ing)

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

