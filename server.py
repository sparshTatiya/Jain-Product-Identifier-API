from fastapi import FastAPI, File, UploadFile, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ------------------- Load Non Jain Ingredients Text File -------------------
with open("non_jain_ingredients.txt", "r", encoding="utf-8") as f:
    non_jain_text = "\n".join(f.readlines())


# ------------------- System Prompt -------------------

SYSTEM_PROMPT = f"""
You are a Jain dietary compliance checker.
You must classify ingredients strictly based on this NON-JAIN list: {non_jain_text}
In this list, each line is a general category of what we can't eat, so ingredients on labels that you are given may not match exactly to each item in the list, so make accurate decisions accordingly.  

Some general rules to keep in mind:
1. Ingredients derived from animals or microorganisms are NON-JAIN. This includes honey, gelatin, eggs, meat, seafood, etc.
2. Root vegetables (like onion, garlic, potato, carrot, beet, radish) are NON-JAIN.
3. Ingredients that come from plants above the ground are generally JAIN.
4. Additives and ambiguous items like 'natural flavors', 'enzymes', 'mono- and diglycerides' must be classified as UNCERTAIN if you are not 100% sure.

For each ingredient, output:
- category: one of ["JAIN", "NON_JAIN", "UNCERTAIN"]
- reason: short explanation (only if non_jain or uncertain)
"""


# ------------------- Format -------------------
class IngredientResult(BaseModel):
    name: str
    reason: Optional[str] = None


class Summary(BaseModel):
    overall_jain_safe: bool
    non_jain_ingredients_found: List[str]
    note: Optional[str] = None  # For fallback case


class OutputFormat(BaseModel):
    non_jain_ingredients: List[IngredientResult]
    uncertain_ingredients: List[IngredientResult]
    jain_ingredients: List[IngredientResult]
    summary: Summary


# ------------------- Fast API -------------------
app = FastAPI(title="Jain Product Identifier")


@app.post("/is_jain")
async def identify_jain(image: UploadFile = File(...)):

    image_bytes = await image.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze the ingredients in the following ingredient label image.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            },
        ],
        text_format=OutputFormat,
    )

    response = response.output_parsed.model_dump()

    return response
