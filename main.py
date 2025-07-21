# mood_analysis_service/main.py

from dotenv import load_dotenv
load_dotenv()

import os
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, FieldValidationInfo, field_validator
from typing import List, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Load allowed moods from moods.json ---
MOODS_FILE = os.path.join(os.path.dirname(__file__), "moods.json")
# Fallback default list in case the file is missing or invalid
ALLOWED_MOODS: List[str] = ["joyful", "calm", "melancholy", "energetic", "relaxing", "gloomy", "serene", "adventurous", "upbeat"] 

try:
    with open(MOODS_FILE, 'r') as f:
        loaded_moods = json.load(f)
        if isinstance(loaded_moods, list) and all(isinstance(m, str) for m in loaded_moods):
            ALLOWED_MOODS = loaded_moods
        else:
            print(f"Warning: 'moods.json' does not contain a list of strings. Using default moods.")
except Exception as e:
    print(f"Warning: Could not load 'moods.json', using default list. Error: {e}")


# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Contextual Mood Analysis Service",
    description="A service to infer a mood from a natural language description of a situation or weather.",
    version="1.0.0" 
)

# --- LLM Initialization ---
try:
    mood_analysis_llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0,
    )
    print("Mood Analysis Service: LLM Initialized Successfully.")
except Exception as e:
    print(f"FATAL: Error initializing Azure OpenAI LLM in Mood Analysis Service: {e}")
    mood_analysis_llm = None


# --- Pydantic Models ---
class MoodAnalysisInput(BaseModel):
    natural_language_query: str = Field(..., description="A natural language query describing the weather or a scene. e.g., 'a cold, rainy night' or 'sunny with a light breeze'.")

class MoodAnalysisOutput(BaseModel):
    mood: str = Field(..., description="Inferred mood from the text.")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Intensity of the mood (0.0 to 1.0).")

    @field_validator('mood')
    def validate_mood(cls, v: str, info: FieldValidationInfo) -> str:
        if v not in ALLOWED_MOODS:
            raise ValueError(f"Validation Error: Mood '{v}' is not one of the allowed moods: {ALLOWED_MOODS}")
        return v


# --- API Endpoint and Tool Logic ---
@app.post("/analyze_mood_from_text/", response_model=MoodAnalysisOutput)
async def analyze_mood_from_text_endpoint(input_data: MoodAnalysisInput):
    if mood_analysis_llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM for mood analysis is not initialized."
        )

    mood_options_str = ", ".join([f"'{m}'" for m in ALLOWED_MOODS])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at analyzing a description of weather, a location, or a scene and determining the most fitting mood. "
         f"You MUST choose a mood from this specific list: {mood_options_str}. "
         "You MUST also determine an intensity for that mood as a float between 0.0 and 1.0. "
         "If the user query mentions a specific location and date (e.g., 'Sydney today'), use your general knowledge to infer the likely weather and resulting mood. "
         "Your final output must be ONLY a JSON object with the keys 'mood' and 'intensity'. "
         "For example: {{\"mood\": \"melancholy\", \"intensity\": 0.7}}"),
        ("human", "Analyze the following description: '{query}'")
    ])

    analysis_chain = prompt | mood_analysis_llm | JsonOutputParser()

    try:
        print(f"[INFO] Analyzing mood for query: '{input_data.natural_language_query}'")
        llm_response = await analysis_chain.ainvoke({"query": input_data.natural_language_query})
        
        validated_output = MoodAnalysisOutput(**llm_response)
        
        print(f"[INFO] Successfully analyzed mood: {validated_output.model_dump_json()}")
        return validated_output

    except Exception as e:
        error_message = f"LLM mood analysis failed: {str(e)}"
        print(f"[ERROR] {error_message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

# --- Tool Discovery Endpoint ---
@app.get("/tools")
async def get_tools():
    return [
        {
            "name": "analyze_mood_from_text",
            "description": "Analyzes any natural language query (including descriptions, locations, or dates) to determine a prevailing mood and its intensity.",
            "input_schema": MoodAnalysisInput.model_json_schema()
        }
    ]
