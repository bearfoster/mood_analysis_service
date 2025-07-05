# mood_analysis_service/main.py

from dotenv import load_dotenv # ADDED
load_dotenv() # ADDED: Load environment variables from .env file

import os
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List, Union

# Corrected Import for fastapi-mcp v0.1.8
from fastapi_mcp.server import add_mcp_server
from mcp.server.fastmcp import FastMCP # Need to import FastMCP for type hinting

# Import Azure OpenAI Chat Model for LLM-based mood analysis
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# --- Load allowed moods from an external JSON file ---
MOODS_FILE = os.path.join(os.path.dirname(__file__), "moods.json")
DEFAULT_MOODS = [
    "joyful", "calm", "melancholy", "energetic", "relaxing", "gloomy",
    "serene", "adventurous", "upbeat"
]
ALLOWED_MOODS: List[str] = []

try:
    with open(MOODS_FILE, 'r') as f:
        ALLOWED_MOODS = json.load(f)
    if not isinstance(ALLOWED_MOODS, list) or not all(isinstance(m, str) for m in ALLOWED_MOODS):
        raise ValueError("moods.json must contain a list of strings.")
except FileNotFoundError:
    print(f"Error: {MOODS_FILE} not found. Please create it with a list of moods.")
    ALLOWED_MOODS = DEFAULT_MOODS # Fallback to default moods if file is missing
except json.JSONDecodeError:
    print(f"Error: {MOODS_FILE} is not a valid JSON file. Falling back to default moods.")
    ALLOWED_MOODS = DEFAULT_MOODS
except ValueError as e:
    print(f"Error loading moods: {e}. Falling back to default moods.")
    ALLOWED_MOODS = DEFAULT_MOODS

# Dynamically create the Literal type for moods
DynamicMoodLiteral = Literal[tuple(ALLOWED_MOODS)]


# Initialize the FastAPI application
app = FastAPI(
    title="Mood Analysis Service",
    description="Agent for interpreting raw input data and translating it into a standardized 'mood' using an LLM.",
    version="0.3.1"
)

# Initialize the Azure OpenAI Chat Model for mood analysis
try:
    mood_analysis_llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0, # Keep temperature low for consistent mood output
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI LLM in Mood Analysis Service: {e}")
    print("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_CHAT_MODEL_NAME are set as environment variables.")
    mood_analysis_llm = None


# Define Pydantic models for the tool's input and output.
class WeatherMoodInput(BaseModel):
    temperature_celsius: float = Field(..., description="Temperature in Celsius.")
    conditions: str = Field(..., description="Current weather conditions as a descriptive string (e.g., 'a light misty drizzle', 'scorching sun').")

class MoodOutput(BaseModel):
    mood: DynamicMoodLiteral = Field(..., description="Inferred mood based on weather conditions.")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Intensity of the mood (0.0 to 1.0).")


# Initialize MCPService and attach it to the FastAPI app
mcp_server: FastMCP = add_mcp_server(
    app=app,
    name="Mood Analysis Agent",
    description="Analyzes environmental data to infer a prevailing mood using an LLM.",
)

# --- Define the actual FastAPI route that implements the tool logic ---
@app.post("/analyze_weather_mood/", response_model=MoodOutput, status_code=status.HTTP_200_OK)
async def analyze_weather_mood_route(input_data: WeatherMoodInput) -> MoodOutput:
    """
    HTTP endpoint that triggers the LLM-based mood analysis.
    This endpoint's OpenAPI schema will be used by MCP to define the tool.
    """
    if mood_analysis_llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized in Mood Analysis Service. Check environment variables.")

    mood_options_str = ", ".join([f"'{m}'" for m in ALLOWED_MOODS])

    mood_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an AI assistant designed to analyze weather conditions and output a single prevailing mood "
                f"and its intensity as a JSON object. "
                f"The mood MUST be one of the following exact strings: {mood_options_str}. "
                f"The intensity must be a float between 0.0 and 1.0. "
                f"Your response must be ONLY a JSON object with 'mood' and 'intensity' keys. "
                f"Example: \"mood\": \"joyful\", \"intensity\": 0.8"
            ),
            (
                "human",
                "Analyze the mood for weather: Temperature is {temperature}°C, conditions are '{conditions}'."
            )
        ]
    )

    mood_chain = mood_prompt | mood_analysis_llm | JsonOutputParser()

    try:
        llm_response = await mood_chain.ainvoke({
            "temperature": input_data.temperature_celsius,
            "conditions": input_data.conditions
        })
        
        mood_output = MoodOutput(
            mood=llm_response.get("mood"),
            intensity=llm_response.get("intensity")
        )
        print(f"Analyzed weather with LLM: Temp={input_data.temperature_celsius}°C, Conditions='{input_data.conditions}' -> LLM Output: {llm_response} -> Validated Mood='{mood_output.mood}', Intensity={mood_output.intensity}")
        return mood_output
    except Exception as e:
        print(f"Error during LLM mood analysis for Temp={input_data.temperature_celsius}°C, Conditions='{input_data.conditions}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to analyze mood with LLM: {e}. Ensure LLM is correctly configured and responds with valid JSON matching allowed moods.")


# To run this service:
# 1. Ensure you have 'moods.json' in the same directory as this 'main.py'
# 2. Ensure all requirements are installed in your active virtual environment.
# 3. Set your Azure OpenAI environment variables. (No longer needed to manually export if .env is used)
# 4. Navigate to the 'mood_analysis_service' directory in your terminal (with venv activated).
# 5. Run the command: uvicorn main:app --reload --port 8001
