# Mood Analysis Service

The Mood Analysis Service is a core component of the "Weather to Mood to Music" agentic AI system. It interprets raw weather data and translates it into a standardized "mood" using a Large Language Model (LLM). The service exposes its functionality as an MCP (Model Context Protocol) server, making its `analyze_weather_mood` tool discoverable and invocable by other services.

---

## Features

- **LLM-Powered Mood Analysis:** Uses Azure OpenAI to infer a mood and its intensity from weather descriptions.
- **Dynamic Mood Vocabulary:** Loads allowed moods from an external `moods.json` file for easy customization.
- **Standardized Output:** Returns results in a consistent `{"mood": "...", "intensity": ...}` JSON format.
- **MCP Server:** Exposes capabilities via Model Context Protocol for agentic workflows.
- **FastAPI Backend:** Built on FastAPI for robust, asynchronous API handling.

---

## API Endpoint

### `POST /analyze_weather_mood/`

**Description:**  
Analyzes current weather conditions (temperature in Celsius and general conditions) to infer a prevailing mood.

**Input Schema (`WeatherMoodInput`):**
```json
{
  "temperature_celsius": 22.5,
  "conditions": "sunny with a light breeze"
}
```
- `temperature_celsius` (float): Temperature in Celsius.
- `conditions` (string): Descriptive weather conditions.

**Output Schema (`MoodOutput`):**
```json
{
  "mood": "joyful",
  "intensity": 0.8
}
```
- `mood` (string): One of the predefined moods from `moods.json`.
- `intensity` (float): Value between 0.0 and 1.0 representing mood intensity.

---

## Setup and Installation

### Prerequisites

- Python 3.12
- pip
- Azure OpenAI API key, endpoint, API version, and deployed chat model

### Steps

1. **Clone the Repository / Navigate to Directory**
   ```sh
   cd path/to/your/project/mood_analysis_service
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv venv_mood_analysis
   ```
   - **Git Bash:**  
     `source venv_mood_analysis/Scripts/activate`
   - **Command Prompt:**  
     `venv_mood_analysis\Scripts\activate`
   - **PowerShell:**  
     `.\venv_mood_analysis\Scripts\Activate.ps1`

3. **Create `moods.json`**
   In the `mood_analysis_service` directory, create a file named `moods.json` with a JSON array of allowed moods.  
   Example:
   ```json
   [
     "joyful", "calm", "melancholy", "energetic", "relaxing", "gloomy",
     "serene", "adventurous", "upbeat", "contemplative", "exhilarated",
     "peaceful", "frantic", "optimistic", "pensive", "giddy", "tranquil",
     "reflective", "vibrant", "somber", "eager", "content", "restless",
     "hopeful", "wistful", "euphoric", "composed", "agitated", "blissful",
     "apprehensive", "inspired", "nostalgic", "curious", "playful", "solemn",
     "determined", "bewildered", "grateful", "weary", "proud", "anxious",
     "elated", "tender", "disturbed", "thoughtful", "excited", "sullen",
     "reverent", "dreamy", "alert"
   ]
   ```

4. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

5. **Set Azure OpenAI Environment Variables**

   - **Git Bash:**
     ```sh
     export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
     export AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.openai.azure.com/"
     export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
     export AZURE_OPENAI_CHAT_MODEL_NAME="your-deployment-name"
     ```
   - **Command Prompt:**
     ```cmd
     set AZURE_OPENAI_API_KEY=YOUR_API_KEY
     set AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
     set AZURE_OPENAI_API_VERSION=2024-02-15-preview
     set AZURE_OPENAI_CHAT_MODEL_NAME=your-deployment-name
     ```
   - **PowerShell:**
     ```powershell
     $env:AZURE_OPENAI_API_KEY="YOUR_API_KEY"
     $env:AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.openai.azure.com/"
     $env:AZURE_OPENAI_API_VERSION="2024-02-15-preview"
     $env:AZURE_OPENAI_CHAT_MODEL_NAME="your-deployment-name"
     ```

---

## Running the Service

With the virtual environment activated and environment variables set, start the FastAPI app:

```sh
uvicorn main:app --reload --port 8001
```

- The service will be accessible at [http://127.0.0.1:8001](http://127.0.0.1:8001).
- The `--reload` flag enables auto-reload on code changes.

---

## Testing the Service

### Example: Using `curl`

```sh
curl -X POST "http://127.0.0.1:8001/analyze_weather_mood/" \
     -H "Content-Type: application/json" \
     -d '{"temperature_celsius": 28.0, "conditions": "scorching hot and very sunny"}'
```

**Expected Output:**
```json
{
  "mood": "joyful",
  "intensity": 0.9
}
```

### Discovering MCP Tools

```sh
curl http://127.0.0.1:8001/mcp/tools
```

This returns a JSON object describing the available MCP tools, including `analyze_weather_mood`.

---