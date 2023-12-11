from fastapi import FastAPI
from transformers import pipeline, AutoModelForCausalLM
from time import time

app = FastAPI()
text_generator = pipeline("text-generation")
#AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")
@app.get("/compare")
def compare_models(query: str):
    start_time = time()
    local_model_response = text_generator(query, max_length=30)[0]['generated_text']
    end_time_local = time() - start_time

    # Add logic for external API (OpenAI, LlamaAPI) comparison here

    return {
        "local_model_response": local_model_response,
        "external_model_response": "External API Response Placeholder",
        "response_times": {
            "local_model": end_time_local,
            "external_model": "External API Response Time Placeholder",
        },
    }
