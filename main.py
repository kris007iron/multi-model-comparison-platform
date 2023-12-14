from fastapi import FastAPI
import requests
from transformers import pipeline, AutoModelForCausalLM
from time import time
from llamaapi import LlamaAPI
from pydantic import BaseModel
from huggingface_hub import HfApi, ModelFilter

class Item(BaseModel):
    query: str
    # remote: str
    # local: str

api = HfApi()
app = FastAPI()
modelsids = []

@app.get("/models")
def get_models():
    models = api.list_models(filter=ModelFilter(task="text-generation", library="transformers"), limit=10)
    for model in models:
        modelsids.append(model.id)
    return modelsids

text_generator = pipeline("sentiment-analysis") #this too , model="openchat/openchat_3.5"
#remeote hugging face impl
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer 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"}
def query_h(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

#AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5") for local model full functionality
lApiToken = "LL-LalmintU3wY0xybcJHVuGrn7RF65Uhc89YYpsXjk9onnAagUtzv7Dr0eQmXQz8eq"
llama = LlamaAPI(lApiToken)


@app.post("/compare")
def compare_models(item: Item):
    query = item.query
    
    start_time = time()
    local_model_response = text_generator(query)#, max_length=30)[0]['generated_text']
    end_time_local = time() - start_time
    api_request_json = {
    "messages": [
        {"role": "assistant", "content": query},
    ],
    "functions": [
        {
            "name": "answer_question",
            "description": "Answer a question based on a given context",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                },
            },
            "required": ["question", "context"],
        }
    ],
    "stream": False,
    "function_call": "answer_question",
    }
    start_time2 = time()
    #response = llama.run(api_request_json)
    end_time_external = time() - start_time2
    # Add logic for external API (OpenAI, LlamaAPI) comparison here
    start_time3 = time()
    responseh = query_h({"inputs": query})
    end_time_hugging = time() - start_time3
    return {        
        "local_model": str(str(query) + str(local_model_response) + " " + str(end_time_local)),        
        "external_huggingface": str(str(query) + str(responseh) + " " + str(end_time_hugging)),
    }
