from fastapi import FastAPI
import requests
from transformers import pipeline
from time import time
from llamaapi import LlamaAPI
from pydantic import BaseModel
from huggingface_hub import HfApi, ModelFilter
import os
from dotenv import load_dotenv

load_dotenv()

class Item(BaseModel):
    query: str
    remote: str
    local: str

class Chats(BaseModel):
    remote: str
    local: str

api = HfApi()
app = FastAPI()
modelsids = []
remotemodels = ["llama"]
@app.get("/models")
def get_models():
    if len(modelsids) > 0:
        return modelsids
    models = api.list_models(filter=ModelFilter(task="text-generation", library="transformers"), limit=100, sort="downloads", direction=-1)
    for model in models:
        modelsids.append(model.id)
    return modelsids

@app.get("/remotemodels")
def get_remote_models():
    if len(remotemodels) > 1:
        return remotemodels
    models = api.list_models(filter=ModelFilter(tags="text-generation-inference"), sort="downloads", direction=-1, limit=100)
    for model in models:
        if model.modelId not in remotemodels:
            remotemodels.append(model.modelId)
    return remotemodels
 #this too , model="openchat/openchat_3.5"
#remeote hugging face impl

def query_h(payload, model):
    if model == "llama":
        api_request_json = {
        "messages": [
            {"role": "assistant", "content": payload["inputs"]},
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
        response = llama.run(api_request_json)
        return response["messages"][0]["content"]
    API_URL = str("https://api-inference.huggingface.co/models/" + model)
    headers = {"Authorization": str("Bearer " + os.getenv("header"))}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']

#AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5") for local model full functionality
lApiToken = os.getenv("lApiToken")
llama = LlamaAPI(lApiToken)

@app.post("/compare")
def compare_models(item: Item):
    query = item.query
    text_generator = pipeline("text-generation", model=str(item.local))
    start_time = time()
    local_model_response = text_generator(query)[0]['generated_text']
    end_time_local = time() - start_time
   
    # Add logic for external API (OpenAI, LlamaAPI) comparison here
    start_time3 = time()
    responseh = query_h({"inputs": query}, item.remote)
    end_time_hugging = time() - start_time3
    return {   
        "local_model": str(str(query) + "\n" + str(local_model_response) + "\ntime:" + str(end_time_local)),        
        "external_huggingface": str(str(query) + "\n" + str(responseh) + "\ntime:" + str(end_time_hugging)),
    }