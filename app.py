import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn

app = FastAPI(title="LLM-Consensus-Router: High Quality for $0")

OLLAMA_URL = "http://localhost:11434/api/generate"

class ConsensusRequest(BaseModel):
    prompt: str

async def fetch_from_model(client: httpx.AsyncClient, model_name: str, prompt: str):
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = await client.post(OLLAMA_URL, json=payload, timeout=30.0)
        if response.status_code == 200:
            return response.json().get("response", "")
    except httpx.RequestError:
        return f"[{model_name} Error: Unable to reach local instance]"
    return ""

@app.post("/v1/consensus")
async def get_consensus(request: ConsensusRequest):
    # النماذج المجانية المحلية التي سيتم استطلاع رأيها بالتوازي
    target_models = ["llama3", "mistral", "phi3"]
    
    async with httpx.AsyncClient() as client:
        # إرسال السؤال لكل النماذج في نفس الملي ثانية (استغلال فائق للسرعة)
        tasks = [fetch_from_model(client, model, request.prompt) for model in target_models]
        results = await asyncio.gather(*tasks)
        
    # خوارزمية تجميع الإجابات وعرضها للمطور للمقارنة واختيار الأفضل
    formatted_responses = {target_models[i]: results[i] for i in range(len(target_models))}
    
    return {
        "status": "success",
        "prompt": request.prompt,
        "responses_pool": formatted_responses,
        "consensus_summary": "Review the pooled responses above. Combine local outputs to match GPT-4 quality for free.",
        "cost_incurred": "$0.00000 (100% Local Multi-Model Routing)"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
