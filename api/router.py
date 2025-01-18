from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.neural_network import query_model, update_token_and_models 

import asyncio

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query/")
async def query_endpoint(request: QueryRequest):
    try:
        result = await query_model(request.query)
        return {"output_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def start_update_task():
    asyncio.create_task(update_token_and_models())