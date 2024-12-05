# api/router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.neural_network import query_model


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
