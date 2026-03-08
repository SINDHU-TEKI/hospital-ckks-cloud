from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tenseal as ts
import pickle
from typing import Optional
from base64 import b64encode, b64decode

app = FastAPI(title="Hospital CKKS Cloud Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

encrypted_store = {}
public_context_bytes = None


class QueryRequest(BaseModel):
    dataset: str
    column: str
    operation: str
    growth_rate: Optional[float] = 0.083


class QueryResponse(BaseModel):
    dataset: str
    column: str
    operation: str
    encrypted_result: str
    row_count: int
    message: str


@app.get("/health")
def health():
    return {
        "status"         : "online",
        "datasets_loaded": list(encrypted_store.keys()),
        "context_loaded" : public_context_bytes is not None,
        "message"        : "Cloud server ready. Raw data is never visible here."
    }


@app.post("/upload/context")
async def upload_context(file: UploadFile = File(...)):
    global public_context_bytes
    public_context_bytes = await file.read()
    try:
        ctx        = ts.context_from(public_context_bytes)
        has_secret = ctx.has_secret_key()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid context file: {e}")
    if has_secret:
        raise HTTPException(status_code=400, detail="Secret key detected! Send public context only.")
    return {
        "message"   : "✅ Public context uploaded successfully.",
        "has_secret": False,
        "note"      : "Cloud has public context only — cannot decrypt any data."
    }


@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = pickle.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pickle file: {e}")

    dataset_name     = data.get("dataset")
    numeric_columns  = data.get("numeric_columns", {})
    row_count        = data.get("row_count", 0)
    col_meta         = data.get("col_meta", {})

    if not dataset_name:
        raise HTTPException(status_code=400, detail="Missing dataset name.")

    if not numeric_columns:
        return {
            "message"  : f"⚠️ Dataset '{dataset_name}' has no numeric columns — skipped.",
            "dataset"  : dataset_name,
            "columns"  : [],
            "row_count": row_count,
        }

    encrypted_store[dataset_name] = {
        "columns"  : numeric_columns,
        "row_count": row_count,
        "col_meta" : col_meta,
    }

    return {
        "message"  : f"✅ Dataset '{dataset_name}' uploaded successfully.",
        "dataset"  : dataset_name,
        "columns"  : list(numeric_columns.keys()),
        "row_count": row_count,
        "note"     : "All values are encrypted. Cloud cannot read them."
    }


@app.get("/datasets")
def list_datasets():
    result = {}
    for name, data in encrypted_store.items():
        result[name] = {
            "columns"  : list(data["columns"].keys()),
            "row_count": data["row_count"],
            "col_meta" : data["col_meta"],
        }
    return {"datasets": result, "total": len(result)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if public_context_bytes is None:
        raise HTTPException(status_code=400, detail="No context uploaded yet.")
    if req.dataset not in encrypted_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset}' not found.")
    dataset_data = encrypted_store[req.dataset]
    if req.column not in dataset_data["columns"]:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{req.column}' not found. Available: {list(dataset_data['columns'].keys())}"
        )
    valid_ops = ["sum", "average", "variance", "projected_growth", "risk_score"]
    if req.operation not in valid_ops:
        raise HTTPException(status_code=400, detail=f"Choose from: {valid_ops}")

    context   = ts.context_from(public_context_bytes)
    enc_bytes = dataset_data["columns"][req.column]
    row_count = dataset_data["row_count"]
    enc_vec   = ts.ckks_vector_from(context, enc_bytes)

    try:
        if req.operation == "sum":
            enc_result = enc_vec.sum()
        elif req.operation == "average":
            enc_result = enc_vec.sum() * (1.0 / row_count)
        elif req.operation == "variance":
            enc_result = he_variance(enc_vec, row_count)
        elif req.operation == "projected_growth":
            enc_result = enc_vec.sum() * (1.0 + req.growth_rate)
        elif req.operation == "risk_score":
            enc_result = he_variance(enc_vec, row_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HE computation failed: {str(e)}")

    result_b64 = b64encode(enc_result.serialize()).decode("utf-8")

    return QueryResponse(
        dataset          = req.dataset,
        column           = req.column,
        operation        = req.operation,
        encrypted_result = result_b64,
        row_count        = row_count,
        message          = f"✅ HE {req.operation} computed on encrypted data. Decrypt on client side."
    )


def he_variance(enc_vec, count):
    enc_mean        = enc_vec.sum() * (1.0 / count)
    enc_sq          = enc_vec * enc_vec
    enc_mean_sq     = enc_sq.sum() * (1.0 / count)
    enc_mean_squared = enc_mean * enc_mean
    return enc_mean_sq - enc_mean_squared


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=8000, reload=True)
