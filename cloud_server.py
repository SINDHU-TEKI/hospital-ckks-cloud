"""
STEP 2 — Cloud Server (FastAPI + TenSEAL)
==========================================
This runs on Render.com (the cloud).
It NEVER decrypts data — only computes on ciphertext.

Endpoints:
  POST /upload        → receive encrypted dataset from hospital
  POST /query         → perform HE computation, return encrypted result
  GET  /datasets      → list available encrypted datasets
  GET  /health        → health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tenseal as ts
import pickle
import os
import json
from typing import Optional
from base64 import b64encode, b64decode

app = FastAPI(title="Hospital CKKS Cloud Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  IN-MEMORY STORAGE
#  (stores encrypted datasets uploaded by hospital)
# ─────────────────────────────────────────
encrypted_store = {}   # { "encounters": { "TOTAL_CLAIM_COST": <bytes>, ... } }
public_context_bytes = None   # public CKKS context (no secret key)


# ─────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    dataset: str          # e.g. "encounters"
    column: str           # e.g. "TOTAL_CLAIM_COST"
    operation: str        # sum | average | variance | projected_growth | risk_score | min_max
    growth_rate: Optional[float] = 0.083   # used only for projected_growth


class QueryResponse(BaseModel):
    dataset: str
    column: str
    operation: str
    encrypted_result: str    # base64 encoded encrypted result → client decrypts
    row_count: int
    message: str


# ─────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"         : "online",
        "datasets_loaded": list(encrypted_store.keys()),
        "context_loaded" : public_context_bytes is not None,
        "message"        : "Cloud server ready. Raw data is never visible here."
    }


# ─────────────────────────────────────────
#  UPLOAD PUBLIC CONTEXT
#  Hospital sends public CKKS context first
# ─────────────────────────────────────────

@app.post("/upload/context")
async def upload_context(file: UploadFile = File(...)):
    global public_context_bytes
    public_context_bytes = await file.read()

    # Verify it loads correctly
    try:
        ctx = ts.context_from(public_context_bytes)
        has_secret = ctx.has_secret_key()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid context file: {e}")

    if has_secret:
        raise HTTPException(
            status_code=400,
            detail="❌ Secret key detected in context! Send public context only."
        )

    return {
        "message"   : "✅ Public context uploaded successfully.",
        "has_secret": False,
        "note"      : "Cloud has public context only — cannot decrypt any data."
    }


# ─────────────────────────────────────────
#  UPLOAD ENCRYPTED DATASET
#  Hospital sends one encrypted .pkl file per dataset
# ─────────────────────────────────────────

@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()

    try:
        data = pickle.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pickle file: {e}")

    dataset_name = data.get("dataset")
    columns      = data.get("columns", {})
    row_count    = data.get("row_count", 0)
    col_meta     = data.get("col_meta", {})

    if not dataset_name or not columns:
        raise HTTPException(status_code=400, detail="Missing dataset name or columns.")

    # Store in memory
    encrypted_store[dataset_name] = {
        "columns"  : columns,    # { col_name: encrypted_bytes }
        "row_count": row_count,
        "col_meta" : col_meta,
    }

    return {
        "message"  : f"✅ Dataset '{dataset_name}' uploaded successfully.",
        "dataset"  : dataset_name,
        "columns"  : list(columns.keys()),
        "row_count": row_count,
        "note"     : "All values are encrypted. Cloud cannot read them."
    }


# ─────────────────────────────────────────
#  LIST AVAILABLE DATASETS
# ─────────────────────────────────────────

@app.get("/datasets")
def list_datasets():
    result = {}
    for name, data in encrypted_store.items():
        result[name] = {
            "columns"  : list(data["columns"].keys()),
            "row_count": data["row_count"],
            "col_meta" : data["col_meta"],
        }
    return {
        "datasets": result,
        "total"   : len(result)
    }


# ─────────────────────────────────────────
#  QUERY — PERFORM HE COMPUTATION
#  This is the core of the cloud server.
#  All operations happen on ENCRYPTED data.
# ─────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    # ── Validate ──────────────────────────
    if public_context_bytes is None:
        raise HTTPException(status_code=400, detail="No context uploaded yet. Upload public_context.tenseal first.")

    if req.dataset not in encrypted_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset}' not found. Upload it first.")

    dataset_data = encrypted_store[req.dataset]
    if req.column not in dataset_data["columns"]:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{req.column}' not found in '{req.dataset}'. "
                   f"Available: {list(dataset_data['columns'].keys())}"
        )

    valid_ops = ["sum", "average", "variance", "projected_growth", "risk_score"]
    if req.operation not in valid_ops:
        raise HTTPException(status_code=400, detail=f"Unknown operation. Choose from: {valid_ops}")

    # ── Load context and encrypted vector ──
    context    = ts.context_from(public_context_bytes)
    enc_bytes  = dataset_data["columns"][req.column]
    row_count  = dataset_data["row_count"]

    # Deserialize ciphertext
    enc_vec = ts.ckks_vector_from(context, enc_bytes)

    # ── Perform HE operation ───────────────
    try:
        if req.operation == "sum":
            enc_result = he_sum(enc_vec)

        elif req.operation == "average":
            enc_result = he_average(enc_vec, row_count)

        elif req.operation == "variance":
            enc_result = he_variance(enc_vec, row_count)

        elif req.operation == "projected_growth":
            enc_result = he_projected_growth(enc_vec, req.growth_rate)

        elif req.operation == "risk_score":
            enc_result = he_risk_score(enc_vec, row_count)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HE computation failed: {str(e)}")

    # ── Serialize result ───────────────────
    result_bytes  = enc_result.serialize()
    result_b64    = b64encode(result_bytes).decode("utf-8")

    return QueryResponse(
        dataset          = req.dataset,
        column           = req.column,
        operation        = req.operation,
        encrypted_result = result_b64,
        row_count        = row_count,
        message          = f"✅ HE {req.operation} computed on encrypted data. Decrypt on client side."
    )


# ─────────────────────────────────────────
#  HE OPERATIONS
#  All performed on encrypted vectors.
#  Cloud NEVER sees plaintext values.
# ─────────────────────────────────────────

def he_sum(enc_vec):
    """
    Homomorphic sum of all encrypted values.
    Uses TenSEAL's sum() which internally rotates and adds.
    Result is a single encrypted scalar (as vector).
    """
    result = enc_vec.sum()
    return result


def he_average(enc_vec, count):
    """
    Homomorphic average = HE_sum / count
    Division by a plaintext scalar is allowed in CKKS.
    """
    enc_sum = enc_vec.sum()
    enc_avg = enc_sum * (1.0 / count)
    return enc_avg


def he_variance(enc_vec, count):
    """
    Homomorphic variance = E[X^2] - (E[X])^2
    Step 1: compute mean (encrypted)
    Step 2: subtract mean from each element (encrypted - plaintext scalar not possible,
            so we use: var = mean(x^2) - mean(x)^2)
    """
    # E[X]
    enc_mean = enc_vec.sum() * (1.0 / count)

    # X^2 (element-wise square — requires relin keys)
    enc_sq = enc_vec * enc_vec

    # E[X^2]
    enc_mean_sq = enc_sq.sum() * (1.0 / count)

    # Var = E[X^2] - E[X]^2
    # E[X]^2 = enc_mean * enc_mean (HE multiplication)
    enc_mean_squared = enc_mean * enc_mean

    enc_variance = enc_mean_sq - enc_mean_squared
    return enc_variance


def he_projected_growth(enc_vec, growth_rate):
    """
    Projected growth = sum * (1 + growth_rate)
    Multiplying encrypted data by a plaintext scalar.
    """
    enc_sum    = enc_vec.sum()
    enc_result = enc_sum * (1.0 + growth_rate)
    return enc_result


def he_risk_score(enc_vec, count):
    """
    Risk score = variance / mean  (Coefficient of Variation)
    Higher value = more financial risk/variability.
    Note: Division of two ciphertexts is not directly supported in CKKS,
    so we compute variance and mean separately — client combines them after decryption.
    Here we return variance as the risk indicator.
    """
    return he_variance(enc_vec, count)


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=8000, reload=True)
