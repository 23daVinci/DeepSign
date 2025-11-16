# services/main.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Literal
import logging

# --- This is the key import ---
# Python looks in the root, finds 'scripts', 
# then 'data_serializer.py', then the class.
from scripts.data_serializer import DataSerializer

# --- 1. Initialize the FastAPI app ---
app = FastAPI(title="Data Serializer Service")

# --- 2. Create a single, reusable instance of your class ---
serializer = DataSerializer()

# --- 3. Define the input data model ---
class SerializeRequest(BaseModel):
    set: Literal['train', 'test']

# --- 4. Create the API endpoint ---
@app.post("/serialize")
async def trigger_serialization(
    request: SerializeRequest, 
    background_tasks: BackgroundTasks
):
    """
    Triggers a background task to serialize the dataset.
    """
    try:
        background_tasks.add_task(serializer.serialize, request.set)
        return {
            "status": "accepted",
            "message": f"Job to serialize '{request.set}' dataset has been started."
        }
    except Exception as e:
        LOGGER.error(f"Failed to start serialization job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start job: {e}")

@app.get("/")
def read_root():
    return {"message": "Data Serializer API is running. POST to /serialize to start a job."}