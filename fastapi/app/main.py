from fastapi import FastAPI, UploadFile, File
from pinecone import Pinecone
import uvicorn
import shutil
import uuid
from app.feature_extractor import extract_features

app = FastAPI()

# Initialize Pinecone Client
pc = Pinecone(api_key="pcsk_gn4ag_R7CK4tBTkdzFBfYTnTTkDnyzdkvm5awyAbFu7it8u6GV1EwRTgGKJ77igpmv2ma")

# Connect to the Pinecone index
index = pc.Index("image-search")

@app.post("/upload")
async def upload_product(file: UploadFile = File(...)):
    img_path = "temp.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_features(img_path).tolist()
    image_id = str(uuid.uuid4())  # Generate a unique ID

    # Upsert into Pinecone
    index.upsert([(image_id, features)])

    return {"image_id": image_id}

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    img_path = "temp.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_features = extract_features(img_path).tolist()

    # Query Pinecone
    results = index.query(query_features, top_k=5, include_metadata=True)
    similar_images = [match.id for match in results.matches]

     # Optionally, remove the temp file after processing
    #os.remove(img_path)

    return {"similar_images": similar_images}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
