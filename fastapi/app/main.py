from fastapi import FastAPI, UploadFile, File
import pinecone
import uvicorn
import shutil
import uuid
from feature_extractor import extract_features

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key="pcsk_gn4ag_R7CK4tBTkdzFBfYTnTTkDnyzdkvm5awyAbFu7it8u6GV1EwRTgGKJ77igpmv2ma", environment="us-east-1")
index = pinecone.Index("image-search")

@app.post("/upload")
async def upload_product(file: UploadFile = File(...)):
    # Save uploaded image
    img_path = "temp.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract features
    features = extract_features(img_path).tolist()

    # Store features in Pinecone
    image_id = str(uuid.uuid4())  # Generate a unique ID
    index.upsert([(image_id, features)])

    return {"image_id": image_id}

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    # Save uploaded image
    img_path = "temp.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract features
    query_features = extract_features(img_path).tolist()

    # Query Pinecone
    results = index.query(query_features, top_k=5, include_metadata=True)

    # Get top 5 similar images
    similar_images = [match.id for match in results.matches]

    return {"similar_images": similar_images}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)