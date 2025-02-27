from fastapi import FastAPI, UploadFile, File
from pinecone import Pinecone
import uvicorn
import shutil
import uuid
from app.feature_extractor import extract_features
from fastapi import FastAPI, UploadFile, File, HTTPException

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
    try:
        # Save uploaded image
        img_path = "temp.jpg"
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Image saved successfully.")

        # Extract features
        query_features = extract_features(img_path).tolist()
        print("Features extracted successfully.")

        # Query Pinecone using keyword arguments
        results = index.query(
            vector=query_features,  # Use keyword argument for the vector
            top_k=5,               # Use keyword argument for top_k
            include_metadata=True  # Use keyword argument for include_metadata
        )
        print("Pinecone query successful.")

        # Get top 5 similar images
        similar_images = [match.id for match in results.matches]
        print("Similar images:", similar_images)

        return {"similar_images": similar_images}
    except Exception as e:
        print("Error in /search endpoint:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
