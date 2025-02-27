from fastapi import FastAPI, UploadFile, File
import pinecone
import uvicorn
import shutil
from feature_extractor import extract_features

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
index = pinecone.Index("image-search")

#USER FLOW

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

###########################################################

#RETAILER FLOW

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)