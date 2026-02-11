

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ocr_utils import extract_text_from_image
from nlp_utils import predict_document_type
import cloudinary
import cloudinary.uploader
import os

app = FastAPI()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

@app.post("/classify_document")
async def classify_document(file: UploadFile = File(...)):


    file_bytes = await file.read()


    text = extract_text_from_image(file_bytes)
    doc_type, confidence = predict_document_type(text)


    FOLDER_MAP = {
        "PAN": "PAN",
        "DRIVING_LICENCE": "DRIVING_LICENCE",
        "INVOICE_OUTWARD": "INVOICE_OUTWARD",
        "INVOICE_INWARD": "INVOICE_INWARD",
        "INVOICE": "INVOICE"
    }

    folder_name = FOLDER_MAP.get(doc_type, "OTHER")


    upload_result = cloudinary.uploader.upload(
        file_bytes,
        folder=f"documents/{folder_name}"
    )

    return JSONResponse(content={
        "document_type": doc_type,
        "confidence": confidence,
        "image_url": upload_result["secure_url"],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
