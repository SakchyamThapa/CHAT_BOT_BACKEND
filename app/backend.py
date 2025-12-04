from fastapi import FastAPI
from app.schema.request import QueryRequest, ProcessPDFRequest
from app.emdeddings.embedding_model import EmbeddingModelClient
from app.config.config_loader import settings
from app.vectorstore.pinecone_db import PineconeClient
from app.utils.utility import clean_final_ans
from app.schema.response import (
    PDFProcessErrorResponse,
    PDFProcessSuccessResponse,
    SuccessResponse,
    ErrorResponse,
    FileUploadSuccessResponse,
    FileUploadErrorResponse,
)
from app.model.chat_model import ChatModel
from loguru import logger
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
from app.preprocessing.pdf_preprocessor import PDFPreprocessor

app = FastAPI(title="MY ChatBot", version="1.0.0")

# objects initialization
embedding_model = None
pinecone_client = None
chat_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModelClient("all-mpnet-base-v2")
        logger.info("Embedding Model Client initialized")
    return embedding_model

def get_pinecone_client():
    global pinecone_client
    if pinecone_client is None:
        pinecone_client = PineconeClient(api_key=settings.PINECONE_API_KEY)
        logger.info("Pinecone Client initialized")
    return pinecone_client

def get_chat_model():
    global chat_model
    if chat_model is None:
        chat_model = ChatModel(model_name=settings.MODEL_NAME)
        logger.info("Chat Model initialized")
    return chat_model


get_embedding_model()
get_pinecone_client()
get_chat_model()


# directory creation for uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def read_root():
    return {"message": "Welcome to MY ChatBot API"}


@app.get("/health")
async def health_check():
    return {"status": "OK"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate only PDF
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Clear uploads directory before saving new file
        for item in os.listdir(UPLOAD_DIR):
            item_path = os.path.join(UPLOAD_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        # Save new file
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            file.file.close()

        logger.info(f"File {file.filename} uploaded successfully to {file_path}")

        return FileUploadSuccessResponse(
            status="success", message="PDF uploaded successfully", file_path=file_path
        )
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return FileUploadErrorResponse(
            code=500,
            status="error",
            message="An error occurred while uploading the PDF with the error: "
            + str(e),
        )


@app.post("/process-uploaded-pdf")
async def process_uploaded_pdf(request: ProcessPDFRequest):
    try:
        # Check if there is any file in the uploads directory
        files = os.listdir(UPLOAD_DIR)
        if not files:
            raise HTTPException(
                status_code=400, detail="No PDF file found in uploads directory"
            )

        file_path = os.path.join(
            UPLOAD_DIR, request.file_name
        )  # Get the specific file path

        pdf_handler = PDFPreprocessor(filepath=file_path)
        pages = pdf_handler.pdf_loader()
        chunked_pages = pdf_handler.chunk_document(pages=pages)

        chunked_embeddings = embedding_model.get_embedding(chunked_pages)
        pinecone_client.create_index(request.index_name, dimension=768)
        pinecone_client.upsert_vectors(
            index_name=request.index_name,
            chunk_text=chunked_pages,
            vectors=chunked_embeddings,
            namespace=request.namespace,
        )

        logger.info(f"File {request.file_name} processed and data stored in Pinecone")

        return PDFProcessSuccessResponse(
            code=200,
            status="success",
            message=f"PDF {request.file_name} processed successfully and data stored in Pinecone",
        )
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}")
        return PDFProcessErrorResponse(
            code=500,
            status="error",
            message="An error occurred while processing the uploaded PDF: " + str(e),
        )


@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        query = request.query
        query_embeddings = embedding_model.get_embedding(query)
        context = pinecone_client.query(
            query_vector=query_embeddings,
            index_name=request.index_name,
            name_space=request.namespace,
        )
        cleaned_context = clean_final_ans(context)
        prompt = chat_model.format_prompt(query=query, context=cleaned_context)
        logger.info(f"Generated prompt: {prompt}")
        final_response = chat_model.generate_response(prompt=prompt)
        return SuccessResponse(
            code=200,
            status="success",
            description="Query processed successfully",
            data=final_response.content,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return ErrorResponse(
            code=500,
            status="error",
            description="An error occurred while processing the query",
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
