from pydantic import BaseModel
from typing import Optional, Any

class SuccessResponse(BaseModel):
    code: int = 200
    status: str = "success"
    description: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    code: int
    status: str = "error"
    description: str
    error: Optional[str] = None

class FileUploadSuccessResponse(BaseModel):
    status: str
    message: str
    file_path: str

class FileUploadErrorResponse(BaseModel):
    code: int = 500
    status: str
    message: str

class PDFProcessSuccessResponse(BaseModel):
    code: int = 200
    status: str
    message: str

class PDFProcessErrorResponse(BaseModel):
    code: int = 500
    status: str
    message: str
