from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    index_name: str
    namespace: str

    
class ProcessPDFRequest(BaseModel):
    file_name: str
    index_name: Optional[str] = None
    namespace: Optional[str] = None