from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class PDFPreprocessor:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def pdf_loader(self):
        loader = PyPDFLoader(self.filepath)
        pages = []
        for doc in loader.lazy_load():
            pages.append(doc)
        logger.success(f"Document Sucessfully Loader")
        return pages

    def chunk_document(
        self, pages: List, chunk_size: int = 500, chunk_overlap: int = 200
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunk_text = text_splitter.split_documents(pages)
        page_content = [doc.page_content for doc in chunk_text]
        logger.success(f"Document Sucessfully Chunked into {len(chunk_text)} chunks")
        return page_content
