from application.rag import VectorStore, LLMHandler
from application.exceptions import UnsafeRequestException

import numpy as np



class RagService:

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def make_query(self, query: str) -> str:

        related_documents = self.vector_store.search(query)
        prompt = self.__bulid_prompt(query, documents=related_documents)
        response = llm_handler.query(prompt)

        return response

