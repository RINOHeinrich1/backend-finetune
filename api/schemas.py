from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class FeedbackRequest(BaseModel):
    question: str
    positive_docs: List[str]
    negative_docs: List[str]
