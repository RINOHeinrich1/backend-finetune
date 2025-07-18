from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class FeedbackDoc(BaseModel):
    text: str
    source: str
    
class FeedbackRequest(BaseModel):
    question: str
    positive_docs: List[FeedbackDoc]
    negative_docs: List[FeedbackDoc]
