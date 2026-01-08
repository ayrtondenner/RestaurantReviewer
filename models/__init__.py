from typing import Optional
from datetime import datetime

class EnrichedReview():
    contributions: int
    grade: int
    title: str
    company: str
    review: str
    cost_grade: Optional[int]
    service_grade: Optional[int]
    food_grade: Optional[int]
    partnership: bool
    date: datetime