from typing import Optional
from datetime import datetime

class TripAdvisorReview():
    nome: str
    cidade_e_estado: Optional[str]
    contribuicoes: int
    nota: int
    titulo: str
    em_companhia_de: Optional[str]
    review: str
    nota_custo: Optional[int]
    nota_atendimento: Optional[int]
    nota_comida: Optional[int]
    nota_ambiente: Optional[int]
    is_parceria_patrocinada: bool
    data_postagem: datetime

    def _grade_validation(self, grade):
        if not isinstance(grade, int):
            raise TypeError("grade must be an int")
        if not (1 <= grade <= 5):
            raise ValueError("grade must be between 1 and 5")

    def __setattr__(self, name, value):
        if name == "nota":
            self._grade_validation(value)
        elif name in {"nota_custo", "nota_atendimento", "nota_comida", "nota_ambiente"}:
            if value is not None:
                self._grade_validation(value)
        super().__setattr__(name, value)

    @property
    def title_len(self) -> int:
        return len(self.titulo)

    @property
    def review_len(self) -> int:
        return len(self.review)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["title_len"] = self.title_len
        d["review_len"] = self.review_len
        return d