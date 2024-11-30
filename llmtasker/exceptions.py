from typing import Any


class LLMTaskerException(Exception):
    def __init__(self, msg: str, code: Any):
        self.msg = msg
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.msg}"

    def __iter__(self):
        # Retourne un itérateur des paires clé-valeur de l'objet
        yield "msg", self.msg
        yield "code", self.code


class LLMTParserException(LLMTaskerException):
    """LLM Tasker Parser Exception."""


class LCClasseOutputParserException(LLMTParserException):
    """LLM Tasker LangChain Output Parser Exception."""
