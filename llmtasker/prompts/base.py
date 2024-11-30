"""Module de base pour formalier un prompt d'une tâche."""

from typing import Any, Union, List, ClassVar, Mapping
import os
from pathlib import Path
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader

from llmtasker.items.base import ItemCollection, T, Classe
from llmtasker.exceptions import LLMTaskerException


class ITaskPrompt:
    """Interface pour générer une nouvelle classe permettant de constuire un prompt."""

    def generate(self) -> Any:
        """Méthode à implémenter pour générer un prompt utilisable.

        Raises:
            NotImplementedError: A implementer

        Returns:
            Any: A définir par l'utilisateur
        """
        raise NotImplementedError


class AbstractBasePrompt(ITaskPrompt, BaseModel):
    """Gestion abstraite d'un prompt pour une tâche.

    Implémentation d'une classe générique permettant de gérer un prompt \
    à partir de:

    * instructions: prompt textuel possiblement avec des variables
    * variables: dictionnaire de variables à intégrer dans le prompt final
    * examples: examples de résultat de la tâche avec le format du package (BaseItem)
    * classes: catégories de classification avec le format du package (Classe). \
        utile pour les classifications (tâches principale du package initialement)
    """

    # model_config: ClassVar[Dict] = {"arbitrary_types_allowed": True}
    TEMPLATE_DIR: ClassVar[str] = "./prompts"

    instructions: str
    variables: Mapping[str, str] = {}
    examples: Union[ItemCollection, List[T]] = []
    classes: List[Classe] = []

    # @classmethod
    # def format_input(cls, input: str = "{input}") -> str:
    #     return cls.INPUT_FORMAT.format(delim=cls.INPUT_DELIM, input=input)

    @classmethod
    def is_jinja_file(cls, template: str) -> bool:
        """Test si le template est un fichier *.jinja.

        Args:
            template (str): chemin du template à partir de `cls.TEMPLATE_DIR`

        Returns:
            bool: True si fchier jinja, False sinon.
        """
        return (
            template.endswith(".jinja")
            and Path(os.path.join(cls.TEMPLATE_DIR, template)).exists()
        )

    @classmethod
    def load_prompt_from_template(cls, template: str, variables: dict = {}) -> str:
        """Retourne le template jinja rempli avec les variables.

        la version de lecture des template jinja avec langchain est trop restreinte
        (par mesure de sécurité justifée!)

        Args:
            template (str): chemin du template relatif à `cls.TEMPLATE_DIR`
            variables (dict, optional): Dictionnaire de variables à remplacer \
                dans le template. Defaults to {}.

        Returns:
            str: chaine de caratères
        """
        env = Environment(loader=FileSystemLoader(cls.TEMPLATE_DIR))
        jinja_template = env.get_template(template)

        return jinja_template.render(variables)

    @classmethod
    def check_input_var_in(cls, text: str, expr: str) -> bool:
        """Vérifie l'existence d'une expression dans un texte.

        Args:
            text (str): texte
            expr (str): expression à rechercher

        Raises:
            LLMTaskerException: exception si l'exp n'est pas présente

        Returns:
            bool: True si l'expression est contenue
        """
        if expr in text:
            return True
        raise LLMTaskerException(
            msg=f"expression '{expr}' should be in instructions.",
            code="check_input_var_in",
        )

    class Config:
        arbitrary_types_allowed = True
