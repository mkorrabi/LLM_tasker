from typing import Any, List, Optional, Union
import json
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableBranch
from langchain_core.runnables.base import RunnableEach
from langchain_core.exceptions import OutputParserException

from llmtasker.items.base import Classe
from llmtasker.exceptions import LCClasseOutputParserException


class LCJsonOutputParser(JsonOutputParser):
    def parse(self, text: str) -> Any:
        raise NotImplementedError("Use LCPydanticOutputParser instead.")


class LCPydanticOutputParser(PydanticOutputParser):
    """Parser d'un JSON représentant un objet python en objet pydantic v2.

    On utilise le parser natif de langchain en surchargeant la méthode `parse()` \
    afin d'éviter l'injection d'anglais dans le prompt.

    """

    # dans le précedent parser on transformait toutes les keys en lower:
    # json_object = convert_keys_to_lower(json_object)
    # il faudrait vérifier si avec la nouvelle implémentation
    # on a le même problème

    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = {k: v for k, v in self.pydantic_object.schema().items()}

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return schema_str


class LCClasseParserUtils:
    """Classe de création d'une sequence de Runnable permettant de parser la sortie dans le cadre d'une classification."""

    def __init__(self, classes: Optional[List[Classe]] = [], closed: bool = True):
        """Constructeur de la classe utils.

        Args:
            classes (List[Classe]): Liste de classe références.
            closed (bool): Les classes sont fixées par l'utilisateur \
                Sinon le LLM ajoute de nouvelles classes.
        """
        self.classes = classes
        self.closed = closed

    def check_if_class_exists(
        self, label: Optional[Union[str, Classe, dict]]
    ) -> Classe:
        """Verifie et retourne la classe correspondant au label.

        Si la classe n'existe pas et que self.closed est à True: la nouvelle classe est ajoutée.

        Args:
            label (Optional[Union[str, Classe, dict]]): Label à rechercher.
                Le label peut être de la forme:

                * une chaine de caractère du label (ex: "label_1")
                * une Classe
                * un dictionnaire avec la clé "label"

        Raises:
            LCClasseOutputParserException: [class_none] la classe n'est pas valide. \
                Le label est à None.
            LCClasseOutputParserException: [label_not_found] la classe n'est pas valide. \
                Pas de clé "label" dans le dictionnaire.
            LCClasseOutputParserException: [class_not_found] le label n'existe pas. \
                Le label n'est pas trouvé dans les classes existantes.

        Returns:
            Classe: Classe correspondante au label
        """

        if not label:
            raise LCClasseOutputParserException(
                msg="La classe n'est pas valide.", code="class_none"
            )
        if isinstance(label, list):
            # handle grouped multi labels classification
            return [self.check_if_class_exists(lab) for lab in label]

        if isinstance(label, Classe):
            label = label.label

        if isinstance(label, dict):
            label = label.get("label", None)
            if not label:
                raise LCClasseOutputParserException(
                    msg="La classe n'est pas valide.", code="label_not_found"
                )

        if not isinstance(label, str):
            raise LCClasseOutputParserException(
                msg="Le nom de la classe n'est pas valide.", code="label_not_str"
            )

        for cl in self.classes:
            if cl.label.lower() == label.strip().lower():
                return cl

        if not self.closed:
            # TODO: prendre en compte la description si le LLM en donne une.
            new_classe = Classe(label=label)
            self.classes.append(new_classe)
            return new_classe

        raise LCClasseOutputParserException(
            msg=f"{label}: le label n'existe pas.", code="class_not_found"
        )

    def runnable_check_if_class_exists(self) -> RunnableLambda:
        """Retourne un runnable langchain à partir de la méthode check_if_class_exists().

        Returns:
            RunnableLambda: runnable langchain
        """
        return RunnableLambda(self.check_if_class_exists)

    def runnable_each_check_if_class_exists(self) -> RunnableEach:
        """Retourne un runnable langchain s'appliquant à une liste \
        à partir de la méthode check_if_class_exists().

        Cela équivaut à une boucle sur la liste en entrée.

        Returns:
            RunnableEach: runnable langchain
        """
        return RunnableEach(bound=self.runnable_check_if_class_exists())

    def parser_from_str(self) -> Runnable:
        """Retourne une runnable/parser pour un label.

        Returns:
            Runnable: runnable langchain
        """
        # parsing from string with one label
        return StrOutputParser() | self.runnable_check_if_class_exists()

    def parser_from_json(self) -> Runnable:
        """Retourne un runnable/parser pour un json (au format string).

        Le json peut représenter une liste de labels.

        Returns:
            Runnable: runnable langchain
        """
        chain = JsonOutputParser() | RunnableBranch(
            # parsing from json with multi label
            (
                lambda x: isinstance(x, list),
                self.runnable_each_check_if_class_exists(),
            ),
            # parsing from json with one label
            self.runnable_check_if_class_exists(),
        )
        return chain

    def parser(self) -> Runnable:
        """Retourne une chaine de parser.

        Retourne un chaine de parser pour gérer l'ensemble des cas:

        * liste de labels ou json
        * un label simple

        Si parser_from_json() échoue, c'est que la sortie n'est pas parsable.
        Dans ce cas, on teste avec parser_from_str().

        Voir les tests unitaires correspondants.

        Returns:
            Runnable: runnable langchain
        """
        return self.parser_from_json().with_fallbacks(
            fallbacks=[self.parser_from_str()],
            exceptions_to_handle=(OutputParserException,),
        )


class LCPydanticParserUtils:
    """Classe de création d'une sequence de Runnable permettant de parser la sortie dans le cadre d'une sortie qui doit être parser en objet pydantic (json)."""

    def __init__(self, pydantic_object):
        """Constructeur de la classe utils.

        Args:
            pydantic_object (BaseModel): Type d'objet pydantic à retourner par le parser.
        """
        self.pydantic_object = pydantic_object

    def parser_from_dict(self) -> Runnable:
        """Retourne un runnable/parser pour gérer un dictionnaire python en entrée.

        Afin de s'appuyer sur les parser de langchain, on dump le dictionnaire en json.

        Returns:
            Runnable: runnable langchain
        """
        return RunnableLambda(func=json.dumps) | self.parser_from_str()

    def parser_from_str(self) -> LCPydanticOutputParser:
        """Retourne un parser pydantic.

        On surcharge celui de langchain.

        Returns:
            LCPydanticOutputParser: parser pydantic langchain
        """
        return LCPydanticOutputParser(pydantic_object=self.pydantic_object)

    def parser(self) -> Runnable:
        """Retourne un runnable/parser pour gérer plusieurs cas.

        Les cas possibles:
        * un json
        * une liste de json

        Returns:
            Runnable: runnable langchain
        """
        fb_chain = JsonOutputParser() | RunnableBranch(
            (
                lambda x: isinstance(x, list),
                RunnableEach(bound=self.parser_from_dict()),
            ),
            self.parser_from_dict(),
        )
        chain = self.parser_from_str().with_fallbacks(
            [fb_chain], exceptions_to_handle=(OutputParserException,)
        )

        return chain
