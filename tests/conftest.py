from pathlib import Path
from typing import List
import pytest

from llmtasker.items.base import BaseItem, Classe


def get_path_with_relation_to_current_file(p: str) -> str:
    return Path(__file__).parent.joinpath(str(p)).resolve()


@pytest.fixture
def jinja_template_filepath(request) -> str:
    filename = request.param
    return get_path_with_relation_to_current_file(f"tests_assets/{filename}")


@pytest.fixture
def base_item_examples() -> List[BaseItem]:
    return [
        BaseItem[str, str](id="1", input="foo_1", output="bar_1"),
        BaseItem[str, str](id="2", input="foo_2", output="bar_2"),
        BaseItem[str, str](id="3", input="foo_3", output="bar_3"),
    ]


@pytest.fixture
def classes_examples() -> List[Classe]:
    return [
        Classe(class_id="1", label="label_1", description="description_1"),
        Classe(class_id="2", label="label_2", description="description_2"),
        Classe(class_id="3", label="label_3", description="description_3"),
    ]


@pytest.fixture
def fstring_prompt_template(request):
    if request.param == "prompt_generic_1":
        return """
        Tu es un classifier.

        Tu dois identifier les animaux. Donne un nom d'animal.

        L'input à classifier est entre |||.
        """
    elif request.param == "prompt_generic_2":
        return """
        Tu es un classifier.

        Tu dois identifier les animaux. Donne un nom d'animal.

        L'input à classifier est entre |||.

        Voici des examples de réponse:
        {examples}
        """
    elif request.param == "prompt_generic_3":
        return """
        Tu es un classifier.

        Tu dois identifier les animaux. Donne un nom d'animal.

        Voici les classes que tu dois identifier avec leur description:
        {classes}

        L'input à classifier est entre |||.
        """
    else:
        return ""


@pytest.fixture
def base_item_inputs() -> List[BaseItem]:
    return [
        BaseItem[str, str](id="1", input="foo_1"),
        BaseItem[str, str](id="2", input="foo_2"),
        BaseItem[str, str](id="3", input="foo_3"),
    ]
