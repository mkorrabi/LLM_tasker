from typing import List
import pytest
import json
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.runnables.base import RunnableEach
from llmtasker.parsers import LCJsonOutputParser, LCPydanticOutputParser
from llmtasker.parsers import LCClasseParserUtils, LCPydanticParserUtils
from llmtasker.exceptions import LCClasseOutputParserException
from llmtasker.items.base import Classe


############
# FIXTURES #
############
@pytest.fixture
def fake_classes():
    return [
        Classe(label="label_1"),
        Classe(label="label_2"),
        Classe(label="label_3"),
    ]


#############################################
# LCClasseParserUtils: parse une sortie LLM #
#############################################


# check_if_class_exists() is the logic method
def test_lcclasseparserutils_check_if_class_exists_closed(fake_classes):
    utils = LCClasseParserUtils(classes=fake_classes)

    with pytest.raises(LCClasseOutputParserException) as e:
        utils.check_if_class_exists("label_1000")
        assert str(e) == "[class_not_found] label_1000: le label n'existe pas."
    with pytest.raises(LCClasseOutputParserException) as e:
        utils.check_if_class_exists(None)
        assert str(e) == "[class_none] La classe n'est pas valide."
    with pytest.raises(LCClasseOutputParserException) as e:
        utils.check_if_class_exists({"foo": "bar"})
        assert str(e) == "[label_not_found] La classe n'est pas valide."

    assert utils.check_if_class_exists("label_1").label == "label_1"
    assert utils.check_if_class_exists({"label": "label_2"}).label == "label_2"
    assert utils.check_if_class_exists(Classe(label="label_2")).label == "label_2"
    assert utils.check_if_class_exists(Classe(label="label_1")) == fake_classes[0]


def test_lcclasseparserutils_check_if_class_exists_open():
    one_class = Classe(label="label_1")
    utils = LCClasseParserUtils(classes=[one_class], closed=False)

    assert utils.check_if_class_exists("label_1") == one_class
    assert utils.check_if_class_exists("label_1") == one_class
    assert utils.check_if_class_exists("label_1000").label == "label_1000"
    assert len(utils.classes) == 2


def test_lcclasseparserutils_without_classes():
    LCClasseParserUtils()


def test_lcclasseparserutils_runnables(fake_classes):
    utils = LCClasseParserUtils(classes=fake_classes)

    assert isinstance(utils.runnable_check_if_class_exists(), Runnable)
    assert isinstance(utils.runnable_check_if_class_exists(), RunnableLambda)
    assert isinstance(utils.runnable_each_check_if_class_exists(), Runnable)
    assert isinstance(utils.runnable_each_check_if_class_exists(), RunnableEach)


def test_lcclasseparserutils_parser_from_str(fake_classes):
    utils = LCClasseParserUtils(classes=fake_classes)
    parser = utils.parser_from_str()
    assert parser.invoke(input="label_1") == fake_classes[0]


def test_lcclasseparserutils_parser_from_json(fake_classes):
    utils = LCClasseParserUtils(classes=fake_classes)
    parser = utils.parser_from_json()
    assert parser.invoke(input=fake_classes[0].model_dump_json()) == fake_classes[0]
    assert parser.invoke(input='{"label": "label_1"}') == fake_classes[0]
    assert (
        parser.invoke(input=json.dumps([cl.model_dump() for cl in fake_classes]))
        == fake_classes
    )
    assert parser.invoke(input='["label_1", "label_2", "label_3"]') == fake_classes
    assert parser.invoke(input='[["label_1", "label_2", "label_3"]]') == [fake_classes]
    assert parser.invoke(input='[["label_1"], ["label_2", "label_3"]]') == [
        [fake_classes[0]],
        fake_classes[1:],
    ]


def test_lcclasseparserutils_parser(mocker, fake_classes):
    utils = LCClasseParserUtils(classes=fake_classes)
    parser = utils.parser()

    assert parser.invoke(input='{"label": "label_1"}') == fake_classes[0]
    assert parser.invoke(input='["label_1", "label_2", "label_3"]') == fake_classes
    assert parser.invoke(input="label_1") == fake_classes[0]
    assert parser.invoke(input='[["label_1"], [{"label": "label_2"}, "label_3"]]') == [
        [fake_classes[0]],
        fake_classes[1:],
    ]


##################################################################################
# LCJsonOutputParser: pas réellement implémnter, utiliser JsonOutputParser sinon #
##################################################################################


def test_lcjsonoutputparser():
    parser = LCJsonOutputParser()
    with pytest.raises(NotImplementedError):
        parser.parse("foo bar")


#########################################################
# LCPydanticOutputParser: retourne un objet pydantic v2 #
#########################################################
# pas besoin de tester la logique car c'est une fonctionnalité
# déjà testée dans le package langchain.


class FakeSubModelPerson(BaseModel):
    name: str
    year_of_birth: int
    roles: List[str]


class FakeModelTeam(BaseModel):
    name: str
    objectives: str
    persons: List[FakeSubModelPerson]


def test_lcpydanticoutputparser():
    persons = [
        FakeSubModelPerson(name="A", year_of_birth=1900, roles=["role_1", "role_2"]),
        FakeSubModelPerson(name="B", year_of_birth=2000, roles=["role_2", "role_3"]),
    ]
    fake_team = FakeModelTeam(name="optimus", objectives="prime", persons=persons)
    parser = LCPydanticOutputParser(pydantic_object=FakeModelTeam)

    assert "Here is the output schema:" not in parser.get_format_instructions()
    assert (
        "The output should be formatted as a JSON instance that conforms to the JSON schema below."
        not in parser.get_format_instructions()
    )
    assert parser.parse(fake_team.model_dump_json()) == fake_team


def test_lcpydanticparserutils_parser():
    persons = [
        FakeSubModelPerson(name="A", year_of_birth=1900, roles=["role_1", "role_2"]),
        FakeSubModelPerson(name="B", year_of_birth=2000, roles=["role_2", "role_3"]),
    ]
    fake_team = FakeModelTeam(name="optimus", objectives="prime", persons=persons)
    utils = LCPydanticParserUtils(pydantic_object=FakeModelTeam)

    # test from json string
    assert (
        utils.parser_from_str().invoke(input=fake_team.model_dump_json()) == fake_team
    )
    # test from python dict
    assert utils.parser_from_dict().invoke(input=fake_team.model_dump()) == fake_team
    # test from json string with main parser
    assert utils.parser().invoke(input=fake_team.model_dump_json()) == fake_team
    # test from json string: list of python dict with main parser
    assert utils.parser().invoke(
        input=json.dumps([fake_team.model_dump(), fake_team.model_dump()])
    ) == [fake_team, fake_team]
