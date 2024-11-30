"""Test unitaires sur la logique de consutrction des prompts"""

import pytest
from llmtasker.prompts.base import AbstractBasePrompt
from llmtasker.exceptions import LLMTaskerException
from llmtasker.prompts.lc_base import (
    LCPromptExamplesMixin,
    LCPromptInputMixin,
    LCPromptClassesMixin,
    LCPromptInstructionsMixin,
    LCPrompt,
)

from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)

################################
# Tests sur AbstractBasePrompt #
################################


def test_prompt_is_not_jinja_file():
    assert not AbstractBasePrompt.is_jinja_file("jinja {{test}}.test")
    assert not AbstractBasePrompt.is_jinja_file("exemple.jinja.test")
    assert not AbstractBasePrompt.is_jinja_file("test_assets/test_no_exists.jinja")


@pytest.mark.parametrize("jinja_template_filepath", ["test.jinja"], indirect=True)
def test_prompt_is_jinja_file(jinja_template_filepath):
    assert AbstractBasePrompt.is_jinja_file(str(jinja_template_filepath))


def test_load_prompt_from_template_not_found():
    with pytest.raises(Exception):
        AbstractBasePrompt.load_prompt_from_template("template_not_found")


@pytest.mark.parametrize("jinja_template_filepath", ["test.jinja"], indirect=True)
def test_load_prompt_from_template_1(jinja_template_filepath):
    AbstractBasePrompt.TEMPLATE_DIR = jinja_template_filepath.parent
    output = AbstractBasePrompt.load_prompt_from_template("test.jinja")
    assert output == "Tu es un classifier.\n\n"


@pytest.mark.parametrize("jinja_template_filepath", ["test.jinja"], indirect=True)
def test_load_prompt_from_template_2(jinja_template_filepath):
    AbstractBasePrompt.TEMPLATE_DIR = jinja_template_filepath.parent
    variables = {"variable_1": "foo", "variable_2": "bar"}
    output = AbstractBasePrompt.load_prompt_from_template(
        "test.jinja", variables=variables
    )
    assert output == "Tu es un classifier.\nfoo\nbar"


def test_check_input_var_in():
    assert AbstractBasePrompt.check_input_var_in("lorem ipsum {foo} bar", "{foo}")


def test_check_input_var_not_in():
    with pytest.raises(LLMTaskerException):
        assert AbstractBasePrompt.check_input_var_in("lorem ipsum foo bar", "{foo}")


#########################################################################
# Tests sur les Mixin: fonctionnalités possibles sur AbstractBasePrompt #
#########################################################################


class FakeLCPromptExamplesMixin(AbstractBasePrompt, LCPromptExamplesMixin):
    pass


class FakeLCPromptInputMixin(AbstractBasePrompt, LCPromptInputMixin):
    pass


class FakeLCPromptClassesMixin(AbstractBasePrompt, LCPromptClassesMixin):
    pass


class FakeLCPromptInstructionsMixin(
    AbstractBasePrompt, LCPromptInstructionsMixin, LCPromptExamplesMixin
):
    pass


class TestLCPromptExamplesMixin:
    @pytest.fixture(autouse=True)
    def setup(self, base_item_examples):
        self.obj = FakeLCPromptExamplesMixin(
            instructions="",
        )
        self.obj.examples = base_item_examples

    def test_generate_examples_with_user_ai_roles(self):
        ex_output = self.obj.generate_examples_with_user_ai_roles()
        assert isinstance(ex_output, list)
        even_indices = list(
            map(
                lambda i: isinstance(ex_output[i], (HumanMessagePromptTemplate)),
                filter(lambda x: x % 2 == 0, range(len(ex_output))),
            )
        )
        odd_indices = list(
            map(
                lambda i: isinstance(ex_output[i], (AIMessage)),
                filter(lambda x: x % 2 != 0, range(len(ex_output))),
            )
        )

        assert all(even_indices)
        assert all(odd_indices)

    def test_generate_examples_in_instructions_with_jinja(self):
        # injection dans les variables
        self.obj.generate_examples_in_instructions_with_jinja()
        assert self.obj.variables.get("examples")
        assert isinstance(self.obj.variables.get("examples"), list)

    def test_generate_examples_in_instructions_with_fstring(self):
        # injection d'une variable example
        self.obj.generate_examples_in_instructions_with_fstring()
        assert self.obj.variables.get("examples")
        assert isinstance(self.obj.variables.get("examples"), str)


class TestLCPromptInputMixin:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.obj = FakeLCPromptInputMixin(instructions="")

    # not necessary (removed and replace by LCPrompt.is_var_in_instructions())
    # def test_is_input_var_in_instructions_1(self):
    #     instruction_msg = SystemMessage(content="En entrée: {input}")
    #     assert self.obj.is_input_var_in_instructions(instruction_msg)
    #     instruction_msg = HumanMessage(content="En entrée: {input}")
    #     assert self.obj.is_input_var_in_instructions(instruction_msg)

    # def test_is_input_var_in_instructions_2(self):
    #     instruction_msg = SystemMessagePromptTemplate.from_template(
    #         template="En entrée: {input}"
    #     )
    #     assert self.obj.is_input_var_in_instructions(instruction_msg)
    #     instruction_msg = HumanMessagePromptTemplate.from_template(
    #         template="En entrée: {input}"
    #     )
    #     assert self.obj.is_input_var_in_instructions(instruction_msg)

    def test_generate_input_with_user_role(self):
        output: HumanMessagePromptTemplate = self.obj.generate_input_with_user_role()
        assert isinstance(output, HumanMessagePromptTemplate)
        assert "input" in output.prompt.input_variables


class TestLCPromptClassesMixin:
    @pytest.fixture(autouse=True)
    def setup(self, classes_examples):
        self.obj = FakeLCPromptClassesMixin(
            instructions="",
        )
        self.obj.classes = classes_examples

    def test_no_classes_in_instructions(self):
        self.obj.classes = []
        self.obj.generate_classes_in_instructions_with_jinja()
        assert self.obj.variables.get("classes") is None

    def test_generate_classes_in_instructions_with_jinja(self):
        self.obj.generate_classes_in_instructions_with_jinja()
        assert self.obj.variables.get("classes")
        assert isinstance(self.obj.variables.get("classes"), list)

    def test_generate_classes_in_instructions_with_fstring(self):
        self.obj.generate_classes_in_instructions_with_fstring()
        assert self.obj.variables.get("classes")
        assert isinstance(self.obj.variables.get("classes"), str)


class TestLCPromptInstructionsMixin:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.obj = FakeLCPromptInstructionsMixin(
            instructions="",
        )

    def test_generate_instructions_with_system_role(self):
        self.obj.instructions = "test system role"
        assert isinstance(
            self.obj.generate_instructions_with_system_role(),
            SystemMessagePromptTemplate,
        )

    def test_generate_instructions_with_user_role(self):
        self.obj.instructions = "test user role"
        assert isinstance(
            self.obj.generate_instructions_with_user_role(),
            HumanMessagePromptTemplate,
        )

    def test_generate_instructions_with_user_role_with_example(
        self, base_item_examples
    ):
        self.obj.instructions = "test user role with examples"
        first_example_content = base_item_examples[0].format_input()
        self.obj.examples = base_item_examples
        examples_messages = self.obj.generate_examples_with_user_ai_roles()
        examples_messages_copy = examples_messages.copy()
        output = self.obj.generate_instructions_with_user_role(examples_messages)
        assert (
            output.prompt.template
            == f"{self.obj.instructions}\n\n{first_example_content}"
        )
        assert examples_messages == examples_messages_copy[1:]


####################################################################################
# Tests sur la classe LCPrompt: proposition flexibles d'implémentation d'un prompt #
####################################################################################


class FakeLCPrompt(LCPrompt):
    TEMPLATE_DIR = "./tests_assets"


@pytest.fixture
def fake_lcprompt_jinja_with_instructions(request, jinja_template_filepath):
    LCPrompt.TEMPLATE_DIR = jinja_template_filepath.parent
    return LCPrompt(
        instructions=jinja_template_filepath.name,
    )


@pytest.fixture
def fake_lcprompt_fstring_with_instructions(request, fstring_prompt_template):
    return LCPrompt(instructions=fstring_prompt_template)


@pytest.mark.parametrize(
    "jinja_template_filepath", ["prompt_generic.jinja"], indirect=True
)
def test_LCPrompt_jinja_output(fake_lcprompt_jinja_with_instructions):
    prompt_generation = fake_lcprompt_jinja_with_instructions.generate()
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert "Human: test" in prompt_generation.format(input="test")


@pytest.mark.parametrize(
    "jinja_template_filepath", ["prompt_generic.jinja"], indirect=True
)
def test_LCPrompt_jinja_examples(
    fake_lcprompt_jinja_with_instructions, base_item_examples
):
    fake_lcprompt_jinja_with_instructions.examples = base_item_examples
    prompt_generation = fake_lcprompt_jinja_with_instructions.generate(
        template_format="jinja2"
    )
    content = prompt_generation.format(input="input final")
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert base_item_examples[0].format_as_example() in content
    assert base_item_examples[1].format_as_example() in content


@pytest.mark.parametrize(
    "jinja_template_filepath", ["prompt_generic.jinja"], indirect=True
)
def test_LCPrompt_jinja_classes(
    fake_lcprompt_jinja_with_instructions, classes_examples
):
    fake_lcprompt_jinja_with_instructions.classes = classes_examples
    prompt_generation = fake_lcprompt_jinja_with_instructions.generate(
        template_format="jinja2"
    )
    content = prompt_generation.format(input="input final")
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert classes_examples[0].format() in content
    assert classes_examples[1].format() in content


@pytest.mark.parametrize("fstring_prompt_template", ["prompt_generic_1"], indirect=True)
def test_LCPrompt_fstring_output(fake_lcprompt_fstring_with_instructions):
    prompt_generation = fake_lcprompt_fstring_with_instructions.generate()
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert "Human: test" in prompt_generation.format(input="test")


@pytest.mark.parametrize("fstring_prompt_template", ["prompt_generic_2"], indirect=True)
def test_LCPrompt_fstring_examples(
    fake_lcprompt_fstring_with_instructions, base_item_examples
):
    fake_lcprompt_fstring_with_instructions.examples = base_item_examples
    prompt_generation = fake_lcprompt_fstring_with_instructions.generate()
    content = prompt_generation.format(input="input final")
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert base_item_examples[0].format_as_example() in content
    assert base_item_examples[1].format_as_example() in content


@pytest.mark.parametrize("fstring_prompt_template", ["prompt_generic_3"], indirect=True)
def test_LCPrompt_fstring_classes(
    fake_lcprompt_fstring_with_instructions, classes_examples
):
    fake_lcprompt_fstring_with_instructions.classes = classes_examples
    prompt_generation = fake_lcprompt_fstring_with_instructions.generate()
    content = prompt_generation.format(input="input final")
    assert isinstance(prompt_generation, ChatPromptTemplate)
    assert classes_examples[0].format() in content
    assert classes_examples[1].format() in content


@pytest.mark.parametrize(
    "jinja_template_filepath", ["prompt_generic.jinja"], indirect=True
)
def test_generate_examples_with_user_ai_roles_is_called(
    mocker, fake_lcprompt_jinja_with_instructions, base_item_examples
):
    spy_my_function = mocker.spy(LCPrompt, "generate_examples_with_user_ai_roles")
    fake_lcprompt_jinja_with_instructions.use_ai_user_role_for_examples = True
    fake_lcprompt_jinja_with_instructions.examples = base_item_examples
    fake_lcprompt_jinja_with_instructions.generate(template_format="jinja2")
    assert spy_my_function.called


@pytest.mark.parametrize(
    "jinja_template_filepath", ["prompt_generic.jinja"], indirect=True
)
def test_generate_instructions_with_user_role_is_called(
    mocker, fake_lcprompt_jinja_with_instructions
):
    spy_my_function = mocker.spy(LCPrompt, "generate_instructions_with_user_role")
    fake_lcprompt_jinja_with_instructions.use_system_prompt = False
    fake_lcprompt_jinja_with_instructions.generate(template_format="jinja2")
    assert spy_my_function.called


def test_input_var_in_instructions(mocker):
    fake_lcprompt = LCPrompt(instructions="Voici l'input à classifier: {input}")
    spy_my_function = mocker.spy(LCPrompt, "is_var_in_instructions")
    fake_lcprompt.input_in_instructions = True
    fake_lcprompt.generate()
    assert spy_my_function.called


def test_input_var_not_in_instructions(mocker):
    fake_lcprompt = LCPrompt(instructions="Voici l'input à classifier: input")
    fake_lcprompt.input_in_instructions = True
    with pytest.raises(LLMTaskerException, match="should be in instructions"):
        fake_lcprompt.generate()


def test_examples_not_in_instructions(base_item_examples):
    fake_lcprompt = LCPrompt(
        instructions="Voici les exemples", examples=base_item_examples
    )
    with pytest.raises(
        LLMTaskerException, match="'{examples}' should be in instructions"
    ):
        fake_lcprompt.generate()


def test_examples_in_instructions(mocker, base_item_examples):
    fake_lcprompt = LCPrompt(
        instructions="Voici les exemples: \n{examples}", examples=base_item_examples
    )
    spy_my_function = mocker.spy(LCPrompt, "is_var_in_instructions")
    fake_lcprompt.generate()

    assert spy_my_function.called


def test_classes_not_in_instructions(classes_examples):
    fake_lcprompt = LCPrompt(instructions="Voici les classes", classes=classes_examples)
    with pytest.raises(
        LLMTaskerException, match="'{classes}' should be in instructions"
    ):
        fake_lcprompt.generate()


def test_classes_in_instructions(mocker, classes_examples):
    fake_lcprompt = LCPrompt(
        instructions="Voici les classes: \n{classes}", classes=classes_examples
    )
    spy_my_function = mocker.spy(LCPrompt, "is_var_in_instructions")
    fake_lcprompt.generate()

    assert spy_my_function.called
