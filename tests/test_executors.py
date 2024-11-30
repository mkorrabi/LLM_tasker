import pytest
from unittest.mock import AsyncMock

from operator import itemgetter
from llmtasker.executors.base import IExecutor
from llmtasker.executors.langchain import (
    LangchainExecutor,
    LCPromptLLMParserExecutor,
)
from llmtasker.exceptions import LLMTaskerException, LLMTParserException
from llmtasker.items.base import ItemCollection, BaseItem
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.runnables.base import Runnable, RunnableMap
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.messages.ai import AIMessage


############
# FIXTURES #
############


@pytest.fixture
def fake_langchain_executor(request):
    has_raw_output = request.param
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||input|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()
    if has_raw_output:
        # fake raw/output
        chain = (
            prompt
            | RunnableMap(raw=llm)
            | RunnablePassthrough.assign(output=itemgetter("raw") | parser)
        )
    else:
        chain = prompt | llm | parser
    return LangchainExecutor(chain=chain, has_raw_output=has_raw_output)


def fake_chat_result(message="output"):
    gen = ChatGeneration(
        message="retor",
        generation_info={},
    )
    return ChatResult(generations=[gen], llm_output={})


@pytest.fixture
def fake_items_collection(base_item_inputs):
    return ItemCollection(base_item_inputs.copy())


#########################################
# Tests sur l'interface d'un executor #
#########################################


class TestInterfaceExecutor:
    @pytest.fixture
    def executor(self):
        return IExecutor()

    def test_execute_one(self, executor, mocker):
        # Créer un objet mock pour BaseItem
        mock_item = mocker.MagicMock(spec=BaseItem[str, str])
        mock_item.output = None  # Initialiser l'attribut output à None

        # Vérifier que la méthode lève NotImplementedError
        with pytest.raises(NotImplementedError):
            executor.execute_one(mock_item)

    def test_execute_batch(self, executor, mocker):
        # Créer une liste d'objets mock pour BaseItem
        mock_items = [mocker.MagicMock(spec=BaseItem[str, str]) for _ in range(5)]

        # Créer un objet ItemCollection à partir de mock_items
        mock_item_collection = mocker.MagicMock(spec=ItemCollection)
        mock_item_collection.__iter__.return_value = iter(mock_items)

        # Vérifier que la méthode lève NotImplementedError
        with pytest.raises(NotImplementedError):
            executor.execute_batch(mock_item_collection)

    @pytest.mark.asyncio
    async def test_execute_aone(self, executor, mocker):
        # Créer un objet mock pour BaseItem
        mock_item = mocker.MagicMock(spec=BaseItem[str, str])

        # Vérifier que la méthode lève NotImplementedError
        with pytest.raises(NotImplementedError):
            await executor.execute_aone(mock_item)

    @pytest.mark.asyncio
    async def test_execute_abatch(self, executor, mocker):
        # Créer une liste d'objets mock pour BaseItem
        mock_items = [mocker.MagicMock(spec=BaseItem) for _ in range(5)]
        for item in mock_items:
            item.output = None  # Initialiser l'attribut output à None

        # Créer un objet ItemCollection à partir de mock_items
        mock_item_collection = mocker.MagicMock(spec=ItemCollection)
        mock_item_collection.__iter__.return_value = iter(mock_items)

        # Vérifier que la méthode lève NotImplementedError
        with pytest.raises(NotImplementedError):
            await executor.execute_abatch(mock_item_collection)


###########################################################
# Tests sur l'implémentation d'un executor avec langchain #
###########################################################


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_chain_instance(fake_langchain_executor):
    assert isinstance(fake_langchain_executor.chain, RunnableWithFallbacks)


# @pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_assign_result_to_item(base_item_inputs):
    fake_exec = LangchainExecutor(chain=FakeChatModel())

    dict_1 = {}
    item = base_item_inputs[0].copy()
    fake_exec.assign_result_to_item(dict_1, item)
    assert item.output is None
    assert item.raw is None
    assert item.error is None

    dict_2 = {"output": "A", "error": {"msg": "B"}}
    item = base_item_inputs[0].copy()
    fake_exec.assign_result_to_item(dict_2, item)
    assert item.output == "A"
    assert item.raw is None
    assert item.error["msg"] == "B"

    dict_3 = {
        "output": "A",
        "error": LLMTaskerException(msg="message", code="foo"),
        "raw": "a",
    }
    item = base_item_inputs[0].copy()
    fake_exec.assign_result_to_item(dict_3, item)
    assert item.output == "A"
    assert item.raw == "a"
    assert item.error.msg == "message"
    assert item.error.code == "foo"


def test_raw_if_error_after_is_none():
    def error_runnable():
        raise Exception("test_raw_if_error_after")

    item = BaseItem(input="test")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||input|||")]
    )
    llm = FakeChatModel()
    chain = (
        prompt
        | RunnableMap(raw=llm)
        | RunnablePassthrough.assign(output=error_runnable)
    )
    fake_exec = LangchainExecutor(chain=chain, has_raw_output=True)

    fake_exec.execute_one(item)

    assert item.output is None
    assert isinstance(item.error, Exception)
    assert item.raw is None


def test_raw_if_error_after_is_not_none():
    def error_runnable():
        raise Exception("test_raw_if_error_after")

    item = BaseItem(input="test")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||input|||")]
    )
    llm = FakeChatModel()
    parser_none = RunnablePassthrough.assign(output=lambda _: None)
    chain = (
        prompt
        | RunnableMap(raw=llm)
        | RunnablePassthrough.assign(output=error_runnable).with_fallbacks(
            [parser_none], exception_key="error"
        )
    )
    fake_exec = LangchainExecutor(chain=chain, has_raw_output=True)

    fake_exec.execute_one(item)

    assert item.output is None
    assert isinstance(item.error, Exception)
    assert isinstance(item.raw, AIMessage)


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_output_and_raw_are_none_after_chain_exception(mocker, fake_langchain_executor):
    mocker.patch(
        "langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception
    )
    item = BaseItem(input="test", raw=AIMessage(content="foo"), output="bar")

    assert item.raw is not None
    assert item.output is not None
    fake_langchain_executor.execute_one(item)
    assert item.error is not None
    assert isinstance(item.error, Exception)
    assert item.raw is None
    assert item.output is None


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_output_and_raw_are_none_after_llm_exception(mocker, fake_langchain_executor):
    mocker.patch(
        "langchain_core.language_models.chat_models.BaseChatModel.invoke",
        side_effect=Exception,
    )
    item = BaseItem(input="test", raw=AIMessage(content="foo"), output="bar")

    assert item.raw is not None
    assert item.output is not None
    fake_langchain_executor.execute_one(item)
    assert item.error is not None
    assert isinstance(item.error, Exception)
    assert item.raw is None
    assert item.output is None


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_one(base_item_inputs, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    fake_exec.execute_one(base_item_inputs[0])
    assert base_item_inputs[0].error is None
    assert base_item_inputs[0].output == "fake response"
    if fake_langchain_executor.has_raw_output:
        assert isinstance(base_item_inputs[0].raw, AIMessage)
        assert base_item_inputs[0].raw.content == "fake response"
    else:
        assert base_item_inputs[0].raw is None


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_one_with_error(base_item_inputs, mocker, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    mocker.patch(
        "langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception
    )
    fake_exec.execute_one(base_item_inputs[0])
    assert base_item_inputs[0].error is not None
    assert isinstance(base_item_inputs[0].error, Exception)


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_one_args(base_item_inputs, mocker, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")
    mock_instance.invoke.return_value = {"output": None, "error": None}

    fake_exec.execute_one(base_item_inputs[0])
    mock_instance.invoke.assert_called_once_with(
        input={"input": base_item_inputs[0].format_input()}, config=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_aone(base_item_inputs, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    await fake_exec.execute_aone(base_item_inputs[0])
    assert base_item_inputs[0].error is None
    assert base_item_inputs[0].output == "fake response"


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_aone_with_error(
    base_item_inputs, mocker, fake_langchain_executor
):
    fake_exec = fake_langchain_executor
    mocker.patch(
        "langchain_core.runnables.RunnableSequence.ainvoke", side_effect=Exception
    )
    await fake_exec.execute_aone(base_item_inputs[0])
    assert base_item_inputs[0].error is not None
    assert isinstance(base_item_inputs[0].error, Exception)


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_aone_args(base_item_inputs, mocker, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")

    mock_instance.ainvoke = AsyncMock(return_value={"output": None, "error": None})

    await fake_exec.execute_aone(base_item_inputs[0])
    mock_instance.ainvoke.assert_called_once_with(
        input={"input": base_item_inputs[0].format_input()}, config=None
    )


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_batch(fake_items_collection, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    output = fake_exec.execute_batch(fake_items_collection)
    assert isinstance(output, ItemCollection)
    for item in fake_items_collection:
        assert item.error is None
        assert item.output == "fake response"


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_batch_with_error(
    fake_items_collection, mocker, fake_langchain_executor
):
    mock_instance = mocker.patch(
        "langchain_core.language_models.chat_models.BaseChatModel.invoke",
    )
    mock_instance.side_effect = [mocker.DEFAULT] * (len(fake_items_collection) - 1) + [
        Exception("test")
    ]
    mock_instance.return_value = "fake response"

    fake_exec = fake_langchain_executor
    output = fake_exec.execute_batch(fake_items_collection)
    assert isinstance(output, ItemCollection)

    # OLD WAY, but not good because of side_effect in parallel
    #
    # assert mock_instance.call_count == len(fake_items_collection)
    # for item in fake_items_collection[:-1]:
    #     assert item.error is None
    #     assert item.output == "fake response"
    # assert fake_items_collection[-1].output is None
    # assert isinstance(fake_items_collection[-1].error, BaseException)

    mock_instance.call_count == len(fake_items_collection)
    error_count = 0
    for i in fake_items_collection:
        if i.output:
            # check if 2 output are "fake_reponse"
            assert i.output == "fake response"
        else:
            # check if 1 error
            assert isinstance(i.error, BaseException)
            error_count += 1

    assert error_count == 1
    # check if order is kept (should do a separate test?)
    for ix, item in enumerate(fake_items_collection):
        assert str(ix + 1) == item.id


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_abatch(fake_items_collection, fake_langchain_executor):
    fake_exec = fake_langchain_executor
    output = await fake_exec.execute_abatch(fake_items_collection)
    assert isinstance(output, ItemCollection)
    for item in fake_items_collection:
        assert item.error is None
        assert item.output == "fake response"


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_abatch_with_error(
    fake_items_collection, mocker, fake_langchain_executor
):
    mock_instance = mocker.patch(
        "langchain_core.language_models.chat_models.BaseChatModel.ainvoke",
    )
    mock_instance.side_effect = [mocker.DEFAULT] * (len(fake_items_collection) - 1) + [
        Exception("test")
    ]
    mock_instance.return_value = "fake response"

    fake_exec = fake_langchain_executor
    output = await fake_exec.execute_abatch(fake_items_collection)
    assert isinstance(output, ItemCollection)
    # for item in fake_items_collection[:-1]:
    #     assert item.error is None
    #     assert item.output == "fake response"
    # assert fake_items_collection[-1].output is None
    # assert isinstance(fake_items_collection[-1].error, BaseException)

    mock_instance.call_count == len(fake_items_collection)
    error_count = 0
    for i in fake_items_collection:
        if i.output:
            # check if 2 output are "fake_reponse"
            assert i.output == "fake response"
        else:
            # check if 1 error
            assert isinstance(i.error, BaseException)
            error_count += 1

    assert error_count == 1
    # check if order is kept (should do a separate test?)
    for ix, item in enumerate(fake_items_collection):
        assert str(ix + 1) == item.id


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_batch_with_batch_size(
    fake_items_collection, mocker, fake_langchain_executor
):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")
    mock_instance.batch.return_value = [
        {"output": "fake_response", "raw": "fake_response", "error": None}
    ] * len(fake_items_collection)

    fake_exec.execute_batch(fake_items_collection, batch_size=32)
    mock_instance.batch.assert_called_once_with(
        [{"input": item.format_input()} for item in fake_items_collection],
        config={"max_concurrency": 32},
        return_exceptions=False,
    )


@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
def test_execute_batch_without_batch_size(
    fake_items_collection, mocker, fake_langchain_executor
):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")
    mock_instance.batch.return_value = [
        {"output": "fake_response", "error": None}
    ] * len(fake_items_collection)

    fake_exec.execute_batch(fake_items_collection)
    mock_instance.batch.assert_called_once_with(
        [{"input": item.format_input()} for item in fake_items_collection],
        config={},
        return_exceptions=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_abatch_with_batch_size(
    fake_items_collection, mocker, fake_langchain_executor
):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")
    mock_instance.abatch = AsyncMock(
        return_value=[{"output": None, "error": None}] * len(fake_items_collection)
    )

    await fake_exec.execute_abatch(fake_items_collection, batch_size=32)
    mock_instance.abatch.assert_called_once_with(
        [{"input": item.format_input()} for item in fake_items_collection],
        config={"max_concurrency": 32},
        return_exceptions=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_langchain_executor", [True, False], indirect=True)
async def test_execute_abatch_without_batch_size(
    fake_items_collection, mocker, fake_langchain_executor
):
    fake_exec = fake_langchain_executor
    mock_instance = mocker.patch.object(fake_exec, "_chain")
    mock_instance.abatch = AsyncMock(
        return_value=[{"output": None, "error": None}] * len(fake_items_collection)
    )

    await fake_exec.execute_abatch(fake_items_collection)
    mock_instance.abatch.assert_called_once_with(
        [{"input": item.format_input()} for item in fake_items_collection],
        config={},
        return_exceptions=False,
    )


#################################################################################
# Tests sur l'implémentation d'un executor PROMPT + LLM + PARSER avec langchain #
#################################################################################


def test_lcpromptllmparserexecutor_chain():
    fake_item = BaseItem[str, str](input="foo")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    chain_raw = prompt | llm
    chain_output = prompt | llm | parser

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_one(fake_item)

    assert isinstance(executor.chain, Runnable)
    assert isinstance(chain_raw.invoke(input="foo"), AIMessage)
    assert isinstance(fake_item.raw, AIMessage)
    assert chain_raw.invoke(input="foo").content == "fake response"
    assert fake_item.raw.content == "fake response"
    assert chain_output.invoke(input="foo") == fake_item.output == "fake response"


def test_lcpromptllmparserexecutor_retry_success(mocker):
    mock_instance = mocker.patch(
        "langchain_core.language_models.chat_models.BaseChatModel.invoke",
    )
    mock_instance.side_effect = [Exception("test")] * 2 + [mocker.DEFAULT]
    mock_instance.return_value = AIMessage("fake response")

    fake_item = BaseItem[str, str](input="foo")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    spy_llm_invoke = mocker.spy(FakeChatModel, "with_retry")

    executor = LCPromptLLMParserExecutor(
        prompt=prompt,
        llm=llm,
        parser=parser,
        wait_exponential_jitter=False,
        retry_if_exception_type=(Exception,),
        stop_after_attempt=3,
    )
    executor.execute_one(fake_item)

    assert fake_item.error is None
    assert fake_item.output == "fake response"
    assert isinstance(fake_item.raw, AIMessage)
    assert fake_item.raw.content == "fake response"
    assert mock_instance.call_count == 3
    spy_llm_invoke.assert_called_once_with(
        llm,
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=False,
        stop_after_attempt=3,
    )


def test_lcpromptllmparserexecutor_retry_failed(mocker):
    mock_instance = mocker.patch(
        "langchain_core.language_models.chat_models.BaseChatModel.invoke",
    )
    mock_instance.side_effect = [Exception("test")] * 2 + [mocker.DEFAULT]
    mock_instance.return_value = AIMessage("fake response")

    fake_item = BaseItem[str, str](input="foo")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    spy_llm_invoke = mocker.spy(FakeChatModel, "with_retry")

    executor = LCPromptLLMParserExecutor(
        prompt=prompt,
        llm=llm,
        parser=parser,
        wait_exponential_jitter=False,
        retry_if_exception_type=(Exception,),
        stop_after_attempt=2,
    )
    executor.execute_one(fake_item)

    assert fake_item.error is not None
    assert "test" in fake_item.error.__str__()
    assert fake_item.output is None
    assert fake_item.raw is None
    assert mock_instance.call_count == 2
    spy_llm_invoke.assert_called_once_with(
        llm,
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=False,
        stop_after_attempt=2,
    )


def test_lcpromptllmparserexecutor_parser_step_success(mocker):
    class FakeParser(BaseOutputParser):
        def parse(self, text: str) -> str:
            return text

    mocker_parser = mocker.spy(FakeParser, "invoke")

    fake_item = BaseItem[str, str](input="foo")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = FakeParser()

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_one(fake_item)

    mocker_parser.assert_called_once()
    assert fake_item.output == "fake response"
    assert isinstance(fake_item.raw, AIMessage)
    assert fake_item.raw.content == "fake response"


def test_lcpromptllmparserexecutor_parser_step_failed(mocker):
    mocker_parser = mocker.patch.object(BaseOutputParser, "invoke")
    mocker_parser.side_effect = LLMTParserException(msg="parser error", code="pytest")

    fake_item = BaseItem[str, str](input="foo")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_one(fake_item)

    mocker_parser.assert_called_once()
    assert "parser error" in fake_item.error.__str__()
    assert isinstance(fake_item.raw, AIMessage)
    assert fake_item.raw.content == "fake response"


def test_lcpromptllmparserexecutor_parser_batch_success(mocker, fake_items_collection):
    class FakeParser(BaseOutputParser):
        def parse(self, text: str) -> str:
            return text

    mocker_parser = mocker.spy(FakeParser, "invoke")

    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = FakeParser()

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_batch(fake_items_collection)

    mocker_parser.call_count == len(fake_items_collection)
    for i in fake_items_collection:
        assert i.output == "fake response"
        assert isinstance(i.raw, AIMessage)
        assert i.raw.content == "fake response"


def test_lcpromptllmparserexecutor_parser_batch_failed(mocker, fake_items_collection):
    mocker_parser = mocker.patch.object(BaseOutputParser, "invoke")
    mocker_parser.side_effect = LLMTParserException(msg="parser error", code="pytest")

    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_batch(fake_items_collection)

    mocker_parser.call_count == len(fake_items_collection)
    for i in fake_items_collection:
        assert isinstance(i.error, LLMTParserException)
        assert isinstance(i.raw, AIMessage)
        assert i.raw.content == "fake response"


def test_lcpromptllmparserexecutor_parser_batch_partially_failed(
    mocker, fake_items_collection
):
    # keep in mind that batch can call invoke on item not in the same order of items
    mocker_parser = mocker.patch.object(BaseOutputParser, "invoke")
    mocker_parser.side_effect = [mocker.DEFAULT] * (len(fake_items_collection) - 1) + [
        LLMTParserException(msg="parser error", code="pytest")
    ]
    mocker_parser.return_value = "fake response"
    prompt = ChatPromptTemplate.from_messages(
        [("system", "prompt system"), ("human", "|||{input}|||")]
    )
    llm = FakeChatModel()
    parser = StrOutputParser()

    executor = LCPromptLLMParserExecutor(prompt=prompt, llm=llm, parser=parser)
    executor.execute_batch(fake_items_collection)

    mocker_parser.call_count == len(fake_items_collection)
    print(fake_items_collection._items)
    error_count = 0
    for i in fake_items_collection:
        assert isinstance(i.raw, AIMessage)
        assert i.raw.content == "fake response"
        if i.output:
            # check if 2 output are "fake_reponse"
            assert i.output == "fake response"
        else:
            # check if 1 error
            assert isinstance(i.error, LLMTParserException)
            error_count += 1

    assert error_count == 1
    # check if order is kept (should do a separate test?)
    for ix, item in enumerate(fake_items_collection):
        assert str(ix + 1) == item.id
