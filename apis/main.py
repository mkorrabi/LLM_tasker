from typing import Annotated, List, Union
from dotenv import load_dotenv, find_dotenv
import os

from fastapi import Body, FastAPI, HTTPException, Response, status
from fastapi.encoders import jsonable_encoder

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langfuse.callback import CallbackHandler

from llmtasker.exceptions import LLMTaskerException
from llmtasker.items.base import T
from llmtasker.tasks.classification import (
    Classification,
    GroupedClassification,
    ClassificationAPIConfig,
    ClassItem,
    MultiClassItem,
)
from llmtasker.tasks.custom_instruction import (
    CustomInstructionAPIConfig,
    CustomPydanticItem,
    CustomInstruction,
)
from llmtasker.utils import jsonschema_to_pydantic

from apis.examples_api import examples_fastapi_classification, examples_fastapi_custom

load_dotenv(find_dotenv(), override=True)

app = FastAPI()

# TODO: ajouter une route pour montrer le prompt entier


def choose_llm(config_llm: dict):
    # check which llm. Default to Azure
    if config_llm.get("openai_api_base"):
        return openai_llm_default(config_llm)
    return azure_llm_default(config_llm)


def azure_llm_default(config_llm: dict):
    # paramètres par défaut:
    if not config_llm.get("azure_endpoint"):
        config_llm["azure_endpoint"] = os.getenv("AZURE_APIM_OPENAI_ENDPOINT")

    if not config_llm.get("api_version"):
        config_llm["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")

    if not config_llm.get("api_key"):
        config_llm["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

    if not config_llm.get("default_headers", {}).get("Ocp-Apim-Subscription-Key"):
        config_llm["default_headers"] = config_llm.get("default_headers", {})
        config_llm["default_headers"]["Ocp-Apim-Subscription-Key"] = os.getenv(
            "AZURE_APIM_KEY"
        )
    if not config_llm.get("default_query", {}).get("project-name"):
        raise HTTPException(
            status_code=422, detail="project name must be filled in llm config"
        )

    return AzureChatOpenAI(**config_llm)


def openai_llm_default(config_llm: dict):
    # paramètres par défaut:
    if not config_llm.get("openai_api_base"):
        config_llm["openai_api_base"] = "DEFAULT"

    if not config_llm.get("api_key"):
        config_llm["api_key"] = "NOT_NEEDED"

    if not config_llm.get("default_query", {}).get("project-name"):
        raise HTTPException(
            status_code=422, detail="project name must be filled in llm config"
        )

    return ChatOpenAI(**config_llm)


def langfuse_handler_task(task_route: str):
    return CallbackHandler(
        secret_key=os.getenv("LANGFUSE_PRIVATE_KEY", default=None),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", default=None),
        host=os.getenv("LANGFUSE_HOST", default=None),
        trace_name=task_route,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/classify", responses={"206": {"description": "One error item is not None."}}
)
async def classify(
    params: Annotated[
        ClassificationAPIConfig,
        Body(openapi_examples=examples_fastapi_classification),
    ],
    response: Response,
) -> List[Union[MultiClassItem, ClassItem]]:
    # callback langfuse
    langfuse_handler = langfuse_handler_task("/classify")
    langchain_config = {"callbacks": [langfuse_handler]}

    # LLM choice
    llm = choose_llm(params.config_llm)

    langfuse_handler.metadata = {
        "project-name": params.config_llm["default_query"]["project-name"]
    }

    if params.config_package.grouped_classification:
        langfuse_handler.tags = [
            "grouped",
            "open" if params.config_package.classification_open else "closed",
        ]
        try:
            task = GroupedClassification.from_config(llm, params.config_package)
            task.langchain_config = langchain_config
            outputs = await task.arun(
                params.inputs,
                params.config_package.batch_size,
                params.config_package.group_size,
            )
        except LLMTaskerException as e:
            raise HTTPException(status_code=422, detail=dict(e))
    else:
        langfuse_handler.tags = [
            "open" if params.config_package.classification_open else "closed",
        ]
        try:
            task = Classification.from_config(llm, params.config_package)
            task.langchain_config = langchain_config
            outputs = await task.arun(params.inputs, params.config_package.batch_size)
        except LLMTaskerException as e:
            raise HTTPException(status_code=422, detail=dict(e))

    error_occurred = any(
        getattr(output, "error", None) is not None for output in outputs._items
    )

    if error_occurred:
        response.status_code = status.HTTP_206_PARTIAL_CONTENT

    return jsonable_encoder(outputs._items)


@app.post("/custom", responses={"206": {"description": "One error item is not None."}})
async def custom(
    params: Annotated[
        CustomInstructionAPIConfig,
        Body(openapi_examples=examples_fastapi_custom),
    ],
    response: Response,
) -> List[T]:
    langfuse_handler = langfuse_handler_task("/custom")
    langchain_config = {"callbacks": [langfuse_handler]}

    # LLM choice
    llm = choose_llm(params.config_llm)

    langfuse_handler.metadata = {
        "project-name": params.config_llm["default_query"]["project-name"]
    }
    langfuse_handler.tags = [params.config_package.method]

    task_class = CustomInstruction

    if params.config_package.json_shema:
        langfuse_handler.tags.append("json_shema")
        new_model = jsonschema_to_pydantic(params.config_package.json_shema)

        class CustomItem(CustomPydanticItem[str, new_model]):
            pass

        # create dynamic class to this particular instruction
        # TODO: store in cache existing class to avoid re-recreating class
        # Cache for storing dynamically created classes
        #
        # define outside this dictionnary:
        # dynamic_class_cache: Dict[str, Type[CustomInstruction]] = {}
        #
        # def create_cache_key(new_model: Type[BaseModel]) -> str:
        #     """Creates a unique cache key based on the new_model's schema."""
        #     schema_str = json.dumps(new_model.schema(), sort_keys=True)
        #     return hashlib.md5(schema_str.encode('utf-8')).hexdigest()

        class DynamicCustomInstruction(CustomInstruction):
            _DEFAULT_BASEITEM_MODEL = CustomItem
            _DEFAULT_OUTPUT_MODEL = new_model

        task_class = DynamicCustomInstruction

    try:
        task = task_class.from_config(llm, params.config_package)
        task.langchain_config = langchain_config
        outputs = await task.arun(params.inputs, params.config_package.batch_size)
    except LLMTaskerException as e:
        raise HTTPException(status_code=422, detail=dict(e))

    error_occurred = any(
        getattr(output, "error", None) is not None for output in outputs._items
    )

    if error_occurred:
        response.status_code = status.HTTP_206_PARTIAL_CONTENT

    return jsonable_encoder(outputs._items)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="error")
