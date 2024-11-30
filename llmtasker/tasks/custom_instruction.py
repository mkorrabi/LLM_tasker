"""Module définissant la tâche d'instruction custom basé sur prompt>llm>parser.

* CustomInstruction: instruction personnalisée

"""

from typing import Any, List, Optional, Dict, Tuple, Union, Literal
import json
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from llmtasker.items.base import BaseItem, ItemCollection
from llmtasker.tasks.base import (
    LCAbstractTask,
    LCAbstractPackageConfig,
    LCAbstractAPIConfig,
)
from llmtasker.parsers import LCPydanticOutputParser
from llmtasker.executors.langchain import LCPromptLLMPydanticExecutor


class CustomDictItem(BaseItem[str, Dict]):
    def format_output(self):
        return json.dumps(self.output, ensure_ascii=False)


class CustomPydanticItem(BaseItem):
    def format_output(self):
        return str(self.output.model_dump_json())


class CustomInstructionConfig(LCAbstractPackageConfig):
    method: Literal["function_calling", "json_mode", "none"] = Field(
        default="json_mode",
        description="Methode de parsing. json_mode est compatible avec tous les LLMs ayant un attribut response_format=json_object. Sinon utiliser none.",
    )
    json_shema: dict = Field(
        default=None,
        description="Shema json représentant un objet pydantic par exemple. Si non renseigné, le dictionnaire renvoyé ne sera pas validé.",
    )


class CustomInstructionAPIConfig(LCAbstractAPIConfig):
    config_package: CustomInstructionConfig = Field(
        description="Configuration du package pour un appel sur mesure."
    )


class CustomInstruction(LCAbstractTask):
    _DEFAULT_EXECUTOR_MODEL = LCPromptLLMPydanticExecutor
    _DEFAULT_BASEITEM_MODEL = CustomDictItem
    _DEFAULT_OUTPUT_MODEL = None

    def __init__(
        self,
        llm: Union[Dict, BaseChatModel],
        instructions: str,
        method: Literal["function_calling", "json_mode", "none"] = "json_mode",
        examples: Union[List[BaseItem], ItemCollection, List[Dict]] = [],
        variables: Dict = {},
        use_system_prompt: bool = True,
        input_in_instructions: bool = False,
        use_ai_user_role_for_examples: bool = False,
        template_format: str = "f-string",
        langchain_config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        retry_if_exception_type: Tuple[type[Exception]] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
        force_rebuild: bool = True,
        *args,
        **kwargs,
    ):
        """Tâche personnalisable.

        Args:
            llm (Union[Dict, BaseChatModel]):  Modèle langchain à fournir par l'utilisateur.
            instructions (str): prompt textuel possiblement avec des variables au format f-string ou jinja2.
            method (Literal[&quot;function_calling&quot;, &quot;json_mode&quot;, &quot;none&quot;], optional): Methode de parsing. json_mode est compatible avec tous les LLMs ayant un attribut response_format=json_object. Sinon utiliser none. Defaults to "json_mode".
            examples (Union[List[BaseItem], ItemCollection, List[Dict]], optional): examples de résultat de la tâche avec le format du package (BaseItem). Defaults to [].
            variables (Dict, optional): dictionnaire de variables à intégrer dans le prompt final. Defaults to {}.
            use_system_prompt (bool, optional): Les instructions sont envoyées avec le rôle system au LLM. Defaults to True.
            input_in_instructions (bool, optional): Si false, l'input est envoyé avec le rôle user au LLM. Si true, il faut ajouter la variable {input} dans l'instruction. Defaults to False.
            use_ai_user_role_for_examples (bool, optional): Si true, les exemples sont envoyés avec les rôles user/AI au LLM (il faut supprimer la variable {example} de l'instruction). Sinon il sont ajoutés à l'instruction. Defaults to False.
            template_format (str, optional): Format d'interpolation des variables (f-string ou jinja2). Defaults to "f-string".
            langchain_config (Optional[Union[RunnableConfig, List[RunnableConfig]]], optional): object config lanchaine (pour les callbacks par ex.). Defaults to None.
            retry_if_exception_type (Tuple[type[Exception]], optional): Indiquer les exceptions relance le LLM (ou l'ensemble de la chaine pour le function calling). Defaults to (Exception,).
            wait_exponential_jitter (bool, optional): Ajouter un temps aléatoire entre chaque essais. Defaults to True.
            stop_after_attempt (int, optional):  Nombre d'essais maximum avant d'abandonner l'item. Defaults to 3.
            force_rebuild (bool, optional): Force la re-création des attributs prompt, parser et executor à leur appel.
        """
        # action before init:
        self.shema = self._DEFAULT_OUTPUT_MODEL
        self.method = method

        # TODO: ne pas re déclarer tous les params
        super().__init__(
            llm,
            instructions,
            None,
            examples,
            variables,
            use_system_prompt,
            input_in_instructions,
            use_ai_user_role_for_examples,
            template_format,
            langchain_config,
            retry_if_exception_type,
            wait_exponential_jitter,
            stop_after_attempt,
            force_rebuild,
        )
        # actions after init:
        if (
            not self.variables.get("json_shema_output")
            and self._DEFAULT_OUTPUT_MODEL
            and self.method != "function_calling"
        ):
            self.variables["json_shema_output"] = self.parser.get_format_instructions()

    def get_executor_params(self):
        return {
            "prompt": self.prompt.generate(template_format=self.template_format),
            "llm": self.llm,
            "schema": self.shema,
            "config": self.langchain_config,
            "retry_if_exception_type": self.retry_if_exception_type,
            "wait_exponential_jitter": self.wait_exponential_jitter,
            "stop_after_attempt": self.stop_after_attempt,
            "method": self.method,
        }

    def init_parser(self):
        return LCPydanticOutputParser(pydantic_object=self.shema)

    def run(
        self,
        inputs: Any,
        batch_size: int = 1,
    ) -> ItemCollection:
        inputs = self.build_base_items(inputs)
        if batch_size > 1:
            return self.executor.execute_batch(inputs, batch_size=batch_size)
        else:
            for input in inputs:
                self.executor.execute_one(input)
        return inputs

    async def arun(
        self,
        inputs: Any,
        batch_size: int = 1,
    ) -> ItemCollection:
        inputs = self.build_base_items(inputs)
        if batch_size > 1:
            return await self.executor.execute_abatch(inputs, batch_size=batch_size)
        else:
            # no async loop because batch_size == 1
            for input in inputs:
                await self.executor.execute_aone(input)
        return inputs
