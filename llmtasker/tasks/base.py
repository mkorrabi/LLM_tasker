from typing import List, Union, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_openai.chat_models import AzureChatOpenAI
from llmtasker.executors.base import T
from llmtasker.items.base import ItemCollection, BaseItem
from llmtasker.prompts.base import AbstractBasePrompt
from llmtasker.prompts.lc_base import LCPrompt
from llmtasker.executors.langchain import (
    LangchainExecutor,
    LCPromptLLMParserExecutor,
)
from llmtasker.parsers import JsonOutputParser
from llmtasker.exceptions import LLMTaskerException


class LCAbstractPackageConfig(BaseModel):
    instructions: str = Field(
        default="Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
        description="instructions au format f-string.",
    )
    examples: List[T] = Field(
        description="Exemples d'entrée au LLM avec resultats attendus.", default=[]
    )
    variables: Dict = Field(
        description="Dictionnaire de variables à remplacer dans l'insutrction",
        default={},
    )
    use_system_prompt: bool = Field(
        default=True,
        description="Les instructions sont envoyées avec le rôle system au LLM",
    )
    input_in_instructions: bool = Field(
        default=False,
        description="Si false, l'input est envoyé avec le rôle user au LLM. Si true, il faut ajouter la variable {input} dans l'instruction.",
    )
    use_ai_user_role_for_examples: bool = Field(
        default=False,
        description="Si true, les exemples sont envoyés avec les rôles user/AI au LLM (il faut supprimer la variable {example} de l'instruction). Sinon il sont ajoutés à l'instruction.",
    )
    batch_size: int = Field(
        default=3, description="Nombre d'appel au LLM en parallèle."
    )


class LCAbstractAPIConfig(BaseModel):
    # config_llm: AzureChatOpenAI
    config_llm: Dict[str, Any] = Field(
        description="Paramètres du modèle utilisant la spécification openai. Voir exemple."
    )
    config_package: LCAbstractPackageConfig = Field(
        description="Configuration du package."
    )
    inputs: List[T] = Field(description="Inputs à traiter.")


class LCAbstractTask:
    _DEFAULT_CHAT_MODEL = AzureChatOpenAI
    _DEFAULT_PROMPT_MODEL = LCPrompt
    _DEFAULT_EXECUTOR_MODEL = LCPromptLLMParserExecutor
    _DEFAULT_BASEITEM_MODEL = BaseItem
    _DEFAULT_OUTPUT_MODEL = BaseModel

    @classmethod
    def from_config(
        cls, config_llm: Dict[str, Any], config_package: LCAbstractPackageConfig
    ) -> "LCAbstractTask":
        return cls(llm=config_llm, **config_package.model_dump())

    def __init__(
        self,
        llm: Union[Dict, BaseChatModel],
        instructions: str,
        parser: Runnable = None,
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
        """Classe abstraite représentant une tâche. Une tâche doit hériter de cette classe.

        Args:
            llm (Union[Dict, BaseChatModel]):  Modèle langchain à fournir par l'utilisateur.
            instructions (str): prompt textuel possiblement avec des variables au format f-string ou jinja2.
            parser (Runnable, optional): Parser au format langchain. llm-taker propore des parsers. Defaults to None.
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
            force_rebuild (bool, optional): Force la re-création des attributs prompt, parser et executor à leur appel. \
                Permet de prendre en compte un changement de paramètre, mais n'est pas recommandé. Defaults to True.
        """
        # config external objects
        self.llm = llm
        self.parser = parser

        # config prompt
        self.instructions = instructions
        self.variables = variables
        self.examples = self.build_base_items(examples)
        self.use_system_prompt = use_system_prompt
        self.input_in_instructions = input_in_instructions
        self.use_ai_user_role_for_examples = use_ai_user_role_for_examples
        self.template_format = template_format

        # config executor
        self.langchain_config = langchain_config
        self.retry_if_exception_type = retry_if_exception_type
        self.wait_exponential_jitter = wait_exponential_jitter
        self.stop_after_attempt = stop_after_attempt

        # config interne
        self.force_rebuild = force_rebuild

    def get_executor_params(self):
        return {
            "llm": self.llm,
            "prompt": self.prompt.generate(template_format=self.template_format),
            "parser": self.parser,
            "config": self.langchain_config,
            "retry_if_exception_type": self.retry_if_exception_type,
            "wait_exponential_jitter": self.wait_exponential_jitter,
            "stop_after_attempt": self.stop_after_attempt,
        }

    def get_prompt_params(self) -> Dict:
        return {
            "instructions": self.instructions,
            "variables": self.variables,
            "examples": self.examples,
            "use_system_prompt": self.use_system_prompt,
            "input_in_instructions": self.input_in_instructions,
            "use_ai_user_role_for_examples": self.use_ai_user_role_for_examples,
        }

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, llm: Union[Dict, BaseChatModel]):
        if isinstance(llm, dict):
            llm = self._DEFAULT_CHAT_MODEL(**llm)
        self._llm = llm

    @property
    def prompt(self):
        if self.force_rebuild:
            self._prompt = None
        if not self._prompt:
            self.prompt = self.get_prompt_params()
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: Union[Dict, AbstractBasePrompt]):
        if isinstance(prompt, dict):
            prompt = self._DEFAULT_PROMPT_MODEL.model_validate(prompt)
        self._prompt = prompt

    @property
    def parser(self):
        if self.force_rebuild:
            self._parser = None
        if not self._parser:
            self.parser = self.init_parser()
        return self._parser

    @parser.setter
    def parser(self, parser: Union[Dict, Runnable] = None):
        self._parser = parser

    @property
    def executor(self):
        if self.force_rebuild:
            self._executor = None
        if not self._executor:
            self.executor = self.get_executor_params()
        return self._executor

    @executor.setter
    def executor(self, executor: Union[Dict, LangchainExecutor]):
        if isinstance(executor, dict):
            executor = self._DEFAULT_EXECUTOR_MODEL(**self.get_executor_params())
        self._executor = executor

    def build_base_items(self, items: Any) -> ItemCollection:
        # une liste d'inputs: on utilise cette fonction en récurrence pour checker les types
        # et construire l'item de base
        if isinstance(items, list):
            return ItemCollection([self.build_base_items(item)[0] for item in items])

        # dictionnaire pour construire un base item grâce à pydantic
        if isinstance(items, dict):
            return ItemCollection(
                [self._default_baseitem_model().model_validate(items)]
            )

        # un objet base item seul
        if isinstance(items, self._default_baseitem_model()) or isinstance(
            items, BaseItem
        ):
            return ItemCollection([items])

        # l'attribut input de base item est de type str
        if isinstance(items, str):
            # cas spécial si l'input est un str: on construit l'item directement
            return ItemCollection([self._default_baseitem_model()(input=items)])

        # l'attribut input de base item est de type pydantic
        if isinstance(items, BaseModel):
            # cas sépcial si l'input est un objet pydantic: on construit l'item directement
            # /!\ un peu trop caché ?
            return ItemCollection([self._default_baseitem_model()(input=items)])

        # c'est déjà une collection typée correctement
        if isinstance(items, ItemCollection):
            return items

        raise LLMTaskerException(
            msg="Impossible to transform inputs or examples with correct class type. Check your types.",
            code="build_base_items",
        )

    def _default_chat_model(self):
        return self._DEFAULT_CHAT_MODEL

    def _default_prompt_model(self):
        return self._DEFAULT_PROMPT_MODEL

    def _default_executor_model(self):
        return self._DEFAULT_EXECUTOR_MODEL

    def _default_baseitem_model(self):
        return self._DEFAULT_BASEITEM_MODEL

    def _default_output_model(self):
        return self._DEFAULT_OUTPUT_MODEL

    def init_parser(self):
        return JsonOutputParser()

    def run(
        self,
        inputs: Union[List[BaseItem], ItemCollection, BaseItem, Dict, List[Dict]],
        batch_size: int = 1,
    ) -> ItemCollection:
        raise NotImplementedError("AbstractTask.run is not implemented.")

    async def arun(
        self,
        inputs: Union[List[BaseItem], ItemCollection, BaseItem, Dict, List[Dict]],
        batch_size: int = 1,
    ) -> ItemCollection:
        raise NotImplementedError("AbstractTask.arun is not implemented.")
