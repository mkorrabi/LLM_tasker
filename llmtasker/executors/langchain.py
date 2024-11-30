"""Module d'implémentation d'un exécuteur avec langchain.

On définit une classe flexible pour gérer une chaine:

* LangchainExecutor

On définit une classe de construction d'une chaine à partir
d'un prompt, llm et parser:

* LCPromptLLMParserExecutor

"""

from typing import Any, Optional, Type, List, Union, Tuple, Literal
from operator import itemgetter
from typing_extensions import deprecated
from pydantic import BaseModel, ValidationError
from langchain_core.messages import BaseMessage
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.runnables.base import (
    RunnableSequence,
    RunnableConfig,
    Runnable,
    RunnableMap,
)
from langchain_core.runnables.passthrough import RunnablePassthrough


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.exceptions import OutputParserException

from llmtasker.executors.base import IExecutor, T
from llmtasker.items.base import ItemCollection
from llmtasker.exceptions import LLMTParserException
from llmtasker.parsers import LCPydanticParserUtils, JsonOutputParser


class LangchainExecutor(IExecutor):
    """Implémentation de l'interface avec langchain.

    La chaine est construite indépendamment.
    """

    def __init__(
        self,
        chain: RunnableSequence,
        config: Union[RunnableConfig, List[RunnableConfig], None] = None,
        has_raw_output: bool = False,
    ):
        """Constructeur de la classe à partir d'une chaine langchain et de la config langchain.

        Possibilité d'ajouter des callbacks grâce à la config.

        Chaque méthode stocke les résultats du traitement d'un `BaseItem` dans `BaseItem.output`.

        Une clé 'raw' est disponible pour stocker les résultats bruts du LLM dans `BaseItem.raw`.

        Si une erreur survient pour un item, l'erreur est stocké dans `BaseItem.error`.

        Args:
            chain (RunnableSequence): Chaine langchain
            config (Union[RunnableConfig, List[RunnableConfig], None], optional): Config langchain. Defaults to None.
            has_raw_output (bool): determiner si la chaine contient déjà les clés raw et output. Defaults to None.
        """
        self.has_raw_output = has_raw_output
        self.chain = chain
        self.config = config

    @property
    def chain(self) -> RunnableSequence:
        return self._chain

    @chain.setter
    def chain(self, chain: RunnableSequence) -> RunnableSequence:
        # def build_chain(self, chain: RunnableSequence):
        # on peut probablement automatiser si raw et output existe dans la chaine en sortie
        # pour le moment on délègue cette information en paramètre:
        if not self.has_raw_output:
            chain = RunnablePassthrough.assign(output=chain, raw=lambda _: None)

        # ajout des erreurs:
        # si erreur, l'output devient None
        output_none = RunnablePassthrough.assign(output=lambda _: None)
        chain = chain.with_fallbacks([output_none], exception_key="error")

        self._chain = chain
        return self._chain

    def assign_result_to_item(self, result: dict, item: T) -> T:
        """Assigne par référence les clés output, error et raw à l'item.

        Args:
            result (dict): dictionnaire avec les clés
            item (T): item

        Returns:
            T: item avec les nouveaux attributs
        """
        # handle pydantic validation error if parser succeed
        # (very rare)
   
        try:
            
            item.output = result.get("output", None)
            item.error = result.get("error", None)
        except ValidationError as e:
            item.output = None
            item.error = e
        item.raw = result.get("raw", None)

    def execute_one(self, item: T) -> T:
        """Implémentation de la méthode pour traiter un `BaseItem`.

        Args:
            item (T): objet héritant de `BaseItem`

        Returns:
            T: objet item avec l'attribut `BaseItem.output` rempli.
        """
        result = self.chain.invoke(
            input={
                "input": item.format_input(),
            },
            config=self.config,
        )
        self.assign_result_to_item(result, item)

        return item

    async def execute_aone(self, item: T) -> T:
        """Implémentation de la méthode asynchrone pour traiter un `BaseItem`.

        Args:
            item (T): objet héritant de `BaseItem`

        Returns:
            T: objet item avec l'attribut `BaseItem.output` rempli.
        """
        result = await self.chain.ainvoke(
            input={
                "input": item.format_input(),
            },
            config=self.config,
        )
        self.assign_result_to_item(result, item)
        return item

    def execute_batch(
        self, items: ItemCollection, batch_size: Optional[int] = None
    ) -> ItemCollection:
        """Implémentation de la méthode pour traiter une collection d'items.

        Dans le cadre de langchain, on utilise cette notion pour définir le nombre \
        de worker maximum (`max_concurrency`) qui font appel à la chaine en parallèle.

        Des threads sont utilisés avec langchain.

        Args:
            items (ItemCollection): Collection d'objet héritant de `BaseItem`.
            batch_size (Optional[int], optional): Taille du batch. Defaults to None.

        Returns:
            ItemCollection: Collection d'objet héritant de `BaseItem` \
                avec l'attribut `BaseItem.output`.
        """
        config = self.config.copy() if self.config else {}
        if batch_size:
            config["max_concurrency"] = batch_size
        items_keys = items.keys()
        results = self.chain.batch(
            [{"input": item.format_input()} for item in items],
            config=config,
            return_exceptions=False,
        )

        for item_id, result in zip(items_keys, results):
            self.assign_result_to_item(result, items[item_id])

        return items

    async def execute_abatch(
        self, items: ItemCollection, batch_size: Optional[int] = None
    ) -> ItemCollection:
        """Implémentation de la méthode asynchrone pour traiter une collection d'items.

        Dans le cadre de langchain, on utilise cette notion pour définir le nombre \
        de task asynchrone maximum (`max_concurrency`) \
        qui font appel à la chaine en parallèle.

        `asyncio.Semaphore` est utilisé côté langchain pour contrôler cette concurrence.

        Args:
            items (ItemCollection): Collection d'objet héritant de `BaseItem`.
            batch_size (Optional[int], optional): Taille du batch. Defaults to None.

        Returns:
            ItemCollection: Collection d'objet héritant de `BaseItem` \
                avec l'attribut `BaseItem.output`.
        """
        config = self.config.copy() if self.config else {}
        if batch_size:
            config["max_concurrency"] = batch_size
        items_keys = items.keys()
        results = await self.chain.abatch(
            [{"input": item.format_input()} for item in items],
            config=config,
            return_exceptions=False,
        )

        for item_id, result in zip(items_keys, results):
            self.assign_result_to_item(result, items[item_id])

        return items


class LCPromptLLMParserExecutor(LangchainExecutor):
    """Classe représentant un executeur langchain à partir de trois étapes.

    Etapes:

    * Prompt
    * LLM
    * Parser

    On considère que c'est le cas le plus courant pour exécuter une tâche.

    La classe ajoute les fonctionnalités de retry et la gestion des erreurs de parsing.

    On utilise une logique de modèle instruct (`BaseChatModel`).
    """

    def __init__(
        self,
        prompt: ChatPromptTemplate,
        llm: BaseChatModel,
        parser: Union[BaseOutputParser, RunnableSequence, Runnable],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        retry_if_exception_type: Tuple[Type[Exception]] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
    ):
        """Construction d'une chaine contrôlée Prompt | LLM | Parser.

        Notes:
        * Si un item échoue en batch, seul l'item est retry.
        * En batch, le nombre d'essais est contrôlé par batch. \
            Cela signifie qu'au deuxième essai, on lance un batch des items \
            ayant échoués au premier essai.

        Args:
            prompt (ChatPromptTemplate): Prompt langchain. Peut être généré par `LCPrompt`.
            llm (BaseChatModel): Modèle langchain à fournir par l'utilisateur.
            parser (Union[BaseOutputParser, RunnableSequence, Runnable]): Parser au format langchain. \
                Exemples: `LCClasseOutputParser`, `LCPydanticOutputParser`, `StrOutputParser`
            config (Union[RunnableConfig, List[RunnableConfig], None], optional): Config langchain. Defaults to None.
            retry_if_exception_type (Tuple[Type[Exception]], optional): Indiquer les exceptions qui relance l'item. \
                Defaults to (Exception,).
            wait_exponential_jitter (bool, optional): Ajouter un temps aléatoire entre chaque essais. Defaults to True.
            stop_after_attempt (int, optional): Nombre d'essais maximum avant d'abandonner l'item. Defaults to 3.
        """
        self.prompt = prompt
        self.llm = llm
        self.parser = parser

        self.config = config

        self.retry_if_exception_type = retry_if_exception_type
        self.wait_exponential_jitter = wait_exponential_jitter
        self.stop_after_attempt = stop_after_attempt

        chain = self.prompt | self.build_llm_step() | self.build_parser_step()
        super().__init__(chain, config, has_raw_output=True)

    def build_llm_step(self, include_raw: bool = True) -> Runnable:
        """Construction de l'étape LLM avec le retry.

        Args:
            include_raw (bool): ajoute les resultats bruts du llm dans une clé 'raw'.

        Returns:
            Runnable: objet Runnable langchain
        """
        step = self.llm.with_retry(
            retry_if_exception_type=self.retry_if_exception_type,
            wait_exponential_jitter=self.wait_exponential_jitter,
            stop_after_attempt=self.stop_after_attempt,
        )
        if include_raw:
            return RunnableMap(raw=step)
        else:
            return step

    def build_parser_step(self) -> Runnable:
        """Construction de l'étape du parser.

        Le parser écrit dans la clé output.
        Un fallback est attaché pour attraper les erreurs et les injecter dans error.
        Cette étape préserve la sortie du llm dans raw.

        Returns:
            Runnable: parser runnable
        """
        fallback_chain = RunnablePassthrough.assign(output=lambda _: None)
        parser_chain = RunnablePassthrough.assign(
            output=itemgetter("raw") | self.parser
        ).with_fallbacks(
            [fallback_chain],
            exception_key="error",
            exceptions_to_handle=(OutputParserException, LLMTParserException),
        )
        return parser_chain

    @deprecated("use build_parser_step()")
    def runnable_parser_step(
        self, input: Union[str, BaseMessage], config: RunnableConfig
    ) -> Any:
        """DEPRECIE CAR NE GERE PAR LES CLES.

        Construction de l'étape parsing avec gestion des erreurs.

        On wrappe l'étape de parsing pour attraper les erreurs et \
        les gérer si besoin. Ici, on ne fait rien de spécial.

        Notes: on implémente avec `.invoke()` mais cela n'empêche pas le \
            comportement attendu en mode asynchrone ou batch. Cela est dûr à \
            l'existance des méthodes `Runnable.batch()` / `Runnable.abatch()`.

        Args:
            input (Union[str, BaseMessage]): Objet représentant un message avec langchain
            config (RunnableConfig): Config langchain.

        Returns:
            Any: réponse du parser ou Exception durant le parsing
        """
        try:
            return self.parser.invoke(input, config=config)
        except (OutputParserException, LLMTParserException) as e:
            # ici on peut insérer une logique de gestion d'erreur du parsing
            # Par exemple attraper les erreurs langchain natives pour renvoyer
            # un objet interne
            raise e


class LCPromptLLMPydanticExecutor(LCPromptLLMParserExecutor):
    # TODO: documenter
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        llm: BaseChatModel,
        schema: Optional[BaseModel] = None,
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        retry_if_exception_type: Tuple[type[Exception]] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
        method: Literal["function_calling", "json_mode", "none"] = "function_calling",
    ):
        self.prompt = prompt
        self.llm = llm
        self.schema = schema

        self.config = config

        self.retry_if_exception_type = retry_if_exception_type
        self.wait_exponential_jitter = wait_exponential_jitter
        self.stop_after_attempt = stop_after_attempt

        self.parser_utils = LCPydanticParserUtils(pydantic_object=self.schema)

        if method in ["json_mode", "none"]:
            if method == "json_mode":
                self.llm = self.llm.bind(response_format={"type": "json_object"})
            if self.schema:
                # on n'utilise une version interne du parsing (+ complète que natif à langchain?)
                parser = self.parser_utils.parser()
            else:
                # TODO: utiliser un parser plus complet ici (similaire LCPydanticParserUtils)
                parser = JsonOutputParser()
            LCPromptLLMParserExecutor.__init__(
                self,
                prompt=self.prompt,
                llm=self.llm,
                parser=parser,
                config=self.config,
                retry_if_exception_type=self.retry_if_exception_type,
                wait_exponential_jitter=self.wait_exponential_jitter,
                stop_after_attempt=self.stop_after_attempt,
            )

        else:
            # on est obligé d'attacher with_structured_output() au llm
            # donc l'ajout du retry n'est pas ciblé à l'appel du LLM mais aussi sur le parser
            # on peut jouer sur les types d'exception pour cibler les erreurs d'appel
            if isinstance(self.schema, dict):
                schema_dict = self.schema
            elif issubclass(self.schema, BaseModel):
                # fix pydantic v2 by using dict schema instead:
                schema_dict = self.schema.model_json_schema()
            else:
                raise TypeError(
                    "schema must be instance of dict or BaseModel pydantic v2"
                )
            self.llm = self.llm.with_structured_output(
                schema=schema_dict, method=method, include_raw=True
            ) | RunnablePassthrough.assign(
                error=itemgetter("parsing_error"), output=itemgetter("parsed")
            )
            chain = self.prompt | self.build_llm_step(include_raw=False)
            LangchainExecutor.__init__(self, chain, config, has_raw_output=True)
