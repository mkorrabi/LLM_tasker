"""Module définissant les différentes tâches de classification.

A date:

Classification:

* ouverte/fermée: `Classification(classification_open=...)`
* mono/multi labels: `Classification(multi_labels=...)`

GroupedClassification:
* groupée: `GroupedClassification(...)`

Il est possible de combinés les options. En revanche, bien faire attention au prompt (cf examples/).

"""

from typing import Dict, List, Union
from typing_extensions import deprecated
from pydantic import Field, RootModel
import json
from llmtasker.items.base import BaseItem, Classe, ItemCollection
from llmtasker.exceptions import LLMTaskerException
from llmtasker.parsers import LCClasseParserUtils
from llmtasker.tasks.base import (
    LCAbstractTask,
    LCAbstractPackageConfig,
    LCAbstractAPIConfig,
)


class ClassItem(BaseItem[str, Classe]):
    pass


class ListClasse(RootModel):
    root: List[Classe]


class MultiClassItem(BaseItem[str, ListClasse]):
    def format_output(self):
        # if full json object
        # return self.output.model_dump_json()
        # for list of str:
        return json.dumps(
            [str(o) if o else None for o in self.output.root], ensure_ascii=False
        )


class ClassificationConfig(LCAbstractPackageConfig):
    labels: List[Classe] = Field(
        description="Liste de labels ou classes pour la classification."
    )
    examples: Dict[str, Union[str, List[str]]] = Field(
        description="Exemples d'entrée au LLM avec resultats attendus.", default={}
    )
    classification_open: bool = Field(
        default=False,
        description="False: Les classes sont fixes, true sinon le LLM peut créer des classes.",
    )
    multi_labels: bool = Field(
        default=False,
        description="False: un label par observation, true plusieurs labels pour une observation.",
    )
    grouped_classification: bool = Field(
        default=False,
        description="Classification groupée. Les inputs sont groupés par paquet dans un message.",
    )
    group_size: int = Field(
        default=5,
        description="Classification groupée. Nombre d'inputs groupés dans un message.",
    )
    grouped_examples: bool = Field(
        default=True,
        description="Classification groupée. Les exemples sont considérés comme groupée plutôt que unitaire (true par défaut).",
    )
    group_size_examples: int = Field(
        default=5,
        description="Nombre d'item groupés en un message pour les examples. Defaults to 5.",
    )


class ClassificationAPIConfig(LCAbstractAPIConfig):
    config_package: ClassificationConfig = Field(
        description="Configuration du package pour la classification."
    )
    inputs: List[Dict] = Field(description="Inputs à classifer.")


class Classification(LCAbstractTask):
    _DEFAULT_BASEITEM_MODEL = ClassItem
    _DEFAULT_OUTPUT_MODEL = Classe

    def __init__(
        self,
        examples: Dict[str, Union[str, List[str]]] = {},
        labels: Union[Dict, List[Dict], Classe, List[Classe]] = [],
        classification_open: bool = False,
        multi_labels: bool = False,
        *args,
        **kwargs,
    ):
        """Classe représentant une classification.

        *args et **kwargs sont les arguments passés à la classe mère: `LCAbstractTask`

        Args:
            examples (Dict[str, Union[str, List[str]]], optional): Exemples d'entrée au LLM avec resultats attendus. Defaults to {}.
            labels (Union[Dict, List[Dict], Classe, List[Classe]], optional): Liste de labels ou classes pour la classification. to [].
            classification_open (bool, optional): False: Les classes sont fixes, true sinon le LLM peut créer des classes. Defaults to False.
            multi_labels (bool, optional): False: un label par observation, true plusieurs labels pour une observation.. Defaults to False.
        """
        self.classification_open = classification_open
        self.multi_labels = multi_labels
        self.classes = self.build_classes(labels)
        self.examples = self.build_examples(examples, self.classes)
        super().__init__(examples=self.examples, *args, **kwargs)

    def get_prompt_params(self) -> Dict:
        return {
            "instructions": self.instructions,
            "variables": self.variables,
            "examples": self.examples,
            "use_system_prompt": self.use_system_prompt,
            "input_in_instructions": self.input_in_instructions,
            "use_ai_user_role_for_examples": self.use_ai_user_role_for_examples,
            "classes": self.classes,
        }

    def _default_baseitem_model(self):
        if self.multi_labels:
            return MultiClassItem
        else:
            return ClassItem

    def init_parser(self):
        closed = not self.classification_open
        utils = LCClasseParserUtils(classes=self.classes, closed=closed)
        return utils.parser()

    def run(
        self,
        inputs: Union[
            str,
            Dict,
            List[Union[Dict, str, ClassItem, MultiClassItem]],
            ClassItem,
            MultiClassItem,
            ItemCollection,
        ],
        batch_size: int = 1,
    ) -> ItemCollection:
        """Execute la tâche sur l'inputs.

        inputs peut être de différentes nature mais au final, les inputs seront transformées en \
        un ItemCollection. C'est à dire une liste de ClassItem ou MultiClassItem. L'opération de classification se fait par référence. \
        Cela signifie que la collection est renvoyée sans en créer une nouvelle.

        Args:
            inputs (Union[ str, Dict, List[Union[Dict, str, ClassItem, MultiClassItem]], ClassItem, MultiClassItem, ItemCollection ]): entrées à classifier.
            batch_size (int, optional): nombre de requêtes au llm en parallèle. Defaults to 1.

        Returns:
            ItemCollection: Liste de ClassItem ou MultiClassItem. Les attributs de ClassItem ou MultiClassItem sont modifiés par référence et non par copy.
        """
        # if inputs is already ItemCollection
        # we use it as inputs
        inputs = self.build_base_items(inputs)

        # compute output (classification is done by reference)
        if batch_size > 1:
            self.executor.execute_batch(inputs, batch_size=batch_size)
        else:
            for item in inputs:
                self.executor.execute_one(item)

        return inputs

    async def arun(
        self,
        inputs: Union[
            str,
            Dict,
            List[Union[Dict, str, ClassItem, MultiClassItem]],
            ClassItem,
            MultiClassItem,
            ItemCollection,
        ],
        batch_size: int = 1,
    ) -> ItemCollection:
        """Execute la tâche sur l'inputs en asynchron.

        inputs peut être de différentes nature mais au final, les inputs seront transformées en \
        un ItemCollection. C'est à dire une liste de ClassItem ou MultiClassItem. L'opération de classification se fait par référence. \
        Cela signifie que la collection est renvoyée sans en créer une nouvelle.

        Args:
            inputs (Union[ str, Dict, List[Union[Dict, str, ClassItem, MultiClassItem]], ClassItem, MultiClassItem, ItemCollection ]): entrées à classifier.
            batch_size (int, optional): nombre de requêtes au llm en parallèle. Defaults to 1.

        Returns:
            ItemCollection: Liste de ClassItem ou MultiClassItem. Les attributs de ClassItem ou MultiClassItem sont modifiés par référence et non par copy.
        """
        # if inputs is already ItemCollection
        # we use it as inputs
        inputs = self.build_base_items(inputs)

        # compute output (classification is done by reference)
        if batch_size > 1:
            await self.executor.execute_abatch(inputs, batch_size=batch_size)
        else:
            # no async loop because batch_size == 1
            for item in inputs:
                await self.executor.execute_aone(item)

        return inputs

    def build_classes(self, classes) -> List[Classe]:
        if isinstance(classes, list):
            return [self.build_classes(classe)[0] for classe in classes]

        if isinstance(classes, dict):
            return [self._default_output_model().model_validate(classes)]

        if isinstance(classes, Classe):
            return [classes]

        return classes

    def build_examples(
        self, examples: Dict[str, Union[str, List[str]]], classes=List[Classe]
    ) -> ItemCollection:
        return_examples = ItemCollection()
        labels_classes = {cl.label: cl for cl in classes}

        # loop over examples in format key, input
        for example_input, example_labels in examples.items():
            if (
                isinstance(example_labels, list)
                and len(example_labels) > 1
                and not self.multi_labels
            ):
                raise LLMTaskerException(
                    msg="You should set multi_labels to True if you have multiple labels.",
                    code="wrong_multiple_labels",
                )
            if isinstance(example_labels, str):
                example_labels = [example_labels]

            # store all classes found for this example:
            current_classes = []
            # loop over labels
            for example_label in example_labels:
                if example_label not in labels_classes.keys():
                    # examples classes must be provided event if open classification
                    raise LLMTaskerException(
                        msg=f"{example_label} not in classes. Check your classes or add them.",
                        code="wrong_classes_in_examples",
                    )
                current_classe = labels_classes[example_label].copy()
                # avoid useless infos for prompt ?
                current_classe.description = None
                current_classes.append(current_classe)
            # handle classes with item:
            if self.multi_labels:
                # output must be a list
                current_item = self._default_baseitem_model()(
                    input=example_input, output=current_classes
                )
            else:
                # output must be a str
                current_item = self._default_baseitem_model()(
                    input=example_input, output=current_classes[0]
                )
            return_examples.add(current_item)

        return return_examples

    @classmethod
    def build_labels_from_list(cls, labels: List[str]) -> List[Classe]:
        # ["l1", ..., "ln"]
        return [Classe(label=label) for label in labels]

    @classmethod
    def build_labels_with_descriptions(
        cls, labels_descs: Dict[str, str]
    ) -> List[Classe]:
        # {"label1": "description1", ..., "label2": "description2"}
        return [
            Classe(label=label, description=desc)
            for label, desc in labels_descs.items()
        ]


class GroupedClassification(Classification):
    def __init__(
        self,
        grouped_examples: bool = True,
        group_size_examples: int = 5,
        *args,
        **kwargs,
    ):
        """Classe représentant une classification groupée. \
            Un message au LLM contient plusieurs entrées à classifier.

        *args et **kwargs sont les arguments passés à la classe mère: `Classification`

        Args:
            grouped_examples (bool, optional): Classification groupée. Les exemples sont considérés comme groupée plutôt que unitaire. Defaults to True.
            group_size_examples (int, optional): Nombre d'item groupés en un message pour les examples. Defaults to 5.
        """
        super().__init__(*args, **kwargs)

        self.grouped_examples = grouped_examples
        self.group_size_examples = group_size_examples

        # format examples
        self.examples = (
            self.examples.create_sub_collections(group_size=self.group_size_examples)
            if self.grouped_examples
            else self.examples
        )

    def run(
        self,
        inputs: Union[
            str,
            Dict,
            List[Union[Dict, str, MultiClassItem, ClassItem]],
            MultiClassItem,
            ClassItem,
            ItemCollection,
        ],
        batch_size: int = 1,
        group_size: int = 5,
    ) -> ItemCollection:
        """Execute la tâche sur l'inputs.

        inputs peut être de différentes nature mais au final, les inputs seront transformées en \
        un ItemCollection. C'est à dire une liste de ClassItem ou MultiClassItem. L'opération de classification se fait par référence. \
        Cela signifie que la collection est renvoyée sans en créer une nouvelle.

        Args:
            inputs (Union[ str, Dict, List[Union[Dict, str, ClassItem, MultiClassItem]], MultiClassItem, ClassItem, ItemCollection ]): entrées à classifier.
            batch_size (int, optional): nombre de requêtes au llm en parallèle. Defaults to 1.
            group_size (int, optional): nombre d'item groupés en un message. Defaults to 5.

        Returns:
            ItemCollection: Liste de ClassItem ou MultiClassItem. Les attributs de ClassItem ou MultiClassItem sont modifiés par référence et non par copy.
        """
        inputs = self.build_base_items(inputs)

        # create sub collection with group size
        group_items_collection = inputs.create_sub_collections(group_size)

        # compute output (classification is done by reference)
        if batch_size > 1:
            self.executor.execute_batch(group_items_collection, batch_size=batch_size)
        else:
            for group_item in group_items_collection:
                self.executor.execute_one(group_item)

        # rebuild original item with output:
        return group_items_collection.flatten_sub_collections()

    async def arun(
        self,
        inputs: Union[
            str,
            Dict,
            List[Union[Dict, str, MultiClassItem, ClassItem]],
            MultiClassItem,
            ClassItem,
            ItemCollection,
        ],
        batch_size: int = 1,
        group_size: int = 5,
    ) -> ItemCollection:
        """Execute la tâche sur l'inputs.

        inputs peut être de différentes nature mais au final, les inputs seront transformées en \
        un ItemCollection. C'est à dire une liste de ClassItem ou MultiClassItem. L'opération de classification se fait par référence. \
        Cela signifie que la collection est renvoyée sans en créer une nouvelle.

        Args:
            inputs (Union[ str, Dict, List[Union[Dict, str, ClassItem, MultiClassItem]], MultiClassItem, ClassItem, ItemCollection ]): entrées à classifier.
            batch_size (int, optional): nombre de requêtes au llm en parallèle. Defaults to 1.
            group_size (int, optional): nombre d'item groupés en un message. Defaults to 5.

        Returns:
            ItemCollection: Liste de ClassItem ou MultiClassItem. Les attributs de ClassItem ou MultiClassItem sont modifiés par référence et non par copy.
        """
        inputs = self.build_base_items(inputs)

        # create sub collection with group size
        group_items_collection = inputs.create_sub_collections(group_size)

        # compute output (classification is done by reference)
        if batch_size > 1:
            await self.executor.execute_abatch(
                group_items_collection, batch_size=batch_size
            )
        else:
            # no async loop because batch_size == 1
            for group_item in group_items_collection:
                await self.executor.execute_aone(group_item)

        # rebuild original item with output:
        return group_items_collection.flatten_sub_collections()


@deprecated("use GroupedClassification instead")
class OpenGroupedClassification(GroupedClassification):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(classification_open=True, multi_labels=False, *args, **kwargs)
