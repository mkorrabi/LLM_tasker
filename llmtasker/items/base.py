from typing import (
    Any,
    List,
    Union,
    Optional,
    Sequence,
    Generic,
    TypeVar,
    Type,
    ClassVar,
    Dict,
)
import collections
import json
from pydantic import BaseModel, Field, model_validator, field_serializer

from llmtasker.exceptions import LLMTaskerException


E = TypeVar("E")
S = TypeVar("S")


class BaseItem(BaseModel, Generic[E, S], validate_assignment=True):
    INPUT_DELIM: ClassVar[Optional[str]] = "|||"
    INPUT_FORMAT: ClassVar[Optional[None]] = "{delim}{input}{delim}"

    id: Optional[Any] = None
    input: E
    output: Optional[S] = Field(default=None)
    error: Optional[Union[Dict, Exception]] = Field(default=None)
    raw: Optional[Union[Dict, Any]] = Field(default=None)

    # model_config = {"arbitrary_types_allowed": True}

    class Config:
        arbitrary_types_allowed = True

    def format_output(self):
        return self.output.__str__()

    def format_input(self):
        return self._format_input(self.input.__str__())

    def format_as_example(self):
        return f"- {self.format_input()}: {self.format_output()}"

    @field_serializer("error", when_used="json")
    def serialize_exception_in_dict(error: Optional[Union[Exception, Dict]] = None):
        if not error:
            return None
        intern_error = isinstance(error, LLMTaskerException)
        if isinstance(error, Exception) and not intern_error:
            error = LLMTaskerException(msg=str(error), code=error.__class__.__name__)

        return dict(error)

    @field_serializer("raw", when_used="json")
    def serialize_raw_in_dict(raw: Optional[Any] = None):
        # spécifique en fonction de l'implémentation
        # si AIMessage => pydantic v1 => dict()
        if not raw:
            return None
        return dict(raw)

    @classmethod
    def _format_input(cls, input: str = "{input}") -> str:
        return cls.INPUT_FORMAT.format(delim=cls.INPUT_DELIM, input=input)


T = TypeVar("T", bound=BaseItem)


class ItemCollection(collections.abc.MutableSequence):
    def __init__(self, items: Optional[Sequence[T]] = None):
        self._items = []
        if items:
            self._items.extend(items)

    def add(self, item: T):
        self.append(item)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setitem__(self, index: int, item: T):
        self._items[index] = item

    def __delitem__(self, index: int):
        del self._items[index]

    def insert(self, index: int, item: T):
        self._items.insert(index, item)

    def keys(self):
        return list(range(0, len(self._items)))

    # def create_sub_collections(self, group_size: int = 2) -> "ItemCollection":
    #     if group_size <= 1:
    #         raise LLMTaskerException(
    #             msg="group size must be greater than 1", code="create_sub_collections"
    #         )
    #     root_collection = ItemCollection()
    #     current_collec_size = len(self)
    #     n_subgroup = current_collec_size // group_size + 1 * (
    #         current_collec_size % group_size > 0
    #     )
    #     for ix in range(n_subgroup):
    #         root_collection.add(
    #             GroupItem.from_items(self[ix * n_subgroup : (ix + 1) * group_size])
    #         )
    #     return root_collection
    def create_sub_collections(self, group_size: int = 2) -> "ItemCollection":
        if group_size <= 1:
            raise LLMTaskerException(
                msg="group size must be greater than 1", 
                code="create_sub_collections"
            )
        
        root_collection = ItemCollection()
        current_collec_size = len(self)
        
        # Si la collection est vide, retourner une collection vide
        if current_collec_size == 0:
            return root_collection
            
        # Calcul du nombre de sous-groupes
        n_subgroup = (current_collec_size + group_size - 1) // group_size
        
        # Création des sous-groupes
        for ix in range(n_subgroup):
            start_idx = ix * group_size
            end_idx = min((ix + 1) * group_size, current_collec_size)
            
            items_subset = self[start_idx:end_idx]
            if items_subset:  # Vérifier que le sous-ensemble n'est pas vide
                root_collection.add(GroupItem.from_items(items_subset))
                
        return root_collection

    def flatten_sub_collections(self) -> "ItemCollection":
        list_items = []
        for group_item in self:
            assert isinstance(group_item, GroupItem)
            list_items.extend(group_item.get_items())
        flatten_collection = ItemCollection(list_items)
        return flatten_collection


class GroupItem(BaseItem[List[E], List[S]]):
    id: Optional[List[Any]] = None
    base_item_cls: Type[BaseItem] = BaseItem[Any, Any]

    @model_validator(mode="after")
    def check_types(self):
        # https://github.com/pydantic/pydantic/discussions/7367

        inputs = self.input
        outputs = self.output
        base_item_cls = self.base_item_cls

        validated_inputs = []
        if inputs:
            for input in inputs:
                validated_inputs.append(
                    base_item_cls.__pydantic_validator__.validate_assignment(
                        base_item_cls.model_construct(), "input", input
                    ).input
                )
            self._set_skip_validation("input", validated_inputs)

        validated_outputs = []
        if outputs:
            for output in outputs:
                validated_outputs.append(
                    base_item_cls.__pydantic_validator__.validate_assignment(
                        base_item_cls.model_construct(), "output", output
                    ).output
                )
            self._set_skip_validation("output", validated_outputs)

        return self

    @model_validator(mode="after")
    def check_lengths(self):
        if self.output:
            assert len(self.input) == len(self.output)

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Workaround to be able to set fields without validation."""
        # https://github.com/pydantic/pydantic/issues/8185
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property):
            attr.__set__(self, value)
        else:
            self.__dict__[name] = value
            self.__pydantic_fields_set__.add(name)

    @classmethod
    def from_items(cls, items: Union[ItemCollection, List[BaseItem]]):
        return cls(
            base_item_cls=items[0].__class__,
            input=[i.input for i in items],
            output=[i.output for i in items],
            id=[i.id for i in items],
        )

    def get_items(self) -> List[BaseItem]:
        items = []
        inputs = self.input
        outputs = [None] * len(inputs)
        if self.output:
            outputs = self.output
        for item_id, inp, out in zip(self.id, inputs, outputs):
            item = self.base_item_cls(
                id=item_id, input=inp, output=out, error=self.error, raw=self.raw
            )
            items.append(item)
        return items

    def format_input(self):
        return json.dumps(self.input, ensure_ascii=False)

    def format_output(self):
        return json.dumps(
            [o.model_dump() if o else None for o in self.output], ensure_ascii=False
        )

    def format_as_example(self):
        return f"{self.format_input()}\n{self.format_output()}"


class Classe(BaseModel):
    class_id: Optional[Union[int, str]] = None
    label: str
    description: Optional[str] = None

    def __str__(self) -> str:
        # cl
        # str(cl) => label
        return self.label

    def format(self) -> str:
        return f"- {self.label}: {self.description}"
