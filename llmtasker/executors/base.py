"""Module de base pour formaliser un exécuteur et ses méthodes à implémenter."""

from typing import Optional, TypeVar
from llmtasker.items.base import BaseItem, ItemCollection

T = TypeVar("T", bound=BaseItem)


class IExecutor:
    """Interface pour un exécuteur de traitement d'éléments."""

    def execute_one(self, item: T) -> T:
        """Méthode pour traiter un `BaseItem`.

        Args:
            item (T): Objet item hérité de `BaseItem`.

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée.

        Returns:
            T: Retourne l'objet item avec l'attribut `BaseItem.output` mis à jour.
        """
        raise NotImplementedError

    def execute_batch(
        self, items: ItemCollection, batch_size: Optional[int] = None
    ) -> ItemCollection:
        """Méthode pour traiter une collection d'items.

        Args:
            items (ItemCollection): Une collection d'objets items.
            batch_size (Optional[int], optional): Taille du batch si nécessaire. Par défaut None.

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée.

        Returns:
            ItemCollection: Retourne la collection d'items traitée.
        """
        raise NotImplementedError

    async def execute_aone(self, item: T) -> T:
        """Méthode asynchrone pour traiter un `BaseItem`.

        Args:
            item (T): Objet item hérité de `BaseItem`.

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée.

        Returns:
            T: Retourne l'objet item avec l'attribut `BaseItem.output` mis à jour.
        """
        raise NotImplementedError

    async def execute_abatch(
        self, items: ItemCollection, batch_size: Optional[int] = None
    ) -> ItemCollection:
        """Méthode asynchrone pour traiter une collection d'items.

        Args:
            items (ItemCollection): Une collection d'objets items.
            batch_size (Optional[int], optional): Taille du batch si nécessaire. Par défaut None.

        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée.

        Returns:
            ItemCollection: Retourne la collection d'items traitée.
        """
        raise NotImplementedError
