"""Script d'example d'usage des tâches définies du package.

Dans le cas de la tâche custom, l'utilisateur doit quand même
définir les objets de retour.

Sinon, on peut utiliser l'objet CustomDictItem (à affiner).

Tâche: custom instruction
Description: on chercher à extraire et identifier si un commentaire est conforme
"""

import os
from typing import List
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from llmtasker.tasks.custom_instruction import CustomInstruction, CustomPydanticItem


# Chargement des variables d'environnement
load_dotenv(find_dotenv(), override=True)


# Définition du format de sortie en deux classes imbriquées
# - Extract
# - Compliance
#
# On utilise la description des champs pour aider le modèle
class Extract(BaseModel):
    """Représente une extraction non conforme"""

    content: str = Field(
        description="Extraction de la séquence de texte non conforme dans le commentaire.",
        default=None,
    )
    reason: str = Field(
        description="Raison pour laquelle l'extraction n'est pas conforme.",
        default=None,
    )


class Compliance(BaseModel):
    """Représente la conformité d'un commentaire"""

    conforme: bool = Field(
        description="Conformité du commentaire. True si valid False sinon.", title=""
    )
    extracts: List[Extract] = Field(description="Liste des Objets Extract", default=[])


# Définition d'un item relatif à la classification
# En entrée "str"
# En sortie "Compliance" => l'objet défini plus haut
#
# On doit définir un format d'affichage de mon objet compliance
# => j'utilise le format json en str pour donner le bon format au LLM
class CustomItem(CustomPydanticItem[str, Compliance]):
    pass


# On paramètre la tâche:
CustomInstruction._DEFAULT_BASEITEM_MODEL = CustomItem
CustomInstruction._DEFAULT_OUTPUT_MODEL = Compliance

# OU on peut réecrire une classe spécifique de cette façon:
# class MyCustomInstruction(CustomInstruction):
#     def _default_baseitem_model(self):
#         return CustomItem

#     def _default_output_model(self):
#         return Compliance


# Définition d'un exemple à partir d'un json
exemple_1 = {
    "input": "bjr j ai eu le mari de m au telephone ils sont separes mais non divorces mme est decedee je suis en attente de l acte de deces notaire maitre gerometta a joigny cdt lisa",
    "output": {
        "conforme": False,
        "extracts": [
            {"content": "non divorces mme est decedee", "reason": "situation familiale"}
        ],
    },
}

# paramètre du modèle LLM Azure
config_azure_llm = {
    "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "deployment_name": "gpt-35-turbo_16k_1106",
    "temperature": 0,
    "max_tokens": 800,
    "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
    "default_query": {"project-name": "llm-tasker"},
}

# Création de la tâche
task = CustomInstruction(
    llm=config_azure_llm,
    instructions="custom_instruction.jinja",
    template_format="jinja2",  # ou "f-string" par défaut
    method="json_mode",
    examples=exemple_1,
    use_ai_user_role_for_examples=True,  # use ai/user role for few shots learning
    stop_after_attempt=2,
)

# Inputs à scorer
i1 = [
    {
        "input": "plus 40 % taux d endettement et vient de recevoir ce jour courrier licenciement economique"
    },
    {"input": "Travaille avec des handicapés"},
]

results = task.run(inputs=i1)


for res in results:
    if res.error:
        print(res.error)
    elif res.output:
        print(res.output)
    elif res.raw:
        print(res.raw)
