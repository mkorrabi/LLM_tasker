"""Script d'example d'usage avancée du package:
C'est à dire qu'on utilise pas les tâches définies.

Description: on chercher à extraire et identifier si un commentaire est conforme
"""

import os
from typing import List
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from llmtasker.items.base import BaseItem
from llmtasker.prompts.lc_base import LCPrompt
from llmtasker.executors.langchain import (
    LCPromptLLMParserExecutor,
    LCPromptLLMPydanticExecutor,
)
from llmtasker.parsers import LCPydanticOutputParser

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


# Définition d'un item spécifique à la tâche custom
# En entrée "str"
# En sortie "Compliance" => l'objet défini plus haut
#
# On doit définir un format d'affichage de mon objet compliance
# => j'utilise le format json en str pour donner le bon format au LLM
class CustomItem(BaseItem[str, Compliance]):
    def format_output(self):
        return str(self.output.model_dump_json())


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

# on utilise la mécanique pydantic
exemple_1 = CustomItem.model_validate(exemple_1)

# le prompt définit une variables "json_shema_output"
# qui permet d'afficher le format JSON attendu
prompt = LCPrompt(
    # instructions="custom_instruction.jinja",
    instructions="custom_instruction.jinja",
    use_system_prompt=True,
    input_in_instructions=False,
    use_ai_user_role_for_examples=True,
    examples=[exemple_1],
)

# le format de mes instructions est écrit avec le moteur de template Jinja
template_format = "jinja2"

# création d'un parser JSON pour transformer la sortie du LLM
parser = LCPydanticOutputParser(pydantic_object=Compliance)

# Injection de la valeur de la variable "json_shema_output" pour créer un prompt
prompt.variables["json_shema_output"] = parser.get_format_instructions()


# exemple du prompt généré
output = prompt.generate(template_format=template_format).format(input="input de test")

print(output)

# Configuration azure
config_azure_llm = {
    "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    # "model": "gpt-35-turbo_16k_1106",
    "deployment_name": "gpt-35-turbo_16k_1106",
    "temperature": 0,
    "max_tokens": 800,
    "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
    "default_query": {"project-name": "llm-tasker"},
}
# instancier le LLM
llm = AzureChatOpenAI(**config_azure_llm)

# instancier l'exécuteur à partir de trois éléments:
# prompt, llm et parser
# l'utilisateur est libre de modifier cette classe
executor = LCPromptLLMParserExecutor(
    prompt.generate(template_format=template_format), llm=llm, parser=parser
)

# OU méthode directe en utilisant une classe pré configuré avec une sortie structurée:
# on peut utiliser le function calling ou json mode
executor = LCPromptLLMPydanticExecutor(
    prompt.generate(template_format=template_format),
    llm=llm,
    schema=Compliance,
    method="function_calling",
    stop_after_attempt=1,
)

# Item à traiter
i1 = CustomItem(
    input="plus 40 % taux d endettement et vient de recevoir ce jour courrier licenciement economique"
)

executor.execute_one(i1)
if i1.error:
    print(i1.error)
else:
    print(i1.format_as_example())
    print(i1.raw)

# >>> |||plus 40 % taux d endettement et vient de recevoir ce jour courrier licenciement economique|||: {"conforme":false,"extracts":[{"content":"plus 40 % taux d endettement","reason":"situation financière"},{"content":"courrier licenciement economique","reason":"situation professionnelle"}]}
