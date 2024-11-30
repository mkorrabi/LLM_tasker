"""Script d'example d'usage avancée du package:
C'est à dire qu'on utilise pas les tâches définies.

Description: on chercher à classifier en un message plusieurs items.

Ce script (de travail) est un peu une utilisation avancée du package.

l'idée de ce script est de faciliter cette possibilité en utilisant `GroupItem`.
Cette classe permet de créer un item qui gère plusieurs inputs/outputs à partir d'un item.

A la fin on récupère nos items avec `GroupItem.get_items()`.
"""

import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from llmtasker.items.base import BaseItem, Classe, GroupItem
from llmtasker.prompts.lc_base import LCPrompt
from llmtasker.executors.langchain import LCPromptLLMParserExecutor
from llmtasker.parsers import LCPydanticParserUtils


# Chargement des variables d'environnement
load_dotenv(find_dotenv())


# définition d'un item dont la sortie (output) est un objet pydantic (Classe)
class MyItem(BaseItem[str, Classe]):
    def format_output(self):
        return str(self.output.model_dump_json())


# items d'exemples
exemples = [
    MyItem(input="Le grand méchant loup", output=Classe(label="loup")),
    MyItem(
        input="on monte sur son dos avec les indiens", output=Classe(label="cheval")
    ),
    MyItem(input="c'est rose et il a une chanson", output=Classe(label="panthère")),
]

# items à classifier en un message
# à partir d'une liste de str
group_item = GroupItem(
    base_item_cls=MyItem,
    input=[
        "elles pondent des oeufs que les humains mangent",
        "Ca a des gros crocs et ca hurle !",
        "c'est la femme du coq",
    ],
)

# à partir d'une liste d'items
group_item = GroupItem.from_items(
    [
        MyItem(id=0, input="elles pondent des oeufs que les humains mangent"),
        MyItem(id=1, input="Ca a des gros crocs et ca hurle !"),
        MyItem(id=2, input="c'est la femme du coq"),
    ]
)

print(group_item.format_as_example())

# ["elles pondent des oeufs que les humains mangent", "Ca a des gros crocs et ca hurle !", "c'est la femme du coq"]
# [null, null, null]

# Prompt en "dur":
instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici des exemples de catégories:
{classes}

Tu DOIS TOUJOURS répondre au format JSON.

# Exemples:
{examples}

# input:
{input}
"""

# l'utilisateur est libre de modifier cette classe
prompt = LCPrompt(
    instructions=instructions,
    use_system_prompt=True,
    input_in_instructions=True,
    use_ai_user_role_for_examples=False,
    examples=exemples,  # si on veut des exemples en liste
    # examples=[GroupItem.from_items(exemples)],  # si on veut des exemples en bloc
    classes=[
        Classe(class_id="test", label="cheval", description="(cheval, poney, chevaux)"),
        Classe(
            class_id="test",
            label="chien",
            description="(chiens, chiennes, meilleur ami de l'homme)",
        ),
    ],
)

template_format = "f-string"

parser = LCPydanticParserUtils(pydantic_object=Classe).parser()

# exemple du prompt généré
output = prompt.generate(template_format=template_format).format(
    input=group_item.format_input()
)
print(output)

# Configuration azure
config_azure_llm = {
    "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "model": "gpt-35-turbo_16k_1106",
    "temperature": 0,
    "max_tokens": 800,
    "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
    "default_query": {"project-name": "llm-tasker"},
}
# instancier le LLM
llm = AzureChatOpenAI(**config_azure_llm)


executor = LCPromptLLMParserExecutor(
    prompt.generate(template_format=template_format),
    llm=llm,
    parser=parser,
)


executor.execute_one(group_item)

print(group_item.format_as_example())

# System:
# Tu es un classifier.

# Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

# Voici des exemples de catégories:
# - cheval: (cheval, poney, chevaux)
# - chien: (chiens, chiennes, meilleur ami de l'homme)

# Tu DOIS TOUJOURS répondre au format JSON.

# # Exemples:
# ["Le grand m\u00e9chant loup", "on monte sur son dos avec les indiens", "c'est rose et il a une chanson"]
# [{"class_id": null, "label": "loup", "description": null}, {"class_id": null, "label": "cheval", "description": null}, {"class_id": null, "label": "panth\u00e8re", "description": null}]

# # input:
# ["elles pondent des oeufs que les humains mangent", "Ca a des gros crocs et ca hurle !", "c'est la femme du coq"]

# ["elles pondent des oeufs que les humains mangent", "Ca a des gros crocs et ca hurle !", "c'est la femme du coq"]
# [{"class_id": null, "label": "poule", "description": null}, {"class_id": null, "label": "loup", "description": null}, {"class_id": null, "label": "poule", "description": null}]

items = group_item.get_items()


# Affichage des résultats
for item in items:
    if item.error:
        print(item.error)
    if item.output:
        print(item.output)
    else:
        print(item.raw)

# for i in items:
#     print(i)
# - |||elles pondent des oeufs que les humains mangent|||: {"class_id":null,"label":"poule","description":null}
# - |||Ca a des gros crocs et ca hurle !|||: {"class_id":null,"label":"loup","description":null}
# - |||c'est la femme du coq|||: {"class_id":null,"label":"poule","description":null}
