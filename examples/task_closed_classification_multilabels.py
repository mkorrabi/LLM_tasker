"""
Script d'example d'usage des tâches définies du package.

Tâche: classification fermée multi labels
Description: on chercher à classifier un item en un message par rapport à des classes prédéfinies. \
Un message peut être classifié avec plusieurs labels.
"""

import os
from dotenv import find_dotenv, load_dotenv

from llmtasker.tasks.classification import Classification
from langchain_openai import OpenAI, ChatOpenAI

load_dotenv(find_dotenv())

# config LLM
llm = ChatOpenAI(model_name="gpt-4o-mini")

# config LLM

# Prompt en "dur":
instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un ou plusieurs noms d'animaux.

Voici les animaux possibles:
{classes}

# Exemples:
{examples}
"""

# construction des classes
labels = [
    {
        "label": "chien",
        "description": "cette classe est utlisée lorsque la phrase en input parle d'un chien",
    },
    {
        "label": "chat",
        "description": "cette classe est utlisée lorsque la phrase en input parle d'un chat",
    },
]

# OU depuis une liste sans descriptions:
# labels = Classification.build_labels_from_list(["chat", "chien", "autre"])

# OU depuis un dictionnaire avec descriptions:
# tmp = {
#     "chien": "cette classe est utlisée lorsque la phrase en input parle d'un chien ",
#     "chat": "cette classe est utlisée lorsque la phrase en input parle d'un chat",
# }
# labels = Classification.build_labels_with_descriptions(tmp)


# Exemples à donner au LLM
examples = {
    "Leur regard perçant semble contenir tous les secrets du monde.": ["chat"],
    "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison": [
        "chien"
    ],
    "Ils ont quatre pattes et doivent être pucés": ["chat", "chien"],
    "Les deux commencent par 'ch'": ["chien", "chat"],
}

# Données à classifier
inputs = [
    "créatures mystérieuses qui aiment se faufiler dans l'obscurité",
    "meilleurs amis de l'homme, toujours fidèles et pleins d'amour.",
    "deux animaux qui vivent souvent avec les humains",
]

# OU
# inputs = [
#     {"input": "créatures mystérieuses qui aiment se faufiler dans l'obscurité"},
#     {"input": "meilleurs amis de l'homme, toujours fidèles et pleins d'amour."},
#     {"input": "deux animaux qui vivent souvent avec les humains"},
# ]


# création de la tâche:
task = Classification(
    examples=examples,
    labels=labels,
    llm=llm,
    instructions=instructions,
    use_ai_user_role_for_examples=False,
    multi_labels=True,
    classification_open=False,  # default
)

# Execution
outputs = task.run(inputs=inputs, batch_size=2)

# Affichage des résultats
for item in outputs:
    if item.error:
        print(item.error)
    if item.output:
        print(item.output.root[0].label)
    else:
        print(item.raw)

# [Classe(class_id=None, label='chat', description="cette classe est utlisée lorsque la phrase en input parle d'un chat")]
# [Classe(class_id=None, label='chien', description="cette classe est utlisée lorsque la phrase en input parle d'un chien")]
# [Classe(class_id=None, label='chat', description="cette classe est utlisée lorsque la phrase en input parle d'un chat"),
#  Classe(class_id=None, label='chien', description="cette classe est utlisée lorsque la phrase en input parle d'un chien")]

# EXEMPLE DU PROMPT ENVOYE
# > print(task.prompt.generate().format(input=task.build_base_items(inputs)[0]))
#
# System:
# Tu es un classifier.

# Tu dois identifier les animaux grâce à la description. Donne un ou plusieurs noms d'animaux.

# Voici les animaux possibles:
# - chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
# - chat: cette classe est utlisée lorsque la phrase en input parle d'un chat

# # Exemples:
# - |||Leur regard perçant semble contenir tous les secrets du monde.|||: [{"class_id": null, "label": "chat", "description": null}]
# - |||Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison|||: [{"class_id": null, "label": "chien", "description": null}]
# - |||Ils ont quatre pattes et doivent être pucés|||: [{"class_id": null, "label": "chat", "description": null}, {"class_id": null, "label": "chien", "description": null}]
#
# Human: |||deux animaux qui vivent souvent avec les humains|||
