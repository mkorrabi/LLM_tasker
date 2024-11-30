"""
Script d'example d'usage des tâches définies du package.

Tâche: classification fermée groupée
Description: on chercher à classifier en un message au LLM plusieurs items en des classes définies.
On appelle "groupée" la notion d'envoyer plusieurs items en un message.
"""

import os
from dotenv import find_dotenv, load_dotenv

# Import de l'API niveau 3
from llmtasker.tasks.classification import GroupedClassification

from langchain_openai import OpenAI, ChatOpenAI

load_dotenv(find_dotenv())
llm = ChatOpenAI(model_name="gpt-4o",temperature=0)

# config LLM


# Ecriture du prompt en dur au format f-string
instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici les animaux possibles:
{classes}

Tu dois absolument répondre sous forme de liste:

# Exemples:
{examples}
"""

# construction des classes
# TODO: on peut utiliser la fonction initial de madjid
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

# construction des exemples
examples = {
    "Leur regard perçant semble contenir tous les secrets du monde.": "chat",
    "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison": "chien",
}


task = GroupedClassification(
    llm=llm, instructions=instructions, labels=labels, examples=examples
)

# TODO: on peut utiliser la fonction initial de madjid
inputs = [
    {"input": "chat et chien"},
    {"input": "meilleurs amis de l'homme, toujours fidèles et pleins d'amour."},
]

# # Verification des données:
# for i in task.build_base_items(inputs):
#     print(i.format_as_example())

# - |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||: None
# - |||meilleurs amis de l'homme, toujours fidèles et pleins d'amour.|||: None


# Execution du classifier
# Paramètres:
# - batch_size: nombre de requête en // (max_concurrency)
# - group_size: nombre d'input regroupé en un message
output = task.run(inputs, batch_size=1, group_size=3)


for o in output:
    print(o.format_as_example())
# - |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||: chat
# - |||meilleurs amis de l'homme, toujours fidèles et pleins d'amour.|||: chien


# EXEMPLE DU PROMPT ENVOYE
# On remarque qu'il y a deux input dans le même message.

# System:
# Tu es un classifier.
#
# Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.
#
# Voici les animaux possibles:
# - chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
# - chat: cette classe est utlisée lorsque la phrase en input parle d'un chat
#
# Tu dois absolument répondre sous forme de liste:
#
# # Exemples:
# ["Leur regard perçant semble contenir tous les secrets du monde.", "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison"]
# [{"class_id": null, "label": "chat", "description": null}, {"class_id": null, "label": "chien", "description": null}]
#
# Human: ["créatures mystérieuses qui aiment se faufiler dans l'obscurité", "meilleurs amis de l'homme, toujours fidèles et pleins d'amour."]
