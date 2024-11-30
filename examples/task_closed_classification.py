"""
Script d'example d'usage des tâches définies du package.

Tâche: classification fermée
Description: on chercher à classifier un item en un message par rapport à des classes prédéfinies.
"""

import os
from dotenv import find_dotenv, load_dotenv

from llmtasker.tasks.classification import Classification

load_dotenv(find_dotenv())

# config LLM
config_azure_llm = {
    "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "model": "gpt-35-turbo_16k_1106",
    "temperature": 0,
    "max_tokens": 800,
    "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
    "default_query": {"project-name": "llm-tasker"},
    "model_kwargs": {
        "seed": 0,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    },
}

# Prompt en "dur":
instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

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
    "Leur regard perçant semble contenir tous les secrets du monde.": "chat",
    "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison": "chien",
}

# Données à classifier
inputs = [
    "créatures mystérieuses qui aiment se faufiler dans l'obscurité",
    "meilleurs amis de l'homme, toujours fidèles et pleins d'amour.",
]

# OU
# inputs = [
#     {"input": "créatures mystérieuses qui aiment se faufiler dans l'obscurité"},
#     {"input": "meilleurs amis de l'homme, toujours fidèles et pleins d'amour."},
# ]


# création de la tâche:
task = Classification(
    examples=examples,
    labels=labels,
    llm=config_azure_llm,
    instructions=instructions,
    use_ai_user_role_for_examples=False,
    multi_labels=False,  # default
    classification_open=False,  # default
)

# Execution
outputs = task.run(inputs=inputs, batch_size=2)

# Affichage des résultats
for item in outputs:
    if item.error:
        print(item.error)
    if item.output:
        print(item.output)
    else:
        print(item.raw)

# chat
# chien

# EXEMPLE DU PROMPT ENVOYE

# System:
# Tu es un classifier.
#
# Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.
#
# Voici les animaux possibles:
# - chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
# - chat: cette classe est utlisée lorsque la phrase en input parle d'un chat
#
# # Exemples:
# - |||Leur regard perçant semble contenir tous les secrets du monde.|||: chat
# - |||Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison|||: chien
#
# Human: |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||
