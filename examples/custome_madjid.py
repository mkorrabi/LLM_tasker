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

from typing import List
from pydantic import BaseModel, Field
from llmtasker.tasks.custom_instruction import CustomInstruction, CustomPydanticItem
from llmtasker.tasks.classification import Classification





# Chargement des variables d'environnement
load_dotenv(find_dotenv())

# config LLM
# config_azure_llm = {
#     "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
#     "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
#     "model": "gpt-35-turbo_16k_1106",
#     "temperature": 0,
#     "max_tokens": 800,
#     "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
#     "default_query": {"project-name": "llm-tasker"},
# }


llm = ChatOpenAI(model_name="gpt-4o-mini")



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

# construction des exemples


class Item(BaseModel):
    """element"""

    class_id: int = Field(
        description="num commentaire", title=""
    )
    label: str=Field(
        description="le label du commentaire", title=""
    )
    description :str=Field(
        description="la description du commentaire", title=""
    )
   

# class multi_labels_item(BaseModel):
#     item:List[Item]=Field(
#         description="liste des classes pour un commentaire", title=""
#     )

class multi_labels_items(BaseModel):
    list_items:List[List[Item]]=Field(
        description="liste des multi-labels de tout les commentaires", title=""
    )



# class CustomItem(CustomPydanticItem[List[str], list_elem]):
#     pass


# # On paramètre la tâche:
# CustomInstruction._DEFAULT_BASEITEM_MODEL = CustomItem
# CustomInstruction._DEFAULT_OUTPUT_MODEL = list_elem

examples_1 = {
    "input": [
        "Le montant de mon remboursement est erroné et je n'arrive pas à obtenir de réponse.",
        "Je comprends pas pourquoi mon remboursement n'est pas complet, ce n'est pas ce qui était promis.",
        "Je n'ai toujours pas reçu de remboursement pour l'article que j'ai renvoyé.",
        "Où est mon colis ? Le suivi est bloqué depuis 3 jours.",
        "Le suivi n'indique pas la date de livraison estimée.",
        "Où est mon colis ? Le délai de livraison est déjà dépassé.",
        "Je suis vraiment déçu par la qualité du tissu, il n'est pas du tout résistant.",
        "Le tissu est de mauvaise qualité et il perd sa forme après un lavage.",
        "Le tissu est très fragile et se déchire facilement. Très mauvaise qualité.",
        "Après avoir validé le paiement, j'ai eu un message d'erreur et la commande est annulée.",
        "Le paiement par carte est très lent et l'option PayPal ne fonctionne pas.",
        "Le site a indiqué un échec de paiement mais a quand même débité ma carte.",
        "Je n'ai pas aimé l'odeur chimique, ça gâche un peu l'expérience d'achat.",
        "Même après plusieurs lavages, l'odeur chimique est toujours présente.",
        "Les vêtements étaient imprégnés d'une forte odeur chimique qui m'a obligé à les laver avant de les porter.",
        "Je suis déçu, le manteau est déjà abîmé à l'arrivée avec des fils qui dépassent.",
        "Le manteau est arrivé avec une petite déchirure près de l'épaule.",
        "J'ai trouvé une déchirure sur la doublure à l'intérieur du manteau.",
        "Le remboursement a bien été effectué, merci pour la réactivité.",
        "Le remboursement est bien arrivé, merci pour votre efficacité.",
        "Remboursement bien reçu, merci pour votre réactivité.",
        "Livraison rapide et sans problème !",
        "Service de livraison rapide et impeccable.",
        "Livraison rapide, très contente du service.",
        "Commande arrivée incomplète, encore une fois.",
        "Commande reçue mais incomplète, je suis déçu.",
        "Manque un vêtement dans ma commande, c'est frustrant.",
        "Colis endommagé à l'arrivée, pas très sérieux.",
        "Mon colis est arrivé mouillé et endommagé.",
        "J'ai reçu mon colis avec des déchirures sur le carton.",
        "J'ai eu des problèmes avec les conditions de retour, ce n'était vraiment pas pratique.",
        "Les conditions de retour m'ont vraiment freiné dans ma décision d'achat.",
        "Les conditions de retour sont trop compliquées, j'aurais aimé plus de simplicité.",
        "Pull de très bonne qualité, il est à la fois chaud et léger, vraiment confortable.",
        "Le pull est d'une qualité supérieure, il est vraiment chaud et agréable à porter.",
        "Ce pull est vraiment de qualité supérieure, le tissu est épais et doux.",
        "Je n'arrive pas à recevoir mon remboursement car mon compte est expiré, que faire?",
        "Le remboursement est refusé car mon compte est expiré. Est-ce que vous pouvez m'aider à régler ce problème?",
        "Le remboursement n'est pas arrivé car mon compte est expiré. Que faire pour le recevoir?",
        "Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
        "Le panier ne se met pas à jour après un changement d'article, j'ai dû fermer et rouvrir la page.",
        "Le panier ne se met pas à jour correctement, j'ai dû rafraîchir plusieurs fois pour voir mes articles.",
        "Est-il possible de demander un remboursement sur un autre compte si le mien est expiré?",
        "Est-ce que le remboursement peut être renvoyé vers un autre compte si le mien est expiré?",
        "Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré?"
    ],
    "output": {
        "list_items": [
            [
                {
                    "class_id": 1,
                    "label": "Problème de remboursement",
                    "description": "Cette classe est utilisée lorsque la phrase en input concerne un problème lié au remboursement."
                }
            ],
            [
                {
                    "class_id": 2,
                    "label": "Retard de livraison",
                    "description": "Cette classe est utilisée lorsque la phrase en input concerne un retard de livraison ou un problème avec le suivi du colis."
                }
            ],
            [
                 {
                    "class_id": 1,
                    "label": "Problème de remboursement",
                    "description": "Cette classe est utilisée lorsque la phrase en input concerne un problème lié au remboursement."
                }
            ],
            [
                {
                    "class_id": 4,
                    "label": "Problèmes de paiement et de commande",
                    "description": "Cette classe est utilisée lorsque la phrase en input concerne des problèmes lors du paiement ou des commandes annulées."
                }
            ],
            [
                {
                    "class_id": 5,
                    "label": "Odeur chimique",
                    "description": "Cette classe est utilisée lorsque la phrase en input parle d'une odeur chimique désagréable liée au produit."
                }
            ],
            [
                {
                    "class_id": 6,
                    "label": "Défaut de qualité",
                    "description": "Cette classe est utilisée lorsque la phrase en input signale un produit abîmé ou présentant un défaut."
                }
            ],
            [
                {
                    "class_id": 7,
                    "label": "Remboursement reçu",
                    "description": "Cette classe est utilisée lorsque la phrase en input confirme la réception d'un remboursement."
                }
            ],
            [
                {
                    "class_id": 8,
                    "label": "Service de livraison rapide",
                    "description": "Cette classe est utilisée lorsque la phrase en input souligne une livraison rapide et efficace."
                }
            ],
            [
                {
                    "class_id": 9,
                    "label": "Articles manquants",
                    "description": "Cette classe est utilisée lorsque la phrase en input mentionne une commande incomplète ou des articles manquants."
                }
            ],
            [
                {
                    "class_id": 10,
                    "label": "Colis endommagé",
                    "description": "Cette classe est utilisée lorsque la phrase en input parle d'un colis reçu endommagé ou mouillé."
                }
            ],
            [
                {
                    "class_id": 11,
                    "label": "Conditions de retour",
                    "description": "Cette classe est utilisée lorsque la phrase en input mentionne des problèmes ou des complications avec les conditions de retour."
                }
            ],
            [
                {
                    "class_id": 12,
                    "label": "Pull de qualité supérieure",
                    "description": "Cette classe est utilisée lorsque la phrase en input loue la qualité supérieure d'un pull."
                }
            ],
            [
                {
                    "class_id": 13,
                    "label": "Problème de remboursement lié à un compte expiré",
                    "description": "Cette classe est utilisée lorsque la phrase en input mentionne un problème de remboursement lié à un compte expiré."
                }
            ],
            [
                {
                    "class_id": 14,
                    "label": "Problèmes de mise à jour du panier",
                    "description": "Cette classe est utilisée lorsque la phrase en input parle de problèmes pour mettre à jour le panier d'achat."
                }
            ]
        ]
    }
}
lebels=[{'label': 'Problème de remboursement', 'description': 'Problème de remboursement'}, {'label': 'Retard de livraison', 'description': 'Retard de livraison'}, {'label': 'Mauvaise qualité.', 'description': 'Mauvaise qualité.'}, {'label': 'Problèmes de paiement et de commande', 'description': 'Problèmes de paiement et de commande'}, {'label': 'Odeur chimique', 'description': 'Odeur chimique'}, {'label': 'Défaut de qualité', 'description': 'Défaut de qualité'}, {'label': 'Remboursement reçu', 'description': 'Remboursement reçu'}, {'label': 'Service de livraison rapide', 'description': 'Service de livraison rapide'}, {'label': 'Articles manquants', 'description': 'Articles manquants'}, {'label': 'Colis endommagé', 'description': 'Colis endommagé'}, {'label': 'Conditions de retour', 'description': 'Conditions de retour'}, {'label': 'Pull de qualité supérieure', 'description': 'Pull de qualité supérieure'}, {'label': 'Problème de remboursement lié à un compte expiré', 'description': 'Problème de remboursement lié à un compte expiré'}, {'label': 'Problèmes de mise à jour du panier', 'description': 'Problèmes de mise à jour du panier'}, {'label': 'Remboursement compte expiré', 'description': 'Remboursement compte expiré'}, {'label': 'Autre', 'description': 'Toute autre classe non présente'}]





instructions="""Classifiez-moi la phrase en une classe. 
Voici les classes : {classes} 
Voici les exemples : {examples}  
Tu DOIS TOUJOURS répondre au format JSON.
{json_shema_output}"""

# Création de la tâche
# task = CustomInstruction(
#     llm=llm,
#     instructions=instructions,
#     template_format="f-string",  # ou "f-string" par défaut
#     method="json_mode",
#     examples=exemple_1,
#     use_ai_user_role_for_examples=True,  # use ai/user role for few shots learning
#     stop_after_attempt=2,
# )

# Inputs à scorer

# Données à classifier
i1 = [
    "créatures mystérieuses qui aiment se faufiler dans l'obscurité",
    "Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré?.",
    "deux animaux qui vivent souvent avec les humains",
    "Est-il possible de demander un remboursement sur un autre compte si le mien est expiré?",
    "Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "panier",
    "salut ça va ",
    "Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré? et en plus Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "panier  ne marche pas et je suis heureux de mon t shirt",
    "créatures mystérieuses qui aiment se faufiler dans l'obscurité",
    "Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré?.",
    "deux animaux qui vivent souvent avec les humains",
    "Est-il possible de demander un remboursement sur un autre compte si le mien est expiré?",
    "Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "panier",
    "salut ça va ",
    "Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré? et en plus Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.",
    "panier  ne marche pas et je suis heureux de mon t shirt",
]

# results = task.run(inputs=i1)


# for res in results:
#     if res.error:
#         print(res.error)
#     elif res.output:
#         print(res.output)
#     elif res.raw:
#         print(res.raw)




import os
from typing import List
from dotenv import find_dotenv, load_dotenv
# from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from llmtasker.items.base import BaseItem
from llmtasker.prompts.lc_base import LCPrompt
from langchain_openai import OpenAI, ChatOpenAI

from llmtasker.executors.langchain import (
    LCPromptLLMParserExecutor,
    LCPromptLLMPydanticExecutor,
)
from llmtasker.parsers import LCPydanticOutputParser

# Chargement des variables d'environnement
load_dotenv(find_dotenv(), override=True)
class CustomItem(BaseItem[List[str], multi_labels_items]):
    def format_output(self):
        return str(self.output.model_dump_json())

# on utilise la mécanique pydantic
exemple_1 = CustomItem.model_validate(examples_1)
from llmtasker.prompts.lc_base import LCPrompt

# le prompt définit une variables "json_shema_output"
# qui permet d'afficher le format JSON attendu
# prompt = LCPrompt(
#     # instructions="custom_instruction.jinja",
#     instructions=instructions,
#     use_system_prompt=True,
#     input_in_instructions=False,
#     use_ai_user_role_for_examples=True,
#     examples=[exemple_1],
# )

# le format de mes instructions est écrit avec le moteur de template Jinja
template_format = "f-string"

from llmtasker.parsers import LCPydanticOutputParser

# création d'un parser JSON pour transformer la sortie du LLM
parser = LCPydanticOutputParser(pydantic_object=multi_labels_items)


# Injection de la valeur de la variable "json_shema_output" pour créer un prompt

# Injection de la valeur de la variable "json_shema_output" pour créer un prompt



# exemple du prompt généré
# output = prompt.generate(template_format=template_format).format(input=["input de test"])



# Configuration azure
# config_azure_llm = {
#     "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
#     "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
#     # "model": "gpt-35-turbo_16k_1106",
#     "deployment_name": "gpt-35-turbo_16k_1106",
#     "temperature": 0,
#     "max_tokens": 800,
#     "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
#     "default_query": {"project-name": "llm-tasker"},
# }
# instancier le LLM
# llm = AzureChatOpenAI(**config_azure_llm)

llm = ChatOpenAI(model_name="gpt-4o-mini")

# instancier l'exécuteur à partir de trois éléments:
# prompt, llm et parser
# l'utilisateur est libre de modifier cette classe

lebels=[{'label': 'Problème de remboursement', 'description': 'Problème de remboursement'}, {'label': 'Retard de livraison', 'description': 'Retard de livraison'}, {'label': 'Mauvaise qualité.', 'description': 'Mauvaise qualité.'}, {'label': 'Problèmes de paiement et de commande', 'description': 'Problèmes de paiement et de commande'}, {'label': 'Odeur chimique', 'description': 'Odeur chimique'}, {'label': 'Défaut de qualité', 'description': 'Défaut de qualité'}, {'label': 'Remboursement reçu', 'description': 'Remboursement reçu'}, {'label': 'Service de livraison rapide', 'description': 'Service de livraison rapide'}, {'label': 'Articles manquants', 'description': 'Articles manquants'}, {'label': 'Colis endommagé', 'description': 'Colis endommagé'}, {'label': 'Conditions de retour', 'description': 'Conditions de retour'}, {'label': 'Pull de qualité supérieure', 'description': 'Pull de qualité supérieure'}, {'label': 'Problème de remboursement lié à un compte expiré', 'description': 'Problème de remboursement lié à un compte expiré'}, {'label': 'Problèmes de mise à jour du panier', 'description': 'Problèmes de mise à jour du panier'}, {'label': 'Remboursement compte expiré', 'description': 'Remboursement compte expiré'}, {'label': 'Autre', 'description': 'Toute autre classe non présente'}]


# OU méthode directe en utilisant une classe pré configuré avec une sortie structurée:
# on peut utiliser le function calling ou json mode
task = CustomInstruction(
    llm=llm,
    instructions=instructions,
    template_format=template_format,  # ou "f-string" par défaut
    method="json_mode",
    use_ai_user_role_for_examples=True,  # use ai/user role for few shots learning
    stop_after_attempt=2,
    variables={"classes":str(lebels),"examples":str(exemple_1),"json_shema_output":parser.get_format_instructions()},
    parser=parser
)

# task = Classification(
#     examples=exemple_1,
#     labels=labels,
#     llm=llm,
#     instructions=instructions,
#     use_ai_user_role_for_examples=False,
#     multi_labels=False,  # default
#     classification_open=False,  # default
# )

# Execution

# Item à traiter
i1 = CustomItem(
    input=["miaw ","je suis un chien et un chat"]
)

# for i in task.build_base_items(i1):
#     print(i.format_as_example())

# - |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||: None
# - |||meilleurs amis de l'homme, toujours fidèles et pleins d'amour.|||: None


# Execution du classifier
# Paramètres:
# - batch_size: nombre de requête en // (max_concurrency)
# - group_size: nombre d'input regroupé en un message
output = task.run(i1, batch_size=1)
print(output[0])




# >>> |||plus 40 % taux d endettement et vient de recevoir ce jour courrier licenciement economique|||: {"conforme":false,"extracts":[{"content":"plus 40 % taux d endettement","reason":"situation financière"},{"content":"courrier licenciement economique","reason":"situation professionnelle"}]}
