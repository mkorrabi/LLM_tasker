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

# config LLM

# Ecriture du prompt en dur au format f-string
# instructions = """
# Tu es un classifier.

# Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

# Voici les animaux possibles:
# {classes}

# Tu dois absolument répondre sous forme de liste:

# # Exemples:
# {examples}
# """
instructions="Classifiez-moi la phrase en une classe. Voici les classes : {classes} Voici les exemples : {examples}"

# construction des classes
# TODO: on peut utiliser la fonction initial de madjid
# labels = [
#     {
#         "label": "chien",
#         "description": "cette classe est utlisée lorsque la phrase en input parle d'un chien",
#     },
#     {
#         "label": "chat",
#         "description": "cette classe est utlisée lorsque la phrase en input parle d'un chat",
#     },
# ]

# construction des exemples
examples = {
    "Leur regard perçant semble contenir tous les secrets du monde.": "chat",
    "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison": "chien",
}
examples_2={"Le montant de mon remboursement est erroné et je n'arrive pas à obtenir de réponse.": 'Problème de remboursement', 
            "Je comprends pas pourquoi mon remboursement n'est pas complet, ce n'est pas ce qui était promis.": 'Problème de remboursement',
            "Je n'ai toujours pas reçu de remboursement pour l'article que j'ai renvoyé.": 'Problème de remboursement',
            'Où est mon colis ? Le suivi est bloqué depuis 3 jours.': 'Retard de livraison',
            "Le suivi n'indique pas la date de livraison estimée.": 'Retard de livraison', 
            'Où est mon colis ? Le délai de livraison est déjà dépassé.': 'Retard de livraison',
            "Je suis vraiment déçu par la qualité du tissu, il n'est pas du tout résistant.": 'Mauvaise qualité.', 
            'Le tissu est de mauvaise qualité et il perd sa forme après un lavage.': 'Mauvaise qualité.', 
            'Le tissu est très fragile et se déchire facilement. Très mauvaise qualité.': 'Mauvaise qualité.', 
            "Après avoir validé le paiement, j'ai eu un message d'erreur et la commande est annulée.": 'Problèmes de paiement et de commande',
            "Le paiement par carte est très lent et l'option PayPal ne fonctionne pas.": 'Problèmes de paiement et de commande',
            'Le site a indiqué un échec de paiement mais a quand même débité ma carte.': 'Problèmes de paiement et de commande',
            "Je n'ai pas aimé l'odeur chimique, ça gâche un peu l'expérience d'achat.": 'Odeur chimique',
            "Même après plusieurs lavages, l'odeur chimique est toujours présente.": 'Odeur chimique',
            "Les vêtements étaient imprégnés d'une forte odeur chimique qui m'a obligé à les laver avant de les porter.": 'Odeur chimique', 
            "Je suis déçu, le manteau est déjà abîmé à l'arrivée avec des fils qui dépassent.": 'Défaut de qualité', 
            "Le manteau est arrivé avec une petite déchirure près de l'épaule.": 'Défaut de qualité', 
            "J'ai trouvé une déchirure sur la doublure à l'intérieur du manteau.": 'Défaut de qualité', 
            'Le remboursement a bien été effectué, merci pour la réactivité.': 'Remboursement reçu', 
            'Le remboursement est bien arrivé, merci pour votre efficacité.': 'Remboursement reçu',
            'Remboursement bien reçu, merci pour votre réactivité.': 'Remboursement reçu', 
            'Livraison rapide et sans problème !': 'Service de livraison rapide', 
            'Service de livraison rapide et impeccable.': 'Service de livraison rapide', 
            'Livraison rapide, très contente du service.': 'Service de livraison rapide',
            'Commande arrivée incomplète, encore une fois.': 'Articles manquants', 
            'Commande reçue mais incomplète, je suis déçu.': 'Articles manquants',
            "Manque un vêtement dans ma commande, c'est frustrant.": 'Articles manquants',
            "Colis endommagé à l'arrivée, pas très sérieux.": 'Colis endommagé',
            'Mon colis est arrivé mouillé et endommagé.': 'Colis endommagé',
            "J'ai reçu mon colis avec des déchirures sur le carton.": 'Colis endommagé',
            "J'ai eu des problèmes avec les conditions de retour, ce n'était vraiment pas pratique.": 'Conditions de retour',
            "Les conditions de retour m'ont vraiment freiné dans ma décision d'achat.": 'Conditions de retour', 
            "Les conditions de retour sont trop compliquées, j'aurais aimé plus de simplicité.": 'Conditions de retour', 
            'Pull de très bonne qualité, il est à la fois chaud et léger, vraiment confortable.': 'Pull de qualité supérieure', 
            "Le pull est d'une qualité supérieure, il est vraiment chaud et agréable à porter.": 'Pull de qualité supérieure', 
            'Ce pull est vraiment de qualité supérieure, le tissu est épais et doux.': 'Pull de qualité supérieure',
            "Je n'arrive pas à recevoir mon remboursement car mon compte est expiré, que faire?": 'Problème de remboursement lié à un compte expiré', 
            "Le remboursement est refusé car mon compte est expiré. Est-ce que vous pouvez m'aider à régler ce problème?": 'Problème de remboursement lié à un compte expiré', 
            "Le remboursement n'est pas arrivé car mon compte est expiré. Que faire pour le recevoir?": 'Problème de remboursement lié à un compte expiré', 
            'Le panier ne se vide pas correctement quand je retire un produit, il faut recharger la page pour voir la modification.': 'Problèmes de mise à jour du panier',
            "Le panier ne se met pas à jour après un changement d'article, j'ai dû fermer et rouvrir la page.": 'Problèmes de mise à jour du panier',
            "Le panier ne se met pas à jour correctement, j'ai dû rafraîchir plusieurs fois pour voir mes articles.": 'Problèmes de mise à jour du panier',
            'Est-il possible de demander un remboursement sur un autre compte si le mien est expiré?': 'Remboursement compte expiré',
            'Est-ce que le remboursement peut être renvoyé vers un autre compte si le mien est expiré?': 'Remboursement compte expiré',
            'Est-ce que le remboursement peut être effectué sur un autre compte si le mien est expiré?': 'Remboursement compte expiré'}

lebels_2=[{'label': 'Problème de remboursement', 'description': 'Problème de remboursement'}, {'label': 'Retard de livraison', 'description': 'Retard de livraison'}, {'label': 'Mauvaise qualité.', 'description': 'Mauvaise qualité.'}, {'label': 'Problèmes de paiement et de commande', 'description': 'Problèmes de paiement et de commande'}, {'label': 'Odeur chimique', 'description': 'Odeur chimique'}, {'label': 'Défaut de qualité', 'description': 'Défaut de qualité'}, {'label': 'Remboursement reçu', 'description': 'Remboursement reçu'}, {'label': 'Service de livraison rapide', 'description': 'Service de livraison rapide'}, {'label': 'Articles manquants', 'description': 'Articles manquants'}, {'label': 'Colis endommagé', 'description': 'Colis endommagé'}, {'label': 'Conditions de retour', 'description': 'Conditions de retour'}, {'label': 'Pull de qualité supérieure', 'description': 'Pull de qualité supérieure'}, {'label': 'Problème de remboursement lié à un compte expiré', 'description': 'Problème de remboursement lié à un compte expiré'}, {'label': 'Problèmes de mise à jour du panier', 'description': 'Problèmes de mise à jour du panier'}, {'label': 'Remboursement compte expiré', 'description': 'Remboursement compte expiré'}, {'label': 'Autre', 'description': 'Toute autre classe non présente'}]


task = GroupedClassification(
    llm=llm, instructions=instructions, labels=lebels_2, examples=examples_2,grouped_examples=True
)

# TODO: on peut utiliser la fonction initial de madjid
# inputs = [
#     {"input": "créatures mystérieuses qui aiment se faufiler dans l'obscurité"},
#     {"input": "meilleurs amis de l'homme, toujours fidèles et pleins d'amour."},
# ]
inputs = [
    {"input": "panier"},

]

# Verification des données:
# for i in task.build_base_items(inputs):
#     print(i.format_as_example())

# - |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||: None
# - |||meilleurs amis de l'homme, toujours fidèles et pleins d'amour.|||: None


# Execution du classifier
# Paramètres:
# - batch_size: nombre de requête en // (max_concurrency)
# - group_size: nombre d'input regroupé en un message
output = task.run(inputs, batch_size=1, group_size=30)


for o in output:
    # print(o.format_as_example())
    print(o.output)
    print(o.error)
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
