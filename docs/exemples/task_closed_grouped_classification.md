# Tâche de classification fermée groupée

La classification fermée groupée permet de classifier plusieurs inputs (textes) en un seul message envoyé au LLM. Les messages sont classifiés à partir d'une liste de classe déjà établie par l'utilisateur.

Le fait d'envoyer plusieurs inputs en un message permet de réduire le nombre d'appels et donc les coûts.

La sortie du LLM doit être une chaine de caractère parsable. Ce qui est adopté aujourd'hui c'est une liste de json représentant les classes.


## Instructions

Les instructions dans cet exemple sont écrites dans au format _f-string_.

```
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici les animaux possibles:
- chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
- chat: cette classe est utlisée lorsque la phrase en input parle d'un chat

Tu dois absolument répondre sous forme de liste:

# Exemples:
["Leur regard perçant semble contenir tous les secrets du monde.", "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison"]
[{"class_id": null, "label": "chat", "description": null}, {"class_id": null, "label": "chien", "description": null}]

```

## Format de Réponse

La réponse à un input de classification doit être structurée sous forme de liste, où chaque élément représente la classification d'un input. Chaque classification doit indiquer clairement l'animal identifié.

### Structure de la Réponse

Chaque réponse est composée d'objets détaillant la classification de l'input spécifique, incluant le type d'animal (label) et toute autre information pertinente.

## Script

```python linenums="1" title="Classification fermée groupée"
--8<-- "examples/task_closed_grouped_classification.py"
```
