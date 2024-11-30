# Tâche classification Fermée Multi labels

Cette section explique comment utiliser le système de classification fermée pour classifier des inputs seuls ou groupés en parallèle (batch), en retournant un objet `ItemCollection`.

## Description de la Classification Fermée Multi labels

La classification fermée multi labels permet de classifier des descriptions textuelles en identifiant les animaux mentionnés dans les inputs. Chaque input est traité pour déterminer les animaux décrits, en fonction d'une liste prédéfinie.

## Instructions de Classification

Les instructions pour la classification sont fournies sous format _f-string_ pour permettre une personnalisation facile en fonction des besoins de classification spécifiques.

### Exemple d'Instructions

Voici les instructions à suivre pour classifier les inputs:

```
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un ou plusieurs noms d'animaux.

Voici les animaux possibles:
- chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
- chat: cette classe est utlisée lorsque la phrase en input parle d'un chat

# Exemples:
- |||Leur regard perçant semble contenir tous les secrets du monde.|||: [{"class_id": null, "label": "chat", "description": null}]
- |||Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison|||: [{"class_id": null, "label": "chien", "description": null}]
- |||Ils ont quatre pattes et doivent être pucés|||: [{"class_id": null, "label": "chat", "description": null}, {"class_id": null, "label": "chien", "description": null}]

|||deux animaux qui vivent souvent avec les humains|||
```


## Script

```python linenums="1" title="Classification fermée multi labels"
--8<-- "examples/task_closed_classification_multilabels.py"
```
