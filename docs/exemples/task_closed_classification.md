# Tâche classification Fermée

Cette section explique comment utiliser le système de classification fermée pour classifier des inputs seuls ou groupés en parallèle (batch), en retournant un objet `ItemCollection`.

## Description de la Classification Fermée

La classification fermée permet de classifier des descriptions textuelles en identifiant les animaux mentionnés dans les inputs. Chaque input est traité pour déterminer quelle catégorie d'animal il décrit, en fonction d'une liste prédéfinie.

## Instructions de Classification

Les instructions pour la classification sont fournies sous format _f-string_ pour permettre une personnalisation facile en fonction des besoins de classification spécifiques.

### Exemple d'Instructions

Voici les instructions à suivre pour classifier les inputs:

```
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici les animaux possibles:
- chien: cette classe est utlisée lorsque la phrase en input parle d'un chien
- chat: cette classe est utlisée lorsque la phrase en input parle d'un chat

# Exemples:
- |||Leur regard perçant semble contenir tous les secrets du monde.|||: chat
- |||Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison|||: chien

Human: |||créatures mystérieuses qui aiment se faufiler dans l'obscurité|||
```

## Script

```python linenums="1" title="Classification fermée"
--8<-- "examples/task_closed_classification.py"
```
