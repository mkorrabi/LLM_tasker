# Open classification

La classification ouverte permet de définir un format de sortie de type `Classe` grâce à la notation `BaseItem[str, Classe]`.
On définit alors une nouvelle classe par simplicité: `ClassItem`. Cette dernière peut surcharger les méthodes `format_output()`, `format_input()` et `format_as_example()` pour afficher l'objet dans un prompt par exemple.

On définit un nouveau parser `MyParser` qui hérite d'un parser de classe `LCClasseOutputParser`. Cela permet de retourner un objet maitrisé de type `Classe`.

## Instructions

Les instructions dans cet exemple sont écrites dans au format _f-string_.


```
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici des exemples de catégories:
{classes}

L'input à classifier est entre |||.

# Exemples:
{examples}

# input:
{input}
```

## Script

```python linenums="1" title="Classification ouverte"
--8<-- "examples/open_classification.py"
```
