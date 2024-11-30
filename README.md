# llm-tasker

> Package IAF pour exécuter des tâches avec des LLMs.

👓📘 [Documentation](http://ia-factory.git-pages.intra.groupama.fr/datalab/iaf-packages/llm-tasker/nextrelease/) (à rédiger)

## Installation

Depuis le repo gitlab:

```bash
# via pip (ssh)
pip install git+git@git.ra1.intra.groupama.fr:ia-factory/datalab/iaf-packages/llm-tasker.git
```

Depuis pypi groupama:
_à renseigner_

```bash
pip install llm-tasker
```

## Exemple d'usage

```python
from langchain_openai.chat_models import AzureChatOpenAI
from llmtasker.tasks.classification import Classification

# config LLM
llm = AzureChatOpenAI(...)

instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici les animaux possibles:
{classes}

# Exemples:
{examples}
"""

examples = {
    "Leur regard perçant semble contenir tous les secrets du monde.": "chat",
    "Leurs queues remuent de joie quand ils voient leur maître rentrer à la maison": "chien",
}

inputs = [
    "créatures mystérieuses qui aiment se faufiler dans l'obscurité",
    "meilleurs amis de l'homme, toujours fidèles et pleins d'amour.",
]

# création de la tâche:
task = Classification(
    llm=llm,
    instructions=instructions,
    examples=examples,
    labels=Classification.build_labels_from_list(["chat", "chien", "autre"])
)

# Execution
outputs = task.run(inputs=inputs)

# Affichage des résultats
for item in outputs:
    if item.error:
        print(item.error)
    if item.output:
        print(item.output)
    else:
        print(item.raw)

```


## Concepts

_à détailler avec le code associé_

Le package est construit autour de plusieurs concepts:

➡️ Le LLM: modèle de langue qui permet de traiter du texte de façon "intelligente".

➡️ Un prompt: Les insutrctions au LLMs qui corresponde à la tâche souhaitée. Le texte en entrée (input) est injecter dans le prompt. La gestion du prompt dans le package permet de gérer les différents rôles des exemples et de l'input.

➡️ Un executeur: c'est un suite logique d'opération autour du LLM pour réaliser une tâche. Un executeur simple consiste à enchaine un prompt, un llm et un parser.

➡️ Un item: concept qui représente une entrée (input) et une sortie (output). L'output représente le résultat de l'executeur. L'item comporte aussi le résultat du llm (raw) et les éventuelles erreurs durant l'execution (error). L'item permet de gérer la façon dont il s'affiche dans le prompt comme exemple ou input.

➡️ Une tâche: elle correspond à une action métier sur du texte (résumé, classification, personnalisée...).


## Tâches

_à détailler avec les noms des classes_

Les tâches sont représentées par des classes prêtes à l'emploi. Par exemple: `from llmtasker.task.classification import Classification`.

- ✅ Classification (input: text, output: un label)

    * Ouverte / Fermée
    * Mono / Multi labels
    * Groupée (plusieurs inputs en un message)

- ✅ Instruction personnalisée (input: text, output: un dictionnaire ou un objet pydantic)

    * mode json
    * function calling


## Fonctionnalités

- ➡️ implémentation langchain
    - ☑️ retry de l'appel llm
    - ☑️ gestion des erreurs (error)
    - ☑️ recupération du résultat du llm (raw)
    - ☑️ parser pydantic/json/classe (output)
    - ☑️ gestion du prompt:
      - ☑️ gestion des variables
      - ☑️ gestion des exemples
      - ☑️ gestion des classes
      - ☑️ choix du rôle pour les examples et l'input
    - ☑️ grouper les inputs en un message au llm
- ☑️ Interface visuel pour la classification
- ☑️ Gitlab-ci (tests, formatter, docs, version, release)
- ☑️ API classification
- ☑️ API instruction personnalisée


## Dev du package

### Abstraction

Gestion de la structure de base du package (classes abstraites et interfaces) sans prendre en compte une implémentation particulière. Pas forcément beaucoup de code. Possibilité d'implémenter avec un client différent de langchain par exemple.


### Implémentation langchain des tâches

Définie une tâche spécifique pour un usage spécifique. Censer être simple à utiliser.

### Implémentation avancée avec langchain

Coeur du package. Permet beaucoup de flexibilité mais n'est pas simple pour l'utilisateur.


## Contribuer

Prérequis:
* installer `poetry`

Puis,
1. Clone le code
2. (optionnel) Créer un environnement virtuel
3. Installer les dépendances de dev: `poetry install --with dev`
4. Installer les precommits: `pre-commit install`

Si besoin d'ajouter un package de dev: `poetry add <nom_package> --group dev`

Sinon: `poetry add <nom_package>`

Nous utilisons les grandes lignes du [gitlab flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html):

* une branche `main` qui suit les évolutions
* des branches de features (`feature/ma-feat`)
* des branches de release pour mettre à jour seulements les infos de release et le tag `release/vX.X.X`
  * Elle doit être d'une durée de vie courte
  * Si des bugfix doivent être ajoutés, c'est possible dans la branche de release
* des tags protégés sous la forme `vX.X.X` qui déclencheront une release sur le pypi interne + documentation


## TODO

* Documenter les tâches
* Développer les tests unitaires des items
* Développer les tests unitaires des tâches
* Factoriser + le code API
* Ajouter un exemple de tâche classification ouverte
* Ajouter des prompts génériques dans `prompts/`
* Détailler la documentation générer par mkdocs
* Développer la fonction run() des tâches en asynchrone
* Ajouter callback langfuse en exemple + API
* Etudier et utiliser [json_repair](https://github.com/mangiucugna/json_repair)
