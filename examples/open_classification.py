"""Script d'example d'usage avancée du package:
C'est à dire qu'on utilise pas les tâches définies.

TODO: ajouter un example avec la classe OpenClassification.

Description: on chercher à classifier un item en donnant des examples de classes
et on laisse le modèle ajouter une classe si besoin.
"""

import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from llmtasker.items.base import BaseItem, Classe, ItemCollection
from llmtasker.prompts.lc_base import LCPrompt
from llmtasker.executors.langchain import LCPromptLLMParserExecutor
from llmtasker.parsers import LCClasseParserUtils


load_dotenv(find_dotenv())


# On utilise un item spécifique à classification
# - input: une chaine de caractère
# - ouptut: un objet pydantic Classe
class ClassItem(BaseItem[str, Classe]):
    def format_as_example(self):
        """Exemple de réecriture d'un objet"""
        return f"input: {self.format_input()} | {self.format_output()}"


# Création de deux exemples:
ex1 = ClassItem(
    input="Le grand méchant loup",
    output=Classe(label="loup"),
)
ex2 = ClassItem(
    input="on monte sur son dos avec les indiens",
    output=Classe(label="cheval"),
)
ex3 = ClassItem(
    input="c'est rose et il a une chanson",
    output=Classe(label="panthère"),
)

exemples = [ex1, ex2]

# Prompt en "dur":
instructions = """
Tu es un classifier.

Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

Voici des exemples de catégories:
{classes}

L'input à classifier est entre |||.

# Exemples:
{examples}

# input:
{input}
"""

# l'utilisateur est libre de modifier cette classe
prompt = LCPrompt(
    instructions=instructions,
    use_system_prompt=True,
    input_in_instructions=True,
    use_ai_user_role_for_examples=False,
    examples=exemples,
    classes=[
        Classe(class_id="test", label="cheval", description="(cheval, poney, chevaux)"),
        Classe(
            class_id="test",
            label="chien",
            description="(chiens, chiennes, meilleur ami de l'homme)",
        ),
    ],
)

template_format = "f-string"

# On défnint un parser de Classe avec le paramètre closed=False
# afin de prendre en compte des classes que le LLM trouve tout seul
utils = LCClasseParserUtils(closed=False)
parser = utils.parser()

# exemple du prompt généré
output = prompt.generate(template_format=template_format).format(
    input=ex1.format_input()
)
print(output)

# System:
# Tu es un classifier.

# Tu dois identifier les animaux grâce à la description. Donne un nom d'animal.

# Voici des exemples de catégories:
# - cheval: (cheval, poney, chevaux)
# - chien: (chiens, chiennes, meilleur ami de l'homme)

# L'input à classifier est entre |||.

# # Exemples:
# input: |||Le grand méchant loup||| | loup
# input: |||on monte sur son dos avec les indiens||| | cheval

# # input:
# |||Le grand méchant loup|||

# Configuration azure
config_azure_llm = {
    "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "model": "gpt-35-turbo_16k_1106",
    "temperature": 0,
    "max_tokens": 800,
    "default_headers": {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_OPENAI")},
    "default_query": {"project-name": "llm-tasker"},
}
# instancier le LLM
llm = AzureChatOpenAI(**config_azure_llm)

# instancier le classifier à partir de trois éléments:
# prompt, llm et parser
# l'utilisateur est libre de modifier cette classe
executor = LCPromptLLMParserExecutor(
    prompt.generate(template_format=template_format), llm=llm, parser=parser
)

# Items à classifier
i1 = ClassItem(id="1", input="elles pondent des oeufs que les humains mangent")
i2 = ClassItem(id="2", input="Ca a des gros crocs et ca hurle !")
i3 = ClassItem(id="3", input="c'est la femme du coq")

# Collection d'item (+ simple à manipuler)
cc = ItemCollection()
cc.add(i1)
cc.add(i2)
cc.add(i3)

# Classifier par batch
executor.execute_batch(cc)

for item in cc:
    print(item.output)

# >>> poule
# >>> loup
# >>> poule

print(utils.classes)
# [Classe(class_id=None, label='poule', description=None), Classe(class_id=None, label='loup', description=None)]
