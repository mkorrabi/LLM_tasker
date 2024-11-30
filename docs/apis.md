# APIs llm-tasker

_à compléter_

## API classification

Cette documentation détaille l'utilisation de l'API de classification fermée qui permet d'envoyer des paramètres via JSON et de recevoir la classification sous forme de JSON.

L'API permet de réaliser une classification en envoyant un JSON contenant les paramètres nécessaires et retourne un JSON avec les résultats de la classification.

### Exemple d'appel via client

```python
import httpx

# Define the URL (local)
# url = "http://digsflrd40.dig.intra.groupama.fr:8008/classify"
# Define the URL (azure webapp)
url = "https://app-llm-tasker-api-np-we.azurewebsites.net/classify"


# Define the headers
headers = {"accept": "application/json", "Content-Type": "application/json"}

# Define the payload
payload = {
    "config_llm": {
        "azure_endpoint": "",
        "api_version": "",
        "api_key": "",
        "default_headers": {"Ocp-Apim-Subscription-Key": ""},
        "default_query": {"project-name": "poc-llm-tasker"},
        "model_name": "gpt-35-turbo_16k_1106",
        "temperature": 0,
        "max_tokens": 2000,
        "model_kwargs": {
            "seed": 200,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
    },
    "config_package": {
        "instructions": "Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
        "labels": [
            {"label": "positif", "description": "intentions positives"},
            {"label": "negatif", "description": "intentions négatives"},
            {"label": "neutre", "description": "intentions ni positives ni négatives"},
        ],
        "examples": {
            "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
            "J'ai rencontré un nouvel ami, c'est génial de développer de nouvelles relations.": "positif",
            "Il pleut encore, ça gâche complètement ma journée.": "negatif",
            "Mon ordinateur a planté et j'ai perdu tout mon travail, quelle catastrophe !": "negatif",
        },
        "use_system_prompt": False,
        "input_in_instructions": False,
        "use_ai_user_role_for_examples": False,
        "batch_size": 1,
    },
    "inputs": [{"input": "je suis heureux"}],
}


# With custom LLM using OpenAI specification;
# payload["config_llm"] = {
#     "openai_api_base": "http://10.203.14.15:80/v1",
#     "default_query": {"project-name": "poc-llm-tasker"},
#     "model_name": "cognitivecomputations/dolphin-2.6-mistral-7b",
#     "temperature": 0,
#     "max_tokens": 2000,
#     "model_kwargs": {
#         "seed": 200,
#         "top_p": 1,
#         "frequency_penalty": 0,
#         "presence_penalty": 0,
#     },
# }

# Make the POST request
response = httpx.post(url, headers=headers, json=payload)

# Print the response
print(response.json())
```


### Exemple d'input

Voici un exemple de JSON à envoyer pour effectuer une classification :

```json
{
    "config_llm": {
        "azure_endpoint": "",
        "api_version": "",
        "api_key": "",
        "streaming": false,
        "default_headers": {"Ocp-Apim-Subscription-Key": ""},
        "default_query": {"project-name": "poc-llm-tasker"},
        "model_name": "gpt-35-turbo_16k_0613",
        "temperature": 0,
        "max_tokens": 2000,
        "model_kwargs": {
            "seed": 200,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
    },
    "config_package": {
        "instructions": "Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
        "classes": [
            {"label": "positif", "description": "intentions positives"},
            {"label": "negatif", "description": "intentions négatives"},
            {"label": "neutre", "description": "intentions ni positives ni négatives"}
        ],
        "examples": {
            "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
            "J'ai rencontré un nouvel ami, c'est génial de développer de nouvelles relations.": "positif",
            "Il pleut encore, ça gâche complètement ma journée.": "negatif",
            "Mon ordinateur a planté et j'ai perdu tout mon travail, quelle catastrophe !": "negatif"
        },
        "use_system_prompt": false,
        "input_in_instructions": false,
        "use_ai_user_role_for_examples": false,
        "batch_size": 1
    },
    "inputs": [{"input": "je suis heureux"}]
}
```

### Exemple d'output

Voici un exemple de JSON que vous recevrez en réponse après la classification :

```json
[
    {
        "id": null,
        "input": "je suis heureux",
        "output": {
            "class_id": null,
            "label": "positif",
            "description": "intentions positives"
        },
        "error": null,
        "raw": {}
    }
]
```


## API instruction personnalisée

paramètres:

* Paramètre `method`:

    * `json_mode`: le paramètre `json_shema` est fortement recommandé mais ce n'est pas obligatoire. Si le `json_shema` n'est pas renseigné, la variable `json_shema_output` du prompt doit être retiré.
    * `function_calling` : le paramètre `json_shema` doit être spécifié. Possible sur les modèles openai.


### Exemple d'appel via client

```python
from typing import Optional
import httpx
from pydantic import BaseModel, Field
from msal import ConfidentialClientApplication

# Paramètres de l'application Azure AD
client_id = "118ac1df-09f0-4cf8-b77d-04fc295b87d6"
client_secret = "4c58Q~cHmrS6zUsU889K9ID4dC8aAWC76ml8ocr8"
tenant_id = "b0558a4e-2a71-4b10-a717-8750998ee43c"
scope = ["api://118ac1df-09f0-4cf8-b77d-04fc295b87d6/.default"]

# Créer une application confidentielle
app = ConfidentialClientApplication(
    client_id,
    authority=f"https://login.microsoftonline.com/{tenant_id}",
    client_credential=client_secret,
)

# Récupération des access token
result = app.acquire_token_for_client(scopes=scope)


# Specific output class:
class Adresse(BaseModel):
    """Représente une adresse postale"""

    ville: str = Field(description="ville")
    code_postale: Optional[int] = Field(description="code postale", default=None)
    rue: Optional[str] = Field(description="nom de la rue avec le numéro", default=None)


# Build examples in json format
addrs1 = Adresse(ville="Paris", code_postale=75000)
addrs2 = Adresse(ville="Besançon", code_postale=25000, rue="22 grande rue")
examples_addrs_parsing = [
    {"input": "J'habite à Paris c'est dans le 75", "output": addrs1.model_dump()},
    {
        "input": "C'est à Besançon proche de la marie, au 22 de la grande rue. Code postal: 25000",
        "output": addrs2.model_dump(),
    },
]


# Define the URL (local)
# url = "http://digsflrd40.dig.intra.groupama.fr:8008/custom"
# Define the URL (azure webapp)
url = "https://app-llm-tasker-api-np-we.azurewebsites.net/custom"

# Define the headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {result['access_token']}",
}

# Define the payload
payload = {
    "config_llm": {
        "azure_endpoint": "",
        "api_version": "",
        "api_key": "",
        "default_headers": {"Ocp-Apim-Subscription-Key": ""},
        "default_query": {"project-name": "poc-llm-tasker"},
        "model_name": "gpt-35-turbo_16k_1106",
        "temperature": 0,
        "max_tokens": 2000,
        "model_kwargs": {
            "seed": 200,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
    },
    "config_package": {
        "instructions": "Extraits les informations de l'adresse. tu DOIS TOUJOURS répondre au format JSON et RIEN d'autres! Avec le shema suivant: \n\n {json_shema_output}.",
        "examples": examples_addrs_parsing,
        # "method": "json_mode",
        "method": "function_calling",
        "use_system_prompt": False,
        "input_in_instructions": False,
        "use_ai_user_role_for_examples": True,
        "batch_size": 5,
        "json_shema": Adresse.model_json_schema(),
    },
    "inputs": [
        {
            "input": "j'habite rue de la fayette, dans un petit appartement proche de la gare de Nîmes"
        },
        {
            "input": "Elle va au travail dans une entreprise à Niort, proche de grande place."
        },
    ],
}


# With custom LLM using OpenAI specification;
# payload["config_llm"] = {
#     "openai_api_base": "http://10.203.14.15:80/v1",
#     "default_query": {"project-name": "poc-llm-tasker"},
#     "model_name": "TheBloke/dolphin-2.6-mixtral-8x7b-GPTQ",
#     "temperature": 0,
#     "max_tokens": 500,
#     "model_kwargs": {
#         "seed": 200,
#         "top_p": 1,
#         "frequency_penalty": 0,
#         "presence_penalty": 0,
#     },
# }

# Make the POST request
response = httpx.post(
    url,
    headers=headers,
    json=payload,
    timeout=300,
)

# Print the response
print(response.json())
```
