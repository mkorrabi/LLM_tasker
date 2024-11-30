# llm-tasker API

_A déplacer dans un projet distinct ultérieurement_

## Build

```bash
az login --use-device-code
az acr login -n g2sacrsp01
```

```bash
cd apis/
DOCKER_BUILDKIT=1 docker build -t llm-tasker-api .
 # current version of llm-tasker ?
docker tag llm-tasker-api g2sacrsp01.azurecr.io/llm-tasker-api:0.2.1
docker push g2sacrsp01.azurecr.io/llm-tasker-api:0.2.1
```

```bash
# local test:
docker run --rm -p 8008:8000 --env-file <path> -t llm-tasker-api
```

```bash
# Création de la webapp
az webapp create \
--resource-group rg-ia-np-we \
--plan ASP-rgianpwe-9d6f \
--name app-llm-tasker-api-np-we \
--deployment-container-image-name g2sacrsp01.azurecr.io/llm-tasker-api:0.1.0 \
--subscription G2S-nonprod \
--vnet /subscriptions/05381c1c-7b12-4287-8895-c8c956398009/resourceGroups/G2S-RG-NONPROD-NETWORK-WE/providers/Microsoft.Network/virtualNetworks/G2S-VNET-NONPROD-WE \
--subnet /subscriptions/05381c1c-7b12-4287-8895-c8c956398009/resourceGroups/G2S-RG-NONPROD-NETWORK-WE/providers/Microsoft.Network/virtualNetworks/G2S-VNET-NONPROD-WE/subnets/G2S-SNET-NONPROD-WE-004
```

```bash
# Configurer la webapp
az webapp config appsettings set \
--name app-llm-tasker-api-np-we \
--resource-group rg-ia-np-we \
--subscription G2S-nonprod \
--settings \
DOCKER_REGISTRY_SERVER_USERNAME=g2sacrsp01 \
DOCKER_REGISTRY_SERVER_PASSWORD=XXXX \
DOCKER_REGISTRY_SERVER_URL=https://g2sacrsp01.azurecr.io \
DOCKER_ENABLE_CI=true \
AZURE_APIM_KEY=XXX \
AZURE_OPENAI_API_VERSION=XXX \
AZURE_OPENAI_API_KEY=XXX \
AZURE_APIM_OPENAI_ENDPOINT=XXX \
LANGFUSE_PRIVATE_KEY=XXX \
LANGFUSE_PUBLIC_KEY=XXX \
LANGFUSE_HOST=XXX \
WEBSITES_PORT=8000 \
PORT=8000
```


## Exemple python

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
