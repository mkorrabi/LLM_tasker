from typing import Optional
from pydantic import BaseModel, Field

examples_fastapi_classification = {
    "exemple_1": {
        "summary": "Classification fermée complète",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "azure_endpoint": "",
                "api_version": "",
                "api_key": "",
                "streaming": False,
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
                    {
                        "label": "neutre",
                        "description": "intentions ni positives ni négatives",
                    },
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
        },
    },
    "exemple_2": {
        "summary": "Classification fermée simple",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "gpt-35-turbo_16k_1106",
                "temperature": 0,
            },
            "config_package": {
                "instructions": "Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
                "labels": [
                    {"label": "positif", "description": "intentions positives"},
                    {"label": "negatif", "description": "intentions négatives"},
                    {
                        "label": "neutre",
                        "description": "intentions ni positives ni négatives",
                    },
                ],
                "examples": {
                    "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
                    "J'ai rencontré un nouvel ami, c'est génial de développer de nouvelles relations.": "positif",
                    "Il pleut encore, ça gâche complètement ma journée.": "negatif",
                    "Mon ordinateur a planté et j'ai perdu tout mon travail, quelle catastrophe !": "negatif",
                },
                "batch_size": 1,
            },
            "inputs": [{"input": "je suis heureux"}],
        },
    },
    "exemple_3": {
        "summary": "Classification fermée groupée",
        "description": "Exemple de requête pour classifier plusieurs inputs dans un message. Coûts réduits.",
        "value": {
            "config_llm": {
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "gpt-35-turbo_16k_1106",
                "temperature": 0,
            },
            "config_package": {
                "instructions": "Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
                "labels": [
                    {"label": "positif", "description": "intentions positives"},
                    {"label": "negatif", "description": "intentions négatives"},
                    {
                        "label": "neutre",
                        "description": "intentions ni positives ni négatives",
                    },
                ],
                "examples": {
                    "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
                    "J'ai rencontré un nouvel ami, c'est génial de développer de nouvelles relations.": "positif",
                    "Il pleut encore, ça gâche complètement ma journée.": "negatif",
                    "Mon ordinateur a planté et j'ai perdu tout mon travail, quelle catastrophe !": "negatif",
                },
                "batch_size": 1,
                "group_size": 5,
                "grouped_classification": True,
            },
            "inputs": [
                {"input": "je suis heureux"},
                {"input": "je suis malheureux"},
                {"input": "je en colère"},
                {"input": "je n'ai pas d'avis"},
            ],
        },
    },
    "exemple_4": {
        "summary": "Classification simple ouverte",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "gpt-35-turbo_16k_1106",
                "temperature": 0,
            },
            "config_package": {
                "instructions": "Classifiez-moi l'intention de la phrase.\nVoici des exemple d'intentions :\n{classes}\n\nVoici des exemples :\n{examples}",
                "labels": [
                    {"label": "positif", "description": "intentions positives"},
                ],
                "examples": {
                    "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
                },
                "batch_size": 1,
                "classification_open": True,
            },
            "inputs": [
                {"input": "je suis malheureux"},
                {"input": "je veux parler à une vraie personne, pas à un SVI !"},
            ],
        },
    },
    "exemple_5": {
        "summary": "Classification simple fermée avec erreur de classe",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "gpt-35-turbo_16k_1106",
                "temperature": 0,
            },
            "config_package": {
                "instructions": "Classifiez-moi l'intention de la phrase.\nVoici des exemple d'intentions :\n{classes}\n\nVoici des exemples :\n{examples}",
                "labels": [
                    {"label": "positif", "description": "intentions positives"},
                ],
                "examples": {
                    "Le soleil brille aujourd'hui, c'est une journée magnifique !": "positif",
                },
                "batch_size": 1,
                "classification_open": False,
            },
            "inputs": [
                {"input": "je suis malheureux"},
                {"input": "je veux parler à une vraie personne, pas à un SVI !"},
            ],
        },
    },
    "exemple_6": {
        "summary": "Classification fermée complète avec LLM interne",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "openai_api_base": "http://10.203.14.15:80/v1",
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "cognitivecomputations/dolphin-2.6-mistral-7b",
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
                    {
                        "label": "neutre",
                        "description": "intentions ni positives ni négatives",
                    },
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
        },
    },
    "exemple_7": {
        "summary": "Classification fermée complète",
        "description": "Exemple de requête pour classifier un ou des inputs en labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "azure_endpoint": "",
                "api_version": "",
                "api_key": "",
                "streaming": False,
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
                    {
                        "label": "neutre",
                        "description": "intentions ni positives ni négatives",
                    },
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
        },
    },
    "exemple_8": {
        "summary": "Classification fermée multi labels",
        "description": "Exemple de requête pour classifier un ou des inputs avec plusieurs labels. Le modèle utilisé est celui configuré par l'API.",
        "value": {
            "config_llm": {
                "default_query": {"project-name": "poc-llm-tasker"},
                "model_name": "gpt-35-turbo_16k_1106",
                "temperature": 0,
            },
            "config_package": {
                "instructions": "Classifiez-moi le type de l'oeuvre en une ou plusieurs classes.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
                "labels": [
                    {
                        "label": "Serie",
                        "description": "oeuvre sous la forme d'une série d'épisodes.",
                    },
                    {
                        "label": "Film",
                        "description": "oeuvre sous la forme d'un film, long métrage.",
                    },
                    {
                        "label": "NSP",
                        "description": "je ne sais pas",
                    },
                ],
                "examples": {
                    "Stranger Things": ["Serie"],
                    "Breaking Bad": ["Serie"],
                    "The Crown": ["Serie"],
                    "The Dark Knight": ["Film"],
                    "Stargate": ["Film", "Serie"],
                    "Limitless": ["Film", "Serie"],
                },
                "multi_labels": True,
                "batch_size": 1,
            },
            "inputs": [
                {"input": "star wars"},
                {"input": "dexter"},
                {"input": "Harry potter"},
            ],
        },
    },
}


class Adresse(BaseModel):
    """Représente une adresse postale"""

    ville: str = Field(description="ville")
    code_postale: Optional[int] = Field(description="code postale", default=None)
    rue: Optional[str] = Field(description="nom de la rue avec le numéro", default=None)


addrs1 = Adresse(ville="Paris", code_postale=75000)
addrs2 = Adresse(ville="Besançon", code_postale=25000, rue="22 grande rue")
examples_addrs_parsing = [
    {"input": "J'habite à Paris c'est dans le 75", "output": addrs1.model_dump()},
    {
        "input": "C'est à Besançon proche de la marie, au 22 de la grande rue. Code postal: 25000",
        "output": addrs2.model_dump(),
    },
]

examples_fastapi_custom = {
    "exemple_1": {
        "summary": "Retourne un objet custom grâce au mode json",
        "description": "Exemple de requête pour envoyer un shema json à suivre en utilisant le mode_json.",
        "value": {
            "config_llm": {
                "azure_endpoint": "",
                "api_version": "",
                "api_key": "",
                "streaming": False,
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
                "use_system_prompt": False,
                "input_in_instructions": False,
                "use_ai_user_role_for_examples": True,
                "batch_size": 1,
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
        },
    },
    "exemple_2": {
        "summary": "Retourne un objet custom grâce au function calling",
        "description": "Exemple de requête pour envoyer un shema json à suivre en utilisant le function calling.",
        "value": {
            "config_llm": {
                "azure_endpoint": "",
                "api_version": "",
                "api_key": "",
                "streaming": False,
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
                "method": "function_calling",
                "examples": examples_addrs_parsing,
                "use_system_prompt": False,
                "input_in_instructions": False,
                "use_ai_user_role_for_examples": True,
                "batch_size": 1,
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
        },
    },
    "exemple_3": {
        "summary": "Retourne un dict custom grâce au json mode",
        "description": "Exemple de requête à suivre en utilisant le function calling. Pas de validation de l'objet retourné côté API.",
        "value": {
            "config_llm": {
                "azure_endpoint": "",
                "api_version": "",
                "api_key": "",
                "streaming": False,
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
                "instructions": "Extraits les informations de l'adresse. tu DOIS TOUJOURS répondre au format JSON et RIEN d'autres!",
                "method": "json_mode",
                "examples": examples_addrs_parsing,
                "use_system_prompt": False,
                "input_in_instructions": False,
                "use_ai_user_role_for_examples": True,
                "batch_size": 1,
            },
            "inputs": [
                {
                    "input": "j'habite rue de la fayette, dans un petit appartement proche de la gare de Nîmes"
                },
                {
                    "input": "Elle va au travail dans une entreprise à Niort, proche de grande place."
                },
            ],
        },
    },
}
