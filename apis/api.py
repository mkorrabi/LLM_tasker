from typing import Annotated, List, Union
from dotenv import load_dotenv, find_dotenv
import os

from fastapi import Body, FastAPI, HTTPException, Response, status
from fastapi.encoders import jsonable_encoder

from langfuse.callback import CallbackHandler

from llmtasker.exceptions import LLMTaskerException
from llmtasker.items.base import T
from llmtasker.tasks.classification import (
    Classification,
    GroupedClassification,
    ClassificationAPIConfig,
    ClassItem,
    MultiClassItem,
)
from llmtasker.tasks.custom_instruction import (
    CustomInstructionAPIConfig,
    CustomPydanticItem,
    CustomInstruction,
)
from llmtasker.utils import jsonschema_to_pydantic

from apis.examples_api import examples_fastapi_classification, examples_fastapi_custom
from langchain_openai import OpenAI, ChatOpenAI
import os
import tempfile
import json
from io import open
import copy
from time import sleep

import gradio as gr
import pandas as pd

# from llmtasker.tasks.classification import Classification
# from langchain_groq import ChatGroq


import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from bertopic.representation import LangChain
from bertopic import BERTopic
import openai
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
from chat2plot import chat2plot
import os
import json 


load_dotenv(find_dotenv(), override=True)

app = FastAPI()

# TODO: ajouter une route pour montrer le prompt entier




BASE_EXEMPLE_FOR_CLASSIFICATION = {
  "inputs": {"inputs": ["test"], "indexes": [0]},
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
    "batch_size": 20,
    "multi_labels": True #(à écrire : true)
  }
}


data_exemple_for_extract_classes={
  "verbatims": [
    "La lune est magnifique ce soir.",
    "Le soleil brille tellement fort aujourd'hui.",
    "J'adore observer les cratères de la lune.",
    "Les couchers de soleil sont toujours spectaculaires.",
    "La lune éclaire la nuit d'une lumière douce.",
    "Le soleil nous réchauffe chaque matin.",
    "Les phases de la lune sont fascinantes.",
    "Rien de tel qu'un bain de soleil sur la plage.",
    "La pleine lune est hypnotisante.",
    "Le lever du soleil est un moment magique.",
    "La lune a une influence sur les marées.",
    "Le soleil est essentiel à la vie sur Terre.",
    "Les éclipses lunaires sont impressionnantes.",
    "Les rayons du soleil à travers les arbres sont magnifiques.",
    "La lune est un symbole de mystère.",
    "Le soleil peut être dangereux sans protection.",
    "Les astronautes ont marché sur la lune.",
    "Le soleil est une étoile parmi des milliards.",
    "La lune inspire de nombreux poètes.",
    "Le soleil est au centre de notre système solaire.",
    "Les nuits de pleine lune sont spéciales.",
    "Le soleil nous donne de la vitamine D.",
    "La lune est notre satellite naturel.",
    "Le soleil peut causer des coups de soleil.",
    "Les légendes sur la lune sont nombreuses.",
    "Le soleil se couche à l'ouest.",
    "La lune est visible même en plein jour parfois.",
    "Le soleil est une source d'énergie renouvelable.",
    "Les loups hurlent à la lune.",
    "Le soleil est incroyablement chaud.",
    "La lune a des mers appelées 'maria'.",
    "Le soleil est à environ 150 millions de kilomètres de la Terre.",
    "Les missions Apollo ont exploré la lune.",
    "Le soleil est composé principalement d'hydrogène et d'hélium.",
    "La lune a des montagnes et des vallées.",
    "Le soleil a des taches solaires.",
    "La lune est souvent associée à la romance.",
    "Le soleil est responsable du climat terrestre.",
    "Les cycles lunaires durent environ 29,5 jours.",
    "Le soleil est une boule de plasma en fusion.",
    "La lune est le cinquième plus grand satellite du système solaire.",
    "Le soleil émet de la lumière et de la chaleur.",
    "Les éclipses solaires sont rares et spectaculaires.",
    "La lune a une face cachée.",
    "Le soleil est en constante activité.",
    "Les marées sont influencées par la lune.",
    "Le soleil est une source de lumière naturelle.",
    "La lune est souvent représentée dans les œuvres d'art.",
    "Le soleil est vital pour la photosynthèse.",
    "Les mythes sur la lune existent dans toutes les cultures.",
    "Le soleil se lève à l'est.",
    "La lune a des phases croissantes et décroissantes.",
    "Le soleil est une étoile de type spectral G2V.",
    "La lune est un sujet populaire en astronomie.",
    "Le soleil est environ 109 fois plus grand que la Terre.",
    "La lune a été visitée par des sondes spatiales.",
    "Le soleil a une durée de vie d'environ 10 milliards d'années."
  ]
}


data_for_viz={
   "df":{
      "inputs":[
         "soleil et la lune",
         "soleil et terre ",
         "terre et lune"
      ],
      "output":[
         ["soleil", "lune"],
         ["soleil", "terre"],
         ["terre", "lune"]
      ]
     
   },
   "query":"fait moi un bart chart des planete qui sortent le plus "
}
     
def setup_models():
    # Configuration OpenAI
    chain = load_qa_chain(ChatOpenAI(model="gpt-4o-mini"), chain_type="stuff")
    representation_models = LangChain(chain, prompt="Donne moi un seul label pour ces documents, et rien d'autre")
    
    # Configuration BERTopic
    stopWords = stopwords.words('french')
    vectorizer_model = CountVectorizer(stop_words=stopWords, ngram_range=(1, 10))
    topic_model = BERTopic(vectorizer_model=vectorizer_model, representation_model=representation_models)

    return topic_model

def get_embeddings( verbatims, batch_size=100):
    """Génère les embeddings par lots avec OpenAI"""
    all_embeddings = []
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") )
    
    for i in range(0, len(verbatims), batch_size):
        batch = verbatims[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
    return np.array(all_embeddings)

def get_topic_info( topic_model):
    """Récupère les informations des topics"""
    if topic_model is None:
        return pd.DataFrame()
    
    topic_info = topic_model.get_topic_info()
    topic_info['Percentage'] = (topic_info['Count'] / topic_info['Count'].sum() * 100).round(2)
    return topic_info






def langfuse_handler_task(task_route: str):
    return CallbackHandler(
        secret_key=os.getenv("LANGFUSE_PRIVATE_KEY", default=None),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", default=None),
        host=os.getenv("LANGFUSE_HOST", default=None),
        trace_name=task_route,
    )

def get_classifier_from_parameters(
    instruct,
    classes,
    examples,
    multi_labels,
):
    

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    ##Gestion des classes ###
    list_labels=[]
    data_class_df = {}
    build_with_def: bool = False
    for classe in classes:
        cl = classe["label"]
        defn = classe["description"]
        if cl != "":
            data_class_df[cl] = defn
            if defn != "":
                build_with_def = True
                list_labels.append({"label":cl,"description":defn})

    if build_with_def:
        lab = Classification.build_labels_with_descriptions(data_class_df)
    else:
        classes = data_class_df.keys()
        lab = Classification.build_labels_from_list(classes)

    

    

    if multi_labels :
        data_class_ex = {}
        for example, cl in examples.items():
            if cl != [] and example != "":
                data_class_ex[example] = cl

        classifier=Classification(
            examples=data_class_ex,
            labels=list_labels,
            llm=llm,
            instructions=instruct,
            use_ai_user_role_for_examples=False,
            input_in_instructions=False,
            multi_labels=True,
            classification_open=False,  # default
        )

    else:

        data_class_ex = {}
        for example, cl in examples.items():
            if cl != [] and example != "":
                data_class_ex[example] = cl[0]


        classifier = Classification(
            llm=llm,
            examples=data_class_ex,
            labels=list_labels,
            instructions=instruct,
            use_system_prompt=False,
            input_in_instructions=False,
            use_ai_user_role_for_examples=False,
        )
 

    return classifier


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/classify", responses={"206": {"description": "One error item is not None."}}
)

async def classify(parameters_json: dict ):
        json_outputs = []
        classifier = get_classifier_from_parameters(
            instruct=parameters_json["config_package"]["instructions"],
            classes=parameters_json["config_package"]["classes"],
            examples=parameters_json["config_package"]["examples"],
            multi_labels=parameters_json["config_package"]["multi_labels"],
        )

        # TODO @madjid: on doit essayer d'être cohérent aveec le json de l'api
        # donc il ne faut pas que les inputs soit sous forme de liste distinctes.
        # fix temporaire (necessite que len(indexes) == len(inputs))
        # (ce qui était déjà le cas avant)
        assert len(parameters_json["inputs"]["inputs"]) == len(parameters_json["inputs"]["indexes"])
        inputs = [
            {"input": inp, "id": ix}
            for inp, ix in zip(parameters_json["inputs"]["inputs"], parameters_json["inputs"]["indexes"])
        ]

        outputs = classifier.run(
                inputs, parameters_json["config_package"]["batch_size"]
            )
        
        for output in outputs:
            if output.error is not None:
                json_outputs.append(
                    {
                        "indexe": output.id,
                        "input": str(output.input),
                        "output": "",
                        "error": str(output.error),
                    }
                )
            else:
                json_outputs.append(
                    {
                        "indexe": output.id,
                        "input": str(output.input),
                        "output":str([classe.label for classe in output.output.root] )if parameters_json["config_package"]["multi_labels"] else str([output.output]) ,
                        "error": output.error,
                    }
                )

        return json_outputs


@app.post(
    "/extract_classes", responses={"206": {"description": "One error item is not None."}}
)

async def process_excel_file(json_input: dict ):
    """Traite le fichier Excel et retourne un message de statut"""
    try:
        topic_model=setup_models()

        
        # Nettoyage des verbatims
        verbatims = json_input["verbatims"]

        
        # Création des embeddings avec OpenAI
        embeddings = get_embeddings(verbatims)

        
        # Entraînement du modèle
        topics, probs = topic_model.fit_transform(verbatims, embeddings=embeddings)
        
        # Récupération des informations sur les topics
        topic_info = get_topic_info(topic_model)

        
        return  topic_info.to_dict()
    
    except Exception as e:
        return f"Erreur lors du traitement du fichier : {str(e)}"



@app.post(
    "/viz_classes", responses={"206": {"description": "One error item is not None."}}
)


async def process_query(json_data: dict):
    
    try:
        
        df= pd.DataFrame.from_dict(json_data["df"])
        #transform df colonne output to real list
        df['output'] = [literal_eval(item) if isinstance(item, str) else item for item in df['output']]

        # Création d'un nouveau DataFrame avec duplication des lignes
        expanded_rows = []
        for _, row in df.iterrows():  # Parcourt chaque ligne
            if isinstance(row['output'], list):  # Vérifie que 'output' est une liste
                for item in row['output']:  # Parcourt chaque élément de la liste
                    new_row = row.copy()  # Copie la ligne originale
                    new_row['output'] = item  # Remplace 'output' par l'élément
                    expanded_rows.append(new_row)  # Ajoute la nouvelle ligne
            else:
                expanded_rows.append(row)  # Si 'output' n'est pas une liste, garde la ligne originale

        # Conversion de la liste en un nouveau DataFrame
        df = pd.DataFrame(expanded_rows)

        llm = ChatOpenAI(model_name="gpt-4o-mini")
        # df = pd.read_excel(r"C:\Users\korra\OneDrive\Bureau\projets\Verbatims\visualisation_test_app\Data\kmeans_sell.xlsx")

        # 2. Pass a dataframe to draw
        plot = chat2plot(df, chat=llm)
        result = plot.query(json_data["query"])
        
        return result.figure.to_json()
    except Exception as e:
        return f"Erreur: {str(e)}"


