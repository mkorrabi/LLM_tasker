import os
import tempfile
import json
from io import open
import copy
from time import sleep

import gradio as gr
import pandas as pd

from llmtasker.tasks.classification import Classification
# from langchain_groq import ChatGroq

from src.interface import Interface

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


BASE_EXEMPLE = {
    "config_llm": {
        "name_model": "llama",
        "temperature": 0,
        "max_tokens": 800,
        "seed": 4009,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    },
    "config_package": {
        "instructions": "Classifiez-moi la phrase en une seule classe.\nVoici les classes :\n{classes}\n\nVoici les exemples :\n{examples}",
        "classes": [
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
        "batch_size": 20,
        "multi_labels":True
    },
}


class loging:
    def __init__(self):
        pass

    def create_greeting(self, request: gr.Request):
        return gr.Markdown(value=f"Thanks for logging in, {request.__getstate__()}")



class TopicSelectionApp:
    def __init__(self):
        self.api_key =os.getenv("OPENAI_API_KEY") 
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.verbatims = None
        self.embeddings = None
        self.client = openai.OpenAI(api_key=self.api_key)
        self.setup_models()
        
    def setup_models(self):
        # Configuration OpenAI
        chain = load_qa_chain(ChatOpenAI(model="gpt-4o-mini"), chain_type="stuff")
        representation_models = LangChain(chain, prompt="Donne moi un seul label pour ces documents, et rien d'autre")
        
        # Configuration BERTopic
        stopWords = stopwords.words('french')
        vectorizer_model = CountVectorizer(stop_words=stopWords, ngram_range=(1, 10))
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model, representation_model=representation_models)

    def get_embeddings(self, texts, batch_size=100):
        """Génère les embeddings par lots avec OpenAI"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)

    def process_excel_file(self, file):
        """Traite le fichier Excel et retourne un message de statut"""
        try:
            # Lecture du fichier Excel
            df = pd.read_excel(file.name)
            
            # Vérification de la présence d'une colonne de verbatims
            verbatim_column = "Verbatims"
            possible_column_names = ['verbatim', 'texte', 'text', 'commentaire', 'comment',"Verbatims"]
            
            for col in df.columns:
                if col.lower() in possible_column_names:
                    verbatim_column = col
                    break
            
            if verbatim_column is None:
                return "Erreur : Aucune colonne de verbatims trouvée. Veuillez utiliser un des noms suivants : verbatim, texte, text, commentaire, comment", None
            
            # Nettoyage des verbatims
            self.verbatims = df[verbatim_column].dropna().astype(str).tolist()
            
            # Création des embeddings avec OpenAI
            self.embeddings = self.get_embeddings(self.verbatims)
            
            # Entraînement du modèle
            self.topics, self.probs = self.topic_model.fit_transform(self.verbatims, embeddings=self.embeddings)
            
            # Récupération des informations sur les topics
            topic_info = self.get_topic_info()
            
            return f"Fichier traité avec succès. {len(self.verbatims)} verbatims analysés. {len(topic_info)} topics identifiés.", topic_info
        
        except Exception as e:
            return f"Erreur lors du traitement du fichier : {str(e)}", None

    def get_topic_info(self):
        """Récupère les informations des topics"""
        if self.topic_model is None:
            return pd.DataFrame()
        
        topic_info = self.topic_model.get_topic_info()
        topic_info['Percentage'] = (topic_info['Count'] / topic_info['Count'].sum() * 100).round(2)
        return topic_info


            
    def update_interface(self,file):
        status, topic_info = self.process_excel_file(file)
        if topic_info is not None:
            topic_names = [topic[0] for topic in topic_info['Representation'].tolist()] 
            print(topic_names)
            return[ gr.CheckboxGroup(choices=topic_names, visible=True, interactive=True), topic_info]
                # topic_info[['Topic', 'Name', 'Percentage']].values.tolist(), ,
            
        return [gr.CheckboxGroup(visible=True,choices=[]),topic_info]


    
    def apply_selection(self,selected):
        if not selected:
            return "Aucun topic sélectionné"
        selected_indices = [i for i, name in enumerate(self.get_topic_info()['Name']) if name in selected]
        return f"Topics sélectionnés : {', '.join(selected)}\nIndices: {selected_indices}"
            
    def convert_df_to_json_format(self,df):
        # Créer la structure des classes
        classes = []
        for _, row in df.iterrows():
            if isinstance(row['Representation'], list) and len(row['Representation']) > 0:
                class_dict = {
                    "label": row['Representation'][0] if row['Representation'][0] else "Non spécifié", #str(row['Name']),
                    "description": row['Representation'][0] if row['Representation'][0] else "Non spécifié"
                }
                classes.append(class_dict)

        autre_exists = any(class_dict["label"] == "Autre" for class_dict in classes)

        # Ajouter "Autre" seulement s'il n'existe pas déjà
        if not autre_exists:
            classes.append({"label": "Autre", "description": "Toute autre classe non présente"})
        
        
        # Créer la structure des examples
        examples = {}
        for _, row in df.iterrows():
            if isinstance(row['Representative_Docs'], list):
                for doc in row['Representative_Docs']:
                    if doc and isinstance(doc, str):
                        examples[doc] =  row['Representation'][0] if row['Representation'][0] else "Non spécifié",#str(row['Name'])
        
        # Construire le JSON final
        final_json = {
            "classes": classes,
            "examples": examples
        }
        
        return final_json
            

    def load_example_params(self, topic_info, selected_topic):
        # status, topic_info = self.process_excel_file(file)
        print(topic_info)
        topic_info["FirstElement"] = topic_info["Representation"].apply(lambda x: x[0].split(',')[0] if x else None)
        filtered_df = topic_info[topic_info["FirstElement"].isin(selected_topic)]
        extract_json=self.convert_df_to_json_format(filtered_df)
        print(extract_json)
        example_params = copy.deepcopy(BASE_EXEMPLE)
        example_params["config_package"]["classes"] = pd.DataFrame(
            extract_json["classes"],
        )

        example_params["config_package"]["classes"].columns = ["Classe", "Définition"]

        example_params["config_package"]["examples"] = pd.DataFrame(
            extract_json["examples"].items(),
            columns=["Exemple", "Classe"],
        )

        values = list(example_params["config_llm"].values()) + list(
            example_params["config_package"].values()
        )
        return values+[ gr.Accordion("Gérer les classes   ", open=True, visible=True)]

        # return interface


class ClassificationFermeeInterface(Interface):
    
    ##### Les fonctions ##########


    def classify_json(
        self,
        inputs_json,
        parameters_json,
    ):
        json_outputs = []
        classifier = self.get_classifier_from_parameters(
            name_model=parameters_json["config_llm"]["name_model"],
            temperature=parameters_json["config_llm"]["temperature"],
            max_tokens=parameters_json["config_llm"]["max_tokens"],
            seed=parameters_json["config_llm"]["seed"],
            top_p=parameters_json["config_llm"]["top_p"],
            frequency_penalty=parameters_json["config_llm"]["frequency_penalty"],
            presence_penalty=parameters_json["config_llm"]["presence_penalty"],
            instruct=parameters_json["config_package"]["instructions"],
            classes=parameters_json["config_package"]["classes"],
            examples=parameters_json["config_package"]["examples"],
            use_system_prompt=parameters_json["config_package"]["use_system_prompt"],
            input_in_instructions=parameters_json["config_package"][
                "input_in_instructions"
            ],
            use_ai_user_role_for_examples=parameters_json["config_package"][
                "use_ai_user_role_for_examples"
            ],
            batch_size=parameters_json["config_package"]["batch_size"],
            multi_labels=parameters_json["config_package"]["multi_labels"],
        )

        # TODO @madjid: on doit essayer d'être cohérent aveec le json de l'api
        # donc il ne faut pas que les inputs soit sous forme de liste distinctes.
        # fix temporaire (necessite que len(indexes) == len(inputs))
        # (ce qui était déjà le cas avant)
        assert len(inputs_json["inputs"]) == len(inputs_json["indexes"])
        inputs = [
            {"input": inp, "id": ix}
            for inp, ix in zip(inputs_json["inputs"], inputs_json["indexes"])
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

    def classify_one(
        self,
        input,
        name_model,
        temperature,
        max_tokens,
        seed,
        top_p,
        frequency_penalty,
        presence_penalty,
        instructions,
        df_classes,
        df_examples,
        use_system_prompt,
        input_in_instructions,
        use_ai_user_role_for_examples,
        batch_size,
        multi_labels
    ):
        parameters_json = self.create_dict_from_parameters(
            name_model=name_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            instruct=instructions,
            df_classes=df_classes,
            df_examples=df_examples,
            use_system_prompt=use_system_prompt,
            input_in_instructions=input_in_instructions,
            use_ai_user_role_for_examples=use_ai_user_role_for_examples,
            batch_size=batch_size,
            multi_labels=multi_labels
        )
        output = self.classify_json(
            inputs_json={"inputs": [input], "indexes": [0]},
            parameters_json=parameters_json,
        )

        return [
            output[0]["output"],
            gr.Textbox(
                label="Error",
                visible=output[0]["error"] is not None,
                value=output[0]["error"],
            ),
        ]

    
    def excel_to_json(self, file_excel):
        df = pd.read_excel(file_excel)
        
        # Liste des noms possibles pour la colonne de verbatims
        possible_column_names = ['verbatim', 'texte', 'text', 'commentaire', 'comment', 'verbatims']
        
        # Chercher la première colonne correspondante
        verbatim_column = None
        for col in df.columns:
            if col.lower() in possible_column_names:
                verbatim_column = col
                break
        
        # Vérifier si une colonne valide a été trouvée
        if verbatim_column is None:
            return {"error": "Aucune colonne de verbatim trouvée. Les noms de colonnes acceptés sont: " + ", ".join(possible_column_names)}
        
        inputs = []
        indexes = []
        
        # Utiliser enumerate pour générer les index
        for i, row in enumerate(df.itertuples()):
            verbatim_index = df.columns.get_loc(verbatim_column) + 1
            inputs.append(row[verbatim_index])
            indexes.append(i)  # Utilisation de i de enumerate au lieu de row[0]
        
        return {"inputs": inputs, "indexes": indexes}

    def json_to_excel(self, json_outputs):
        df = pd.DataFrame(json_outputs)
        global_path = os.path.expanduser(os.environ["GRADIO_TEMP_DIR"])
        new_file = os.path.join(global_path, "output.xlsx")
        df.to_excel(new_file)
        return [new_file, df]

    def classify_excel(
        self,
        file_excel,
        name_model,
        temperature,
        max_tokens,
        seed,
        top_p,
        frequency_penalty,
        presence_penalty,
        instructions,
        df_classes,
        df_examples,
        use_system_prompt,
        input_in_instructions,
        use_ai_user_role_for_examples,
        batch_size,
        multi_labels,
    ):

        inputs_json = self.excel_to_json(file_excel)
        parameters_json = self.create_dict_from_parameters(
            name_model=name_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            instruct=instructions,
            df_classes=df_classes,
            df_examples=df_examples,
            use_system_prompt=use_system_prompt,
            input_in_instructions=input_in_instructions,
            use_ai_user_role_for_examples=use_ai_user_role_for_examples,
            batch_size=batch_size,
            multi_labels=multi_labels
        )
        json_outputs = self.classify_json(
            inputs_json=inputs_json, parameters_json=parameters_json
        )

        return [
            gr.DownloadButton(
                label="Download output.xlsx",
                visible=True,
                value=self.json_to_excel(json_outputs)[0],
            ),
            gr.DataFrame(
                value=self.json_to_excel(json_outputs)[1], label="Classes", visible=True
            ),
        ]

    def download_excel_file(self):
        return [gr.UploadButton(visible=True), gr.DownloadButton(visible=False)]

    def get_classifier_from_parameters(
        self,
        name_model,
        temperature,
        max_tokens,
        seed,
        top_p,
        frequency_penalty,
        presence_penalty,
        instruct,
        classes,
        examples,
        use_system_prompt,
        input_in_instructions,
        use_ai_user_role_for_examples,
        batch_size,
        multi_labels,
    ):
        ####LLM parameter ########
        if name_model == "llama":
            model = "llama3-8b-8192"
        else:
            model = "mixtral-8x7b-32768"

        # config_azure_llm = {
        #     "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
        #     "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        #     "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        #     "streaming": False,
        #     "default_headers": {
        #         "Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_KEY"),
        #     },
        #     "default_query": {
        #         "project-name": "poc-llm-tasker"
        #     },  # TODO: ajouter le nom du projet en fonction de l'utilisateur
        #     "model_name": model,
        #     "temperature": temperature,
        #     "max_tokens": max_tokens,
        #     "model_kwargs": {
        #         "seed": seed,
        #         "top_p": top_p,
        #         "frequency_penalty": frequency_penalty,
        #         "presence_penalty": presence_penalty,
        #     },
        # }

        # llm = ChatGroq(
        #     model= model,
        #     api_key="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",
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     timeout=None,
        #     max_retries=2,

        # )
        # llm = OpenAI(model="gpt-3.5-turbo-1106",api_key=os.getenv("OPENAI_API_KEY"))
        llm = ChatOpenAI(model_name="gpt-4o-mini")

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
                use_ai_user_role_for_examples=use_ai_user_role_for_examples,
                input_in_instructions=input_in_instructions,
                multi_labels=True,
                classification_open=False,  # default
            )
           
            # classifier=GroupedClassification(
            #     examples=data_class_ex,
            #     labels=list_labels,
            #     llm=llm,
            #     instructions=instruct,
            #     use_ai_user_role_for_examples=False,
            #     multi_labels=True,
            #     classification_open=False,  # default
            #     group_size_examples=4, ###  /!\ si on augmente ce nombre le llm se pered et renvoie n'importe quoi 
            # )
            

          



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
                use_system_prompt=use_system_prompt,
                input_in_instructions=input_in_instructions,
                use_ai_user_role_for_examples=use_ai_user_role_for_examples,
            )
            # classifier=GroupedClassification(
            #     examples=data_class_ex,
            #     labels=list_labels,
            #     llm=llm,
            #     instructions=instruct,
            #     use_ai_user_role_for_examples=False,
            #     multi_labels=False,
            #     classification_open=False,  # default
            #     group_size_examples=4, ###  /!\ si on augmente ce nombre le llm se pered et renvoie n'importe quoi 
            # )

        
            
            

        return classifier

    ####checking json format#####
    def check_json_format(self, file):
        file_name, file_extension = os.path.splitext(file)
        if file_extension not in [".json", ".JSON"]:
            raise gr.Error(message="Le fichier charger n'est pas un fichier JSON")

    def init_json(self, file):
        # Specify the path to your JSON file
        self.check_json_format(file)
        try:
            # Open the JSON file
            with open(file,encoding='utf-8') as json_file:
                json_prompt = json.load(json_file,)
           

            name_model = json_prompt["config_llm"]["name_model"]
            temperature = json_prompt["config_llm"]["temperature"]
            max_tokens = json_prompt["config_llm"]["max_tokens"]
            seed = json_prompt["config_llm"]["seed"]
            top_p = json_prompt["config_llm"]["top_p"]
            frequency_penalty = json_prompt["config_llm"]["frequency_penalty"]
            presence_penalty = json_prompt["config_llm"]["presence_penalty"]
            df_classes = pd.DataFrame(
                json_prompt["config_package"]["classes"],
            )
            df_classes.columns = ["Classe", "Définition"]
            df_examples = pd.DataFrame(
                json_prompt["config_package"]["examples"].items(),
                columns=["Exemple", "Classe"],
            )
            instructions = json_prompt["config_package"]["instructions"]
            use_system_prompt = json_prompt["config_package"]["use_system_prompt"]
            input_in_instructions = json_prompt["config_package"][
                "input_in_instructions"
            ]
            use_ai_user_role_for_examples = json_prompt["config_package"][
                "use_ai_user_role_for_examples"
            ]
            batch_size = json_prompt["config_package"]["batch_size"]
            multi_labels=json_prompt["config_package"]["multi_labels"]

        except Exception:
            raise gr.Error(
                message=str(
                    "Un problème est survenu lors de la lecture du fichier JSON, merci de vérifiez que le JSON est au bon format. 😊"
                )
            )

        return [
            name_model,
            temperature,
            max_tokens,
            seed,
            top_p,
            frequency_penalty,
            presence_penalty,
            instructions,
            df_classes,
            df_examples,
            use_system_prompt,
            input_in_instructions,
            use_ai_user_role_for_examples,
            batch_size,
            multi_labels,
        ]

    def dump_parameters_into_tempfile(
        self, parameters: dict
    ) -> tempfile.NamedTemporaryFile:
        with tempfile.NamedTemporaryFile(
            dir=self.iface.GRADIO_CACHE,
            suffix=".json",
            mode="w+",
            delete=False,
            encoding="utf-8",
        ) as tfile:
            json.dump(parameters, tfile, ensure_ascii=False, indent=True)
            tfile.flush()
            return tfile.name

    def create_dict_from_parameters(
        self,
        name_model,
        temperature,
        max_tokens,
        seed,
        top_p,
        frequency_penalty,
        presence_penalty,
        instruct,
        df_classes,
        df_examples,
        use_system_prompt,
        input_in_instructions,
        use_ai_user_role_for_examples,
        batch_size,
        multi_labels,
    ):
        data_class_df = []
        for cl, defn in zip(
            df_classes["Classe"].to_list(), df_classes["Définition"].to_list()
        ):
            if cl != "":
                data_class_df.append({"label": cl, "description": defn})

        data_class_ex = {}
        for cl, defn in zip(
            df_examples["Exemple"].to_list(), df_examples["Classe"].to_list()
        ):
            if cl != "":
                data_class_ex[cl] = defn

        data_dict = {
            "config_llm": {
                "name_model": name_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            },
            "config_package": {
                "classes": data_class_df,
                "examples": data_class_ex,
                "instructions": instruct,
                "use_system_prompt": use_system_prompt,
                "input_in_instructions": input_in_instructions,
                "use_ai_user_role_for_examples": use_ai_user_role_for_examples,
                "batch_size": batch_size,
                "multi_labels":multi_labels,
            },
        }
        return data_dict

    def handle_dump_parameters(
        self,
        name_model,
        temperature,
        max_tokens,
        seed,
        top_p,
        frequency_penalty,
        presence_penalty,
        instruct,
        df_classes,
        df_examples,
        use_system_prompt,
        input_in_instructions,
        use_ai_user_role_for_examples,
        batch_size,
        multi_labels,
    ):
        data_dict = self.create_dict_from_parameters(
            name_model=name_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            instruct=instruct,
            df_classes=df_classes,
            df_examples=df_examples,
            use_system_prompt=use_system_prompt,
            input_in_instructions=input_in_instructions,
            use_ai_user_role_for_examples=use_ai_user_role_for_examples,
            batch_size=batch_size,
            multi_labels=multi_labels,
        )

        json_file_path = self.dump_parameters_into_tempfile(data_dict)

        return [
            gr.JSON(
                label="Les paramètres sauvegardés en Json",
                visible=False,
                show_label=False,
                # value=json.dumps(
                #     data_dict, ensure_ascii=False, indent=True, skipkeys=True
                # ),
                value=data_dict,
            ),
            json_file_path,
        ]

    def load_example_params(self, extract_json):
        example_params = copy.deepcopy(BASE_EXEMPLE)
        example_params["config_package"]["classes"] = pd.DataFrame(
            extract_json["classes"],
        )

        example_params["config_package"]["classes"].columns = ["Classe", "Définition"]

        example_params["config_package"]["examples"] = pd.DataFrame(
            extract_json["examples"].items(),
            columns=["Exemple", "Classe"],
        )

        values = list(example_params["config_llm"].values()) + list(
            example_params["config_package"].values()
        )
        return values

    def load_example_params_2(self):
        example_params = copy.deepcopy(BASE_EXEMPLE)
        example_params["config_package"]["use_ai_user_role_for_examples"] = True
        example_params["config_package"]["instructions"] = (
            "Classifie moi la phrase en une seule classe.\nVoici les classes: \n{classes}"
        )
        example_params["config_package"]["classes"] = pd.DataFrame(
            example_params["config_package"]["classes"],
        )
        example_params["config_package"]["classes"].columns = ["Classe", "Définition"]

        example_params["config_package"]["examples"] = pd.DataFrame(
            example_params["config_package"]["examples"].items(),
            columns=["Exemple", "Classe"],
        )

        values = list(example_params["config_llm"].values()) + list(
            example_params["config_package"].values()
        )

        return values

    ####checking#####
    def check_excel(self, file):
        file_name, file_extension = os.path.splitext(file)
        if file_extension not in [".xls", ".xlsx"]:
            raise gr.Error(message="Le fichier charger n'est pas un fichier Excel")
        
class viisualisation():
    def __init__(self):
        pass
    
    
    def process_multiclass_data(self,df):
        # Convertir les chaînes de caractères en listes
        try:
            all_classes = []
            for output in df['output']:
                # Convertir la chaîne en liste
                classes = literal_eval(output)
                all_classes.extend(classes)
            
            # Compter les occurrences de chaque classe
            class_counts = pd.Series(all_classes).value_counts()
            return class_counts
        except:
            return pd.Series({'Erreur': 0})

    def create_visualization(self,visualization_type,data):
        # Obtenir les comptages des classes
        class_counts = self.process_multiclass_data(data)
        
        if visualization_type == "Graphique en barres":
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                labels={'x': 'Classe', 'y': 'Nombre d\'occurrences'},
                title='Distribution des classes'
            )
            
        elif visualization_type == "Camembert":
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Distribution des classes'
            )
            
        elif visualization_type == "Treemap":
            fig = px.treemap(
                names=class_counts.index,
                values=class_counts.values,
                title='Distribution des classes'
            )
            
        elif visualization_type == "Heatmap des co-occurrences":
            # Créer une matrice de co-occurrences
            all_classes = sorted(list(set([item for sublist in [literal_eval(x) for x in data['output']] for item in sublist])))
            cooc_matrix = pd.DataFrame(0, index=all_classes, columns=all_classes)
            
            for output in data['output']:
                classes = literal_eval(output)
                for c1 in classes:
                    for c2 in classes:
                        cooc_matrix.loc[c1, c2] += 1
            
            fig = px.imshow(
                cooc_matrix,
                labels=dict(x="Classe", y="Classe", color="Co-occurrences"),
                title="Matrice de co-occurrences des classes"
            )
        
        return 
    

    def process_query(self, query, df):
        
        try:
            
            os.environ["OPENAI_API_KEY"] = "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"

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
            result = plot.query(query)

            return gr.Plot(value=result.figure, visible=True)
        except Exception as e:
            return f"Erreur: {str(e)}"



    ############extract 
class extract_and_classify(ClassificationFermeeInterface,TopicSelectionApp):

    def body(self):
        app = TopicSelectionApp()
        viz=viisualisation()
        log=loging()

        with gr.Blocks() as interface:
            gr.Markdown("# Analyse et Sélection des Topics")
            
            with gr.Row():
                # Zone de chargement du fichier
                file_input = gr.File(
                    label="Charger un fichier Excel",
                    file_types=[".xlsx", ".xls"]
                )
                
      
            
            # with gr.Row():
                # Tableau des topics
                # topic_df = gr.DataFrame(
                #     label="Topics identifiés",
                #     headers=['Topic', 'Name', 'Percentage'],
                #     interactive=False
                # )
            
          
                # Sélection des topics
            selected_topics = gr.CheckboxGroup(
                    choices=[],
                    label="Sélectionner les topics pour la classification",
                    visible=True

                )
            topic_info=gr.DataFrame(visible=False)

            
            # Bouton pour appliquer la sélection
            apply_btn = gr.Button("Appliquer la sélection")
            # result_text = gr.Textbox(label="Résultat de la sélection")
          
        ####GPT choice #######

    
                ### le fichier JSON #######
    
            parameters_upload_btn = gr.UploadButton(
                label="Charger fichier de paramétrage JSON",
                file_count="single",
                visible=False,
                scale=5,
                variant="primary",
            )
            load_example_params_1 = gr.Button(
                value="Charger exemple 1",
                visible=False,
                scale=1,
                variant="secondary",
            )
            load_example_params_2 = gr.Button(
                value="Charger exemple 2",
                visible=False,
                scale=1,
                variant="secondary",
            )
            parameters_text_area = gr.JSON(
                label="Les paramètres sauvegardés en Json",
                visible=False,
                show_label=False,
            )

            parameters_download_btn = gr.DownloadButton(
                label="Télécharger les paramètres sauvegardés en Json",
                visible=False,
            )

            
            name_model = gr.Radio(
                choices=["llama", "Mixtral"],
                label="Modèle",
                info="Conseil: GPT3.5 est plus rapide et moins coûteux ",
                value="GPT3.5",
                container=False,
                visible=False,
            )

            
            # Initialize the checkbox (initially hidden)
            temperature = gr.Slider(
                0,
                1,
                value=0,
                step=0.1,
                label="température",
                info="Choisissez la température du modèle",
                visible=False,
            )
            seed = gr.Number(
                visible=False,
                label="Seed",
                info="Avec la même Seed, le modèle fait de son mieux les requêtes répétées avec les mêmes paramètres et devraient renvoyer le même résultat. ",
                value=0,
                minimum=0,
            )
            max_tokens = gr.Number(
                visible=False,
                label="Max_tokens",
                info="Choisissez Le nombre maximum de jetons pouvant être générés",
                value=2000,
                minimum=1,
            )
            top_p = gr.Slider(
                0,
                1,
                value=1,
                step=0.1,
                label="top_p",
                info="0,1 signifie que seuls les jetons comprenant la masse de probabilité supérieure de 10% sont pris en compte.",
                visible=False,
            )
            frequency_penalty = gr.Slider(
                -2,
                2,
                value=0,
                step=0.1,
                label="frequency_penalty",
                info="Les valeurs positives pénalisent les nouveaux jetons en fonction de leur fréquence existante",
                visible=False,
            )
            presence_penalty = gr.Slider(
                -2,
                2,
                value=0,
                step=0.1,
                label="presence_penalty",
                info=" Les valeurs positives pénalisent les nouveaux jetons en fonction de leur apparition ou non dans le texte jusqu’à présent, ce qui augmente la probabilité que le modèle parle de nouveaux sujets.",
                visible=False,
            )

        with gr.Accordion("Gérer les classes   ", open=True, visible=False)  as crd :
            
            instructions = gr.TextArea(
                value="Classifie moi la phrase en une seule classe.\nVoici les classes: \n{classes}\n\nVoici les exemples: \n{examples}",
                container=False,
                show_label=False,
                placeholder="Classifie moi la phrase en une seule classe.\nVoici les classes: \n{classes}\n\nVoici les exemples: \n{examples}",
                info="Les instructions permettent de définir les règles que le modèle doit suivre",
                visible=False,
            )
                    # with gr.Column(scale=1):
                    #     gr.HTML(
                    #         """
                    #         <b>Informations</b>
                    #         <br/>
                    #         <ul>
                    #             <li><code>{classes}</code> permet d'injecter les classes et leur définition.</li>
                    #             <li><code>{examples}</code> permet d'injecter les exemples si <i>Utiliser le role AI pour les exemples</i> n'est pas coché.</li>
                    #         </ul>
                    #         """
                        # )

            with gr.Tab("Les classes "):
                ##### les classes #######
                df_classes = gr.DataFrame(
                    headers=["Classe", "Définition"],
                    datatype=["str", "str"],
                    row_count=5,
                    col_count=(2, "fixed"),
                    visible=True
                )

            with gr.Tab("Les exemples "):
                # ###### Les exemples ###########
                df_examples = gr.DataFrame(
                    headers=["Exemple", "Classe"],
                    datatype=["str", "str"],
                    row_count=5,
                    col_count=(2, "fixed"),
                    visible=True
                )
            
            # Initialize the checkbox (initially hidden)
            use_system_prompt = gr.Checkbox(
                visible=False, label="Système prompt", 
            )
            input_in_instructions = gr.Checkbox(
                visible=False, label="Input est dans le prompt", 
            )
            use_ai_user_role_for_examples = gr.Checkbox(
                visible=False, label="Utiliser le role AI pour les exemples", 
            )
                

    #### update if any  change #######


    
    #### classification avec un input texte ########

        with gr.Accordion("classification avec un input texte", open=False):
            with gr.Row():
                with gr.Column():
                    input = gr.Textbox(label="Input Text")
                    classify_btn = gr.Button(value="Classifier")

                with gr.Column():
                    output = gr.Textbox(label="classe")
                    Error = gr.Textbox(label="Error", visible=False)


        ###### classification avec un excel ########
        with gr.Accordion("classification avec un fichier en input ", open=False):
            file_obj = gr.File(
                label="Charger un fichier Excel File",
                file_count="single",
                file_types=[".xls", ".xlsx"],
            )


        batch_size = gr.Number(
            visible=True,
            label="Nombre de requêtes simulatanées",
            info="Choisissez le nombre de requêtes simulatanées",
            value=1,
            minimum=1,
        )
        multi_labels=gr.Checkbox(
                    visible=True, label="Multi-labels"
                )
                

        classify_file_btn = gr.Button(visible=True,value="Classifier")

        ##### les resultats #######
        df = gr.DataFrame(label="Classes", visible=False)
        download_excel_btn = gr.DownloadButton(
            label="Télécharger", visible=False
        )

        # Création de l'interface Gradio

        gr.Markdown("# Visualisation des Classes")
        
        # with gr.Row():
        #     visualization_choice = gr.Dropdown(
        #         choices=["Graphique en barres", "Camembert", "Treemap"],
        #         value="Graphique en barres",
        #         label="Choisissez le type de visualisation"
        #     )
        
        with gr.Row():
            inputs=gr.Textbox(label="Entrez votre requête")
            submite = gr.Button(visible=True,value="submite")
        outputs = gr.Plot(visible=False)

        user_ip = gr.Markdown(value="Not logged in")
        
        


        # idea: we could use session state to store values one day ...
        # order is important !
        user_parameters = [
            name_model,
            temperature,
            max_tokens,
            seed,
            top_p,
            frequency_penalty,
            presence_penalty,
            instructions,
            df_classes,
            df_examples,
            use_system_prompt,
            input_in_instructions,
            use_ai_user_role_for_examples,
            batch_size,
            multi_labels,
        ]

        interface.load(log.create_greeting, inputs=None, outputs=user_ip)
        
        

        for user_param in user_parameters:
            user_param.change(
                self.handle_dump_parameters,
                inputs=user_parameters,
                outputs=[parameters_text_area, parameters_download_btn],
            )

        #### load examples params ####
        # load_example_params_1.click(
        #     self.load_example_params_1, None, user_parameters
        # )
        load_example_params_2.click(
            self.load_example_params_2, None, user_parameters
        )

        #### Handle upload parameters #######

        parameters_upload_btn.upload(
            self.init_json, inputs=parameters_upload_btn, outputs=user_parameters
        )

        #####extract 
        file_input.change(
        fn=app.update_interface,    
        inputs=[file_input],
        outputs=[ selected_topics, topic_info])

        apply_btn.click(
        fn=app.load_example_params,
        inputs=[topic_info,selected_topics],
        outputs=user_parameters+[crd])
        

        file_input.upload(self.check_excel, inputs=file_input, outputs=None)
    
        # file_input.change(
        #     self.load_example_params, extract_json, user_parameters)
        
        
        classify_file_btn.click(
            self.classify_excel,
            inputs=[file_obj] + user_parameters,
            outputs=[download_excel_btn, df],
        )

        

        classify_btn.click(
            self.classify_one,
            inputs=[input] + user_parameters,
            outputs=[output, Error],
        )

        

        
        

        # classify_file_btn.click(
        #     self.classify_excel,
        #     inputs=[file_obj] + user_parameters,
        #     outputs=[download_excel_btn, df],
        # )
        download_excel_btn.click(
            self.download_excel_file, outputs=[file_input, download_excel_btn]
        )

        submite.click(
            fn=viz.process_query,
            inputs=[inputs,df],
            outputs=outputs
        )
