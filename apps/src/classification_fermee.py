import os
import tempfile
import json
from io import open
import copy

import gradio as gr
import pandas as pd

from llmtasker.tasks.classification import Classification
from src.interface import Interface


BASE_EXEMPLE = {
    "config_llm": {
        "name_model": "GPT3.5",
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
        "batch_size": 1,
    },
}


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
        # à voir avec dump json
        # return output
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
                        "output": str(output.output),
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
        inputs = []
        indexes = []

        for row in df.itertuples():
            inputs.append(row[1])
            indexes.append(row[0])

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
    ):
        ####LLM parameter ########
        if name_model == "GPT3.5":
            model = "gpt-35-turbo_16k_0613"
        else:
            model = "gpt-4-128k-turbo-openai-chatgpt-np-fr"

        config_azure_llm = {
            "azure_endpoint": os.getenv("AZURE_APIM_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "streaming": False,
            "default_headers": {
                "Ocp-Apim-Subscription-Key": os.getenv("AZURE_APIM_KEY"),
            },
            "default_query": {
                "project-name": "poc-llm-tasker"
            },  # TODO: ajouter le nom du projet en fonction de l'utilisateur
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model_kwargs": {
                "seed": seed,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            },
        }

        ##Gestion des classes ###
        data_class_df = {}
        build_with_def: bool = False
        for classe in classes:
            cl = classe["label"]
            defn = classe["description"]
            if cl != "":
                data_class_df[cl] = defn
                if defn != "":
                    build_with_def = True

        if build_with_def:
            lab = Classification.build_labels_with_descriptions(data_class_df)
        else:
            classes = data_class_df.keys()
            lab = Classification.build_labels_from_list(classes)

        data_class_ex = {}
        for example, cl in examples.items():
            if cl != "" and example != "":
                data_class_ex[example] = cl

        classifier = Classification(
            llm=config_azure_llm,
            examples=data_class_ex,
            labels=lab,
            instructions=instruct,
            use_system_prompt=use_system_prompt,
            input_in_instructions=input_in_instructions,
            use_ai_user_role_for_examples=use_ai_user_role_for_examples,
        )

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
            with open(file) as json_file:
                json_prompt = json.load(json_file)

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
        )

        json_file_path = self.dump_parameters_into_tempfile(data_dict)

        return [
            gr.JSON(
                label="Les paramètres sauvegardés en Json",
                visible=True,
                show_label=False,
                # value=json.dumps(
                #     data_dict, ensure_ascii=False, indent=True, skipkeys=True
                # ),
                value=data_dict,
            ),
            json_file_path,
        ]

    def load_example_params_1(self):
        example_params = copy.deepcopy(BASE_EXEMPLE)
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

    def body(self):
        ####GPT choice #######

        with gr.Tab("Classification fermée "):
            with gr.Accordion("Paramétrage JSON  ", open=False):
                ### le fichier JSON #######
                with gr.Row():
                    parameters_upload_btn = gr.UploadButton(
                        label="Charger fichier de paramétrage JSON",
                        file_count="single",
                        visible=True,
                        scale=5,
                        variant="primary",
                    )
                    load_example_params_1 = gr.Button(
                        value="Charger exemple 1",
                        visible=True,
                        scale=1,
                        variant="secondary",
                    )
                    load_example_params_2 = gr.Button(
                        value="Charger exemple 2",
                        visible=True,
                        scale=1,
                        variant="secondary",
                    )
                parameters_text_area = gr.JSON(
                    label="Les paramètres sauvegardés en Json",
                    visible=True,
                    show_label=False,
                )

                parameters_download_btn = gr.DownloadButton(
                    label="Télécharger les paramètres sauvegardés en Json",
                    visible=True,
                )

            with gr.Accordion("1. Choix du modèle ", open=True):
                name_model = gr.Radio(
                    choices=["GPT3.5", "GPT4"],
                    label="Modèle",
                    info="Conseil: GPT3.5 est plus rapide et moins coûteux ",
                    value="GPT3.5",
                    container=False,
                )

                #### Paramètres avancés ####
                with gr.Accordion("Paramètres avancés ", open=False):
                    # Initialize the checkbox (initially hidden)
                    temperature = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.1,
                        label="température",
                        info="Choisissez la température du modèle",
                    )
                    seed = gr.Number(
                        visible=True,
                        label="Seed",
                        info="Avec la même Seed, le modèle fait de son mieux les requêtes répétées avec les mêmes paramètres et devraient renvoyer le même résultat. ",
                        value=0,
                        minimum=0,
                    )
                    max_tokens = gr.Number(
                        visible=True,
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
                    )
                    frequency_penalty = gr.Slider(
                        -2,
                        2,
                        value=0,
                        step=0.1,
                        label="frequency_penalty",
                        info="Les valeurs positives pénalisent les nouveaux jetons en fonction de leur fréquence existante",
                    )
                    presence_penalty = gr.Slider(
                        -2,
                        2,
                        value=0,
                        step=0.1,
                        label="presence_penalty",
                        info=" Les valeurs positives pénalisent les nouveaux jetons en fonction de leur apparition ou non dans le texte jusqu’à présent, ce qui augmente la probabilité que le modèle parle de nouveaux sujets.",
                    )

            with gr.Accordion("2. Paramètres du prompt ", open=True):
                with gr.Tab("Instructions "):
                    #### les instructions #######
                    with gr.Row():
                        with gr.Column(scale=3):
                            instructions = gr.TextArea(
                                value="Classifie moi la phrase en une seule classe.\nVoici les classes: \n{classes}\n\nVoici les exemples: \n{examples}",
                                container=False,
                                show_label=False,
                                placeholder="Classifie moi la phrase en une seule classe.\nVoici les classes: \n{classes}\n\nVoici les exemples: \n{examples}",
                                info="Les instructions permettent de définir les règles que le modèle doit suivre",
                            )
                        with gr.Column(scale=1):
                            gr.HTML(
                                """
                                <b>Informations</b>
                                <br/>
                                <ul>
                                    <li><code>{classes}</code> permet d'injecter les classes et leur définition.</li>
                                    <li><code>{examples}</code> permet d'injecter les exemples si <i>Utiliser le role AI pour les exemples</i> n'est pas coché.</li>
                                </ul>
                                """
                            )

                with gr.Tab("Les classes "):
                    ##### les classes #######
                    df_classes = gr.DataFrame(
                        headers=["Classe", "Définition"],
                        datatype=["str", "str"],
                        row_count=5,
                        col_count=(2, "fixed"),
                    )

                with gr.Tab("Les exemples "):
                    # ###### Les exemples ###########
                    df_examples = gr.DataFrame(
                        headers=["Exemple", "Classe"],
                        datatype=["str", "str"],
                        row_count=5,
                        col_count=(2, "fixed"),
                    )
                with gr.Tab("Paramètres avancés "):
                    # Initialize the checkbox (initially hidden)
                    use_system_prompt = gr.Checkbox(
                        visible=True, label="Système prompt"
                    )
                    input_in_instructions = gr.Checkbox(
                        visible=True, label="Input est dans le prompt"
                    )
                    use_ai_user_role_for_examples = gr.Checkbox(
                        visible=True, label="Utiliser le role AI pour les exemples"
                    )

            #### update if any  change #######
            batch_size = gr.Number(
                visible=False,
                label="Nombre de requêtes simulatanées",
                info="Choisissez le nombre de requêtes simulatanées",
                value=1,
                minimum=1,
            )

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
            ]

            for user_param in user_parameters:
                user_param.change(
                    self.handle_dump_parameters,
                    inputs=user_parameters,
                    outputs=[parameters_text_area, parameters_download_btn],
                )

            #### load examples params ####
            load_example_params_1.click(
                self.load_example_params_1, None, user_parameters
            )
            load_example_params_2.click(
                self.load_example_params_2, None, user_parameters
            )

            #### Handle upload parameters #######

            parameters_upload_btn.upload(
                self.init_json, inputs=parameters_upload_btn, outputs=user_parameters
            )

            #### classification avec un input texte ########

            with gr.Accordion("classification avec un input texte", open=False):
                with gr.Row():
                    with gr.Column():
                        input = gr.Textbox(label="Input Text")
                        classify_btn = gr.Button(value="Classifier")

                    with gr.Column():
                        output = gr.Textbox(label="classe")
                        Error = gr.Textbox(label="Error", visible=False)

                classify_btn.click(
                    self.classify_one,
                    inputs=[input] + user_parameters,
                    outputs=[output, Error],
                )

            ###### classification avec un excel ########
            with gr.Accordion("classification avec un fichier en input ", open=False):
                file_obj = gr.File(
                    label="Charger un fichier Excel File",
                    file_count="single",
                    file_types=[".xls", ".xlsx"],
                )

                file_obj.upload(self.check_excel, inputs=file_obj, outputs=None)
                batch_size = gr.Number(
                    visible=True,
                    label="Nombre de requêtes simulatanées",
                    info="Choisissez le nombre de requêtes simulatanées",
                    value=1,
                    minimum=1,
                )
                classify_file_btn = gr.Button(value="Classifier")

                ##### les resultats #######
                df = gr.DataFrame(label="Classes", visible=False)
                download_excel_btn = gr.DownloadButton(
                    label="Télécharger", visible=False
                )

            classify_file_btn.click(
                self.classify_excel,
                inputs=[file_obj] + user_parameters,
                outputs=[download_excel_btn, df],
            )
            download_excel_btn.click(
                self.download_excel_file, outputs=[file_obj, download_excel_btn]
            )
