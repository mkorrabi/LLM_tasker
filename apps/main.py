from src.classification_fermee_ import extract_and_classify

TITLE = "Classification  par LLM"
AUTH_MESSAGE = "Bienvenue sur l'interface du projet Classification par LLM."


if __name__ == "__main__":
    my_interface = extract_and_classify(TITLE, AUTH_MESSAGE)

    my_interface()
