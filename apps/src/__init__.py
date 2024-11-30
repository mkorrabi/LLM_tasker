"""Chargement des variables d'environnements avant l'import de gradio"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)
