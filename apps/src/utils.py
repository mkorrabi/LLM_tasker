import os


class CredentialsError(Exception):
    pass


def load_credentials():
    if not os.environ["AUTH_CREDENTIALS"]:
        raise CredentialsError(
            "La variable d'environnement 'AUTH_CREDENTIALS' n'est pas définie ou est vide."
        )

    auth_credentials = os.getenv("AUTH_CREDENTIALS")
    if auth_credentials:
        credentials_list = auth_credentials.split(";")
    else:
        raise CredentialsError(
            "La variable d'environnement 'AUTH_CREDENTIALS' n'est pas définie ou est vide."
        )

    username_password_list = [
        tuple(creds.split(":")) for creds in credentials_list if ":" in creds
    ]

    return username_password_list
