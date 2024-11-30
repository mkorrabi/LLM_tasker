# Use the official Python image from the Docker Hub
FROM python:3.9-slim-bookworm

# prefer non root user to launch app
ARG USERNAME=appuser

# usefull if mounted volumes
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# add user bin
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# env for Azure
ENV AZURE_APIM_OPENAI_ENDPOINT=
ENV AZURE_OPENAI_API_KEY=XXX
ENV AZURE_OPENAI_API_VERSION=
ENV AZURE_APIM_KEY=

# enf for langfuse
ENV LANGFUSE_PUBLIC_KEY=
ENV LANGFUSE_PRIVATE_KEY=
ENV LANGFUSE_HOST=

# config poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory in the container
WORKDIR /code

# Create the non root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USER_UID:$USER_GID /code

# change user
USER $USERNAME

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY ./pyproject.toml /code/

# Install Poetry
RUN pip config --user set global.index-url http://registry.beluga.intra.groupama.fr/repository/pypi/simple && \
    pip config --user set install.trusted-host registry.beluga.intra.groupama.fr && \
    pip install --user --no-cache-dir poetry

# Install dependencies
RUN poetry install --no-root --only main

# Copy the rest of the application code to the working directory
COPY main.py /code/
# fix path temporary because of arbo
COPY examples_api.py /code/apis/

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
