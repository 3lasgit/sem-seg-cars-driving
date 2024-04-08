FROM python:3.10-slim

# Install poetry
RUN pip install poetry

# ENV POETRY_NO_INTERACTION=1 \
#     POETRY_VIRTUALENVS_IN_PROJECT=1 \
#     POETRY_VIRTUALENVS_CREATE=1 \
#     POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy and install dependencies using poetry
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy the rest of your application code
COPY cnn/task_unet_model.py ./cnn/task_unet_model.py
# Create this file to allow poetry to function 
RUN touch README.md
RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR


# Set the entrypoint for running your training script
ENTRYPOINT ["poetry", "run", "python", "-m", "cnn.task_unet_model"]