FROM python:3.8.9

RUN pip install poetry==1.1.6 \
    && poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock /app/

WORKDIR /app
RUN poetry install --no-root --no-interaction --no-ansi
