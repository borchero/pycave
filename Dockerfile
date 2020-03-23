FROM python:3.7

RUN pip install pylint twine

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
