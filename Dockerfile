FROM python:3.7

RUN pip install pylint twine sphinx sphinx-rtd-theme
RUN pip install hmmlearn==0.2.3

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
