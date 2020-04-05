FROM python:3.7

RUN pip install pylint twine sphinx sphinx-rtd-theme

# Cache PyTorch download
RUN pip install "torch>=1.4.0,<2.0.0"

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
