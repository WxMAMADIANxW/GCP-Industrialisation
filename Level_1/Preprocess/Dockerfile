FROM python:3.7-slim

USER root

WORKDIR /Preprocess

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "run.py" ]