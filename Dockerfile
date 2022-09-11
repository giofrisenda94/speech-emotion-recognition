#FROM python:3.8.12-bullseye
FROM tensorflow/tensorflow:2.9.1
COPY functions functions
COPY requirements_silicon.txt requirements.txt
COPY model_final.h5 model_final.h5
RUN pip install -r requirements.txt
RUN pip install gcloud
CMD uvicorn functions.api.api_interface:app --host 0.0.0.0 --port $PORT
