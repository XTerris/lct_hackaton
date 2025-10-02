FROM ubuntu:22.04

RUN mkdir /app
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip curl netcat

ADD requirements.txt /app

RUN python3 -m pip install -r requirements.txt

RUN mkdir /app/onnx
RUN mkdir /app/fine_tuned_ruRoberta_ner

ADD onnx/roberta.onnx /app/onnx
ADD fine_tuned_ruRoberta_ner_v8/* /app/fine_tuned_ruRoberta_ner
ADD ner_data_utils.py eval_onnx.py gunicorn_conf.py /app

ADD run.sh main.py /app

CMD PYTHONPATH=/app /app/run.sh
