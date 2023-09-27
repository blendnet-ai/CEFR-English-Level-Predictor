FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

USER root
RUN apt update && apt install -y python3-dev python3-pip nano && rm -rf /opt/miniconda

USER dockeruser
COPY requirements.txt .
RUN /usr/bin/python3 -m pip install -r requirements.txt
COPY cefr_predictor/ cefr_predictor/
COPY entrypoint.py /
COPY sample.txt /

EXPOSE 8080


ENTRYPOINT ["sleep", "infinity"]

# CMD streamlit run --server.port 8080 --server.enableCORS false CEFR_Predictor.py 
#--server.maxUploadSize 50 
