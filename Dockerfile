FROM python:3.7-slim

#USER root
RUN apt update && apt install -y  python3-pip vim # && rm -rf /opt/miniconda

#USER dockeruser
COPY requirements.txt .
#RUN /usr/bin/python3 -m pip install Cython==0.29.36  --break-system-packages
RUN python -m pip install -r requirements.txt --break-system-packages
COPY cefr_predictor/ cefr_predictor/
COPY entrypoint.py /
COPY predict.py /
COPY fastapi_server.py /
COPY sample.txt /

EXPOSE 8080


ENTRYPOINT ["sleep", "infinity"]

# CMD streamlit run --server.port 8080 --server.enableCORS false CEFR_Predictor.py
#--server.maxUploadSize 50