FROM python:3.7-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY cefr_predictor/ cefr_predictor/
COPY entrypoint.py /

EXPOSE 8080

CMD ["sleep", "infinity"]

# CMD streamlit run --server.port 8080 --server.enableCORS false CEFR_Predictor.py 
#--server.maxUploadSize 50 
