FROM python:3.11-slim

WORKDIR /app

COPY app.py .

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY clean_text.py .

COPY model.pkl .

COPY tfidf.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0" , '--port' , "8000"]
