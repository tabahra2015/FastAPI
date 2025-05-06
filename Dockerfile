FROM python:3.10-slim

RUN apt-get update && apt-get install -y libsndfile1
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
