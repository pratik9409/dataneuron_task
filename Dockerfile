FROM python:3.11.4

RUN mkdir /app

WORKDIR /app

ADD . /app

RUN pip install --default-timeout=3600 -r requirements.txt

CMD ["python", "app.py"]