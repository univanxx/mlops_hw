FROM python:3.8
COPY . /flask_app
WORKDIR /flask_app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0
