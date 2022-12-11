FROM python:3.8
COPY . /dockerdir
WORKDIR /dockerdir
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0
