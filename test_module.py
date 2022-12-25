import pytest
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

from model_utils import train_model, make_predictions
from db_utils import save_model, delete_model, get_all_models

# Подключение для mock-ания БД-шки для тестов 
POSTGRES_DB = "mlops_db"
POSTGRES_USER = "bestuser"
POSTGRES_PASSWORD = "hellokek"
POSTGRES_HOST = "localhost"

CONNECTION_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

client = create_engine(CONNECTION_URL)
client.execute("TRUNCATE TABLE models_logs;")
client.dispose()

# Класс замоканных аргументов
class Args():
    def __init__(self, experiment_id, model_type, model_params = "{}", file="data/train.csv"):
        self.experiment_id = experiment_id
        self.model_type = model_type
        self.model_params = model_params
        self.file=file

# Проверка того, как сохраняются уникальные модель и эксперимент
def test_saving_first():
    args = Args(0, 'SVC')
    model = train_model(args)
    _, output = save_model(model, args)
    assert output == 200

# Проверка того, как перезаписываются уникальные модель и эксперимент
def test_saving_second():
    args = Args(0, 'SVC')
    model = train_model(args)
    _, output = save_model(model, args)
    assert output == 202

# Проверка того, как реагирует сервис на то, что пытаются удалить не тот номер эксперимента, либо не ту модель
def test_delete_model_bad():
    args = Args(228, 'SVC')
    _, output_first = delete_model(args)
    args = Args(0, 'Swaaaag')
    _, output_second = delete_model(args)
    assert output_first == output_second == 403

# Проверка работы предсказаний модели
def test_predictions_good():
    args = Args(0, 'SVC')
    _, output = make_predictions(args)
    assert output == 200

# Проверка того, что будет, если параметры модели кривые
def test_predictions_bad():
    args = Args(228, 'SVC')
    _, output_first = make_predictions(args)
    args = Args(0, 'Swaaaag')
    _, output_second = make_predictions(args)
    assert output_first == output_second == 403

# Проверка того, как удаляются модель и эксперимент
def test_delete_model_good():
    args = Args(0, 'SVC')
    _, output = delete_model(args)
    assert output == 200

# Проверка на вывод всех обученных моделей при условии того, что их нет
def test_get_models_bad():
    output = get_all_models()
    assert output == 403