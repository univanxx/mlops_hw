# Для работы с БД-шкой
from sqlalchemy import create_engine
import psycopg2
import pickle
from io import BytesIO

# Подключение к БД-шке
POSTGRES_DB = "mlops_db"
POSTGRES_USER = "bestuser"
POSTGRES_PASSWORD = "hellokek"
POSTGRES_HOST = "db"

CONNECTION_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

# Сохранение модели в БД-шку
def save_model(model, args):
    existed = False
    client = create_engine(CONNECTION_URL)
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)

    if client.execute("SELECT * FROM models_logs WHERE experiment_id = %s AND model_name = %s",(args.experiment_id, args.model_type)).fetchone() != None:
        client.execute("DELETE FROM models_logs WHERE experiment_id = %s AND model_name = %s",(args.experiment_id, args.model_type))
        existed = True

    client.execute(
        f"""
            INSERT INTO models_logs (experiment_id, model_weights, model_name)
            VALUES (%s, %s, %s)
        """,
        (args.experiment_id, psycopg2.Binary(buffer.read()), args.model_type)
    )
    client.dispose()

    if existed:
        return 'Модель обучена, но такой эксперимент уже существует!', 202
    else:
        return 'Обучение прошло успешно!', 200

# Удаление модели из БД-шки
def delete_model(args):
    client = create_engine(CONNECTION_URL)

    if client.execute("SELECT * FROM models_logs WHERE experiment_id = %s AND model_name = %s",(args.experiment_id, args.model_type)).fetchone() == None:
        return 'Модели с таким экспериментом нет!', 403
    
    client = create_engine(CONNECTION_URL)
    client.execute(
        f"""
            DELETE FROM models_logs
            WHERE experiment_id = %s AND model_name = %s
        """,
        (args.experiment_id, args.model_type)
    )
    client.dispose()

    return 'Эксперимент удалён', 200

# Загрузка модели из БД-шки
def get_model(args):
    client = create_engine(CONNECTION_URL)
    model_raw = client.execute(
    f"""SELECT model_weights FROM models_logs WHERE model_name = '{args.model_type}'
    AND experiment_id = '{args.experiment_id}';
    """).fetchone()
    client.dispose()

    if model_raw == None:
        return 403
    else:
        model = pickle.loads(model_raw[0])
    
    return model

# Получение списка всех обученных моделей
def get_all_models():
    client = create_engine(CONNECTION_URL)
    all_logs = client.execute("SELECT experiment_id, model_name FROM models_logs")
    client.dispose()
    results = {'experiment_id':[], 'model_name':[]}
    if all_logs == None:
        return 403
    else:
        for log in all_logs:  
            log = dict(log)
            results['experiment_id'].append(log['experiment_id'])
            results['model_name'].append(log['model_name'])
    return results