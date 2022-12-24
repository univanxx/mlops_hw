import json
import ast
# Модели - 3 класса
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# Stuff для обучения модели
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
# Работа с данными
import pandas as pd
from db_stuff import get_model


def load_model(params, name):
    # Преобразую параметры в формат json
    model_params = json.loads(json.dumps(ast.literal_eval(params)))
    # Загружаю модель с параметрами, исходя из названия
    if name == 'SVC':
        return SVC(**model_params)
    elif name == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(**model_params)
    elif name == 'LogisticRegression':
        return LogisticRegression(**model_params)

# Удаляем NaN-ы из данных
def prepare_data(df, for_train=True):
    columns = ['Age', 'Embarked', 'Pclass', 'Sex']
    df = df.dropna()
    if for_train:
        return df[columns], df['Survived']
    else:
        return df[columns]

# Обучение модели
def train_model(args):
    base_model = load_model(args.model_params, args.model_type)
    if base_model == 0:
        return 403
    # Пайплайн с преобразованием данных
    model = make_pipeline(
        ColumnTransformer([
            ('ohe', OneHotEncoder(), ['Pclass', 'Embarked']),
            ('binarizer', OrdinalEncoder(), ['Sex'])
        ],
            remainder='passthrough'),
        base_model
    )
    # Чтение данных и предобработка
    try:
        data = pd.read_csv(args.file)
    except FileNotFoundError:
        return 400
    train_data, train_target = prepare_data(data)
    # Фит модели и сохранение
    model.fit(train_data, train_target)
    return model

# Инференс модели
def make_predictions(args):
    inference_data = pd.read_csv(args.file)
    inference_data = prepare_data(inference_data, for_train=False)
    # Загрузка модели
    model = get_model(args)
    if model == 403:
        return 'Модели с таким экспериментом нет!', 403
    # Предсказания
    preds = model.predict(inference_data)
    return {'Предсказания модели': preds.tolist()}, 200