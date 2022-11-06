import json
import ast
# Модели - 3 класса
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def load_model(params, name):
    # Преобразую параметры в формат json
    model_params = json.loads(json.dumps(ast.literal_eval(params)))
    # Загружаю модель с параметрами, исходя из названия
    try:
        if 'SVC' in name:
            return SVC(**model_params)
        elif 'GradientBoostingClassifier' in name:
            return GradientBoostingClassifier(**model_params)
        elif 'LogisticRegression' in name:
            return LogisticRegression(**model_params)
    except TypeError:
        return 0
