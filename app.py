# Основные библиотеки
import pandas as pd
import os
import pickle
import json
import ast

# Для работы с данными и моделью
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from model_loading import load_model
# Для построения REST-API
from flask import Flask, jsonify
from flask_restx import Api, Resource

app = Flask(__name__)
api = Api(app)

# Список возможных моделей
models = {

    "SVC": {
        "tutorial": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
    },

    "GradientBoostingClassifier": {
        "tutorial": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"
    },

    "LogisticRegression": {
        "tutorial": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    },
}


# Удаляем NaN-ы из данных
def prepare_data(df, for_train=True):
    columns = ['Age', 'Embarked', 'Pclass', 'Sex']
    df = df.dropna()
    if for_train:
        return df[columns], df['Survived']
    else:
        return df[columns]


# Парсер и класс для обучения модели
model_train = api.parser()
model_train.add_argument('filename', type='str', default='train.csv', required=True)
model_train.add_argument('experiment_id',
                         type=str, default="0", required=False, location='args')
model_train.add_argument('model_type',
                         required=True,
                         choices=['SVC', 'GradientBoostingClassifier', 'LogisticRegression'],
                         location='args'
                         )
model_train.add_argument('model_params',
                         required=True,
                         location='json'
                         )


@api.route('/model_train', methods=['PUT'], doc={'description': 'Запустить обучение выбранной модели на датасете Titanic'})
@api.expect(model_train)
class ModelTrain(Resource):
    # Параметры
    @api.doc(params={'filename': f'Название файла в формате CSV с полями Age, Embarked, Pclass, Sex, Survived'})
    @api.doc(params={'experiment_id': f'Номер эксперимента'})
    # @api.doc(params={'model_params': f'Параметры модели'})
    @api.doc(params={'model_type': f'Тип модели'})
    # Результаты выполнения
    @api.doc(responses={403: 'Модель с такими параметрами не определена!'})
    @api.doc(responses={200: 'Обучение прошло успешно!'})
    @api.doc(responses={202: 'Модель обучена, но такой эксперимент уже существует!'})
    def put(self):

        args = model_train.parse_args()
        # Загрузка модели по её параметрам
        base_model = load_model(args.model_params, args.model_type)
        print(base_model)
        if base_model == 0:
            api.abort(403, message="Модель с такими параметрами не определена!")

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
        data = pd.read_csv(args.file)
        X, y = prepare_data(data)
        # Фит модели и сохранение
        model.fit(X, y)
        # Сохранение результата
        save_filename = "train_results/" + args.model_type + "_" + args.experiment_id + ".pkl"
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        # Смотрим, проводилось ли ранее обучение по заданному номеру эксперимента
        if os.path.isfile(save_filename):
            pickle.dump(model, open(save_filename, 'wb'))
            return 'Модель обучена, но такой эксперимент уже существует!', 202
        else:
            pickle.dump(model, open(save_filename, 'wb'))
            return 'Обучение прошло успешно!', 200


# Класс для получения списка моделей
@api.route('/show_models', methods=['GET'], doc={'description': 'Получить список возможных моделей'})
class GetModel(Resource):
    # Смотрим список всех возможных типов моделей
    def get(self):
        return jsonify(models)


# Парсер и класс для удаления экспериментов
model_delete = api.parser()
model_delete.add_argument('experiment_id',
                          type=str, default="0", required=False, location='args')
model_delete.add_argument('model_type',
                          required=True,
                          choices=['SVC', 'GradientBoostingClassifier', 'LogisticRegression'],
                          location='args'
                          )


@api.route('/model_delete', methods=['DELETE'], doc={'description': 'Удалить эксперимент'})
@api.expect(model_delete)
class DeleteModel(Resource):
    # Удаляем эскперимент
    @api.doc(params={'experiment_id': f'Номер эксперимента для удаления'})
    @api.doc(params={'model_type': f'Название модели'})
    @api.doc(responses={403: 'Модели с таким экспериментом нет!'})
    @api.doc(responses={200: 'Эксперимент удалён'})
    def delete(self):
        args = model_delete.parse_args()
        delete_filename = "train_results/" + args.model_type + "_" + args.experiment_id + ".pkl"
        try:
            os.remove(delete_filename)
        except FileNotFoundError:
            return 'Модели с таким экспериментом нет!', 403
        return 'Эксперимент удалён', 200


model_predict = api.parser()
model_predict.add_argument('filename', type='str', default='train.csv', required=True, location='args')
model_predict.add_argument('experiment_id',
                           type=str, default="0", required=True, location='args')
model_predict.add_argument('model_type',
                           required=True,
                           choices=['SVC', 'GradientBoostingClassifier', 'LogisticRegression'],
                           location='args'
                           )


# Парсер и класс для предсказания
@api.route('/model_predict', methods=['POST'], doc={'description': 'Предсказание на обученной модели'})
@api.expect(model_predict)
class Predict(Resource):
    @api.doc(params={'filename': f'Название файла в формате CSV с полями Age, Embarked, Pclass, Sex'})
    @api.doc(params={'experiment_id': f'Номер эксперимента'})
    @api.doc(params={'model_type': f'Тип модели'})
    @api.doc(responses={200: 'Предсказания модели'})
    def post(self):
        args = model_predict.parse_args()
        # Загрузка и предобработка данных
        data = pd.read_csv(args.file)
        X = prepare_data(data, for_train=False)
        # Загрузка модели
        model = pickle.load(open('train_results/' + args.model_type + "_" + str(args.experiment_id) + '.pkl', 'rb'))
        # Предсказания
        preds = model.predict(X)
        return {'Предсказания модели': preds.tolist()}, 200


if __name__ == '__main__':
    app.run(debug=True)
