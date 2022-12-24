# Основные библиотеки
import os
import pickle
from io import BytesIO
# Для построения REST-API
from flask import Flask, jsonify
from flask_restx import Api, Resource
# Для работы с БД-шкой
from sqlalchemy import create_engine
import psycopg2
# Мои модули для работы с моделями и БД-шкой
from model_stuff import train_model, make_predictions
from db_stuff import save_model, delete_model, get_all_models

app = Flask(__name__)
api = Api(app)



# Парсер и класс для обучения модели
model_train = api.parser()
model_train.add_argument('file', type=str, default='data/train.csv', required=True)
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


@api.route('/model_train', methods=['PUT'], doc={'description': 'Запустить обучение выбранной модели на датасете Titanic. ОБЯЗАТЕЛЬНО ВВЕСТИ ПАРАМЕТРЫ МОДЕЛИ, ЛИБО ОСТАВИТЬ ПРОСТО {}'})
@api.expect(model_train)
class ModelTrain(Resource):
    # Параметры
    @api.doc(params={'file': f'Название файла в формате CSV с полями Age, Embarked, Pclass, Sex, Survived'})
    @api.doc(params={'experiment_id': f'Номер эксперимента'})
    @api.doc(params={'model_type': f'Тип модели'})
    # Результаты выполнения
    @api.doc(responses={403: 'Модель с такими параметрами не определена!'})
    @api.doc(responses={400: 'Файл с таким именем не найден'})
    @api.doc(responses={200: 'Обучение прошло успешно!'})
    @api.doc(responses={202: 'Модель обучена, но такой эксперимент уже существует!'})
    def put(self):
        args = model_train.parse_args()
        # Построение пайплайна и обучение модели
        model = train_model(args)
        if model == 400:
            api.abort(400, 'Файл с данными с таким именем не найден')
        elif model == 403:
            api.abort(403, message="Модель с такими параметрами не определена!")
        # Сохранение результата 
        return save_model(model, args)


# Класс для получения списка обученных моделей
@api.route('/show_models', methods=['GET'], doc={'description': 'Получить список возможных моделей'})
class GetModel(Resource):
    # Смотрим список всех возможных типов моделей
    def get(self):
        models = get_all_models()
        if models == 403:
            return 'Тут ничего нет!', 403
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
        return delete_model(args)


model_predict = api.parser()
model_predict.add_argument('file', type=str, default='data/train.csv', required=True, location='args')
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
    @api.doc(params={'file': f'Название файла в формате CSV с полями Age, Embarked, Pclass, Sex'})
    @api.doc(params={'experiment_id': f'Номер эксперимента'})
    @api.doc(params={'model_type': f'Тип модели'})
    @api.doc(responses={200: 'Успешное предсказание модели'})
    @api.doc(responses={403: 'Модели с таким экспериментом нет!'})
    def post(self):
        args = model_predict.parse_args()
        return make_predictions(args)


if __name__ == '__main__':
    app.run(debug=True)
