import pickle
import pandas as pd
import numpy as np
import lightfm
from lightfm import LightFM
from lightfm.data import Dataset

class RecSys():
    """
    Класс для подготовки данных и для обучения модели.
    Класс всегда принимает, датасет transactions и product
    products.csv
    product_id - уникальный идентификатор товара
    product_name - название товара
    aisle_id - уникальный идентификатор подкатегории
    department_id - уникальный идентификатор категории
    aisle - название подкатегории
    department - название категории
    transactions.csv

    order_id - уникальный идентификатор транзакции
    user_id - уникальный идентификатор покупателя
    order_number - номер транзакции в истории покупок данного пользователя
    order_dow - день недели транзакции
    order_hour_of_day - час совершения транзакции
    days_since_prior_order - количество дней с совершения предыдущей транзакции данным пользователем
    product_id - уникальный идентификатор товара
    add_to_cart_order - номер под которым данный товар был добавлен в корзину
    reordered - был ли товар "перезаказан"

    Функция __init__ инициализирует переменные и создает словари для кодирования и докедирования пользователей и товаров.
    Флаг use_features по умолчанию False, если True, то используем следующие фичи
    user_features = дни между покупками, уникальные дни в которые покупал пользователь, уникальные подкатегории товара
    item_feautres = категории товаров по популярности
    """

    def __init__(self, transactions, product, use_features=False):
        # приходит тюпл
        __temp = transactions,
        # получаем датасет из тюпла
        self.__transactions = __temp[0]
        # датасет продуктов
        self.__product = product
        # флаг использования фичей
        self.__use_features = use_features
        # инициализруем переменную пользовательских фичей
        self.__user_features = None
        # инициализруем переменную фичей продуктов
        self.__item_features = None
        # инициалзируем переменную в которую в дальнейшем запишем датасет с топом покупок для пользователя
        self.__top_product_to_user = None
        # инициализируем класс датасет из lightfm
        self.dataset = Dataset()
        # создадим множество user_id для кодировки ID
        self.__user_id_set = set(transactions['user_id'])
        # создадим множество product_id для кодировки ID
        self.__product_id_set = set(transactions['product_id'])
        # словарь ключ словаря содержит значение user_id/product_id значение ключа содержит
        self.__dict_user_id = {}
        # словари ключ словаря содержит значение user_id/product_id значение ключа
        self.__dict_product_id = {}
        # словарь для раскодировки
        self.__encode_user_id = {}
        # словарь для раскодировки
        self.__encode_product_id = {}
        # кодируем user_id
        for idx, values in enumerate(self.__user_id_set):
            self.__dict_user_id.update({values: idx})
        # кодируем product_id
        for idx, values in enumerate(self.__product_id_set):
            self.__dict_product_id.update({values: idx})
        # раскодируем user_id
        for i in self.__dict_user_id.keys():
            self.__encode_user_id.update({self.__dict_user_id.get(i): i})
        # раскодируем product_id
        for i in self.__dict_product_id.keys():
            self.__encode_product_id.update({self.__dict_product_id.get(i): i})

    def encode_user(self, code):
        """
        Функция для декодировки пользователя
        :param code: code user_id, int
        :return: real user_id,int
        """
        return self.__encode_user_id.get(code)

    def encode_product(self, code):
        """
        Функция для декодировки product_id
        :param code: code product_id, int
        :return: real product_id, int
        """
        return self.__encode_product_id.get(code)

    def code_user(self, code):
        """
        Функция для кодирования пользователя по id
        :param code: real user_id, int
        :return: code user_idm, int
        """
        return self.__dict_user_id.get(code)

    def code_product(self, code):
        """
        Функция для кодирования продуктов по id
        :param code: real product_id, int
        :return: code product_id, int
        """
        return self.__dict_product_id.get(code)

    def __generate_int_id(self, dataframe, id_col_name):
        """
        Функция для генерации уникального идентификатора пользователей
        :param dataframe: Фрейм данных Pandas для пользователей.
        :param id_col_name: Имя столбца кодированных идентификаторов.
        :return: Dataframe
            Обновленный фрейм данных, содержащий новый столбец идентификатора
        """
        new_dataframe = dataframe.assign(
            int_id_col_name=np.arange(len(dataframe))
        ).reset_index(drop=True)
        return new_dataframe.rename(columns={'int_id_col_name': id_col_name})

    def __create_features(self, dataframe, features_name, id_col_name):
        """
        Генерация фичей, которые будут готовы для загрузки в lightfm
        :param dataframe: Фрейм данных Pandas, содержащий фичи
        :param features_name: Список имен столбцов фичей, доступных в фрейме данных
        :param id_col_name: Имя столбца кодированных идентификаторов.
        :return: pandas серия
            Серия pandas, содержащая особенности процесса
            которые готовы для подачи в lightfm.
            Формат каждого значения
             (user_id, ['feature_1', 'feature_2', 'feature_3'])
            Ex. -> (1, ['military', 'army', '5'])
        """
        features = dataframe[features_name].apply(
            lambda x: ','.join(x.map(str)), axis=1)
        features = features.str.split(',')
        features = list(zip(dataframe[id_col_name], features))
        return features

    def __generate_feature_list(self, dataframe, features_name):
        """
        Функция для создания списка  для сопоставления
        :param dataframe: Фрейм данных Pandas для пользователей или продуктов.
        :param features_name: Список имен столбцов фичей, доступных в фрейме данных.
        :return: Список всех возможностей разметок
        """
        features = dataframe[features_name].apply(
            lambda x: ','.join(x.map(str)), axis=1)
        features = features.str.split(',')
        features = features.apply(pd.Series).stack().reset_index(drop=True)
        return features

    def __prepare_data(self):
        """
        Функция для подготовки колонки user_product_tuple для дальнейшего создания матрицы взаимодействия,
        пользовательских признаков и признаков товаров
        :return:
        датасет df_merge с колонкой
        df_merge, product_f, user_f
        """
        # отсортируем данные
        self.__transactions.sort_values('order_id', inplace=True)
        # заменим первое посещение на -1 так как 0 есть в датасете когда посещение не первое
        self.__transactions.days_since_prior_order.fillna(-1, inplace=True)
        # получим количество дней между покупками
        user_f = self.__transactions.groupby(['user_id'], as_index=False).agg(
            {'days_since_prior_order': 'median', 'order_id': 'nunique'})
        # обьеденим датасеты
        user_f_list_aisle = self.__transactions.merge(self.__product, how='inner', left_on='product_id',
                                                      right_on='product_id')
        # подготовим 2 фичи, список уникальных aisle_id и список уникальных дней покупок
        user_f_list_aisle = user_f_list_aisle.groupby('user_id', as_index=False).agg(
            {'order_dow': 'unique', 'aisle_id': 'unique'})
        # обьеденим датасеты
        user_f = user_f.merge(user_f_list_aisle, how='inner', left_on='user_id', right_on='user_id')
        # переименуем столбцы
        user_f.rename(columns={'order_dow': 'unique_order_dow', 'aisle_id': 'unique_aisle_id'}, inplace=True)
        # посчитаем сколько раз купили каждый товар для создания категории
        t_product = self.__transactions.groupby('product_id', as_index=False).agg({'user_id': 'count'})
        # функция которая проставляет категории по критериям
        f = lambda x: 0 if x['user_id'] <= 46 else (1 if 46 <= x['user_id'] <= 204 else 2)
        # создадим столбец категории
        t_product['category'] = t_product.apply(f, axis=1)
        # добавляем фичи в датасет с товарами
        product_f = self.__product.merge(t_product, how='inner')
        # переименуем столбцы
        product_f.rename(columns={'user_id': 'p_amount'}, inplace=True)
        # отсортируем
        product_f.sort_values(by='p_amount', ascending=False)
        # кодируем product_id, user_id
        product_f = self.__generate_int_id(product_f, 'product_id_num')
        user_f = self.__generate_int_id(user_f, 'user_id_num')
        # создадим фичи и получим разметки
        product_f['product_feature'] = self.__create_features(product_f, ['category'], 'product_id_num')
        user_f['days_since_prior_order'] = user_f.days_since_prior_order.astype('int')
        user_f['user_feature'] = self.__create_features(user_f, ['days_since_prior_order', 'unique_order_dow',
                                                                 'unique_aisle_id'], 'user_id_num')
        user_feature_list = self.__generate_feature_list(user_f,
                                                         ['days_since_prior_order', 'unique_order_dow',
                                                          'unique_aisle_id'])
        product_feature_list = self.__generate_feature_list(product_f, ['category'])
        # инициализируем класс dataset lightfm
        self.dataset.fit(
            set(user_f['user_id_num']),
            set(product_f['product_id_num']),
            item_features=product_feature_list,
            user_features=user_feature_list)
        # создадим датасет df_merge для этого обьеденим полученные выше датасеты
        df_merge = self.__transactions.merge(product_f, how='inner', left_on='product_id', right_on='product_id')
        df_merge = df_merge.merge(user_f, how='inner', left_on='user_id', right_on='user_id')
        # получим вес для матрицы взаимодействия, получим сколько раз купил каждый пользователь определенный товар
        df_weight = df_merge.groupby(['user_id', 'product_id'], as_index=False).agg({'order_id_x': 'count'})
        # переименуем колонки
        df_weight.rename(columns={'order_id_x': 'product_amount'}, inplace=True)
        # получим сумму покупок по пользователю это будет наши 100% вес всегда от 0 до 1
        df_weight_2 = df_weight.groupby('user_id', as_index=False).sum()
        # переименуем колонку
        df_weight_2.rename(columns={'product_amount': 'total_amount'}, inplace=True)
        # обьеденим датасеты
        df_weight_res = df_weight.merge(df_weight_2, how='inner', left_on='user_id', right_on='user_id')
        # переименуем колонку
        df_weight_res.rename(columns={'product_id_x': 'product_id'}, inplace=True)
        # оставим только нужные столбцы
        df_weight_res = df_weight_res[['user_id', 'product_id', 'product_amount', 'total_amount']]
        # посчитаем вес количество покупок / на сумму покупок и округляем до 3го знака после запятой
        df_weight_res['weight'] = round(df_weight_res['product_amount'] / df_weight_res['total_amount'], 3)
        # оставим только нужные столбцы
        df_weight_res = df_weight_res[['user_id', 'product_id', 'weight']]
        # обьеденим датасеты
        df_merge = df_merge.merge(df_weight_res, how='inner', left_on=['user_id', 'product_id'],
                                  right_on=['user_id', 'product_id'])
        # подготовим тюпл для создания матрицы взаимодействий
        df_merge['user_product_tuple'] = list(zip(
            df_merge.user_id_num, df_merge.product_id_num, df_merge.weight))
        # запишем топ покупок пользователя в переменную
        self.__top_product_to_user = df_weight.sort_values(by=['user_id', 'product_amount'], ascending=False)
        return df_merge, product_f, user_f

    def __prepare_matrix_to_model(self, df, product_f, user_f):
        """
        Функция для подготовки матрицы взаимодействия, весов и фичей для пользователя и продуктов
        :param df: датасет с колонкой  user_product_tuple (iterable of (user_id, item_id) or (user_id, item_id, weight))
        :param product_f: датасет с колонкой фичей товаров, формата (iterable of the form) – (item id, [list of feature names])
        or (item id, {feature name: feature weight}).
        :param user_f: датасет с колонкой пользовательский фичей  (iterable of the form) – (
        user id, [list of feature names]) or
        (user id, {feature name: feature weight})
        :return:
        #матрица взаимодействия, пользовательские фичи, товара фичи, формат coomatrix
        """
        # получим матрицу взаимодействия и веса
        interactions, weights = self.dataset.build_interactions(
            df['user_product_tuple'])
        #  получим фичи продуктов
        product_features = self.dataset.build_item_features(
            product_f['product_feature'])
        #  получим фичи товаров
        user_features = self.dataset.build_user_features(
            user_f['user_feature'])
        # переназначим переменную класса в полученные пользовательские фичи для дальнейшего использования
        self.__user_features = user_features
        # переназначим переменную класса в полученные фичи товаров для дальнейшего использования
        self.__item_features = product_features
        return interactions, weights, product_features, user_features

    def __create_model(self, no_components, learning_rate, loss, random_state):
        """
        Функция для создания модели lightfm
        :param no_components: количество компонентов
        :param learning_rate: скорость обучения
        :param loss: лосс функция
        :param random_state: число для восспроизводимости
        :return: модель lightfm
        """
        model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state)
        return model

    def fit_model(self, use_features=False, epochs=100, num_threads=4, verbose=True, no_components=300,
                  learning_rate=0.05, loss='warp', random_state=2019):
        """
        Функция для обучения модель, если флаг use_features = True, то обучаем модель с фичами иначе без фичей
        :param use_features: использовать фичи, Bool
        :param epochs: количество эпох, int
        :param num_threads: количество процесов, int
        :param verbose: вывести лог, Bool
        :param no_components: количество компонентов, int
        :param learning_rate: скорость обучения, float
        :param loss: лосс функция, str
        :param random_state: число для воспроизводимости, int
        :return: обученная модель
        """
        # подготовим данные
        df_merge, prodcut_f, user_f = self.__prepare_data()
        # получим матрицы
        interactions, weights, product_features, user_features = self.__prepare_matrix_to_model(df_merge, prodcut_f,
                                                                                                user_f)
        # создадим модель
        model = self.__create_model(no_components, learning_rate, loss, random_state)
        # переназначим переменную использования фичей
        self.__use_feautres = use_features
        # если True обучаем с фичами
        if use_features:
            model.fit(
                interactions,
                sample_weight=weights,
                user_features=user_features,
                item_features=product_features,
                epochs=epochs,
                num_threads=num_threads,
                verbose=verbose)
            return model
        else:
            # обучаем без фичей
            model.fit(
                interactions,
                sample_weight=weights,
                epochs=epochs,
                num_threads=num_threads,
                verbose=verbose)
            return model

    def __load_model(self, model=None, patch=None,use_top_product_to_user=True):
        """
        Функция для загрузки модели по пути
        :param model: модель, lightfm
        :param patch: путь модели, str
        :return: загруженная или текущая модель
        """
        # получим топ пользователей
        if self.__top_product_to_user is None and patch is not None and use_top_product_to_user == True:
            self.__prepare_data()
        # если в use_features True и patch не пустой, то загружаем модель с фичами
        if self.__use_features and patch is not None:
            f = open(patch + 'model.pkl', 'rb')
            model = pickle.load(f)
            f.close()
            f = open(patch + 'product_features.pkl', 'rb')
            self.__item_features = pickle.load(f)
            f.close()
            f = open(patch + '/user_features.pkl', 'rb')
            self.__user_features = pickle.load(f)
            f.close()
            return model
        # иначе use_features False и patch не пустой, то загружаем без фичей
        elif patch is not None:
            f = open(patch + 'model.pkl', 'rb')
            model = pickle.load(f)
            f.close()
            return model
        else:
            # иначе возращаем существующую модель
            return model

    def __sample_recommendation(self, model, n_items, user_ids, k, k_top, use_top_product_to_user):
        """
        Функция для получения рекомендации
        :param model: модель
        :param n_items: количество всего товаров
        :param user_ids: список пользователей, если один то [user]
        :param k: количество рекомендуемых товаров
        :param k_top: сколько брать товаров из топа пользователя
        :param use_top_product_to_user: флаг для использования топ историй покупок, bool
        :return: датасет формата {'user_id':[recommended product]}
        """
        # создадим датасет который будет дополнять и возвращать
        df_res = pd.DataFrame(columns=['user_id', 'product_id'])
        # если n_items не None, создаем массив длинною n_items
        if n_items is not None:
            n_items = np.arange(n_items)
        # иначе получаем количество уникальных товаров из фунции декодрирования, список ключей
        # это и есть количество items
        n_items = np.arange(len(self.__encode_product_id.keys()))
        # если флаг использование топ покупок пользователю True
        if use_top_product_to_user:
            # если use_feautres True, то делаем прогноз с фичами
            if self.__use_features:
                # идем в цикле по списку пользователей
                for user in tqdm(user_ids):
                    # получаем оценки прогноза
                    scores = model.predict(
                        user,
                        n_items,
                        user_features=self.__user_features,
                        item_features=self.__item_features,
                    )
                    # получаем реальный user_id пользователя
                    usr = self.encode_user(user)
                    # декодируем 6 товаров, остальное берем из топа покупок
                    product_list = [self.encode_product(product) for product in np.argsort(-scores)[:k - k_top]]
                    # получаем список покупок по пользователю
                    top_10_usr = list(
                        self.__top_product_to_user[self.__top_product_to_user['user_id'] == usr]['product_id'][:100])
                    # переменная которая считаем сколько товаров мы добавили в product_list из top_10_usr
                    app = 0
                    for item in top_10_usr:
                        # если товаров меньше 10, то заходим
                        if len(product_list) <= k - 1:
                            # если данного товара нет в списке, то добавляем и делаем app += 1
                            if product_list.count(item) == 0:
                                product_list.append(item)
                                app += 1
                        # иначе прерываем цикл
                        else:
                            break
                    # если мы не добавили не один товар, то просто берем из топ 10 четыре товара и добавляем
                    # в product_list
                    if app == 0:
                        for item in top_10_usr[:k_top]:
                            product_list.append(item)
                    # запишем все в итоговый датасет
                    df_res = df_res.append({'user_id': usr, 'product_id': product_list}, ignore_index=True)
                return df_res
            else:
                # иначе делаем прогноз без фичей
                # идем в цикле по списку пользователей
                for user in tqdm(user_ids):
                    # получаем оценки прогноза
                    scores = model.predict(
                        user,
                        n_items
                    )
                    # получаем реальный user_id пользователя
                    usr = self.encode_user(user)
                    # декодируем 6 товаров, остальное берем из топа покупок
                    product_list = [self.encode_product(product) for product in np.argsort(-scores)[:k - k_top]]
                    # получаем список покупок по пользователю
                    top_10_usr = list(
                        self.__top_product_to_user[self.__top_product_to_user['user_id'] == usr]['product_id'][:100])
                    # переменная которая считаем сколько товаров мы добавили в product_list из top_10_usr
                    app = 0
                    for item in top_10_usr:
                        # если товаров меньше 10, то заходим
                        if len(product_list) <= k - 1:
                            # если данного товара нет в списке, то добавляем и делаем app += 1
                            if product_list.count(item) == 0:
                                product_list.append(item)
                                app += 1
                        # иначе прерываем цикл
                        else:
                            break
                    # если мы не добавили не один товар, то просто берем из топ 10 четыре товара и добавляем
                    # в product_list
                    if app == 0:
                        for item in top_10_usr[:k_top]:
                            product_list.append(item)
                    # запишем все в итоговый датасет
                    df_res = df_res.append({'user_id': usr, 'product_id': product_list}, ignore_index=True)
                return df_res

        else:
            # если use_feautres True, то делаем прогноз с фичами
            if self.__use_features:
                for user in tqdm(user_ids):
                    scores = model.predict(
                        user,
                        n_items,
                        user_features=self.__user_features,
                        item_features=self.__item_features,
                    )
                    product_list = [self.encode_product(product) for product in np.argsort(-scores)[:k]]
                    df_res = df_res.append({'user_id': self.encode_user(user), 'product_id': product_list},
                                           ignore_index=True)
                return df_res
            else:
                # иначе делаем прогноз без фичей
                for user in tqdm(user_ids):
                    scores = model.predict(
                        user,
                        n_items
                    )
                    product_list = [self.encode_product(product) for product in np.argsort(-scores)[:k]]
                    df_res = df_res.append({'user_id': self.encode_user(user), 'product_id': product_list},
                                           ignore_index=True)
                return df_res

    def predict_model(self, user_ids, model=None, n_items=None, patch=None, k=10, k_top=4, use_top_product_to_user=True):
        """
        Функция прогнозирования
        :param k_top: сколько брать товаров из топа пользователя
        :param use_top_product_to_user: флаг для использования топ историй покупок, bool
        :param user_ids: список пользователей, если один то [user],list
        :param model: ligfhtfm model,
        :param n_items: количество всего товаров,int
        :param patch: путь до модели, тогда model должна быть None, str
        :param k: количество рекомендуемых товаров, int
        :return: датасет формата {'user_id':[recommended product]}
        """
        model = self.__load_model(model, patch, use_top_product_to_user)
        predict = self.__sample_recommendation(model=model, user_ids=user_ids, n_items=n_items, k=k, k_top=k_top,
                                               use_top_product_to_user=use_top_product_to_user)
        return predict


