# RecSys
Для использования данного класса, необходимо установить библиотеку LightFM pip install lightfm.

За основу взята библиотека LightFM ссылка на документацию https://making.lyst.com/lightfm/docs/home.html

Файл EDA.ipynb это разведочный анализ.

Файл RecSys.py это класс реализации модели.
Получить файлы transactions.csv и product.csv можно по ссылке https://www.kaggle.com/c/skillbox-recommender-system/data
Обученная модель без использования фичей с результатом 0.226 на платформе kaggle можно получить по ссылке https://drive.google.com/drive/folders/1PnBfB0fSDEEd98DryfZVg8nQZhQMV1E8?usp=sharing
Пример использования класса.
Если не хотим использовать фичи, то флаг должен быть установлен use_features=False, по умолчанию False
Инициализируем класс
recsys = RecSys(transactions,product,use_features=False)
Обучаем модель
model = recsys.fit(use_features=False, epochs=110, num_threads=4, verbose=True, no_components=310,
                  learning_rate=0.05, loss='warp', random_state=2019)                
Получаем прогноз в формате user_id:[n_product_id]
:param k_top: сколько брать товаров из топа пользователя
:param use_top_product_to_user: флаг для использования топ историй покупок, bool
:param user_ids: список пользователей, если один то [user],list
:param model: ligfhtfm model,
:param n_items: количество всего товаров,int
:param patch: путь до модели, тогда model должна быть None, str
:param k: количество рекомендуемых товаров, int
predict = recsys.predict_model(user_ids, model=model, n_items=None, patch=None, k=10, k_top=4, use_top_product_to_user=True)
Получение прогноза с загрузкой модели, без обучения, модель должна быть в формате дампа pickle, если установить флаг use_features = True, то в папке с моделью должны также присутствовать два файла product_features.pkl и user_features.pkl 
Инициализируем класс
recsys = RecSys(transactions,product,use_features=False)
predict = recsys.predict_model(user_ids, n_items=None, patch='/model/', k=10, k_top=4, use_top_product_to_user=True)
Результат полученный на платформе Kaggle, без использования топа истории покупок и фичей, составляет 0.207 с использованием топа истории покупок составляет 0.226, использование простого ALS 0.128, подход item_to_item 0.119, подход user_to_user 0.117
