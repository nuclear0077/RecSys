from RecSys import RecSys
import pandas as pd

# загрузим датасеты
transactions = pd.read_csv('transactions.csv')
product = pd.read_csv('products.csv')
# инициализируем класс
recsys = RecSys(transactions, product, use_features=False)
# обучим модель
model = recsys.recsys.fit(use_features=False,
                          epochs=110,
                          num_threads=4,
                          verbose=True,
                          no_components=310,
                          learning_rate=0.05,
                          loss='warp',
                          random_state=2019)
# получим прогноз для 3х пользователей
predict = recsys.predict_model(user_ids=[0, 1, 100],
                               model=model,
                               n_items=None,
                               patch=None,
                               k=10,
                               k_top=4,
                               use_top_product_to_user=True)
