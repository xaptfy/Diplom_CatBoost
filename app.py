import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from catboost import CatBoostClassifier

# Загрузите данные и обучите модель
model = CatBoostClassifier().load_model('./cat_model.save')

# Функция, которая преобразует Признак Region, с целью уменьшения количества категорий этого признака
def region_transform(data):
    region_mapping = {'Центральный': 'Центральный, Север и Юг',
                  'Приволжский': 'Сибирь и Дальний Восток',
                  'Северо-Западный': 'Центральный, Север и Юг',
                  'Южный': 'Центральный, Север и Юг',
                  'Уральский': 'Сибирь и Дальний Восток',
                  'Северо-Кавказский': 'Центральный, Север и Юг',
                  'Сибирский': 'Сибирь и Дальний Восток',
                  'Не из России': 'Не из России',
                  'Дальневосточный': 'Сибирь и Дальний Восток'}

    data['Region'] = data['Region'].apply(lambda x: region_mapping[x])


# Функция, которая преобразует Признак Attention to Desing, с целью уменьшения количества категорий этого признака
def attention_design_transform(data):
    data['Attention to Desing'] = data['Attention to Desing'].apply(lambda x: 'Иногда' if x == 'Редко' else x)


# Функция, которая преобразует Признак User Decisions, с целью уменьшения количества категорий этого признака
def user_decision_transform(data):
    data['User Decisions'] = data["User Decisions"].apply(lambda x: 'Редко' if x == 'Никогда' else x)


# Функция, которая прнимает на вход данные и список из преобразований и применяет их к копии данных
def data_transform(data, transforms=[]):
    data = data.copy()

    for _, transform in transforms:
        transform(data)

    return data


data = pd.read_csv("./train (1).csv", sep=';')

features_vals = [(col, data[col].unique()) for col in data.iloc[:, :-1].columns]

# Создайте заголовок
st.title("Интерфейс для предсказаний")

# Создание панельки для ввода признаков
st.sidebar.header("Введите значения признаков:")
feature_values = {}

# Добавьте элементы вводных полей в зависимости от ваших признаков
for feature, vals in features_vals: # Замените на названия ваших признаков
    feature_values[feature] = st.sidebar.selectbox(feature, vals)

input_data = pd.DataFrame([feature_values])

label_encoder = joblib.load('./label_en.save')

transformations = [('User Decisions', user_decision_transform),
                   ('Region', region_transform),
                   ('Attention to Desing', attention_design_transform)]

# Преобразование признаков в нужный вид
data_tr = data_transform(input_data, transformations)

x = data_tr

preprocessor = joblib.load('./preprocessor.save')

x_en = pd.DataFrame(preprocessor.transform(x), columns=x.columns).astype('int32')

# Создайте кнопку для запуска предсказания
if st.button("Сделать предсказание"):
    # Сделайте предсказание
    prediction = model.predict(x_en)

    # Получите распределение классов
    class_probs = model.predict_proba(x_en)

    # Вывод предсказанного класса
    pred_label = label_encoder.inverse_transform(prediction)
    st.write("Предсказанный класс:", f':blue-background{pred_label}')

    # Визуализация результатов
    fig, ax = plt.subplots()
    ax.bar(range(len(class_probs[0])), class_probs[0])
    ax.set_xticks(range(len(class_probs[0])))
    ax.set_xticklabels([f'{cls}' for cls in label_encoder.classes_])
    ax.set_ylabel('Логарифм Вероятности')
    ax.set_title('Распределение классов')
    plt.yscale('log')
    st.pyplot(fig)