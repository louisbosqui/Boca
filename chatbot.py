import random
import json
import pickle
import numpy as np
import requests
import time

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.h5')


def cleanup_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = cleanup_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    err_threshold = 0.2
    results = [[i, r] for i, r in enumerate(res) if r > err_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    if tag == 'weather':
        result = get_weather()
    elif tag == 'time':
        result = get_time()
    elif tag == 'date':
        result = get_date()
    else:
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result


def get_weather():
    print("Enter the name of a city, I'll give you the weather there : ")
    city = input("")
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid=11eb65b60469f741cf1f01edba7764bf&units=metric'
    response = requests.get(url)
    if response.status_code == 404:
        return "I'm sorry, I didn't find the city you are looking for !"
    elif response.status_code == 200:
        return format_weather_data(city, response.json())
    else:
        return "I'm sorry, it looks like I am indisposed to give you the weather right now. Check your internet connection!"


def format_weather_data(city, data):
    coord_lon = data['coord']['lon']
    coord_lat = data['coord']['lat']
    weather_desc = data['weather'][0]['main']
    real_temp = data['main']['temp']
    human_temp = data['main']['feels_like']
    return f'The weather in {city}({coord_lon},{coord_lat}) is best described with : {weather_desc}. The temperature is {real_temp}°C, but feels more like {human_temp}°C. Ask me something else!'


def get_time():
    result = time.localtime(time.time())
    return f'The time right now is {result.tm_hour}:{result.tm_min}'


def get_date():
    named_tuple = time.localtime()
    time_string = time.strftime("%d/%m/%Y", named_tuple)
    return f"Today's date is {time_string}"


print("Bot is running")
while True:
    message = input("")
    predicted_intents = predict_class(message)
    res = get_response(predicted_intents, intents)
    print(res)
