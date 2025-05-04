import os
import sys
import torch
import queue
import sounddevice as sd
import re
import time
import json
from pathlib import Path
from vosk import Model, KaldiRecognizer
from agent import ask_agent
from memory import save_to_memory

# Константы
SAMPLE_RATE = 16000
VOSK_MODEL_PATH = "model_small"
SILERO_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'silero_models')
MODELS_URLS = {
    'ru': 'https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
    'en': 'https://models.silero.ai/models/tts/en/v3_en.pt'
}
MODEL_PATHS = {
    'ru': os.path.join(SILERO_MODELS_DIR, 'ru', 'model.pt'),
    'en': os.path.join(SILERO_MODELS_DIR, 'en', 'model.pt')
}

# Глобальные переменные для TTS
models = {}
tts_model_loaded = False
pyttsx3_engine = None

# Попытка импорта резервной библиотеки TTS
try:
    import pyttsx3
    pyttsx3_available = True
except ImportError:
    pyttsx3_available = False
    print("ПРЕДУПРЕЖДЕНИЕ: pyttsx3 не установлен, запасной TTS будет недоступен")

#---------- Функции для озвучивания текста (Silero TTS) ----------#

def init_pyttsx3():
    """Инициализация резервной системы pyttsx3"""
    global pyttsx3_engine
    if pyttsx3_available:
        try:
            pyttsx3_engine = pyttsx3.init()
            # Настройка голоса
            voices = pyttsx3_engine.getProperty('voices')
            for voice in voices:
                if 'russian' in str(voice).lower() or 'ru' in str(voice).lower():
                    pyttsx3_engine.setProperty('voice', voice.id)
                    break
            return True
        except Exception as e:
            print(f"Ошибка инициализации pyttsx3: {e}")
    return False

def download_model(lang):
    """Загрузка модели из интернета, если она отсутствует"""
    model_path = MODEL_PATHS[lang]
    model_url = MODELS_URLS[lang]
    
    # Создаем директорию, если не существует
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.isfile(model_path):
        print(f"Загружаю модель {lang} из {model_url}")
        try:
            torch.hub.download_url_to_file(model_url, model_path)
            print(f"Модель {lang} успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели {lang}: {e}")
            return False
    return True

def load_model(lang):
    """Загрузка модели из локального файла"""
    global models, tts_model_loaded
    
    if lang in models:
        return True
        
    model_path = MODEL_PATHS[lang]
    
    try:
        if os.path.isfile(model_path):
            model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            model.to('cpu')
            models[lang] = model
            tts_model_loaded = True
            return True
        else:
            print(f"Файл модели {lang} не найден")
            return False
    except Exception as e:
        print(f"Ошибка загрузки модели {lang}: {e}")
        return False

def init_tts():
    """Инициализация всей системы TTS"""
    global tts_model_loaded
    
    # Инициализация pyttsx3 как резервной системы
    pyttsx3_initialized = init_pyttsx3()
    
    # Пытаемся загрузить русскую модель
    if download_model('ru') and load_model('ru'):
        tts_model_loaded = True
    
    # Пытаемся загрузить английскую модель
    download_model('en') and load_model('en')

def split_text_into_chunks(text, max_chunk_size=1000):
    """Делит текст на части, длина каждой не превышает max_chunk_size символов"""
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Если добавление очередного предложения не превысит лимит,
        # то добавляем его к текущему фрагменту
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def detect_language(text):
    """Простое определение языка текста"""
    # Подсчитываем кириллические символы
    cyrillic_count = sum(1 for char in text if 'а' <= char.lower() <= 'я' or char.lower() in 'ёіїєґ')
    
    # Если более 50% символов кириллические, считаем текст русским
    if cyrillic_count / max(1, len(text)) > 0.5:
        return 'ru'
    else:
        return 'en'

def speak_text_silero(text, speaker='baya', sample_rate=48000, lang=None):
    """Озвучивание текста с помощью Silero TTS"""
    global models
    
    if not text:
        return False
    
    # Определяем язык, если не указан
    if lang is None:
        lang = detect_language(text)
    
    # Проверяем, загружена ли нужная модель
    if lang not in models:
        if not load_model(lang):
            return False
    
    try:
        # Разбиваем текст на части, если он длинный
        chunks = split_text_into_chunks(text)
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(0.3)  # Пауза между частями
                
            audio = models[lang].apply_tts(
                text=chunk, 
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=True,
                put_yo=True
            )
            
            sd.play(audio, sample_rate)
            sd.wait()
        
        return True
    except Exception as e:
        print(f"Ошибка при синтезе речи через Silero: {e}")
        return False

def speak_text_pyttsx3(text):
    """Озвучивание текста с помощью pyttsx3"""
    global pyttsx3_engine
    
    if not text or not pyttsx3_engine:
        return False
    
    try:
        pyttsx3_engine.say(text)
        pyttsx3_engine.runAndWait()
        return True
    except Exception as e:
        print(f"Ошибка при синтезе речи через pyttsx3: {e}")
        return False

def speak_text(text, speaker='baya'):
    """Основная функция озвучивания текста"""
    if not text:
        return
    
    # Пытаемся озвучить через Silero
    if tts_model_loaded and speak_text_silero(text, speaker):
        return
    
    # Если не получилось, используем pyttsx3
    if speak_text_pyttsx3(text):
        return
    
    # Если ничего не сработало
    print("Не удалось озвучить текст:", text[:50] + "..." if len(text) > 50 else text)

#---------- Функции для распознавания речи (Vosk) ----------#

def check_vosk_model():
    """Проверка наличия модели распознавания речи"""
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"ОШИБКА: Модель распознавания речи не найдена в {VOSK_MODEL_PATH}")
        return False
    return True

def recognize_speech():
    """Распознавание речи с микрофона"""
    if not check_vosk_model():
        raise Exception("Модель распознавания речи не найдена")
    
    try:
        model = Model(VOSK_MODEL_PATH)
        q = queue.Queue()

        def callback(indata, frames, time, status):
            if status:
                print("Ошибка:", status, file=sys.stderr)
            q.put(bytes(indata))

        print("🎤 Скажи что-нибудь (Ctrl+C для выхода)...")
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                              channels=1, callback=callback):
            rec = KaldiRecognizer(model, SAMPLE_RATE)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    return result.get("text", "")
    except Exception as e:
        print(f"Ошибка при распознавании речи: {e}")
        raise

def run_voice():
    """Запуск голосового интерфейса в консоли"""
    # Проверяем наличие модели распознавания речи
    if not check_vosk_model():
        print("Модель распознавания речи не найдена.")
        raise Exception("Модель распознавания речи не найдена")
    
    # Инициализируем систему TTS
    init_tts()
    
    try:
        print("🔊 Голосовой режим запущен. Нажмите Ctrl+C для выхода.")
        while True:
            try:
                phrase = recognize_speech()
                if not phrase:
                    continue
                print("Вы:", phrase)
                save_to_memory("Пользователь", phrase)

                response = ask_agent(phrase)
                print("Агент:", response)
                speak_text(response)
                save_to_memory("Агент", response)
                
            except Exception as e:
                print(f"Ошибка в цикле распознавания: {e}")
                print("Попробуйте снова...")

    except KeyboardInterrupt:
        print("\nГолосовой режим завершён.")

# Инициализируем TTS при импорте модуля
init_tts() 