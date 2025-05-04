import os
import sys
import json
import time
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from datetime import datetime

class OnlineTranscriber:
    def __init__(self):
        self.model = None
        self.mic_recognizer = None
        self.system_recognizer = None
        self.temp_dir = tempfile.mkdtemp()
        self.language = "ru"  # По умолчанию русский
        self.vosk_model_path = "model_small"  # Путь к модели Vosk
        self.sample_rate = 16000
        
        # Очереди для аудиоданных
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        # Флаги для управления потоками
        self.is_running = False
        self.capture_mic = True
        self.capture_system = True
        
        # Накопленный текст
        self.transcript = []
        
        # Результаты распознавания
        self.results_callback = None
        
    def load_model(self):
        """Загрузка модели Vosk"""
        try:
            if not os.path.exists(self.vosk_model_path):
                raise ValueError(f"Путь к модели Vosk не существует: {self.vosk_model_path}")
                
            print(f"Загрузка модели Vosk из {self.vosk_model_path}...")
            self.model = Model(self.vosk_model_path)
            
            # Создаем распознаватели
            self.mic_recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.system_recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            print("Модель Vosk успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели Vosk: {str(e)}")
            return False
    
    def mic_callback(self, indata, frames, time, status):
        """Callback для захвата аудио с микрофона"""
        if status:
            print(f"Статус микрофона: {status}")
        if self.capture_mic:
            self.mic_queue.put(bytes(indata))
    
    def system_callback(self, indata, frames, time, status):
        """Callback для захвата системного аудио"""
        if status:
            print(f"Статус системного аудио: {status}")
        if self.capture_system:
            self.system_queue.put(bytes(indata))
    
    def process_mic_audio(self):
        """Обработка аудио с микрофона"""
        print("Начало обработки аудио с микрофона")
        
        while self.is_running:
            try:
                data = self.mic_queue.get(timeout=1)
                if self.mic_recognizer.AcceptWaveform(data):
                    result = json.loads(self.mic_recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        speaker = "Вы"
                        entry = {"time": timestamp, "speaker": speaker, "text": text}
                        self.transcript.append(entry)
                        
                        if self.results_callback:
                            self.results_callback(entry)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Ошибка при обработке аудио с микрофона: {str(e)}")
    
    def process_system_audio(self):
        """Обработка системного аудио"""
        print("Начало обработки системного аудио")
        
        while self.is_running:
            try:
                data = self.system_queue.get(timeout=1)
                if self.system_recognizer.AcceptWaveform(data):
                    result = json.loads(self.system_recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        speaker = "Собеседник"
                        entry = {"time": timestamp, "speaker": speaker, "text": text}
                        self.transcript.append(entry)
                        
                        if self.results_callback:
                            self.results_callback(entry)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Ошибка при обработке системного аудио: {str(e)}")
    
    def start_transcription(self, results_callback=None, capture_mic=True, capture_system=True):
        """
        Запуск одновременной транскрибации с микрофона и системного аудио
        
        Args:
            results_callback: Функция обратного вызова для получения результатов
            capture_mic: Захватывать аудио с микрофона
            capture_system: Захватывать системное аудио
        """
        if not self.model:
            if not self.load_model():
                return False, "Не удалось загрузить модель Vosk"
        
        self.results_callback = results_callback
        self.capture_mic = capture_mic
        self.capture_system = capture_system
        self.is_running = True
        self.transcript = []
        
        # Потоки для обработки аудио
        self.mic_thread = None
        self.system_thread = None
        
        try:
            # Запуск потоков обработки аудио
            if capture_mic:
                self.mic_thread = threading.Thread(target=self.process_mic_audio)
                self.mic_thread.daemon = True
                self.mic_thread.start()
                
                # Запуск потока для чтения с микрофона
                self.mic_stream = sd.InputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    callback=self.mic_callback,
                    dtype='int16'
                )
                self.mic_stream.start()
            
            if capture_system:
                self.system_thread = threading.Thread(target=self.process_system_audio)
                self.system_thread.daemon = True
                self.system_thread.start()
                
                # Для захвата системного аудио нужен дополнительный код в зависимости от ОС
                if sys.platform.startswith('win'):
                    # На Windows нужна дополнительная настройка
                    # Здесь мы предполагаем, что у пользователя уже настроен loopback или virtual cable
                    try:
                        # Пытаемся найти устройство для захвата системного аудио
                        devices = sd.query_devices()
                        system_device = None
                        
                        for i, device in enumerate(devices):
                            if 'CABLE Output' in device['name'] or 'Stereo Mix' in device['name'] or 'Loopback' in device['name']:
                                system_device = i
                                break
                        
                        if system_device is not None:
                            self.system_stream = sd.InputStream(
                                device=system_device,
                                channels=1,
                                samplerate=self.sample_rate,
                                callback=self.system_callback,
                                dtype='int16'
                            )
                            self.system_stream.start()
                        else:
                            print("Предупреждение: Не найдено устройство для захвата системного аудио.")
                            print("Для захвата системного звука установите Virtual Audio Cable или включите Stereo Mix.")
                    except Exception as e:
                        print(f"Ошибка при запуске захвата системного аудио: {str(e)}")
                else:
                    # Для Linux и macOS нужен другой подход
                    print("Захват системного аудио на Linux/macOS требует дополнительной настройки")
            
            print("Транскрибация запущена успешно")
            return True, "Транскрибация запущена успешно"
            
        except Exception as e:
            self.stop_transcription()
            return False, f"Ошибка при запуске транскрибации: {str(e)}"
    
    def stop_transcription(self):
        """Остановка транскрибации"""
        self.is_running = False
        
        # Останавливаем потоки и очищаем ресурсы
        if hasattr(self, 'mic_stream') and self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
        
        if hasattr(self, 'system_stream') and self.system_stream:
            self.system_stream.stop()
            self.system_stream.close()
        
        # Ждем завершения потоков
        if self.mic_thread and self.mic_thread.is_alive():
            self.mic_thread.join(timeout=1)
        
        if self.system_thread and self.system_thread.is_alive():
            self.system_thread.join(timeout=1)
        
        print("Транскрибация остановлена")
        return self.get_transcript()
    
    def get_transcript(self):
        """Получение полного транскрипта"""
        return self.transcript
    
    def save_transcript(self, file_path=None):
        """Сохранение транскрипта в файл"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.temp_dir, f"transcript_{timestamp}.txt")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in self.transcript:
                    f.write(f"[{entry['time']}] {entry['speaker']}: {entry['text']}\n")
            
            print(f"Транскрипт сохранен в {file_path}")
            return True, file_path
        except Exception as e:
            print(f"Ошибка при сохранении транскрипта: {str(e)}")
            return False, str(e)
    
    def get_system_audio_devices():
        """Получение списка устройств для захвата системного аудио"""
        devices = sd.query_devices()
        system_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Только устройства с входными каналами
                name = device['name']
                system_devices.append({"id": i, "name": name})
                
                # Отмечаем потенциальные устройства для захвата системного звука
                if any(keyword in name for keyword in ['CABLE', 'Mix', 'Loopback', 'VAC', 'VB-Audio']):
                    system_devices[-1]["is_system"] = True
                else:
                    system_devices[-1]["is_system"] = False
        
        return system_devices 