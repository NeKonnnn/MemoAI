import os
import tempfile
import subprocess
from vosk import Model, KaldiRecognizer
import wave
import json
import pytube
from moviepy.editor import VideoFileClip

class Transcriber:
    def __init__(self):
        self.model = None
        self.temp_dir = tempfile.mkdtemp()
        self.language = "ru"  # По умолчанию используем русский
        self.model_size = "model_small"  # Путь к модели Vosk
        
    def load_model(self, model_path=None):
        """Загрузка модели для транскрибации"""
        if model_path:
            self.model_size = model_path
            
        try:
            print(f"Загрузка модели Vosk ({self.model_size})...")
            self.model = Model(self.model_size)
            print("Модель Vosk успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели Vosk: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Транскрибация аудио файла"""
        if not self.model:
            success = self.load_model()
            if not success:
                return False, "Не удалось загрузить модель транскрибации"
        
        try:
            print(f"Начинаю транскрибацию файла: {audio_path}")
            
            # Преобразуем аудио в WAV 16кГц 16бит моно если нужно
            wav_path = os.path.join(self.temp_dir, "audio_for_transcription.wav")
            
            # Используем ffmpeg для конвертации
            command = [
                "ffmpeg", 
                "-y",  # Перезаписывать существующие файлы
                "-i", audio_path,  # Входной файл
                "-ar", "16000",  # Частота дискретизации 16 кГц
                "-ac", "1",      # Моно
                "-bits_per_raw_sample", "16",  # 16 бит
                wav_path  # Выходной файл
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Открываем WAV файл для распознавания
            wf = wave.open(wav_path, "rb")
            # Создаем распознаватель
            rec = KaldiRecognizer(self.model, wf.getframerate())
            
            # Собираем результаты транскрибации
            result_text = []
            
            # Читаем аудио по частям и распознаем
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    if 'text' in part_result and part_result['text'].strip():
                        result_text.append(part_result['text'])
            
            # Получаем финальный результат
            part_result = json.loads(rec.FinalResult())
            if 'text' in part_result and part_result['text'].strip():
                result_text.append(part_result['text'])
            
            full_text = " ".join(result_text)
            return True, full_text
            
        except Exception as e:
            print(f"Ошибка при транскрибации аудио: {str(e)}")
            return False, f"Ошибка при транскрибации: {str(e)}"
    
    def extract_audio_from_video(self, video_path):
        """Извлечение аудио из видео файла"""
        try:
            audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
            
            # Извлекаем аудио с помощью moviepy
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, 
                                       codec='pcm_s16le',  # 16-бит PCM
                                       ffmpeg_params=["-ar", "16000", "-ac", "1"])  # 16кГц моно
            video.close()
            
            return True, audio_path
        except Exception as e:
            print(f"Ошибка при извлечении аудио из видео: {str(e)}")
            return False, f"Ошибка при извлечении аудио: {str(e)}"
    
    def transcribe_video(self, video_path):
        """Транскрибация видео файла"""
        # Извлекаем аудио из видео
        success, audio_path = self.extract_audio_from_video(video_path)
        if not success:
            return False, audio_path  # возвращаем сообщение об ошибке
        
        # Транскрибируем полученное аудио
        return self.transcribe_audio(audio_path)
    
    def download_youtube(self, url):
        """Загрузка видео с YouTube"""
        try:
            print(f"Загрузка видео с YouTube: {url}")
            yt = pytube.YouTube(url)
            video_path = os.path.join(self.temp_dir, "youtube_video.mp4")
            
            # Загружаем видео с максимально доступным качеством
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(output_path=self.temp_dir, filename="youtube_video.mp4")
            
            print(f"Видео успешно загружено: {video_path}")
            return True, video_path
        except Exception as e:
            print(f"Ошибка при загрузке видео с YouTube: {str(e)}")
            return False, f"Ошибка при загрузке видео: {str(e)}"
    
    def transcribe_zoom_meeting(self, zoom_recording_path):
        """Транскрибация записи Zoom"""
        # Zoom сохраняет записи в формате mp4, поэтому используем метод для транскрибации видео
        return self.transcribe_video(zoom_recording_path)
    
    def transcribe_youtube(self, url):
        """Транскрибация видео с YouTube"""
        # Загружаем видео
        success, video_path = self.download_youtube(url)
        if not success:
            return False, video_path  # Возвращаем сообщение об ошибке
        
        # Транскрибируем загруженное видео
        return self.transcribe_video(video_path)
    
    def transcribe_streaming_audio(self, audio_stream_url):
        """Транскрибация потокового аудио"""
        try:
            # Сохраняем поток во временный файл
            temp_audio_file = os.path.join(self.temp_dir, "stream_audio.wav")
            
            # Используем ffmpeg для захвата потока
            command = [
                "ffmpeg", 
                "-y",  # Перезаписывать существующие файлы
                "-i", audio_stream_url,  # Входной поток
                "-ar", "16000",  # Частота 16 кГц
                "-ac", "1",      # Моно
                "-f", "wav",     # Формат WAV
                temp_audio_file  # Выходной файл
            ]
            
            # Запускаем команду
            subprocess.run(command, check=True)
            
            # Транскрибируем сохраненный аудиофайл
            return self.transcribe_audio(temp_audio_file)
        except Exception as e:
            print(f"Ошибка при транскрибации потокового аудио: {str(e)}")
            return False, f"Ошибка при транскрибации потока: {str(e)}"
    
    def process_audio_file(self, file_path):
        """Обработка аудио файла"""
        # Проверяем расширение файла
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.mp3', '.wav', '.m4a', '.aac', '.flac']:
            # Транскрибируем аудио напрямую
            return self.transcribe_audio(file_path)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Если это видеофайл, извлекаем аудио и транскрибируем
            return self.transcribe_video(file_path)
        else:
            return False, f"Неподдерживаемый формат файла: {file_extension}"
    
    def set_language(self, language_code):
        """Установка языка для транскрибации"""
        self.language = language_code
        print(f"Установлен язык транскрибации: {language_code}")
        
    def set_model_size(self, model_path):
        """Установка размера модели Vosk"""
        if os.path.exists(model_path):
            self.model_size = model_path
            self.model = None  # Сбрасываем текущую модель
            print(f"Установлен путь к модели Vosk: {model_path}")
            return True
        else:
            print(f"Некорректный путь к модели: {model_path}")
            return False
    
    def clean_temp_files(self):
        """Очистка временных файлов"""
        try:
            for file_name in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print("Временные файлы очищены")
            return True
        except Exception as e:
            print(f"Ошибка при очистке временных файлов: {str(e)}")
            return False 