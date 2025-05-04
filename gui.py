import sys
import os
import threading
import json
import glob
import queue
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QLineEdit, QFileDialog, QMessageBox, QTabWidget,
                            QListWidget, QListWidgetItem, QFormLayout, QDialog,
                            QFrame, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QRadioButton, QButtonGroup, QProgressBar,
                            QGroupBox, QSplitter)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QObject, pyqtSignal, QThread, QDateTime
from PyQt6.QtGui import QFont, QIcon, QColor, QTextCursor

from agent import ask_agent, update_model_settings, model_settings
from memory import save_to_memory
from voice import speak_text, check_vosk_model, VOSK_MODEL_PATH
from document_processor import DocumentProcessor
from transcriber import Transcriber
from online_transcription import OnlineTranscriber

# Константы
CONFIG_FILE = "settings.json"
MODELS_DIR = "models"

# Класс для обработки сигналов
class Signals(QObject):
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    voice_recognized = pyqtSignal(str)
    voice_error = pyqtSignal(str)
    voice_response_ready = pyqtSignal(str)
    document_processed = pyqtSignal(bool, str)
    transcription_complete = pyqtSignal(bool, str)
    progress_update = pyqtSignal(int)
    online_transcription_result = pyqtSignal(dict)
    streaming_chunk_ready = pyqtSignal(str, str)  # сигнал для стриминга (chunk, accumulated_text)

# Класс для фонового получения ответов от модели
class AgentThread(QThread):
    def __init__(self, signals, message, for_voice=False, streaming=None):
        super().__init__()
        self.signals = signals
        self.message = message
        self.for_voice = for_voice
        # Если streaming не указан явно, берем из настроек модели
        self.streaming = streaming if streaming is not None else model_settings.get("streaming", False)
        
    def run(self):
        try:
            # Функция обратного вызова для потоковой генерации
            def stream_callback(chunk, accumulated_text):
                self.signals.streaming_chunk_ready.emit(chunk, accumulated_text)
            
            # Получаем ответ от модели
            response = ask_agent(
                self.message, 
                streaming=self.streaming,
                stream_callback=stream_callback if self.streaming else None
            )
            
            # Отправляем сигнал с полным ответом
            if self.for_voice:
                self.signals.voice_response_ready.emit(response)
            else:
                self.signals.response_ready.emit(response)
            
            # Сохраняем в историю
            save_to_memory("Агент", response)
            
        except Exception as e:
            # Отправляем сигнал с ошибкой
            self.signals.error_occurred.emit(str(e))

# Класс для работы с документами в фоновом режиме
class DocumentThread(QThread):
    def __init__(self, signals, doc_processor, file_path=None, query=None):
        super().__init__()
        self.signals = signals
        self.doc_processor = doc_processor
        self.file_path = file_path
        self.query = query
        
    def run(self):
        if self.file_path:
            # Обработка документа
            success, message = self.doc_processor.process_document(self.file_path)
            self.signals.document_processed.emit(success, message)
        elif self.query:
            # Запрос к документам
            response = self.doc_processor.process_query(self.query, ask_agent)
            self.signals.response_ready.emit(response)
            
            # Сохраняем в историю
            save_to_memory("Агент", response)

# Класс для транскрибации в фоновом режиме
class TranscriptionThread(QThread):
    def __init__(self, signals, transcriber, file_path=None, youtube_url=None):
        super().__init__()
        self.signals = signals
        self.transcriber = transcriber
        self.file_path = file_path
        self.youtube_url = youtube_url
        
    def run(self):
        try:
            # Начинаем с 10% прогресса
            self.signals.progress_update.emit(10)
            
            if self.file_path:
                # Транскрибация файла
                self.signals.progress_update.emit(30)
                success, text = self.transcriber.process_audio_file(self.file_path)
            elif self.youtube_url:
                # Транскрибация YouTube
                self.signals.progress_update.emit(30)
                success, text = self.transcriber.transcribe_youtube(self.youtube_url)
            else:
                success, text = False, "Не указан источник для транскрибации"
            
            # Сигнализируем о завершении
            self.signals.progress_update.emit(100)
            self.signals.transcription_complete.emit(success, text)
            
        except Exception as e:
            self.signals.progress_update.emit(100)
            self.signals.transcription_complete.emit(False, f"Ошибка при транскрибации: {str(e)}")

# Класс для распознавания голоса в отдельном потоке
class VoiceRecognitionThread(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.running = True
        self.paused = False
        self.pause_condition = threading.Condition()
        
    def run(self):
        """Основной метод для запуска распознавания речи"""
        if not check_vosk_model():
            self.signals.voice_error.emit("Модель распознавания речи не найдена")
            return
            
        try:
            model = Model(VOSK_MODEL_PATH)
            q = queue.Queue()
            
            def callback(indata, frames, time, status):
                if status:
                    print("Ошибка:", status, file=sys.stderr)
                if self.running and not self.paused:
                    q.put(bytes(indata))
            
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                                  channels=1, callback=callback):
                rec = KaldiRecognizer(model, SAMPLE_RATE)
                
                while self.running:
                    # Проверяем, не поставлен ли поток на паузу
                    with self.pause_condition:
                        if self.paused:
                            self.pause_condition.wait()  # Ждем возобновления
                            continue
                    
                    try:
                        data = q.get(timeout=0.5)  # Таймаут, чтобы периодически проверять running
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            if text:  # Если распознан непустой текст
                                self.signals.voice_recognized.emit(text)
                    except queue.Empty:
                        pass  # Просто продолжаем, если данных нет
                        
        except Exception as e:
            self.signals.voice_error.emit(f"Ошибка при распознавании речи: {str(e)}")
    
    def stop(self):
        """Остановка потока распознавания"""
        self.running = False
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()
        self.wait()
    
    def pause(self):
        """Приостановка распознавания"""
        with self.pause_condition:
            self.paused = True
    
    def resume(self):
        """Возобновление распознавания"""
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

class ModelConfig:
    """Класс для управления конфигурацией моделей"""
    def __init__(self):
        self.config = {
            "models": [],
            "current_model": "",
            "voice_speaker": "baya",
            "theme": "light"  # Добавляем настройку темы по умолчанию - светлая
        }
        self.load_config()
        
    def load_config(self):
        """Загрузка конфигурации из файла"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            
            # Если список моделей пуст, сканируем директорию моделей
            if not self.config["models"]:
                self.scan_for_models()
                
            # Если текущая модель не указана, но есть модели, устанавливаем первую как текущую
            if not self.config["current_model"] and self.config["models"]:
                self.config["current_model"] = self.config["models"][0]["path"]
                
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
    
    def save_config(self):
        """Сохранение конфигурации в файл"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
    
    def scan_for_models(self):
        """Сканирование директории для поиска моделей"""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
            
        # Находим все .gguf файлы в директории моделей
        model_files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
        
        # Обновляем список моделей
        self.config["models"] = []
        for model_path in model_files:
            model_name = os.path.basename(model_path)
            self.config["models"].append({
                "name": model_name,
                "path": model_path
            })
            
        # Если нашли хотя бы одну модель, устанавливаем её как текущую
        if self.config["models"] and not self.config["current_model"]:
            self.config["current_model"] = self.config["models"][0]["path"]
            
        self.save_config()
    
    def add_model(self, model_path):
        """Добавление новой модели в конфигурацию"""
        # Проверяем, существует ли такая модель в конфигурации
        if any(model["path"] == model_path for model in self.config["models"]):
            return False
            
        # Добавляем новую модель
        model_name = os.path.basename(model_path)
        self.config["models"].append({
            "name": model_name,
            "path": model_path
        })
        
        # Если это первая модель, устанавливаем её как текущую
        if len(self.config["models"]) == 1:
            self.config["current_model"] = model_path
            
        self.save_config()
        return True
    
    def set_current_model(self, model_path):
        """Установка текущей модели"""
        if any(model["path"] == model_path for model in self.config["models"]):
            self.config["current_model"] = model_path
            self.save_config()
            return True
        return False
    
    def get_current_model(self):
        """Получение информации о текущей модели"""
        if not self.config["current_model"]:
            return None
            
        for model in self.config["models"]:
            if model["path"] == self.config["current_model"]:
                return model
                
        return None

class AddModelDialog(QDialog):
    """Диалог добавления новой модели"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавление новой модели")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Форма для ввода данных
        form_layout = QFormLayout()
        
        # Путь к файлу
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Путь к файлу модели")
        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(self.browse_file)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_button)
        
        form_layout.addRow("Файл модели:", path_layout)
        
        # Кнопки
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        
        add_button = QPushButton("Добавить")
        add_button.clicked.connect(self.accept)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(add_button)
        
        # Сборка интерфейса
        layout.addLayout(form_layout)
        layout.addStretch()
        layout.addLayout(button_layout)
    
    def browse_file(self):
        """Выбор файла модели"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            "",
            "GGUF модели (*.gguf)"
        )
        
        if file_path:
            self.path_edit.setText(file_path)
    
    def get_model_path(self):
        """Получение пути к модели"""
        return self.path_edit.text()

class ModelSettingsDialog(QDialog):
    """Диалог настроек модели LLM"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки LLM модели")
        self.setMinimumWidth(500)
        
        # Получаем текущие настройки
        self.current_settings = model_settings.get_all()
        
        layout = QVBoxLayout(self)
        
        # Создаем форму для настроек
        form_layout = QFormLayout()
        
        # Выбор устройства (CPU/GPU)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU"])
        self.device_combo.setCurrentIndex(1 if self.current_settings.get("use_gpu", False) else 0)
        form_layout.addRow("Устройство вычислений:", self.device_combo)
        
        # Размер контекста
        self.context_size_spin = QSpinBox()
        self.context_size_spin.setRange(512, 16384)
        self.context_size_spin.setSingleStep(512)
        self.context_size_spin.setValue(self.current_settings["context_size"])
        form_layout.addRow("Размер контекста:", self.context_size_spin)
        
        # Размер выходного текста
        self.output_tokens_spin = QSpinBox()
        self.output_tokens_spin.setRange(128, 4096)
        self.output_tokens_spin.setSingleStep(128)
        self.output_tokens_spin.setValue(self.current_settings["output_tokens"])
        form_layout.addRow("Размер выходного текста:", self.output_tokens_spin)
        
        # Размер батча
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(32, 1024)
        self.batch_size_spin.setSingleStep(32)
        self.batch_size_spin.setValue(self.current_settings["batch_size"])
        form_layout.addRow("Размер батча:", self.batch_size_spin)
        
        # Количество потоков
        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 16)
        self.n_threads_spin.setValue(self.current_settings["n_threads"])
        form_layout.addRow("Количество потоков:", self.n_threads_spin)
        
        # Температура
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 1.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setValue(self.current_settings["temperature"])
        form_layout.addRow("Температура:", self.temperature_spin)
        
        # Top-p
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setValue(self.current_settings["top_p"])
        form_layout.addRow("Top-p:", self.top_p_spin)
        
        # Штраф за повторения
        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(1.0, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setDecimals(2)
        self.repeat_penalty_spin.setValue(self.current_settings["repeat_penalty"])
        form_layout.addRow("Штраф за повторения:", self.repeat_penalty_spin)
        
        # Подробный вывод
        self.verbose_combo = QComboBox()
        self.verbose_combo.addItems(["Включен", "Выключен"])
        self.verbose_combo.setCurrentIndex(0 if self.current_settings["verbose"] else 1)
        form_layout.addRow("Подробный вывод:", self.verbose_combo)
        
        # Потоковая генерация
        self.streaming_combo = QComboBox()
        self.streaming_combo.addItems(["Включена", "Выключена"])
        self.streaming_combo.setCurrentIndex(0 if self.current_settings.get("streaming", False) else 1)
        form_layout.addRow("Потоковая генерация:", self.streaming_combo)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        # Кнопка сброса к настройкам по умолчанию
        reset_button = QPushButton("Сбросить к значениям по умолчанию")
        reset_button.clicked.connect(self.reset_to_defaults)
        
        # Кнопка отмены
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        
        # Кнопка сохранения
        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self.accept)
        
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        
        # Добавляем форму и кнопки в основной макет
        layout.addLayout(form_layout)
        layout.addStretch()
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Сброс настроек к значениям по умолчанию"""
        # Сбрасываем значения в форме
        self.device_combo.setCurrentIndex(0)  # CPU по умолчанию
        self.context_size_spin.setValue(2048)
        self.output_tokens_spin.setValue(512)
        self.batch_size_spin.setValue(512)
        self.n_threads_spin.setValue(2)
        self.temperature_spin.setValue(0.7)
        self.top_p_spin.setValue(0.95)
        self.repeat_penalty_spin.setValue(1.05)
        self.verbose_combo.setCurrentIndex(0)
        self.streaming_combo.setCurrentIndex(0)  # Потоковая генерация включена по умолчанию
    
    def get_settings(self):
        """Получение настроек из формы"""
        return {
            "context_size": self.context_size_spin.value(),
            "output_tokens": self.output_tokens_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "n_threads": self.n_threads_spin.value(),
            "temperature": self.temperature_spin.value(),
            "top_p": self.top_p_spin.value(),
            "repeat_penalty": self.repeat_penalty_spin.value(),
            "verbose": self.verbose_combo.currentIndex() == 0,
            "use_gpu": self.device_combo.currentIndex() == 1,  # GPU выбран, если индекс = 1
            "use_mmap": True,  # Оставляем эти параметры неизменными
            "use_mlock": False,
            "streaming": self.streaming_combo.currentIndex() == 0  # Streaming включен, если индекс = 0
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Инициализация объектов для работы с документами и транскрибацией
        self.doc_processor = DocumentProcessor()
        self.transcriber = Transcriber()
        self.online_transcriber = OnlineTranscriber()
        
        # Настройка сигналов
        self.signals = Signals()
        self.signals.response_ready.connect(self.handle_response)
        self.signals.error_occurred.connect(self.handle_error)
        self.signals.voice_recognized.connect(self.handle_voice_recognition)
        self.signals.voice_error.connect(self.handle_voice_error)
        self.signals.voice_response_ready.connect(self.handle_voice_response)
        self.signals.document_processed.connect(self.handle_document_processed)
        self.signals.transcription_complete.connect(self.handle_transcription_complete)
        self.signals.progress_update.connect(self.update_progress_bar)
        self.signals.online_transcription_result.connect(self.handle_online_transcription)
        self.signals.streaming_chunk_ready.connect(self.handle_streaming_chunk)
        
        # Флаг для отслеживания активной потоковой генерации
        self.streaming_active = False
        self.current_stream_message = ""
        
        # Настройка предпочтений
        self.model_config = ModelConfig()
        
        # Настройка главного окна
        self.setWindowTitle("MemoAI")
        self.setMinimumSize(900, 600)
        self.setWindowIcon(QIcon("assets/icon.ico"))
        
        # Центральный виджет и компоновка
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Создаем основную горизонтальную компоновку
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Создаем боковую панель
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(200)
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame)
        self.sidebar_layout.setContentsMargins(10, 20, 10, 20)
        self.sidebar_layout.setSpacing(10)
        
        # Создаем основную рабочую область
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Добавляем панели в главную компоновку
        self.main_layout.addWidget(self.sidebar_frame)
        self.main_layout.addWidget(self.content_frame)
        
        # Настраиваем боковую панель
        self.setup_sidebar()
        
        # Настраиваем заголовок
        self.header_frame = QFrame()
        self.header_layout = QHBoxLayout(self.header_frame)
        self.header_layout.setContentsMargins(20, 10, 20, 10)
        self.content_layout.addWidget(self.header_frame)
        self.setup_header()
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.content_layout.addWidget(self.tabs)
        
        # Добавляем вкладку с чатом
        self.chat_tab = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_tab)
        self.tabs.addTab(self.chat_tab, "Текстовый чат")
        self.setup_chat_tab()
        
        # Добавляем вкладку с голосом
        self.voice_tab = QWidget()
        self.voice_layout = QVBoxLayout(self.voice_tab)
        self.tabs.addTab(self.voice_tab, "Голосовой режим")
        self.setup_voice_tab()
        
        # Добавляем вкладку для работы с документами
        self.docs_tab = QWidget()
        self.docs_layout = QVBoxLayout(self.docs_tab)
        self.tabs.addTab(self.docs_tab, "Документы")
        self.setup_docs_tab()
        
        # Добавляем вкладку для транскрибации
        self.transcribe_tab = QWidget()
        self.transcribe_layout = QVBoxLayout(self.transcribe_tab)
        self.tabs.addTab(self.transcribe_tab, "Транскрибация")
        self.setup_transcribe_tab()
        
        # Добавляем вкладку для онлайн-транскрибации совещаний
        self.setup_online_transcribe_tab()
        
        # Применяем тему
        self.apply_theme()
        
        # Инициализация голосового распознавания
        self.voice_recognition_thread = None
        self.recognition_active = False
    
    def setup_sidebar(self):
        """Настройка боковой панели (шторки)"""
        # Заголовок
        sidebar_title = QLabel("MemoAI")
        sidebar_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.sidebar_layout.addWidget(sidebar_title)
        
        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.sidebar_layout.addWidget(separator)
        
        # Кнопка настроек моделей
        models_button = QPushButton("Модели")
        models_button.setMinimumHeight(40)
        models_button.clicked.connect(self.show_models_dialog)
        self.sidebar_layout.addWidget(models_button)
        
        # Кнопка настроек LLM
        llm_settings_button = QPushButton("Настройки LLM")
        llm_settings_button.setMinimumHeight(40)
        llm_settings_button.clicked.connect(self.show_llm_settings)
        self.sidebar_layout.addWidget(llm_settings_button)
        
        # Кнопка настроек голосового режима
        voice_button = QPushButton("Голосовой режим")
        voice_button.setMinimumHeight(40)
        voice_button.clicked.connect(self.show_voice_settings)
        self.sidebar_layout.addWidget(voice_button)
        
        # Кнопка настроек интерфейса
        interface_button = QPushButton("Интерфейс")
        interface_button.setMinimumHeight(40)
        interface_button.clicked.connect(self.show_interface_settings)
        self.sidebar_layout.addWidget(interface_button)
        
        # Добавляем растягивающий элемент
        self.sidebar_layout.addStretch()
        
        # Информация о текущей модели
        current_model = self.model_config.get_current_model()
        model_name = current_model["name"] if current_model else "Нет"
        model_info = QLabel(f"Текущая модель:\n{model_name}")
        model_info.setWordWrap(True)
        model_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_layout.addWidget(model_info)
    
    def setup_header(self):
        """Настройка верхней панели"""
        # Кнопка открытия/закрытия шторки
        toggle_button = QPushButton("☰")
        toggle_button.setFixedSize(40, 40)
        toggle_button.clicked.connect(self.toggle_sidebar)
        self.header_layout.addWidget(toggle_button)
        
        # Заголовок
        title = QLabel("MemoAI Ассистент")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.header_layout.addWidget(title)
        
        # Добавляем растягивающий элемент
        self.header_layout.addStretch()
    
    def setup_chat_tab(self):
        """Настройка вкладки чата"""
        # История чата
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 11))
        self.chat_layout.addWidget(self.chat_history)
        
        # Поле ввода и кнопка отправки
        input_layout = QHBoxLayout()
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Введите сообщение...")
        self.chat_input.setFont(QFont("Arial", 11))
        self.chat_input.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Отправить")
        self.send_button.setFixedWidth(100)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        
        self.chat_layout.addLayout(input_layout)
        
        # Добавляем приветственное сообщение
        self.append_message("Ассистент", "Привет! Я ваш AI-ассистент. Чем могу помочь?")
    
    def setup_voice_tab(self):
        """Настройка вкладки голосового чата"""
        # История голосового чата
        self.voice_history = QTextEdit()
        self.voice_history.setReadOnly(True)
        self.voice_history.setFont(QFont("Arial", 11))
        self.voice_layout.addWidget(self.voice_history)
        
        # Панель управления голосовым режимом
        control_layout = QHBoxLayout()
        
        # Статус
        self.voice_status = QLabel("Ожидание...")
        
        # Кнопка включения/выключения распознавания
        self.voice_toggle_button = QPushButton("🎤 Начать прослушивание")
        self.voice_toggle_button.setMinimumHeight(40)
        self.voice_toggle_button.clicked.connect(self.toggle_voice_recognition)
        
        control_layout.addWidget(self.voice_status)
        control_layout.addStretch()
        control_layout.addWidget(self.voice_toggle_button)
        
        self.voice_layout.addLayout(control_layout)
        
        # Добавляем приветственное сообщение
        self.append_voice_message("Ассистент", "Привет! Нажмите кнопку микрофона, чтобы начать голосовое общение.")
    
    def setup_docs_tab(self):
        """Настройка вкладки для работы с документами"""
        # Создаем разделитель для документов и чата
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.docs_layout.addWidget(splitter)
        
        # Левая панель для документов
        docs_panel = QWidget()
        docs_panel_layout = QVBoxLayout(docs_panel)
        
        # Заголовок
        docs_header = QLabel("Загруженные документы")
        docs_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        docs_panel_layout.addWidget(docs_header)
        
        # Кнопки управления документами
        docs_controls = QHBoxLayout()
        
        self.load_doc_btn = QPushButton("Загрузить документ")
        self.load_doc_btn.clicked.connect(self.load_document)
        docs_controls.addWidget(self.load_doc_btn)
        
        self.clear_docs_btn = QPushButton("Очистить")
        self.clear_docs_btn.clicked.connect(self.clear_documents)
        docs_controls.addWidget(self.clear_docs_btn)
        
        docs_panel_layout.addLayout(docs_controls)
        
        # Список документов
        self.docs_list = QListWidget()
        docs_panel_layout.addWidget(self.docs_list)
        
        # Правая панель для чата с документами
        chat_panel = QWidget()
        chat_panel_layout = QVBoxLayout(chat_panel)
        
        # Заголовок
        chat_docs_header = QLabel("Запросы к документам")
        chat_docs_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        chat_panel_layout.addWidget(chat_docs_header)
        
        # Область чата
        self.docs_chat_area = QTextEdit()
        self.docs_chat_area.setReadOnly(True)
        chat_panel_layout.addWidget(self.docs_chat_area)
        
        # Поле ввода и кнопка отправки
        input_layout = QHBoxLayout()
        
        self.docs_input = QLineEdit()
        self.docs_input.setPlaceholderText("Введите запрос к документам...")
        self.docs_input.returnPressed.connect(self.send_docs_query)
        input_layout.addWidget(self.docs_input)
        
        self.docs_send_btn = QPushButton("Отправить")
        self.docs_send_btn.clicked.connect(self.send_docs_query)
        input_layout.addWidget(self.docs_send_btn)
        
        chat_panel_layout.addLayout(input_layout)
        
        # Добавляем панели в разделитель
        splitter.addWidget(docs_panel)
        splitter.addWidget(chat_panel)
        splitter.setSizes([300, 600])  # Начальные размеры панелей
    
    def setup_transcribe_tab(self):
        """Настройка вкладки для транскрибации"""
        # Верхняя панель с настройками
        settings_group = QGroupBox("Настройки транскрибации")
        settings_layout = QFormLayout(settings_group)
        
        # Выбор размера модели
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_size_combo.setCurrentText("base")
        self.model_size_combo.currentTextChanged.connect(self.change_model_size)
        settings_layout.addRow("Размер модели:", self.model_size_combo)
        
        # Выбор языка
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ru", "en", "auto"])
        self.language_combo.setCurrentText("ru")
        self.language_combo.currentTextChanged.connect(self.change_transcription_language)
        settings_layout.addRow("Язык:", self.language_combo)
        
        self.transcribe_layout.addWidget(settings_group)
        
        # Панель источников
        source_group = QGroupBox("Источник для транскрибации")
        source_layout = QVBoxLayout(source_group)
        
        # Кнопки выбора источника
        self.source_radio_group = QButtonGroup(self)
        
        self.file_radio = QRadioButton("Файл (аудио/видео)")
        self.file_radio.setChecked(True)
        self.source_radio_group.addButton(self.file_radio)
        source_layout.addWidget(self.file_radio)
        
        self.youtube_radio = QRadioButton("YouTube")
        self.source_radio_group.addButton(self.youtube_radio)
        source_layout.addWidget(self.youtube_radio)
        
        # Поле ввода для URL и кнопка выбора файла
        input_layout = QHBoxLayout()
        
        self.transcribe_input = QLineEdit()
        self.transcribe_input.setPlaceholderText("Введите URL YouTube или выберите файл...")
        input_layout.addWidget(self.transcribe_input)
        
        self.browse_file_btn = QPushButton("Выбрать файл")
        self.browse_file_btn.clicked.connect(self.browse_media_file)
        input_layout.addWidget(self.browse_file_btn)
        
        source_layout.addLayout(input_layout)
        
        # Кнопка начала транскрибации и индикатор прогресса
        transcribe_controls = QHBoxLayout()
        
        self.start_transcribe_btn = QPushButton("Начать транскрибацию")
        self.start_transcribe_btn.clicked.connect(self.start_transcription)
        transcribe_controls.addWidget(self.start_transcribe_btn)
        
        self.transcribe_progress = QProgressBar()
        self.transcribe_progress.setRange(0, 100)
        self.transcribe_progress.setValue(0)
        transcribe_controls.addWidget(self.transcribe_progress)
        
        source_layout.addLayout(transcribe_controls)
        
        self.transcribe_layout.addWidget(source_group)
        
        # Область результатов
        result_group = QGroupBox("Результаты транскрибации")
        result_layout = QVBoxLayout(result_group)
        
        self.transcribe_result = QTextEdit()
        self.transcribe_result.setReadOnly(True)
        result_layout.addWidget(self.transcribe_result)
        
        # Кнопки действий с результатами
        action_layout = QHBoxLayout()
        
        self.copy_result_btn = QPushButton("Копировать")
        self.copy_result_btn.clicked.connect(self.copy_transcription)
        action_layout.addWidget(self.copy_result_btn)
        
        self.save_result_btn = QPushButton("Сохранить в файл")
        self.save_result_btn.clicked.connect(self.save_transcription)
        action_layout.addWidget(self.save_result_btn)
        
        result_layout.addLayout(action_layout)
        
        self.transcribe_layout.addWidget(result_group)
    
    def setup_online_transcribe_tab(self):
        """Настройка вкладки для онлайн-транскрибации совещаний"""
        # Создаем вкладку
        self.online_transcribe_tab = QWidget()
        self.online_transcribe_layout = QVBoxLayout(self.online_transcribe_tab)
        
        # Добавляем вкладку в основной виджет с вкладками
        self.tabs.addTab(self.online_transcribe_tab, QIcon("assets/online.png"), "Совещания")
        
        # Верхняя панель с настройками
        settings_group = QGroupBox("Настройки записи совещания")
        settings_layout = QVBoxLayout(settings_group)
        
        # Настройки источников аудио
        sources_form = QFormLayout()
        
        # Чекбокс для микрофона
        self.mic_checkbox = QCheckBox("Записывать микрофон (ваш голос)")
        self.mic_checkbox.setChecked(True)
        sources_form.addRow("Записывать микрофон:", self.mic_checkbox)
        
        # Чекбокс для системного звука
        self.system_audio_checkbox = QCheckBox("Записывать аудио системы (голоса собеседников)")
        self.system_audio_checkbox.setChecked(True)
        sources_form.addRow("Записывать аудио системы:", self.system_audio_checkbox)
        
        # Выбор устройства для системного звука
        audio_devices_layout = QHBoxLayout()
        self.system_device_combo = QComboBox()
        self.refresh_audio_devices_btn = QPushButton("Обновить список")
        self.refresh_audio_devices_btn.clicked.connect(self.refresh_audio_devices)
        
        audio_devices_layout.addWidget(QLabel("Устройство для захвата аудио системы:"))
        audio_devices_layout.addWidget(self.system_device_combo)
        audio_devices_layout.addWidget(self.refresh_audio_devices_btn)
        
        sources_form.addRow("", audio_devices_layout)
        
        # Информация о настройке аудио
        info_label = QLabel("Для записи аудио системы (звук из динамиков) необходимо:\n"
                          "1. В Windows: Включите 'Стерео микшер' в настройках звука или\n"
                          "2. Установите виртуальный аудиокабель (например, VB-Cable)\n"
                          "3. Выберите соответствующее устройство в списке выше")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        sources_form.addRow(info_label)
        
        # Кнопки управления записью
        controls_layout = QHBoxLayout()
        
        self.start_meeting_btn = QPushButton("Начать запись совещания")
        self.start_meeting_btn.clicked.connect(self.start_online_transcription)
        self.start_meeting_btn.setMinimumHeight(40)
        
        self.stop_meeting_btn = QPushButton("Остановить запись")
        self.stop_meeting_btn.clicked.connect(self.stop_online_transcription)
        self.stop_meeting_btn.setEnabled(False)
        self.stop_meeting_btn.setMinimumHeight(40)
        
        controls_layout.addWidget(self.start_meeting_btn)
        controls_layout.addWidget(self.stop_meeting_btn)
        
        sources_form.addRow("", controls_layout)
        
        # Добавляем настройки на вкладку
        settings_layout.addLayout(sources_form)
        self.online_transcribe_layout.addWidget(settings_group)
        
        # Область для отображения транскрипции в реальном времени
        transcript_group = QGroupBox("Стенограмма совещания")
        transcript_layout = QVBoxLayout(transcript_group)
        
        self.online_transcript_area = QTextEdit()
        self.online_transcript_area.setReadOnly(True)
        transcript_layout.addWidget(self.online_transcript_area)
        
        # Кнопки для сохранения транскрипта
        save_layout = QHBoxLayout()
        
        self.copy_transcript_btn = QPushButton("Копировать стенограмму")
        self.copy_transcript_btn.clicked.connect(self.copy_online_transcript)
        
        self.save_transcript_btn = QPushButton("Сохранить в файл")
        self.save_transcript_btn.clicked.connect(self.save_online_transcript)
        
        save_layout.addWidget(self.copy_transcript_btn)
        save_layout.addWidget(self.save_transcript_btn)
        
        transcript_layout.addLayout(save_layout)
        
        # Добавляем область транскрипции на вкладку
        self.online_transcribe_layout.addWidget(transcript_group)
        
        # Заполняем список устройств
        self.refresh_audio_devices()
    
    def refresh_audio_devices(self):
        """Обновление списка аудиоустройств"""
        try:
            self.system_device_combo.clear()
            
            import sounddevice as sd
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Только устройства с входными каналами
                    device_name = device['name']
                    is_system = any(keyword in device_name for keyword in ['CABLE', 'Mix', 'Loopback', 'VAC', 'VB-Audio'])
                    
                    # Добавляем индикатор для устройств, которые могут захватывать системный звук
                    if is_system:
                        self.system_device_combo.addItem(f"✓ {device_name} (Системный звук)", i)
                    else:
                        self.system_device_combo.addItem(device_name, i)
            
            # Выбираем первое "системное" устройство, если есть
            for i in range(self.system_device_combo.count()):
                if "✓" in self.system_device_combo.itemText(i):
                    self.system_device_combo.setCurrentIndex(i)
                    break
                    
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось получить список аудиоустройств: {str(e)}")
    
    def start_online_transcription(self):
        """Запуск онлайн-транскрибации совещания"""
        # Проверяем выбранные источники
        capture_mic = self.mic_checkbox.isChecked()
        capture_system = self.system_audio_checkbox.isChecked()
        
        if not capture_mic and not capture_system:
            QMessageBox.warning(self, "Внимание", "Выберите хотя бы один источник аудио для записи")
            return
            
        # Получаем выбранное устройство для системного звука
        system_device = None
        if capture_system and self.system_device_combo.currentData() is not None:
            system_device = self.system_device_combo.currentData()
        
        try:
            # Очищаем текстовую область
            self.online_transcript_area.clear()
            
            # Запускаем транскрибацию
            success, message = self.online_transcriber.start_transcription(
                results_callback=self.handle_real_time_transcript, 
                capture_mic=capture_mic, 
                capture_system=capture_system
            )
            
            if success:
                # Меняем состояние кнопок
                self.start_meeting_btn.setEnabled(False)
                self.stop_meeting_btn.setEnabled(True)
                self.mic_checkbox.setEnabled(False)
                self.system_audio_checkbox.setEnabled(False)
                self.system_device_combo.setEnabled(False)
                self.refresh_audio_devices_btn.setEnabled(False)
                
                # Добавляем сообщение о начале записи
                self.append_online_transcript({
                    "time": QDateTime.currentDateTime().toString("HH:mm:ss"),
                    "speaker": "Система",
                    "text": "Запись совещания началась. Говорите в микрофон."
                })
            else:
                QMessageBox.warning(self, "Ошибка", f"Не удалось запустить транскрибацию: {message}")
                
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при запуске транскрибации: {str(e)}")
    
    def handle_real_time_transcript(self, entry):
        """Обработка результатов онлайн-транскрибации"""
        # Отправляем данные через сигнал для безопасного обновления UI
        self.signals.online_transcription_result.emit(entry)
    
    def handle_online_transcription(self, entry):
        """Обработка результатов онлайн-транскрибации в UI потоке"""
        self.append_online_transcript(entry)
    
    def append_online_transcript(self, entry):
        """Добавление записи в область транскрипции"""
        time_str = entry["time"]
        speaker = entry["speaker"]
        text = entry["text"]
        
        # Определяем цвет для разных говорящих
        if speaker == "Вы":
            color = "#0066cc"
        elif speaker == "Собеседник":
            color = "#cc6600"
        else:  # Система
            color = "#666666"
        
        # Добавляем запись в текстовую область
        cursor = self.online_transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Форматируем текст
        cursor.insertHtml(f'<p><span style="color: {color};"><b>[{time_str}] {speaker}:</b></span> {text}</p>')
        
        # Прокручиваем вниз
        self.online_transcript_area.setTextCursor(cursor)
        self.online_transcript_area.ensureCursorVisible()
    
    def stop_online_transcription(self):
        """Остановка онлайн-транскрибации"""
        try:
            # Останавливаем транскрибацию
            transcript = self.online_transcriber.stop_transcription()
            
            # Меняем состояние кнопок
            self.start_meeting_btn.setEnabled(True)
            self.stop_meeting_btn.setEnabled(False)
            self.mic_checkbox.setEnabled(True)
            self.system_audio_checkbox.setEnabled(True)
            self.system_device_combo.setEnabled(True)
            self.refresh_audio_devices_btn.setEnabled(True)
            
            # Добавляем сообщение о завершении записи
            self.append_online_transcript({
                "time": QDateTime.currentDateTime().toString("HH:mm:ss"),
                "speaker": "Система",
                "text": f"Запись совещания завершена. Всего записано {len(transcript)} фрагментов."
            })
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при остановке транскрибации: {str(e)}")
    
    def copy_online_transcript(self):
        """Копирование стенограммы в буфер обмена"""
        text = self.online_transcript_area.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Скопировано", "Стенограмма скопирована в буфер обмена")
    
    def save_online_transcript(self):
        """Сохранение стенограммы в файл"""
        # Запрашиваем путь для сохранения
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("Текстовые файлы (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                file_path = filenames[0]
                success, message = self.online_transcriber.save_transcript(file_path)
                
                if success:
                    QMessageBox.information(self, "Сохранено", f"Стенограмма сохранена в файл: {file_path}")
                else:
                    QMessageBox.warning(self, "Ошибка", f"Ошибка при сохранении стенограммы: {message}")
    
    def toggle_sidebar(self):
        """Открытие/закрытие боковой панели"""
        # Текущая ширина
        current_width = self.sidebar_frame.width()
        
        # Целевая ширина
        target_width = 200 if current_width == 0 else 0
        
        # Создаем анимацию
        self.animation = QPropertyAnimation(self.sidebar_frame, b"minimumWidth")
        self.animation.setDuration(200)
        self.animation.setStartValue(current_width)
        self.animation.setEndValue(target_width)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()
        
        # Дублируем анимацию для максимальной ширины
        self.animation2 = QPropertyAnimation(self.sidebar_frame, b"maximumWidth")
        self.animation2.setDuration(200)
        self.animation2.setStartValue(current_width)
        self.animation2.setEndValue(target_width)
        self.animation2.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation2.start()
    
    def show_models_dialog(self):
        """Показывает диалог управления моделями"""
        # Закрываем боковую панель
        if self.sidebar_frame.width() > 0:
            self.toggle_sidebar()
        
        # Создаем диалог
        dialog = QDialog(self)
        dialog.setWindowTitle("Управление моделями")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Список моделей
        models_label = QLabel("Доступные модели:")
        models_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(models_label)
        
        self.models_list = QListWidget()
        self.refresh_models_list()
        layout.addWidget(self.models_list)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        add_button = QPushButton("Добавить")
        add_button.clicked.connect(self.add_model)
        
        remove_button = QPushButton("Удалить")
        remove_button.clicked.connect(self.remove_model)
        
        set_current_button = QPushButton("Установить как текущую")
        set_current_button.clicked.connect(self.set_current_model)
        
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addWidget(set_current_button)
        
        layout.addLayout(buttons_layout)
        
        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec()
    
    def refresh_models_list(self):
        """Обновление списка моделей"""
        self.models_list.clear()
        
        current_model_path = self.model_config.config["current_model"]
        
        for model in self.model_config.config["models"]:
            item = QListWidgetItem(model["name"])
            item.setData(Qt.ItemDataRole.UserRole, model["path"])
            
            # Если это текущая модель, выделяем её
            if model["path"] == current_model_path:
                item.setText(f"✓ {model['name']} (текущая)")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                
            self.models_list.addItem(item)
    
    def add_model(self):
        """Добавление новой модели"""
        dialog = AddModelDialog(self)
        
        if dialog.exec():
            model_path = dialog.get_model_path()
            
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Путь к файлу модели не указан")
                return
                
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Ошибка", "Указанный файл не существует")
                return
                
            if not model_path.lower().endswith(".gguf"):
                QMessageBox.warning(self, "Ошибка", "Файл должен иметь расширение .gguf")
                return
                
            # Добавляем модель
            success = self.model_config.add_model(model_path)
            
            if success:
                QMessageBox.information(self, "Успех", "Модель успешно добавлена")
                self.refresh_models_list()
            else:
                QMessageBox.warning(self, "Ошибка", "Такая модель уже добавлена")
    
    def remove_model(self):
        """Удаление выбранной модели"""
        selected_items = self.models_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Выберите модель для удаления")
            return
            
        selected_item = selected_items[0]
        model_path = selected_item.data(Qt.ItemDataRole.UserRole)
        
        # Подтверждение удаления
        confirm = QMessageBox.question(
            self, 
            "Подтверждение", 
            "Вы уверены, что хотите удалить выбранную модель из списка?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Удаляем модель из списка
            self.model_config.config["models"] = [
                model for model in self.model_config.config["models"] 
                if model["path"] != model_path
            ]
            
            # Если удаляется текущая модель, выбираем новую
            if self.model_config.config["current_model"] == model_path:
                if self.model_config.config["models"]:
                    self.model_config.config["current_model"] = self.model_config.config["models"][0]["path"]
                else:
                    self.model_config.config["current_model"] = ""
                    
            self.model_config.save_config()
            self.refresh_models_list()
    
    def set_current_model(self):
        """Установка выбранной модели как текущей"""
        selected_items = self.models_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Выберите модель")
            return
            
        selected_item = selected_items[0]
        model_path = selected_item.data(Qt.ItemDataRole.UserRole)
        
        success = self.model_config.set_current_model(model_path)
        
        if success:
            QMessageBox.information(self, "Успех", "Модель установлена как текущая")
            self.refresh_models_list()
    
    def show_voice_settings(self):
        """Настройки голосового режима"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки голосового режима")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Выбор голоса
        voice_layout = QFormLayout()
        voice_combo = QComboBox()
        voice_combo.addItems(["baya", "xenia", "kseniya", "aidar", "eugene"])
        current_voice = self.model_config.config.get("voice_speaker", "baya")
        voice_combo.setCurrentText(current_voice)
        
        voice_layout.addRow("Голос для синтеза:", voice_combo)
        
        # Кнопка тестирования
        test_button = QPushButton("Тест голоса")
        test_button.clicked.connect(lambda: self.test_voice(voice_combo.currentText()))
        
        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(lambda: dialog.accept())
        
        # Кнопки внизу
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(test_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(close_button)
        
        # Добавляем всё в основной макет
        layout.addLayout(voice_layout)
        layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Сохраняем выбранный голос при закрытии
        def save_voice():
            self.model_config.config["voice_speaker"] = voice_combo.currentText()
            self.model_config.save_config()
        
        dialog.accepted.connect(save_voice)
        
        dialog.exec()
    
    def test_voice(self, voice_name):
        """Тестирование выбранного голоса"""
        threading.Thread(
            target=speak_text,
            args=("Это тестовое сообщение для проверки голоса " + voice_name, voice_name),
            daemon=True
        ).start()
    
    def toggle_voice_recognition(self):
        """Включение/выключение распознавания речи"""
        if self.recognition_active:
            self.stop_voice_recognition()
        else:
            self.start_voice_recognition()
            
    def start_voice_recognition(self):
        """Запуск распознавания речи"""
        if not check_vosk_model():
            self.handle_voice_error("Модель распознавания речи не найдена в директории model_small")
            return
            
        try:
            self.recognition_active = True
            self.voice_toggle_button.setText("⏹ Остановить прослушивание")
            self.voice_status.setText("Слушаю... Говорите в микрофон")
            
            # Создаем и запускаем поток распознавания речи
            self.voice_recognition_thread = VoiceRecognitionThread(self.signals)
            self.voice_recognition_thread.start()
            
            # Добавляем информационное сообщение
            self.append_voice_message("Система", "Микрофон активирован. Говорите.")
            
        except Exception as e:
            self.handle_voice_error(f"Ошибка при запуске распознавания: {str(e)}")
    
    def stop_voice_recognition(self):
        """Остановка распознавания речи"""
        self.recognition_active = False
        self.voice_toggle_button.setText("🎤 Начать прослушивание")
        self.voice_status.setText("Ожидание...")
        
        if self.voice_recognition_thread:
            self.voice_recognition_thread.stop()
            self.voice_recognition_thread = None
            
        # Добавляем информационное сообщение
        self.append_voice_message("Система", "Микрофон отключен.")
        
    def handle_voice_recognition(self, text):
        """Обработка распознанного голосового ввода"""
        # Добавляем текст пользователя в историю
        self.append_voice_message("Вы", text)
        
        # Сохраняем в историю
        save_to_memory("Пользователь", text)
        
        # Меняем статус и останавливаем распознавание на время ответа
        self.voice_status.setText("Генерирую ответ...")
        self.recognition_active = True
        
        # Сбрасываем флаг потоковой генерации
        self.streaming_active = False
        
        # Получаем настройку потоковой генерации
        use_streaming = model_settings.get("streaming", False)
        
        # Если стриминг отключен, показываем индикатор загрузки
        if not use_streaming:
            self.voice_history.append('<span style="color: #888888;">Ассистент печатает...</span>')
        
        # Приостанавливаем распознавание речи на время ответа
        if self.voice_recognition_thread:
            self.voice_recognition_thread.pause()
        
        # Запускаем обработку сообщения в отдельном потоке
        self.agent_thread = AgentThread(self.signals, text, for_voice=True)
        self.agent_thread.start()
    
    def handle_response(self, response):
        """Обработка ответа от модели"""
        # Восстанавливаем элементы интерфейса
        self.send_button.setEnabled(True)
        self.chat_input.setEnabled(True)
        self.chat_input.setFocus()
        
        # Если был потоковый режим, то полный ответ уже отображен
        if self.streaming_active:
            self.streaming_active = False
            return
        
        # Если не было потокового режима (стриминг отключен)
        current_tab_index = self.tabs.currentIndex()
        
        # Удаляем сообщение "Ассистент печатает..."
        if current_tab_index == 0:  # Текстовый чат
            html = self.chat_history.toHtml()
            html = html.replace('<span style="color: #888888;">Ассистент печатает...</span>', '')
            self.chat_history.setHtml(html)
            self.append_message("Агент", response)
        elif current_tab_index == 2:  # Документы
            html = self.docs_chat_area.toHtml()
            html = html.replace('<span style="color: #888888;">Ассистент печатает...</span>', '')
            self.docs_chat_area.setHtml(html)
            self.docs_send_btn.setEnabled(True)
            self.append_docs_message("Агент", response)
    
    def handle_error(self, error):
        """Обработчик ошибки"""
        # Восстанавливаем кнопку
        self.send_button.setEnabled(True)
        
        # Добавляем сообщение об ошибке в историю чата
        self.append_message("Ошибка", error)
        
        # Восстанавливаем интерфейс
        self.chat_input.setEnabled(True)
        self.chat_input.setFocus()
    
    def send_message(self):
        """Отправка сообщения в чат"""
        # Получаем текст из поля ввода
        message = self.chat_input.text().strip()
        
        # Если сообщение пустое, ничего не делаем
        if not message:
            return
            
        # Добавляем сообщение пользователя в историю чата
        self.append_message("Пользователь", message)
        
        # Очищаем поле ввода
        self.chat_input.clear()
        
        # Добавляем индикатор "ассистент печатает..."
        self.chat_history.append('<span style="color: #888888;">Ассистент печатает...</span>')
        
        # Отключаем кнопку отправки на время генерации ответа
        self.send_button.setEnabled(False)
        
        # Создаем поток для обработки сообщения
        # Получаем настройки из конфигурации
        streaming = self.model_config.config.get("streaming", False)
        self.agent_thread = AgentThread(self.signals, message, streaming=streaming)
        self.agent_thread.finished.connect(lambda: self.send_button.setEnabled(True))
        self.agent_thread.start()
    
    def load_document(self):
        """Загрузка документа"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Документы (*.pdf *.docx *.xlsx *.xls)")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                # Запускаем обработку в отдельном потоке
                self.doc_thread = DocumentThread(self.signals, self.doc_processor, file_path=filenames[0])
                self.doc_thread.start()
                
                # Деактивируем кнопку на время обработки
                self.load_doc_btn.setEnabled(False)
                self.load_doc_btn.setText("Загрузка...")
    
    def clear_documents(self):
        """Очистка загруженных документов"""
        result = self.doc_processor.clear_documents()
        self.docs_list.clear()
        self.append_docs_message("Система", result)
    
    def handle_document_processed(self, success, message):
        """Обработка результата обработки документа"""
        # Восстанавливаем кнопку
        self.load_doc_btn.setEnabled(True)
        self.load_doc_btn.setText("Загрузить документ")
        
        if success:
            # Обновляем список документов
            self.docs_list.clear()
            for doc_name in self.doc_processor.get_document_list():
                self.docs_list.addItem(doc_name)
            
            # Добавляем сообщение об успехе
            self.append_docs_message("Система", message)
        else:
            # Отображаем ошибку
            QMessageBox.warning(self, "Ошибка", message)
    
    def send_docs_query(self):
        """Отправка запроса к документам"""
        query = self.docs_input.text().strip()
        if not query:
            return
        
        # Очищаем поле ввода
        self.docs_input.clear()
        
        # Добавляем запрос в историю чата
        self.append_docs_message("Вы", query)
        
        # Проверяем наличие загруженных документов
        if not self.doc_processor.get_document_list():
            self.append_docs_message("Система", "Нет загруженных документов. Пожалуйста, загрузите документы перед выполнением запроса.")
            return
        
        # Сбрасываем флаг потоковой генерации
        self.streaming_active = False
        
        # Получаем настройку потоковой генерации
        use_streaming = model_settings.get("streaming", False)
        
        # Если стриминг отключен, показываем индикатор загрузки
        if not use_streaming:
            self.docs_chat_area.append('<span style="color: #888888;">Ассистент печатает...</span>')
        
        # Запускаем обработку в отдельном потоке
        self.doc_thread = DocumentThread(self.signals, self.doc_processor, query=query)
        self.doc_thread.start()
        
        # Деактивируем кнопку на время обработки
        self.docs_send_btn.setEnabled(False)
    
    def append_docs_message(self, sender, message):
        """Добавление сообщения в историю чата с документами"""
        color = "#0066cc" if sender == "Вы" else "#009933"
        if sender == "Ошибка":
            color = "#cc0000"
        elif sender == "Система":
            color = "#888888"
            
        timestamp = QDateTime.currentDateTime().toString("HH:mm")
        
        cursor = self.docs_chat_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Форматируем HTML для сообщения с улучшенным стилем
        html = f'<p><span style="font-weight: bold; color: {color};">[{timestamp}] {sender}:</span> '
        
        # Добавляем сообщение с переносами строк
        message_formatted = message.replace('\n', '<br/>')
        html += f'{message_formatted}</p>'
        
        # Вставляем HTML
        cursor.insertHtml(html)
        
        # Прокручиваем вниз
        self.docs_chat_area.setTextCursor(cursor)
        self.docs_chat_area.ensureCursorVisible()
    
    def browse_media_file(self):
        """Выбор медиа-файла для транскрибации"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Медиа файлы (*.mp3 *.wav *.mp4 *.avi *.mov *.m4a *.flac *.webm *.mkv)")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                self.transcribe_input.setText(filenames[0])
                # Переключаем радиокнопку на файловый режим
                self.file_radio.setChecked(True)
    
    def start_transcription(self):
        """Начало процесса транскрибации"""
        # Получаем источник
        source = self.transcribe_input.text().strip()
        if not source:
            QMessageBox.warning(self, "Ошибка", "Укажите источник для транскрибации")
            return
        
        # Определяем тип источника
        is_file = self.file_radio.isChecked()
        is_youtube = self.youtube_radio.isChecked()
        
        # Проверка валидности источника
        if is_file and not os.path.exists(source):
            QMessageBox.warning(self, "Ошибка", "Указанный файл не существует")
            return
        
        if is_youtube and not (source.startswith("http://") or source.startswith("https://")):
            QMessageBox.warning(self, "Ошибка", "Неверный формат URL")
            return
        
        # Запускаем транскрибацию в отдельном потоке
        self.transcribe_thread = TranscriptionThread(
            self.signals, 
            self.transcriber,
            file_path=source if is_file else None,
            youtube_url=source if is_youtube else None
        )
        self.transcribe_thread.start()
        
        # Деактивируем кнопку на время транскрибации
        self.start_transcribe_btn.setEnabled(False)
        self.start_transcribe_btn.setText("Транскрибация...")
        
        # Сбрасываем прогресс
        self.transcribe_progress.setValue(0)
    
    def update_progress_bar(self, value):
        """Обновление индикатора прогресса"""
        self.transcribe_progress.setValue(value)
    
    def handle_transcription_complete(self, success, text):
        """Обработка результата транскрибации"""
        # Восстанавливаем кнопку
        self.start_transcribe_btn.setEnabled(True)
        self.start_transcribe_btn.setText("Начать транскрибацию")
        
        if success:
            # Отображаем результат
            self.transcribe_result.setPlainText(text)
        else:
            # Отображаем ошибку
            QMessageBox.warning(self, "Ошибка", text)
            self.transcribe_result.setPlainText(f"Ошибка транскрибации: {text}")
    
    def copy_transcription(self):
        """Копирование результата транскрибации в буфер обмена"""
        text = self.transcribe_result.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Скопировано", "Результат транскрибации скопирован в буфер обмена")
    
    def save_transcription(self):
        """Сохранение результата транскрибации в файл"""
        text = self.transcribe_result.toPlainText()
        if not text:
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter("Текстовые файлы (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                try:
                    with open(filenames[0], 'w', encoding='utf-8') as f:
                        f.write(text)
                    QMessageBox.information(self, "Сохранено", f"Результат транскрибации сохранен в файл {filenames[0]}")
                except Exception as e:
                    QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def change_model_size(self, size):
        """Изменение размера модели для транскрибации"""
        self.transcriber.set_model_size(size)
    
    def change_transcription_language(self, language):
        """Изменение языка для транскрибации"""
        self.transcriber.set_language(language)
    
    def apply_theme(self):
        """Применение выбранной темы к приложению"""
        theme = self.model_config.config.get("theme", "light")
        
        if theme == "light":
            self.apply_light_theme()
        else:
            self.apply_dark_theme()
    
    def apply_light_theme(self):
        """Применение светлой темы"""
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #f8f8f8;
            }
            
            QWidget#header {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
            
            QTextEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #d0e8fa;
                color: #333333;
            }
            
            QLineEdit {
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                selection-background-color: #d0e8fa;
                background-color: white;
                color: #333333;
            }
            
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            
            QPushButton:hover {
                background-color: #3a76d8;
            }
            
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                background-color: white;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            
            QTabBar::tab {
                background-color: #e8e8e8;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 2px;
                color: #333333;
            }
            
            QTabBar::tab:selected {
                background-color: white;
            }
            
            QListWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 5px;
                selection-background-color: #e0e0e0;
                color: #333333;
            }
            
            QListWidget::item {
                padding: 5px;
                border-radius: 4px;
            }
            
            QListWidget::item:selected {
                background-color: #e0e0e0;
            }
            
            QLabel {
                color: #333333;
            }
            
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 4px;
                color: #333333;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #e0e0e0;
                color: #333333;
            }
        """)
    
    def apply_dark_theme(self):
        """Применение темной темы"""
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #2d2d2d;
            }
            
            QWidget#header {
                background-color: #333333;
                border-bottom: 1px solid #444444;
            }
            
            QTextEdit {
                background-color: #3d3d3d;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #505050;
                color: #e0e0e0;
            }
            
            QLineEdit {
                padding: 10px;
                border: 1px solid #444444;
                border-radius: 8px;
                selection-background-color: #505050;
                background-color: #3d3d3d;
                color: #e0e0e0;
            }
            
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            
            QPushButton:hover {
                background-color: #5a96f8;
            }
            
            QPushButton:pressed {
                background-color: #3a76d8;
            }
            
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #333333;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 2px;
                color: #e0e0e0;
            }
            
            QTabBar::tab:selected {
                background-color: #333333;
            }
            
            QListWidget {
                background-color: #3d3d3d;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 5px;
                selection-background-color: #505050;
                color: #e0e0e0;
            }
            
            QListWidget::item {
                padding: 5px;
                border-radius: 4px;
            }
            
            QListWidget::item:selected {
                background-color: #505050;
            }
            
            QLabel {
                color: #e0e0e0;
            }
            
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
                color: #e0e0e0;
            }
            
            QComboBox::drop-down {
                border: none;
            }
            
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                selection-background-color: #505050;
                color: #e0e0e0;
            }
        """)
    
    def show_llm_settings(self):
        """Показать диалог настроек LLM"""
        # Закрываем боковую панель
        if self.sidebar_frame.width() > 0:
            self.toggle_sidebar()
        
        # Создаем и показываем диалог настроек
        dialog = ModelSettingsDialog(self)
        
        if dialog.exec():
            # Если пользователь нажал "Сохранить", применяем новые настройки
            new_settings = dialog.get_settings()
            
            # Показываем диалог с информацией о перезагрузке модели
            QMessageBox.information(
                self,
                "Перезагрузка модели",
                "Настройки сохранены. Модель будет перезагружена с новыми параметрами."
            )
            
            # Применяем новые настройки
            try:
                update_model_settings(new_settings)
                QMessageBox.information(
                    self,
                    "Успех",
                    "Модель успешно перезагружена с новыми настройками."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    f"Не удалось перезагрузить модель: {str(e)}"
                )
    
    def show_interface_settings(self):
        """Показать диалог настроек интерфейса"""
        # Закрываем боковую панель
        if self.sidebar_frame.width() > 0:
            self.toggle_sidebar()
        
        # Создаем диалог
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки интерфейса")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Настройка темы
        theme_layout = QFormLayout()
        theme_combo = QComboBox()
        theme_combo.addItems(["Светлая тема", "Темная тема"])
        current_theme = self.model_config.config.get("theme", "light")
        theme_combo.setCurrentIndex(0 if current_theme == "light" else 1)
        
        theme_layout.addRow("Тема оформления:", theme_combo)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(dialog.reject)
        
        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(dialog.accept)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(save_button)
        
        # Добавляем в основной макет
        layout.addLayout(theme_layout)
        layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Сохраняем настройки при принятии
        if dialog.exec():
            new_theme = "light" if theme_combo.currentIndex() == 0 else "dark"
            if new_theme != self.model_config.config.get("theme", "light"):
                self.model_config.config["theme"] = new_theme
                self.model_config.save_config()
                self.apply_theme()
                QMessageBox.information(
                    self,
                    "Тема изменена",
                    "Тема оформления успешно изменена."
                )

    def handle_streaming_chunk(self, chunk, accumulated_text):
        """Обработка фрагмента потоковой генерации"""
        # Определяем, на какой вкладке происходит потоковая генерация
        current_tab_index = self.tabs.currentIndex()
        
        # Обновляем текст в соответствующей вкладке
        if current_tab_index == 0:  # Текстовый чат
            self.update_streaming_message_in_chat(chunk, accumulated_text)
        elif current_tab_index == 1:  # Голосовой чат
            self.update_streaming_message_in_voice(chunk, accumulated_text)
        elif current_tab_index == 2:  # Документы
            self.update_streaming_message_in_docs(chunk, accumulated_text)

    def update_streaming_message_in_chat(self, chunk, accumulated_text):
        """Обновление потокового сообщения в текстовом чате"""
        # Если это первый фрагмент сообщения
        if not self.streaming_active:
            self.streaming_active = False
            # Добавляем новое сообщение с начальным текстом
            color = "#009933"  # цвет для сообщений ассистента
            timestamp = QDateTime.currentDateTime().toString("HH:mm")
            
            html = f'''
            <div style="margin-bottom: 10px;">
                <div style="white-space: pre-wrap;">
                    <span style="font-weight: bold; color: {color};">[{timestamp}] Агент:</span> <span id="streaming-message">{chunk}</span>
                </div>
            </div>
            '''
            
            # Добавляем сообщение в историю чата
            self.chat_history.append(html)
            self.current_stream_message = chunk
        else:
            # Обновляем существующее сообщение
            html = self.chat_history.toHtml()
            
            # Заменяем содержимое span с id="streaming-message" на новый текст
            updated_html = html.replace(
                f'<span id="streaming-message">{self.current_stream_message}</span>', 
                f'<span id="streaming-message">{accumulated_text}</span>'
            )
            
            self.chat_history.setHtml(updated_html)
            self.current_stream_message = accumulated_text
            
            # Прокручиваем до конца
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_history.setTextCursor(cursor)

    def update_streaming_message_in_voice(self, chunk, accumulated_text):
        """Обновление потокового сообщения в голосовом чате"""
        # Если это первый фрагмент сообщения
        if not self.streaming_active:
            self.streaming_active = False
            # Добавляем новое сообщение с начальным текстом
            color = "#009933"  # цвет для сообщений ассистента
            timestamp = QDateTime.currentDateTime().toString("HH:mm")
            
            html = f'''
            <div style="margin-bottom: 10px;">
                <div style="white-space: pre-wrap;">
                    <span style="font-weight: bold; color: {color};">[{timestamp}] Ассистент:</span> <span id="streaming-voice-message">{chunk}</span>
                </div>
            </div>
            '''
            
            # Добавляем сообщение в историю голосового чата
            self.voice_history.append(html)
            self.current_stream_message = chunk
        else:
            # Обновляем существующее сообщение
            html = self.voice_history.toHtml()
            
            # Заменяем содержимое span с id="streaming-voice-message" на новый текст
            updated_html = html.replace(
                f'<span id="streaming-voice-message">{self.current_stream_message}</span>', 
                f'<span id="streaming-voice-message">{accumulated_text}</span>'
            )
            
            self.voice_history.setHtml(updated_html)
            self.current_stream_message = accumulated_text
            
            # Прокручиваем до конца
            cursor = self.voice_history.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.voice_history.setTextCursor(cursor)

    def update_streaming_message_in_docs(self, chunk, accumulated_text):
        """Обновление потокового сообщения в чате документов"""
        # Если это первый фрагмент сообщения
        if not self.streaming_active:
            self.streaming_active = False
            # Добавляем новое сообщение с начальным текстом
            color = "#009933"  # цвет для сообщений ассистента
            timestamp = QDateTime.currentDateTime().toString("HH:mm")
            
            cursor = self.docs_chat_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            
            # Форматируем HTML для сообщения
            html = f'<p><span style="font-weight: bold; color: {color};">[{timestamp}] Агент:</span> '
            html += f'<span id="streaming-docs-message">{chunk}</span></p>'
            
            # Вставляем HTML
            cursor.insertHtml(html)
            self.current_stream_message = chunk
        else:
            # Обновляем существующее сообщение
            html = self.docs_chat_area.toHtml()
            
            # Заменяем содержимое span с id="streaming-docs-message" на новый текст
            updated_html = html.replace(
                f'<span id="streaming-docs-message">{self.current_stream_message}</span>', 
                f'<span id="streaming-docs-message">{accumulated_text}</span>'
            )
            
            self.docs_chat_area.setHtml(updated_html)
            self.current_stream_message = accumulated_text
            
            # Прокручиваем до конца
            cursor = self.docs_chat_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.docs_chat_area.setTextCursor(cursor)

    def append_message(self, sender, message, error=False):
        """Добавление сообщения в историю чата"""
        # Определяем цвет в зависимости от отправителя
        if error:
            color = "#FF0000"  # красный для ошибок
        elif sender == "Пользователь":
            color = "#0066CC"  # синий для пользователя
        else:
            color = "#009933"  # зеленый для ассистента
        
        # Форматируем текущее время
        timestamp = QDateTime.currentDateTime().toString("HH:mm")
        
        # Форматируем сообщение в HTML
        formatted_message = message.replace("\n", "<br>")
        html = f'''
        <div style="margin-bottom: 10px;">
            <div style="white-space: pre-wrap;">
                <span style="font-weight: bold; color: {color};">[{timestamp}] {sender}:</span> {formatted_message}
            </div>
        </div>
        '''
        
        # Добавляем сообщение в историю чата
        self.chat_history.append(html)
        
        # Прокручиваем до конца
        self.chat_history.moveCursor(QTextCursor.MoveOperation.End)

    def append_voice_message(self, sender, message, error=False):
        """Добавление сообщения в историю голосового чата"""
        # Определяем цвет в зависимости от отправителя
        if error:
            color = "#FF0000"  # красный для ошибок
        elif sender == "Пользователь":
            color = "#0066CC"  # синий для пользователя
        else:
            color = "#009933"  # зеленый для ассистента
        
        # Форматируем текущее время
        timestamp = QDateTime.currentDateTime().toString("HH:mm")
        
        # Форматируем сообщение в HTML
        formatted_message = message.replace("\n", "<br>")
        html = f'''
        <div style="margin-bottom: 10px;">
            <div style="white-space: pre-wrap;">
                <span style="font-weight: bold; color: {color};">[{timestamp}] {sender}:</span> {formatted_message}
            </div>
        </div>
        '''
        
        # Добавляем сообщение в историю голосового чата
        self.voice_history.append(html)
        
        # Прокручиваем до конца
        self.voice_history.moveCursor(QTextCursor.MoveOperation.End)

    def handle_voice_response(self, response):
        """Обработка ответа от модели для голосового режима"""
        # Если был потоковый режим, то полный ответ уже отображен
        if self.streaming_active:
            self.streaming_active = False
        else:
            # Удаляем сообщение "Ассистент печатает..." если оно есть
            html = self.voice_history.toHtml()
            html = html.replace('<span style="color: #888888;">Ассистент печатает...</span>', '')
            self.voice_history.setHtml(html)
            
            # Добавляем ответ в историю
            self.append_voice_message("Ассистент", response)
        
        # Озвучиваем ответ
        speaker = self.model_config.config.get("voice_speaker", "baya")
        threading.Thread(target=self.speak_and_resume, args=(response, speaker), daemon=True).start()

    def speak_and_resume(self, text, speaker="baya"):
        """Озвучивание текста с последующим возобновлением распознавания"""
        try:
            # Если распознавание активно, приостанавливаем его на время озвучивания
            if self.recognition_active and self.voice_recognition_thread:
                self.voice_recognition_thread.pause()
            
            # Озвучиваем текст при помощи голосового синтезатора
            voice = Voice(speaker)
            voice.say(text)
            
            # Возобновляем распознавание, если оно было активно
            if self.recognition_active and self.voice_recognition_thread:
                self.voice_recognition_thread.resume()
        except Exception as e:
            print(f"Ошибка при озвучивании: {e}")
            # Все равно возобновляем распознавание в случае ошибки
            if self.recognition_active and self.voice_recognition_thread:
                self.voice_recognition_thread.resume()

    def handle_voice_error(self, error):
        """Обработка ошибок голосового режима"""
        # Отображаем ошибку в истории голосового чата
        self.append_voice_message("Система", f"Ошибка: {error}", error=True)
        
        # Если распознавание голоса активно, останавливаем его
        if self.recognition_active:
            self.stop_voice_recognition()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 