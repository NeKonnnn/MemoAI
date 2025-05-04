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
                            QFrame, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QObject, pyqtSignal, QThread, QDateTime
from PyQt6.QtGui import QFont, QIcon, QColor, QTextCursor

from agent import ask_agent, update_model_settings, model_settings
from memory import save_to_memory
from voice import speak_text, check_vosk_model, VOSK_MODEL_PATH

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

# Класс для фонового получения ответов от модели
class AgentThread(QThread):
    def __init__(self, signals, message, for_voice=False):
        super().__init__()
        self.signals = signals
        self.message = message
        self.for_voice = for_voice
        
    def run(self):
        try:
            # Получаем ответ от модели
            response = ask_agent(self.message)
            
            # Отправляем сигнал с ответом
            if self.for_voice:
                self.signals.voice_response_ready.emit(response)
            else:
                self.signals.response_ready.emit(response)
            
            # Сохраняем в историю
            save_to_memory("Агент", response)
            
        except Exception as e:
            # Отправляем сигнал с ошибкой
            self.signals.error_occurred.emit(str(e))

# Класс для распознавания речи в фоновом режиме
class VoiceRecognitionThread(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.running = False
        self.paused = False  # Флаг для приостановки распознавания
        self.queue = queue.Queue()
        
    def run(self):
        try:
            from vosk import Model, KaldiRecognizer
            import sounddevice as sd
            
            # Проверяем наличие модели
            if not check_vosk_model():
                self.signals.voice_error.emit("Модель распознавания речи не найдена")
                return
                
            # Инициализация модели распознавания речи
            model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(model, 16000)
            
            # Запуск аудио потока
            with sd.RawInputStream(
                samplerate=16000, 
                blocksize=8000, 
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            ):
                self.running = True
                
                while self.running:
                    # Если распознавание приостановлено, ждем
                    if self.paused:
                        time.sleep(0.1)
                        continue
                        
                    try:
                        data = self.queue.get(timeout=1)
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "").strip()
                            if text:
                                self.signals.voice_recognized.emit(text)
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            self.signals.voice_error.emit(str(e))
            
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Ошибка статуса: {status}")
        if not self.paused:  # Добавляем данные только если не на паузе
            self.queue.put(bytes(indata))
        
    def pause(self):
        """Приостановить распознавание"""
        self.paused = True
        
    def resume(self):
        """Возобновить распознавание"""
        self.paused = False
        
    def stop(self):
        """Полностью остановить распознавание"""
        self.running = False
        self.wait()

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
        self.context_size_spin.setValue(2048)
        self.output_tokens_spin.setValue(512)
        self.batch_size_spin.setValue(512)
        self.n_threads_spin.setValue(2)
        self.temperature_spin.setValue(0.7)
        self.top_p_spin.setValue(0.95)
        self.repeat_penalty_spin.setValue(1.05)
        self.verbose_combo.setCurrentIndex(0)
    
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
            "use_mmap": True,  # Оставляем эти параметры неизменными
            "use_mlock": False
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Инициализация конфигурации моделей
        self.model_config = ModelConfig()
        
        # Создаем объект сигналов
        self.signals = Signals()
        self.signals.response_ready.connect(self.handle_response)
        self.signals.error_occurred.connect(self.handle_error)
        self.signals.voice_recognized.connect(self.handle_voice_recognition)
        self.signals.voice_error.connect(self.handle_voice_error)
        self.signals.voice_response_ready.connect(self.handle_voice_response)
        
        # Инициализация переменных для голосового режима
        self.voice_recognition_thread = None
        self.is_listening = False
        self.is_responding = False  # Флаг, указывающий, что модель генерирует ответ
        
        # Базовая настройка окна
        self.setWindowTitle("MemoAI Ассистент")
        self.setMinimumSize(1000, 700)
        
        # Создаем центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Основная компоновка
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Создаем боковую панель (шторку)
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(0)  # Изначально скрыта
        self.sidebar.setMinimumWidth(0)
        self.sidebar.setMaximumWidth(250)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.setup_sidebar()
        
        # Создаем основной контент
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем верхнюю панель
        self.header = QWidget()
        self.header.setFixedHeight(60)
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(10, 5, 10, 5)
        self.setup_header()
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # Более компактный вид
        
        # Вкладка чата
        self.chat_tab = QWidget()
        self.setup_chat_tab()
        self.tabs.addTab(self.chat_tab, "Текстовый чат")
        
        # Вкладка голосового чата
        self.voice_tab = QWidget()
        self.setup_voice_tab()
        self.tabs.addTab(self.voice_tab, "Голосовой чат")
        
        # Добавляем элементы в основной контент
        self.content_layout.addWidget(self.header)
        self.content_layout.addWidget(self.tabs)
        
        # Добавляем шторку и основной контент в главную компоновку
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.content)
        
        # Применяем тему
        self.apply_theme()
    
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
        layout = QVBoxLayout(self.chat_tab)
        
        # История чата
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 11))
        layout.addWidget(self.chat_history)
        
        # Поле ввода и кнопка отправки
        input_layout = QHBoxLayout()
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Введите сообщение...")
        self.chat_input.setFont(QFont("Arial", 11))
        self.chat_input.returnPressed.connect(self.send_message)
        
        send_button = QPushButton("Отправить")
        send_button.setFixedWidth(100)
        send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        
        layout.addLayout(input_layout)
        
        # Добавляем приветственное сообщение
        self.append_message("Ассистент", "Привет! Я ваш AI-ассистент. Чем могу помочь?")
    
    def setup_voice_tab(self):
        """Настройка вкладки голосового чата"""
        layout = QVBoxLayout(self.voice_tab)
        
        # История голосового чата
        self.voice_history = QTextEdit()
        self.voice_history.setReadOnly(True)
        self.voice_history.setFont(QFont("Arial", 11))
        layout.addWidget(self.voice_history)
        
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
        
        layout.addLayout(control_layout)
        
        # Добавляем приветственное сообщение
        self.append_voice_message("Ассистент", "Привет! Нажмите кнопку микрофона, чтобы начать голосовое общение.")
    
    def toggle_sidebar(self):
        """Открытие/закрытие боковой панели"""
        # Текущая ширина
        current_width = self.sidebar.width()
        
        # Целевая ширина
        target_width = 250 if current_width == 0 else 0
        
        # Создаем анимацию
        self.animation = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.animation.setDuration(200)
        self.animation.setStartValue(current_width)
        self.animation.setEndValue(target_width)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()
        
        # Дублируем анимацию для максимальной ширины
        self.animation2 = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.animation2.setDuration(200)
        self.animation2.setStartValue(current_width)
        self.animation2.setEndValue(target_width)
        self.animation2.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation2.start()
    
    def show_models_dialog(self):
        """Показывает диалог управления моделями"""
        # Закрываем боковую панель
        if self.sidebar.width() > 0:
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
        if self.is_listening:
            self.stop_voice_recognition()
        else:
            self.start_voice_recognition()
            
    def start_voice_recognition(self):
        """Запуск распознавания речи"""
        if not check_vosk_model():
            self.handle_voice_error("Модель распознавания речи не найдена в директории model_small")
            return
            
        try:
            self.is_listening = True
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
        self.is_listening = False
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
        self.is_responding = True
        
        # Приостанавливаем распознавание речи на время ответа
        if self.voice_recognition_thread:
            self.voice_recognition_thread.pause()
        
        # Запускаем обработку сообщения в отдельном потоке
        self.agent_thread = AgentThread(self.signals, text, for_voice=True)
        self.agent_thread.start()
    
    def handle_voice_response(self, response):
        """Обработка ответа от модели для голосового режима"""
        # Добавляем ответ в историю
        self.append_voice_message("Ассистент", response)
        
        # Озвучиваем ответ
        speaker = self.model_config.config.get("voice_speaker", "baya")
        threading.Thread(target=self.speak_and_resume, args=(response, speaker), daemon=True).start()
        
    def speak_and_resume(self, text, speaker):
        """Озвучивает текст и возобновляет прослушивание"""
        # Озвучиваем текст
        speak_text(text, speaker)
        
        # Возобновляем прослушивание
        self.is_responding = False
        
        # Возвращаем статус в исходное состояние
        if self.is_listening:
            self.voice_status.setText("Слушаю... Говорите в микрофон")
            # Возобновляем распознавание речи
            if self.voice_recognition_thread:
                self.voice_recognition_thread.resume()
        else:
            self.voice_status.setText("Ожидание...")
    
    def handle_voice_error(self, error_message):
        """Обработка ошибок голосового режима"""
        self.append_voice_message("Ошибка", error_message)
        self.stop_voice_recognition()
    
    def append_voice_message(self, sender, message):
        """Добавление сообщения в историю голосового чата"""
        color = "#0066cc" if sender == "Вы" else "#009933"
        if sender == "Ошибка":
            color = "#cc0000"
        elif sender == "Система":
            color = "#888888"
            
        timestamp = QDateTime.currentDateTime().toString("hh:mm")
        self.voice_history.append(f'<span style="color: {color};">[{timestamp}] <b>{sender}:</b></span> {message}')
        self.voice_history.append("<br>")
        
        # Прокручиваем до конца
        cursor = self.voice_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.voice_history.setTextCursor(cursor)
    
    def append_message(self, sender, message):
        """Добавление сообщения в историю чата"""
        color = "#0066cc" if sender == "Вы" else "#009933"
        self.chat_history.append(f'<span style="font-weight: bold; color: {color};">{sender}:</span> {message}')
        self.chat_history.append('<br>')  # Пустая строка после сообщения
    
    def send_message(self):
        """Отправка сообщения"""
        message = self.chat_input.text().strip()
        if not message:
            return
        
        # Очищаем поле ввода
        self.chat_input.clear()
        
        # Добавляем сообщение пользователя
        self.append_message("Вы", message)
        
        # Сохраняем в историю
        save_to_memory("Пользователь", message)
        
        # Блокируем интерфейс
        self.chat_input.setEnabled(False)
        
        # Показываем индикатор
        self.chat_history.append('<span style="color: #888888;">Ассистент печатает...</span>')
        
        # Запускаем поток для получения ответа
        self.agent_thread = AgentThread(self.signals, message)
        self.agent_thread.start()
    
    def handle_response(self, response):
        """Обработчик ответа от модели"""
        # Удаляем индикатор "печатает..."
        html_content = self.chat_history.toHtml()
        html_content = html_content.replace(
            '<span style="color: #888888;">Ассистент печатает...</span>', 
            ''
        )
        self.chat_history.setHtml(html_content)
        
        # Добавляем ответ безопасным способом
        color = "#009933"
        self.chat_history.append(f'<span style="font-weight: bold; color: {color};">Ассистент:</span> {response}')
        self.chat_history.append('<br>')  # Пустая строка после сообщения
        
        # Восстанавливаем интерфейс
        self.chat_input.setEnabled(True)
        self.chat_input.setFocus()
    
    def handle_error(self, error):
        """Обработчик ошибки"""
        # Удаляем индикатор "печатает..."
        html_content = self.chat_history.toHtml()
        html_content = html_content.replace(
            '<span style="color: #888888;">Ассистент печатает...</span>', 
            ''
        )
        self.chat_history.setHtml(html_content)
        
        # Добавляем сообщение об ошибке безопасным способом
        self.chat_history.append(f'<span style="color: #cc0000;"><b>Ошибка:</b> {error}</span>')
        self.chat_history.append('<br>')  # Пустая строка после сообщения
        
        # Восстанавливаем интерфейс
        self.chat_input.setEnabled(True)
        self.chat_input.setFocus()
    
    def get_response(self, message):
        """Этот метод больше не используется, оставлен для сохранения истории"""
        pass
    
    def show_llm_settings(self):
        """Показать диалог настроек LLM"""
        # Закрываем боковую панель
        if self.sidebar.width() > 0:
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
        if self.sidebar.width() > 0:
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
    
    def apply_stylesheet(self):
        """Этот метод больше не используется, вместо него применяется apply_theme"""
        self.apply_theme()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 