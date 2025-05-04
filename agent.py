from llama_cpp import Llama
from config import MODEL_PATH
import os
import glob
import json

# Класс для хранения настроек модели
class ModelSettings:
    def __init__(self):
        self.settings_file = "llm_settings.json"
        # Настройки модели по умолчанию
        self.default_settings = {
            "context_size": 2048,      # Размер контекста
            "output_tokens": 512,      # Размер выходного текста
            "batch_size": 512,         # Размер батча
            "n_threads": 2,            # Количество потоков
            "use_mmap": True,          # Использовать mmap
            "use_mlock": False,        # Блокировать в памяти
            "verbose": True,           # Подробный вывод
            "temperature": 0.7,        # Температура генерации
            "top_p": 0.95,             # Top-p sampling
            "repeat_penalty": 1.05,    # Штраф за повторения
            "use_gpu": False,          # Использовать GPU
            "streaming": True         # Использовать потоковую генерацию
        }
        self.settings = self.default_settings.copy()
        self.load_settings()
    
    def load_settings(self):
        """Загрузка настроек из файла"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    self.settings.update(loaded_settings)
                print("Настройки модели загружены")
        except Exception as e:
            print(f"Ошибка при загрузке настроек модели: {str(e)}")
    
    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print("Настройки модели сохранены")
        except Exception as e:
            print(f"Ошибка при сохранении настроек модели: {str(e)}")
    
    def get(self, key, default=None):
        """Получение значения настройки"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Установка значения настройки"""
        if key in self.settings:
            self.settings[key] = value
            self.save_settings()
            return True
        return False
    
    def reset_to_defaults(self):
        """Сброс настроек к значениям по умолчанию"""
        self.settings = self.default_settings.copy()
        self.save_settings()
    
    def get_all(self):
        """Получение всех настроек"""
        return self.settings.copy()

# Создаем экземпляр класса настроек
model_settings = ModelSettings()

# Настройки модели
MODEL_CONTEXT_SIZE = model_settings.get("context_size")
DEFAULT_OUTPUT_TOKENS = model_settings.get("output_tokens")
VERBOSE_OUTPUT = model_settings.get("verbose")

# Поиск доступных моделей
def find_available_model():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print(f"Директория с моделями не существует: {models_dir}")
        return None
    
    # Ищем модели с расширением .gguf
    model_files = glob.glob(os.path.join(models_dir, '*.gguf'))
    if model_files:
        print(f"Найдена модель: {model_files[0]}")
        return model_files[0]
    
    print("Модели в формате GGUF не найдены в директории models/")
    return None

# Инициализация модели с проверкой существования файла
llm = None

def initialize_model():
    """Инициализация модели с текущими настройками"""
    global llm
    
    # Освобождаем ресурсы, если модель уже была загружена
    if llm is not None:
        del llm
        llm = None
        
    try:
        model_to_use = MODEL_PATH
        if not os.path.exists(model_to_use):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Модель по указанному пути не найдена: {model_to_use}")
            model_to_use = find_available_model()
        
        if model_to_use and os.path.exists(model_to_use):
            use_gpu = model_settings.get("use_gpu", False)
            device_type = "GPU" if use_gpu else "CPU"
            print(f"Загружаю модель из: {model_to_use} (устройство: {device_type})")
            try:
                # Параметры для модели с текущими настройками
                llm = Llama(
                    model_path=model_to_use,
                    n_ctx=model_settings.get("context_size"),
                    n_batch=model_settings.get("batch_size"),
                    use_mmap=model_settings.get("use_mmap"),
                    use_mlock=model_settings.get("use_mlock"),
                    verbose=model_settings.get("verbose"),
                    seed=42,                          # Фиксированное зерно для стабильности
                    n_threads=model_settings.get("n_threads"),
                    use_gpu=use_gpu
                )
                print(f"Модель успешно загружена на {device_type} с контекстным окном {model_settings.get('context_size')} токенов!")
                return True
            except Exception as e:
                print(f"ОШИБКА: Не удалось загрузить модель: {str(e)}")
                raise
        else:
            print("ОШИБКА: Не удалось найти подходящую модель.")
            raise ValueError("Модель не найдена")
    except Exception as e:
        print(f"ОШИБКА при загрузке модели: {str(e)}")
        raise

try:
    initialize_model()
except Exception as e:
    print(f"ОШИБКА при инициализации модели: {str(e)}")

def update_model_settings(new_settings):
    """Обновление настроек модели и перезагрузка"""
    global model_settings, MODEL_CONTEXT_SIZE, DEFAULT_OUTPUT_TOKENS, VERBOSE_OUTPUT
    
    # Обновляем настройки
    for key, value in new_settings.items():
        model_settings.set(key, value)
    
    # Обновляем глобальные переменные
    MODEL_CONTEXT_SIZE = model_settings.get("context_size")
    DEFAULT_OUTPUT_TOKENS = model_settings.get("output_tokens")
    VERBOSE_OUTPUT = model_settings.get("verbose")
    
    # Перезагружаем модель с новыми настройками
    return initialize_model()

def prepare_prompt(text, system_prompt=None):
    """Подготовка промпта в правильном формате"""
    if system_prompt is None:
        system_prompt = "Ты умный и полезный русскоязычный ассистент. Отвечай подробно и по существу на заданный вопрос."
    
    # Базовый шаблон для чата
    return f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{text.strip()}
<|im_end|>
<|im_start|>assistant
"""

def ask_agent(prompt, history=None, max_tokens=None, streaming=False, stream_callback=None):
    if llm is None:
        raise ValueError("Модель не загружена. Пожалуйста, убедитесь, что модель инициализирована.")
    
    # Если не указано количество токенов, берем из настроек
    if max_tokens is None:
        max_tokens = model_settings.get("output_tokens")
    
    # Формируем вход (можно добавить историю позже)
    try:
        print(f"[LLM] Получен запрос: {prompt.strip()[:50]}...")
        print(f"[LLM] Режим потоковой генерации: {'включен' if streaming else 'выключен'}")
        
        # Используем правильный формат запроса
        full_prompt = prepare_prompt(prompt)
        
        # Если включен режим потоковой генерации
        if streaming and stream_callback:
            print("[LLM] Запускаем потоковую генерацию")
            # Инициализируем переменную для накопления текста
            accumulated_text = ""
            
            # Создаем генератор для потоковой обработки
            generator = llm(
                full_prompt,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
                temperature=model_settings.get("temperature"),
                top_p=model_settings.get("top_p"),
                repeat_penalty=model_settings.get("repeat_penalty"),
                stream=True  # Включаем потоковую генерацию
            )
            
            # Обрабатываем каждый фрагмент
            chunk_counter = 0
            for output in generator:
                chunk = output["choices"][0]["text"]
                accumulated_text += chunk
                chunk_counter += 1
                
                if chunk_counter <= 3 or chunk_counter % 10 == 0:
                    print(f"[LLM] Фрагмент {chunk_counter}: '{chunk}', длина: {len(chunk)}")
                    if len(accumulated_text) <= 100:
                        print(f"[LLM] Накопленный текст: '{accumulated_text}'")
                    else:
                        print(f"[LLM] Накопленный текст (первые 50 символов): '{accumulated_text[:50]}...'")
                
                # Вызываем колбэк с текущим фрагментом
                stream_callback(chunk, accumulated_text)
            
            print(f"[LLM] Потоковая генерация завершена, всего фрагментов: {chunk_counter}")
            if len(accumulated_text) <= 100:
                print(f"[LLM] Итоговый текст: '{accumulated_text}'")
            else:
                print(f"[LLM] Итоговый текст (первые 100 символов): '{accumulated_text[:100]}...'")
            
            return accumulated_text
        else:
            # Обычная генерация без стриминга
            print("[LLM] Запускаем обычную генерацию")
            output = llm(
                full_prompt,
                max_tokens=max_tokens,     # Размер ответа
                stop=["<|im_end|>", "<|im_start|>"],  # Стоп-токены для формата чата
                echo=False,                # Не возвращать входной текст
                temperature=model_settings.get("temperature"),
                top_p=model_settings.get("top_p"),
                repeat_penalty=model_settings.get("repeat_penalty")
            )
            
            generated_text = output["choices"][0]["text"].strip()
            
            if len(generated_text) <= 100:
                print(f"[LLM] Генерация завершена, результат: '{generated_text}'")
            else:
                print(f"[LLM] Генерация завершена, результат (первые 100 символов): '{generated_text[:100]}...'")
            
            # Обрабатываем случай пустого вывода
            if not generated_text:
                print("[LLM] Модель вернула пустой ответ, пробуем повторно с другими параметрами...")
                # Более безопасные параметры для повторной попытки
                output = llm(
                    prompt.strip(),  # Более простой формат
                    max_tokens=256,  # Уменьшенное число токенов
                    temperature=0.5, # Более низкая температура
                    echo=False
                )
                generated_text = output["choices"][0]["text"].strip()
                
                if len(generated_text) <= 100:
                    print(f"[LLM] Повторная генерация завершена, результат: '{generated_text}'")
                else:
                    print(f"[LLM] Повторная генерация завершена, результат (первые 100 символов): '{generated_text[:100]}...'")
            
            return generated_text
    except Exception as e:
        print(f"ОШИБКА при генерации ответа: {str(e)}")
        # Вместо непосредственной передачи ошибки, возвращаем сообщение об ошибке
        return f"Извините, произошла ошибка при генерации ответа: {str(e)}. Попробуйте задать вопрос иначе или позже."