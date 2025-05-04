from PIL import Image, ImageDraw, ImageFont
import os

# Убедимся, что директория assets существует
if not os.path.exists("assets"):
    os.makedirs("assets")

# Размер логотипа
width, height = 200, 200
background_color = (240, 240, 240)  # Светло-серый фон

# Создаем новое изображение
img = Image.new('RGB', (width, height), background_color)
draw = ImageDraw.Draw(img)

# Рисуем круг для фона
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 2 - 10
circle_color = (70, 130, 180)  # Синий цвет
draw.ellipse((center_x - radius, center_y - radius, 
               center_x + radius, center_y + radius), 
               fill=circle_color)

# Добавляем текст (если есть шрифт)
try:
    # Пытаемся найти системный шрифт
    font_size = 120
    font = ImageFont.truetype("arial.ttf", font_size)
    text = "M"
    text_color = (255, 255, 255)  # Белый цвет
    
    # Центрируем текст
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    position = ((width - text_width) // 2, (height - text_height) // 2 - 10)
    
    draw.text(position, text, font=font, fill=text_color)
except Exception as e:
    print(f"Не удалось загрузить шрифт: {e}")
    # Вместо этого нарисуем простой символ
    draw.rectangle((center_x - 30, center_y - 30, center_x + 30, center_y + 30), 
                  fill=(255, 255, 255))

# Сохраняем как PNG и ICO
img.save("assets/logo.png")
print("Логотип создан в assets/logo.png")

# Создаем версию для иконки
sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
try:
    img.save("assets/icon.ico", sizes=sizes)
    print("Иконка создана в assets/icon.ico")
except Exception as e:
    print(f"Не удалось создать .ico файл: {e}")
    # Создаем альтернативную иконку
    img.resize((32, 32)).save("assets/icon.png")
    print("Альтернативная иконка создана в assets/icon.png") 