import sys
import os
import argparse

def run_cli():
    """Запускает текстовый режим в консоли"""
    from agent import ask_agent
    from memory import save_to_memory
    
    print("MemoAI в текстовом режиме. Нажмите Ctrl+C для выхода.")
    
    try:
        while True:
            user_input = input("\nВы: ")
            save_to_memory("Пользователь", user_input)
            
            print("\nОбработка запроса...")
            response = ask_agent(user_input)
            
            print(f"\nАссистент: {response}")
            save_to_memory("Агент", response)
    except KeyboardInterrupt:
        print("\nРабота текстового режима завершена.")

def run_voice():
    """Запускает голосовой режим в консоли"""
    from voice import run_voice
    
    try:
        run_voice()
    except Exception as e:
        print(f"Ошибка в голосовом режиме: {e}")
        print("Переключаюсь на текстовый режим...")
        run_cli()

def run_gui():
    """Запускает графический интерфейс"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from gui import MainWindow
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

def main():
    # По умолчанию запускаем GUI
    try:
        # Проверяем, есть ли аргументы командной строки
        if len(sys.argv) > 1:
            # Создаем парсер аргументов командной строки
            parser = argparse.ArgumentParser(description="MemoAI - персональный ассистент с различными режимами работы")
            parser.add_argument('mode', nargs='?', choices=['text', 'voice', 'gui'], 
                                help='Режим работы: text - текстовый, voice - голосовой, gui - графический интерфейс')
            
            args = parser.parse_args()
            
            # Запускаем указанный режим
            if args.mode == 'text':
                run_cli()
            elif args.mode == 'voice':
                run_voice()
            elif args.mode == 'gui' or args.mode is None:
                run_gui()
        else:
            # Если аргументов нет, запускаем GUI
            run_gui()
    
    except Exception as e:
        print(f"Не удалось запустить графический интерфейс: {e}")
        # Предлагаем выбор режима
        print("\nВыберите режим:")
        print("1 - Текстовый режим")
        print("2 - Голосовой режим")
        
        choice = input("Ваш выбор: ")
        
        if choice == "1":
            run_cli()
        elif choice == "2":
            run_voice()
        else:
            print("Неизвестный выбор. Запускаю текстовый режим.")
            run_cli()

if __name__ == "__main__":
    main()