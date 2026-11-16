"""Skrypt weryfikujący przygotowanie środowiska oraz połączenie z modelem Gemma w Google AI Studio."""

import os
import sys
import time

MODEL_NAME = "gemma-2-2b-it"


def main() -> None:
    print("=== Weryfikacja środowiska (Warsztat 4: Kupiec LLM - Google AI Studio) ===")
    print(f"Wersja Python: {sys.version.split()[0]}")

    try:
        import google.generativeai as genai
        import pydantic
        print(f"[OK] Biblioteka google-generativeai zainstalowana.")
        print(f"[OK] Biblioteka pydantic ({pydantic.__version__}) zainstalowana.")
    except ImportError as e:
        print(f"[BŁĄD] Brak wymaganej biblioteki: {e}")
        print("Uruchom: pip install -r requirements.txt")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[BŁĄD] Brak klucza API w zmiennych środowiskowych!")
        print("Ustaw klucz poleceniem:")
        print(" Windows PowerShell: $env:GEMINI_API_KEY=\"Twój_Klucz\"")
        print(" Linux/macOS: export GEMINI_API_KEY=\"Twój_Klucz\"")
        sys.exit(1)
    
    print("[OK] Zmienna środowiskowa GEMINI_API_KEY jest ustawiona.")
    genai.configure(api_key=api_key)

    print(f"\nSprawdzanie połączenia z Google AI Studio i modelu '{MODEL_NAME}'...")
    try:
        start = time.perf_counter()
        
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(
            "Odpowiedz jednym słowem: działa?",
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        elapsed = time.perf_counter() - start
        content = response.text.strip()
        print(f"[OK] Odpowiedź chmury Google ({elapsed:.2f} s): {content}")
        print("\nŚrodowisko jest w pełni gotowe do pracy!")
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wywołać modelu '{MODEL_NAME}' w AI Studio: {e}")
        print("Upewnij się, że:")
        print("  1. Masz połączenie z internetem.")
        print("  2. Twój klucz API jest poprawny (skopiowany z https://aistudio.google.com/api-keys).")
        sys.exit(1)


if __name__ == "__main__":
    main()
