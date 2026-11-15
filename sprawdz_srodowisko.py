"""Skrypt weryfikujący przygotowanie środowiska oraz połączenie z modelem Google Gemma w Ollama."""

import sys
import time

MODEL_NAME = "gemma2:2b"


def main() -> None:
    print("=== Weryfikacja środowiska (Warsztat 4: Kupiec LLM - Google Gemma) ===")
    print(f"Wersja Python: {sys.version.split()[0]}")

    try:
        import ollama
        import pydantic
        print(f"[OK] Biblioteka ollama zainstalowana.")
        print(f"[OK] Biblioteka pydantic ({pydantic.__version__}) zainstalowana.")
    except ImportError as e:
        print(f"[BŁĄD] Brak wymaganej biblioteki: {e}")
        print("Uruchom: pip install -r requirements.txt")
        sys.exit(1)

    print(f"\nSprawdzanie połączenia z serwerem Ollama i modelu '{MODEL_NAME}'...")
    try:
        start = time.perf_counter()
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Odpowiedz jednym słowem: działa?"}],
            options={"temperature": 0.1, "num_predict": 15},
        )
        elapsed = time.perf_counter() - start
        content = response["message"]["content"].strip()
        print(f"[OK] Odpowiedź modelu ({elapsed:.2f} s): {content}")
        print("\nŚrodowisko jest w pełni gotowe do pracy!")
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wywołać modelu '{MODEL_NAME}': {e}")
        print("Upewnij się, że:")
        print("  1. Aplikacja Ollama jest uruchomiona w tle (`ollama serve`).")
        print(f"  2. Model został pobrany przed zajęciami (`ollama pull {MODEL_NAME}`).")
        sys.exit(1)


if __name__ == "__main__":
    main()
