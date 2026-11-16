"""Skrypt weryfikujący przygotowanie środowiska oraz połączenie z modelem Gemma w Google AI Studio."""

import os
import sys
import time

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
        sys.exit(1)
    
    print("[OK] Zmienna środowiskowa GEMINI_API_KEY jest ustawiona.")
    genai.configure(api_key=api_key)

    print("\nWyszukiwanie dostępnych modeli Gemma w Twoim Google AI Studio...")
    available_gemma_models = []
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods and "gemma" in m.name.lower():
                available_gemma_models.append(m.name.replace("models/", ""))
    except Exception as e:
        print(f"[BŁĄD] Nie udało się pobrać listy modeli: {e}")
        sys.exit(1)

    if not available_gemma_models:
        print("[BŁĄD] Nie znaleziono żadnych modeli z rodziny 'Gemma' dostępnych dla Twojego konta w AI Studio.")
        sys.exit(1)
        
    print(f"[OK] Znaleziono obsługiwane modele Gemma: {', '.join(available_gemma_models)}")
    
    model_name = available_gemma_models[0]
    
    for m in available_gemma_models:
        if "gemma-4" in m or "gemma-2b" in m:
            model_name = m
            break

    print(f"\nTestowanie połączenia z modelem '{model_name}'...")
    try:
        start = time.perf_counter()
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            "Odpowiedz jednym słowem: działa?",
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        elapsed = time.perf_counter() - start
        content = response.text.strip()
        print(f"[OK] Odpowiedź chmury Google ({elapsed:.2f} s): {content}")
        print(f"\nŚrodowisko jest w pełni gotowe do pracy.")
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wywołać modelu '{model_name}' w AI Studio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
