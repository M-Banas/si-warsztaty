# Warsztat 4 — Rozmowa z kupcem sterowanym przez LLM (Google AI Studio)

Wymagany Python 3.11 lub nowszy oraz aktywne połączenie z internetem.

## 1. Wygenerowanie klucza API (przed zajęciami)

Zaloguj się na swoje konto Google i wygeneruj klucz API w usłudze Google AI Studio:
### https://aistudio.google.com/api-keys

Skopiuj wygenerowany klucz. Na zajęciach będziemy używać potężnego modelu **Gemma 2 2B (`gemma-2-2b-it`)** udostępnianego przez chmurę Google.

## 2. Przygotowanie środowiska Python

Uruchom terminal w tym folderze:

```bash
python -m venv .venv
```

Aktywacja środowiska:
- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

Instalacja bibliotek:

```bash
python -m pip install -r requirements.txt
```

## 3. Ustawienie klucza API i weryfikacja środowiska

Zanim uruchomisz skrypt weryfikacyjny lub notatnik, musisz przypisać swój klucz do zmiennej środowiskowej:

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="WKLEJ_TUTAJ_SWOJ_KLUCZ_API"
python sprawdz_srodowisko.py
```

**Linux / macOS:**
```bash
export GEMINI_API_KEY="WKLEJ_TUTAJ_SWOJ_KLUCZ_API"
python sprawdz_srodowisko.py
```

## 4. Uruchomienie notatnika

W tym samym terminalu (gdzie aktywowano środowisko i ustawiono zmienną `GEMINI_API_KEY`) wpisz:
```bash
python -m notebook
```

Otwórz plik `warsztat4.ipynb` i uzupełniaj miejsca oznaczone `# TODO` zgodnie z harmonogramem zajęć. Pamiętaj o bieżącym uzupełnianiu pliku `sprawozdanie_szablon.md`.
