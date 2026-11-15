# Warsztat 4 — Rozmowa z kupcem sterowanym przez LLM (Google Gemma)

Wymagany Python 3.11 lub nowszy oraz zainstalowana i uruchomiona aplikacja **Ollama**:

### https://ollama.com/download

## 1. Przygotowanie modelu Google Gemma (przed zajęciami)

Upewnij się, że serwer Ollama działa w tle, a następnie pobierz lekki model instrukcyjny Google Gemma (działa sprawnie na CPU typowego laptopa):

```bash
ollama pull gemma2:2b
```
*(Alternatywnie, jeśli prowadzący zalecił mniejszy model: `ollama pull gemma3:1b` lub `gemma2:9b` dla mocniejszych maszyn).*

## 2. Przygotowanie środowiska Python

Uruchom terminal w tym folderze:

```bash
python -m venv .venv
```

Aktywacja środowiska:
- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

Instalacja bibliotek i weryfikacja środowiska:

```bash
python -m pip install -r requirements.txt
python sprawdz_srodowisko.py
```

## 3. Uruchomienie notatnika

```bash
python -m notebook
```

Otwórz plik `warsztat4.ipynb` i uzupełniaj miejsca oznaczone `# TODO` zgodnie z harmonogramem zajęć. Pamiętaj o bieżącym uzupełnianiu pliku `sprawozdanie_szablon.md`.
