# Warsztat 4 — Rozmowa z kupcem sterowanym przez LLM

## 1. Temat i autorzy
Uzupełnione elementy kodu: MerchantResponse, build_merchant_prompt.


**Temat:** Rozmowa z kupcem sterowanym przez LLM — połączenie swobodnego dialogu ze ściśle kontrolowanym stanem gry  
**Data:** _[uzupełnijcie]_  
**Numer/nazwa zespołu:** _[uzupełnijcie]_  
**Członkowie zespołu (imiona i nazwiska):** _[uzupełnijcie]_  
**Przewodnik z zespołu liderskiego (imię i nazwisko):** _[uzupełnijcie]_

## 2. Cel ćwiczenia

**Cel:** Zbudować dialog z kupcem NPC, w którym model LLM zwraca odpowiedź JSON (`reply`, `action`), a silnik gry w Pythonie waliduje odpowiedź i sam decyduje o zmianie stanu (`coins`, `potions`).

## 3. Przebieg prac (chronologicznie)

### 3.1. Etap 0–10 min: środowisko i podział odpowiedzialności

**Model i środowisko:**
| Parametr | Użyta wartość |
|---|---|
| Model LLM | `gemma-4-26b-a4b-it` (Google AI Studio API) |
| Temperatura (`temperature`) | `0.2` |
| Limit czasu (`timeout_sec`) | `20.0` s |
| Dozwolone akcje | `talk`, `buy_potion` |
| Domyślny stan gry | 15 monet, 2 mikstury, cena 10 monet |

**Podział odpowiedzialności:**
- **Za co odpowiada model LLM:** _[uzupełnijcie]_  
- **Za co odpowiada logika programu (Python):** _[uzupełnijcie]_  
- **Dlaczego nie można ufać samemu `reply` przy zmianie stanu gry:** _[uzupełnijcie]_

### 3.2. Etap 10–30 min: prompt systemowy i schemat JSON

**Uzupełnione elementy kodu:** `MerchantResponse`, `build_merchant_prompt`.

**Wymuszanie formatu odpowiedzi:** JSON Schema (`MerchantResponse.model_json_schema()`).

**Skrócona treść finalnego promptu systemowego kupca:**  
_(to ma być treść z funkcji `build_merchant_prompt(...)` po Waszych zmianach — nie zmyślony opis i nie cały kod, tylko sam tekst instrukcji)_

```text
[TUTAJ WKLEJCIE SWÓJ FINALNY PROMPT SYSTEMOWY DLA KUPCA]
```

### 3.3. Etap 30–55 min: walidacja i deterministyczna logika gry

**Po uzupełnieniu:** `parse_and_validate_response`, `process_turn`.

**Opis działania (2–5 zdań):**  
_[uzupełnijcie: co sprawdzacie w JSON, jak obsługujecie nieznaną akcję, timeout i kiedy dokładnie zmienia się stan]_ 

**Testy odporności walidatora wykonane w notatniku:**
- uszkodzony JSON,
- akcja spoza listy dozwolonej,
- halucynacja zakupu przy niewystarczających środkach.

**Wniosek z testów odporności:** _[uzupełnijcie]_ 

### 3.4. Etap 55–75 min: 6 scenariuszy obowiązkowych

**Źródło scenariuszy:** `data/test_scenarios.json`.

| Nr | Scenariusz | Wypowiedź gracza | Stan początkowy | JSON poprawny? | `action` modelu | Decyzja programu | Stan końcowy |
|---:|---|---|---|---|---|---|---|
| 1 | Pytanie o ofertę | _[uzupełnijcie]_ | 15 monet, 2 mikstury | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |
| 2 | Poprawny zakup | _[uzupełnijcie]_ | 15 monet, 2 mikstury | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |
| 3 | Ponowny zakup (brak monet) | _[uzupełnijcie]_ | 5 monet, 1 mikstura | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |
| 4 | Zakup (pusty magazyn) | _[uzupełnijcie]_ | 15 monet, 0 mikstur | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |
| 5 | Niejasna wypowiedź | _[uzupełnijcie]_ | 15 monet, 2 mikstury | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |
| 6 | Nieobsługiwana czynność | _[uzupełnijcie]_ | 15 monet, 2 mikstury | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ |

### 3.5. Etap 75–90 min: podsumowanie wyników

**Analiza trzech warstw poprawności:**
- **Poprawność formatu (JSON Schema):** _[uzupełnijcie]_  
- **Poprawność rozpoznania intencji (`action`):** _[uzupełnijcie]_  
- **Poprawność zmian stanu gry:** _[uzupełnijcie]_  

## 4. Zakres dodatkowy (opcjonalnie)

**Druga postać i styl wypowiedzi:** _[uzupełnijcie]_  
**Wpływ stylu na poprawność JSON i `action`:** _[uzupełnijcie]_  
**Krótki komentarz porównawczy do kupca Barnaby:** _[uzupełnijcie]_ 

## 5. Podsumowanie i wnioski

**Co zadziałało najlepiej:** _[uzupełnijcie]_  
**Najważniejsze ograniczenia i trudności:** _[uzupełnijcie]_  
**Dlaczego wzorzec LLM + deterministyczny silnik jest bezpieczniejszy niż sam LLM:** _[uzupełnijcie]_  
**Stan rozwiązania na koniec 90 minut:** _[uzupełnijcie]_ 

**Wkład członków zespołu:**

| Imię i nazwisko | Wykonane zadania podczas warsztatu |
|---|---|
| _[osoba 1]_ | _[uzupełnijcie]_ |
| _[osoba 2]_ | _[uzupełnijcie]_ |
| _[osoba 3]_ | _[uzupełnijcie]_ |
| _[osoba 4, jeśli dotyczy]_ | _[uzupełnijcie]_ |
| _[osoba 5, jeśli dotyczy]_ | _[uzupełnijcie]_ |