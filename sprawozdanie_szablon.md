# Warsztat 4 — Rozmowa z kupcem sterowanym przez LLM (Google AI Studio)

## 1. Temat i autorzy

**Temat:** Rozmowa z kupcem sterowanym przez LLM — połączenie swobodnego dialogu ze ściśle określonym stanem gry  
**Data:** ....................................  
**Numer/nazwa zespołu:** ....................................  
**Członkowie zespołu (imiona i nazwiska):** ........................................................................  
**Przewodnik z zespołu liderskiego (imię i nazwisko):** ....................................................  

---

## 2. Cel i wykonane zmiany

**Cel ćwiczenia:**  
Zbudowanie tekstowego dialogu z kupcem w grze RPG z wykorzystaniem potężnego modelu językowego Gemma udostępnianego przez chmurę Google AI Studio. Poznanie i praktyczna implementacja podziału odpowiedzialności pomiędzy generatywny model językowy (rozpoznawanie intencji gracza i generowanie odpowiedzi w roli postaci w formacie JSON) a deterministyczną logikę programu (walidacja struktury JSON, weryfikacja dozwolonych akcji oraz rozstrzyganie transakcji na podstawie stanu gry: salda monet i zapasu mikstur).

**Podział odpowiedzialności (0–10 min):**
- **Za co odpowiada model LLM (`gemma-2-2b-it`):** ........................................................................
- **Za co odpowiada logika programu (Python):** ........................................................................
- **Dlaczego program nie może ufać samej treści wypowiedzi modelu (`reply`) przy zmianie stanu gry?** ........................................................................

**Wykonane zmiany i uzupełnione elementy kodu:**
1. **Instrukcja systemowa (`build_merchant_prompt`) i schemat JSON (`MerchantResponse`):**  
   *(Wklej poniżej krótką treść użytego promptu systemowego dla kupca bez przeklejania całego kodu notatnika)*
   ```text
   [TUTAJ WKLEJ TREŚĆ SWOJEGO PROMPTU SYSTEMOWEGO]
   ```
2. **Walidacja odpowiedzi i logika transakcji (`parse_and_validate_response`, `process_turn`):**  
   *(Opisz w 2–4 zdaniach, jakie warunki sprawdza program przed odjęciem monet i wydaniem mikstury oraz jak reaguje na niepoprawny JSON, nieznaną akcję lub timeout)*  
   ................................................................................................................................................

---

## 3. Eksperyment

**Środowisko i model:**
| Parametr | Użyta wartość |
|---|---|
| Nazwa i wersja modelu LLM | `gemma-2-2b-it` (Google Gemma) |
| Temperatura generowania (`temperature`) | `0.2` |
| Ziarno losowości (`seed`) | `42` |
| Limit czasu oczekiwania (`timeout_sec`) | `20.0` s |
| Wymuszanie schematu odpowiedzi (`format`) | JSON Schema (`MerchantResponse.model_json_schema()`) |
| Domyślny stan początkowy gry | Gracz: `15` monet, Kupiec: `2` mikstury, Cena: `10` monet |
| Dozwolone akcje (`ALLOWED_ACTIONS`) | `["talk", "buy_potion"]` |

**Scenariusze testowe:**  
Przeprowadzono 6 obowiązkowych prób obejmujących: (1) pytanie o ofertę, (2) poprawny zakup mikstury, (3) ponowny zakup bez wystarczających środków (kontynuujący stan po 2. próbie: 5 monet, 1 mikstura), (4) zakup przy pustym magazynie (0 mikstur), (5) niejasną wypowiedź gracza oraz (6) żądanie nieobsługiwanej czynności (np. sprzedaż przedmiotu / kredyt).

---

## 4. Wyniki

### 4.1. Tabela sześciu testów obowiązkowych

> Wpisz rzeczywiste wyniki uzyskane podczas eksperymentów (możesz wygenerować tabelę pomocniczą funkcją `print_markdown_table(test_results)` w notatniku).

| Nr | Scenariusz | Wypowiedź gracza | Stan początkowy | Poprawny JSON? | Proponowana akcja (`action`) | Odpowiedź modelu (`reply` - skrót) | Decyzja programu (walidacja i stan) | Stan końcowy |
|---:|---|---|---|---|---|---|---|---|
| 1 | Pytanie o ofertę | | 15 monet, 2 mikstury | | | | | |
| 2 | Poprawny zakup | | 15 monet, 2 mikstury | | | | | |
| 3 | Ponowny zakup (brak monet) | | 5 monet, 1 mikstura | | | | | |
| 4 | Zakup (pusty magazyn) | | 15 monet, 0 mikstur | | | | | |
| 5 | Niejasna wypowiedź | | 15 monet, 2 mikstury | | | | | |
| 6 | Nieobsługiwana czynność | | 15 monet, 2 mikstury | | | | | |

### 4.2. Analiza rozdziału trzech warstw poprawności

- **Poprawność formatu (JSON Schema):** Ile razy na 6 prób model zwrócił poprawny syntaktycznie JSON zgodny ze schematem? Czy testy odporności walidatora (błędny JSON, nieznana akcja, timeout) zachowały nienaruszony stan gry?  
  ................................................................................................................................................
- **Poprawność rozpoznania intencji (`action`):** W których scenariuszach model trafnie rozpoznał intencję (`talk` vs `buy_potion`), a w których zaproponował błędną akcję (np. `buy_potion` mimo braku monet lub braku towaru)?  
  ................................................................................................................................................
- **Poprawność zmiany stanu gry:** Czy mimo ewentualnych błędnych propozycji modelu (halucynacji transakcji w polu `reply` lub nadmiarowego `buy_potion`) stan gry w programie zmienił się wyłącznie w próbie nr 2?  
  ................................................................................................................................................

### 4.3. Zakres dodatkowy (opcjonalnie — dla zespołów, które skończyły wcześniej)

**Opis drugiej postaci (styl wypowiedzi):** ........................................................................  
**Zaobserwowany wpływ zmiany stylu na poprawność formatu JSON i rozpoznawanie akcji `action`:**  
................................................................................................................................................

---

## 5. Wnioski i ograniczenia

**Co zadziałało i jakie obserwacje to potwierdzają:**  
................................................................................................................................................

**Co nie zadziałało / napotkane trudności i ograniczenia modelu (np. niespójność między `reply` a decyzją silnika gry):**  
................................................................................................................................................

**Wkład poszczególnych członków zespołu:**

| Imię i nazwisko | Wykonane zadania podczas warsztatu |
|---|---|
| | |
| | |
| | |
| | |
