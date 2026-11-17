# Warsztat 4: Rozmowa z kupcem sterowanym przez LLM (sprawozdanie)

## 1. Zespół

- **Temat:** Rozmowa z kupcem sterowanym przez LLM — połączenie swobodnego dialogu ze ścisłym stanem gry
- **Data:** _[uzupełnijcie]_
- **Numer lub nazwa zespołu:** _[uzupełnijcie]_
- **Skład zespołu:** _[uzupełnijcie]_
- **Opiekun z zespołu liderskiego:** _[uzupełnijcie]_

---

## 2. Podział zadań między model a kod i konfiguracja (0–10 min)

### 2.1. Za co odpowiada model, a za co kod gry
- **Zadania modelu językowego (LLM):** _[uzupełnijcie — np. wcielenie się w kupca, napisanie odpowiedzi dla gracza (`reply`) i odczytanie jego zamiaru (`action`)]_
- **Zadania kodu w Pythonie (`GameState`, `process_turn`):** _[uzupełnijcie — np. pilnowanie liczby monet, ceny i stanu magazynu, sprawdzenie poprawności JSON-a i dozwolonych akcji, decyzja o zakupie oraz odjęcie monet i towaru]_
- **Dlaczego nie możemy polegać wyłącznie na tekście odpowiedzi (`reply`):** _[uzupełnijcie — np. model może zmyślić stan konta, dać rabat, sprzedać miksturę na kredyt albo obiecać towar, którego już nie ma w magazynie]_

### 2.2. Wybrany model i ustawienia
> _Uwaga: nawet przy tych samych ustawieniach kolejne uruchomienia mogą dawać nieco inne wypowiedzi kupca (`reply`)._

| Parametr | Wartość użyta w testach |
|---|---|
| Dostawca API (`PROVIDER_BASE_URL`) | `https://openrouter.ai/api/v1` _(lub inny)_ |
| Model (`model.value`) | _[uzupełnijcie, np. `nvidia/nemotron-3.5-super-vl:free`]_ |
| Temperatura (`temperature`) | `0.2` _(lub inna)_ |
| `top_p` | `1.0` _(lub inna)_ |
| Limit tokenów (`max_output_tokens`) | `256` _(lub inna)_ |
| Tryb rozumowania (`reasoning_effort`) | `none` _(lub inny)_ |
| Limit czasu (`timeout_sec`) | `20.0` s |
| Stan początkowy (`GameState`) | `coins = 15`, `potions = 2`, `potion_price = 10` |
| Dozwolone akcje (`ALLOWED_ACTIONS`) | `("talk", "buy_potion")` |

---

## 3. Instrukcja dla modelu i format JSON (10–30 min)

### 3.1. Struktura odpowiedzi (`TODO 1` — `MerchantResponse`)
Klasa `MerchantResponse` opisująca wymagany schemat JSON (`strict=True`, `extra="forbid"`):

```python
class MerchantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Wklejcie tutaj swoją definicję pól reply oraz action z TODO 1
```

### 3.2. Instrukcja systemowa dla kupca (`TODO 2` — `build_merchant_prompt`)
Funkcja `build_merchant_prompt(state: GameState)` lub sam tekst instrukcji z opisem roli kupca, aktualnego stanu gry (`state.coins`, `state.potions`, `state.potion_price`), zasad sklepu i wyboru akcji (`"talk"` albo `"buy_potion"`):

```python
def build_merchant_prompt(state: GameState) -> str:
    # Wklejcie tutaj swoją funkcję z TODO 2 (lub sam wygenerowany tekst instrukcji)
```

### 3.3. Pierwsza próba (pytanie o towar i prośba o zakup)
- **Pytanie o ofertę:** _[napiszcie krótko, czy model zwrócił poprawny JSON z `"action": "talk"`]_
- **Prośba o zakup jednej mikstury:** _[napiszcie krótko, czy model zwrócił poprawny JSON z `"action": "buy_potion"`]_

---

## 4. Walidacja i logika zakupów (30–55 min)

### 4.1. Sprawdzanie odpowiedzi modelu (`TODO 3` — `parse_and_validate_response`)
```python
def parse_and_validate_response(raw_text: str) -> tuple[bool, bool, str | None, str, str | None]:
    # Wklejcie tutaj swój kod funkcji parse_and_validate_response z TODO 3
```

### 4.2. Obsługa zakupu i zmiana stanu gry (`TODO 4` — `process_turn`)
```python
    # Wklejcie tutaj uzupełnioną część funkcji process_turn z TODO 4
```

### 4.3. Testy na sztucznych odpowiedziach (`mock_raw_output`)
- **Test 1 — Ucięty JSON (`res_bad_json`):** _[napiszcie, co zwrócił program i czy stan gry pozostał bez zmian]_
- **Test 2 — Nieznana akcja, np. `"give_free_gold"` (`res_bad_action`):** _[napiszcie, co zwrócił program i czy stan gry pozostał bez zmian]_
- **Test 3 — Zgoda na zakup w `reply` mimo braku monet (`res_poor`, 5 monet):** _[opiszcie, jak zachował się program, gdy model zwrócił `"action": "buy_potion"`, a gracz miał za mało złota]_

---

## 5. Sześć testów obowiązkowych (55–75 min, `TODO 5`)

> **Wskazówka:** Gotową tabelę możecie wypisać w notatniku funkcją `print_markdown_table(test_results)` i wkleić poniżej.
>
> Próba nr 3 korzysta ze stanu gry po próbie nr 2 (`use_previous_state=True`), a pozostałe startują z osobnych ustawień początkowych.

| Nr | Scenariusz | Wypowiedź gracza | Stan początkowy | JSON OK | Proponowana akcja | Odpowiedź modelu (skrót) | Decyzja programu | Stan końcowy |
|---:|---|---|---|---|---|---|---|---|
| 1 | 1. Pytanie o ofertę | 'Witaj kupcze! Co masz na sprzedaż i ile kosztuje?' | 15 monet, 2 mikstur | _[Tak/Nie]_ | _`talk`_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 15 monet, 2 mikstur |
| 2 | 2. Poprawny zakup | _[wpiszcie z TODO 5]_ | 15 monet, 2 mikstur | _[Tak/Nie]_ | _`buy_potion`_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 5 monet, 1 mikstur |
| 3 | 3. Ponowny zakup (brak monet) | _[wpiszcie z TODO 5]_ | 5 monet, 1 mikstur | _[Tak/Nie]_ | _[`talk` / `buy_potion`]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 5 monet, 1 mikstur |
| 4 | 4. Zakup przy pustym magazynie | _[wpiszcie z TODO 5]_ | 15 monet, 0 mikstur | _[Tak/Nie]_ | _[`talk` / `buy_potion`]_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 15 monet, 0 mikstur |
| 5 | 5. Niejasna wypowiedź | _[wpiszcie z TODO 5]_ | 15 monet, 2 mikstur | _[Tak/Nie]_ | _`talk`_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 15 monet, 2 mikstur |
| 6 | 6. Nieobsługiwana czynność | _[wpiszcie z TODO 5]_ | 15 monet, 2 mikstur | _[Tak/Nie]_ | _`talk`_ | _[uzupełnijcie]_ | _[uzupełnijcie]_ | 15 monet, 2 mikstur |

---

## 6. Obserwacje i wnioski (75–90 min)

W podsumowaniu oceńcie osobno trzy rzeczy:

1. **Poprawność formatu JSON:**
   _[Czy w każdej z 6 prób model zwrócił poprawny JSON zgodny z `MerchantResponse`? Czy zdarzyły się puste odpowiedzi, dodatkowy tekst poza nawiasami albo przekroczenie czasu (`timeout`)?]_

2. **Rozpoznanie zamiaru gracza (`proposed_action`):**
   _[Czy model dobrze odróżniał zwykłą rozmowę, niejasne uwagi i nietypowe prośby (`"talk"`) od chęci zakupu (`"buy_potion"`)? Co zrobił w próbach 3 i 4 — sam odmówił (`"talk"`), czy przekazał `"buy_potion"`, mimo że gracz nie miał monet lub skończył się towar?]_

3. **Ochrona stanu gry w Pythonie:**
   _[Czy liczba monet i mikstur zmieniła się tylko w próbie 2? Jak program zachował się wtedy, gdy model zwrócił poprawny JSON, ale zaproponował niedozwolony zakup lub obiecał coś w polu `reply`?]_

4. **Główne wnioski (do krótkiego dema):**
   _[W 2–4 zdaniach: po co w grach i programach użytkowych łączy się swobodną rozmowę z modelem z twardym sprawdzaniem reguł w kodzie?]_

---

## 7. Zadania dodatkowe (dla chętnych)

### 7.1. Druga postać o innym stylu mówienia (`TODO 6` — `build_alt_character_prompt`)
- **Kim jest nowa postać i jak mówi:** _[uzupełnijcie, np. zrzędliwy goblin, wyniosły elfi alchemik]_
- **Instrukcja systemowa (`build_alt_character_prompt`):**

```python
def build_alt_character_prompt(state: GameState) -> str:
    # Wklejcie tutaj swoją funkcję z TODO 6
```

- **Porównanie z podstawowym kupcem (6 scenariuszy):**
  _[Czy przez inny styl mówienia model zaczął psuć format JSON albo gorzej dobierać pole `action`?]_

### 7.2. Zabawa parametrami generowania (`temperature`, `top_p`, `max_output_tokens`, `reasoning_effort`)
- **Co zmieniło się po ruszeniu suwaków:**
  _[Opcjonalnie: opiszcie krótko, co działo się po podbiciu temperatury albo mocnym przycięciu `max_output_tokens` (np. do 16–32 tokenów) i jak poradził sobie z tym walidator]_

---

## 8. Podział pracy w zespole

| Imię i nazwisko | Co zrobiła dana osoba |
|---|---|
| _[Osoba 1]_ | _[uzupełnijcie]_ |
| _[Osoba 2]_ | _[uzupełnijcie]_ |
| _[Osoba 3]_ | _[uzupełnijcie]_ |
| _[Osoba 4 — jeśli dotyczy]_ | _[uzupełnijcie]_ |
