# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Warsztat 4: Rozmowa z kupcem sterowanym przez LLM (OpenRouter / OpenAI Responses API)

# %% [markdown]
# ### Cel ćwiczenia
#
# W grach fabularnych (RPG) gracze chętnie prowadzą swobodne rozmowy z postaciami niezależnymi (NPC), jednak stan świata gry — liczba monet w sakiewce, zawartość ekwipunku czy zapasy w sklepie — musi podlegać ścisłym, deterministycznym regułom.
#
# Waszym zadaniem jest zbudowanie tekstowego systemu dialogu z kupcem w grze przy użyciu modelu **Gemma 4 (`google/gemma-4-26b-a4b-it`)** uruchamianego przez **OpenRouter** (`https://openrouter.ai/api/v1`) z wykorzystaniem **OpenAI Responses API** (bez obciążania Waszych procesorów):
# 1. Gracz wpisuje dowolną wypowiedź (np. pytanie o towar, prośbę o zakup mikstury, niejasną uwagę lub żądanie nieobsługiwanej akcji).
# 2. Model językowy odgrywa rolę kupca, rozpoznaje intencję gracza i zwraca ustrukturyzowaną odpowiedź w formacie **JSON** zawierającą pola `"reply"` (wypowiedź kupca) oraz `"action"` (proponowana akcja: `"talk"` lub `"buy_potion"`).
# 3. **Program w języku Python** waliduje strukturę JSON, sprawdza czy akcja należy do listy dozwolonych i **samodzielnie rozstrzyga**, czy transakcja jest możliwa (weryfikuje liczbę monet gracza oraz stan magazynu kupca), a następnie aktualizuje stan gry albo odrzuca zakup.
#
# Celem warsztatu jest poznanie bezpiecznego wzorca integracji swobodnej generacji LLM ze ściśle kontrolowaną maszyną stanów gry.

# %% [markdown]
# ### Stan początkowy i reguły gry
#
# - **Stan początkowy (`GameState`):**
#   - Gracz posiada **15 monet** (`coins = 15`).
#   - Kupiec posiada w magazynie **2 mikstury** (`potions = 2`).
#   - Każda mikstura kosztuje **10 monet** (`potion_price = 10`).
# - **Dozwolone akcje (`ALLOWED_ACTIONS`):**
#   - `"talk"` — zwykła rozmowa, pytanie o ofertę, brak transakcji lub odrzucenie prośby przez kupca.
#   - `"buy_potion"` — intencja zakupu dokładnie **jednej** mikstury.
# - **Format wyjściowy modelu (JSON):**
#   ```json
#   {
#     "reply": "Tekst odpowiedzi kupca w świecie gry...",
#     "action": "talk"
#   }
#   ```
# - **Złota zasada architektury:** Saldo monet, cena i zapas mikstur są utrzymywane **wyłącznie przez program**. Niepoprawny JSON, nieznana akcja spoza listy lub przekroczenie czasu oczekiwania (`timeout`) **nigdy nie zmieniają stanu gry**. Komunikat potwierdzający lub odrzucający transakcję powstaje na podstawie decyzji programu, a nie samej wypowiedzi modelu.

# %% [markdown]
# ### 1. Przygotowanie środowiska i podział odpowiedzialności (0–10 min)
#
# Notatnik jest w pełni **samowystarczalny** i przystosowany do uruchomienia w **Google Colab** (nie wymaga żadnych zewnętrznych plików z repozytorium).
#
# W tej części:
# 1. Instalujemy wymagane pakiety (`openai`, `pydantic`).
# 2. Definiujemy wbudowane klasy stanu gry oraz funkcje pomocnicze do komunikacji z API i prezentacji wyników.
# 3. Konfigurujemy klucz `OPENROUTER_API_KEY` wygenerowany w [OpenRouter](https://openrouter.ai/keys) (endpoint: `https://openrouter.ai/api/v1`) — w Google Colab najwygodniej zapisać go w zakładce **Sekrety** (ikona klucza po lewej stronie) pod nazwą `OPENROUTER_API_KEY` albo wkleić w polu formularza / oknie po uruchomieniu komórki.
# 4. Omawiamy w zespole podział odpowiedzialności między model językowy a logikę gry i sprawdzamy połączenie z API.

# %%
# Instalacja wymaganych bibliotek w Google Colab / Jupyter
# %pip install -q "openai>=1.66.0" "pydantic>=2.0"

# %%
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Literal

from openai import APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
ALLOWED_ACTIONS = ("talk", "buy_potion")
client: OpenAI | None = None


@dataclass
class GameState:
    """Ściśle określony stan gry utrzymywany wyłącznie przez program."""

    coins: int = 15
    potions: int = 2
    potion_price: int = 10

    def copy(self) -> "GameState":
        """Zwraca niezależną kopię stanu gry."""
        return GameState(
            coins=self.coins,
            potions=self.potions,
            potion_price=self.potion_price,
        )

    def format_short(self) -> str:
        """Zwraca krótki zapis stanu do tabeli wyników."""
        return f"monety={self.coins}, mikstury={self.potions} (cena={self.potion_price})"


@dataclass
class TurnResult:
    """Pełny zapis pojedynczej tury rozmowy i decyzji programu."""

    scenario_name: str
    player_input: str
    state_before: GameState
    state_after: GameState
    raw_output: str
    json_valid: bool
    proposed_action: str | None
    merchant_reply: str
    action_valid: bool
    tx_allowed: bool
    system_message: str
    elapsed_sec: float
    timed_out: bool = False
    error: str | None = None


def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.2,
    seed: int = 42,
    timeout_sec: float = 20.0,
) -> tuple[str, float, bool, str | None]:
    """Wywołuje model przez OpenRouter z użyciem OpenAI Responses API i wymuszonym formatem JSON."""
    del seed  # Responses API nie obsługuje parametru seed

    def _invoke() -> str:
        active_client = client
        if active_client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Brak skonfigurowanego klienta OpenRouter (klucz OPENROUTER_API_KEY).")
            active_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

        schema_instruction = ""
        if schema:
            schema_payload = dict(schema)
            if schema_payload.get("type") == "object" and "additionalProperties" not in schema_payload:
                schema_payload["additionalProperties"] = False
            text_config: dict[str, Any] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_payload.get("title", "MerchantResponse"),
                    "strict": True,
                    "schema": schema_payload,
                }
            }
            schema_instruction = (
                "\n\nMUSISZ ODPOWIEDZIEĆ W ŚCISŁYM FORMACIE JSON ZGODNYM Z PONIŻSZYM SCHEMATEM:\n"
                f"{json.dumps(schema_payload, ensure_ascii=False)}"
            )
        else:
            text_config = {"format": {"type": "json_object"}}

        response = active_client.responses.create(
            model=model,
            instructions=f"{system_prompt}{schema_instruction}",
            input=user_message,
            temperature=temperature,
            text=text_config,
            timeout=timeout_sec,
        )
        return response.output_text or ""

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            raw_text = future.result(timeout=timeout_sec)
            elapsed = time.perf_counter() - start
            return raw_text, elapsed, False, None
        except (FuturesTimeoutError, APITimeoutError):
            elapsed = time.perf_counter() - start
            return "", elapsed, True, f"Przekroczono limit czasu ({timeout_sec} s)"
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return "", elapsed, False, str(exc)


def show_dialogue_screen(turn: TurnResult, character_name: str = "Kupiec Barnaba") -> None:
    """Wyświetla czytelny ekran tekstowy pojedynczej wymiany zdań i rozstrzygnięcia gry."""
    border = "=" * 78
    sep = "-" * 78
    print(border)
    print(f" SCENARIUSZ: {turn.scenario_name}")
    print(sep)
    print(
        f" [STAN PRZED]  Gracz: {turn.state_before.coins} monet | "
        f"Magazyn kupca: {turn.state_before.potions} szt. (cena: {turn.state_before.potion_price} monet)"
    )
    print(f" [GRACZ]       '{turn.player_input}'")
    print(sep)
    if turn.timed_out:
        print(f" [MODEL LLM]   TIMEOUT ({turn.elapsed_sec:.2f} s) - {turn.error}")
    elif turn.error and not turn.raw_output:
        print(f" [MODEL LLM]   BŁĄD ({turn.elapsed_sec:.2f} s) - {turn.error}")
    else:
        print(f" [SUROWY JSON] {turn.raw_output.strip()} ({turn.elapsed_sec:.2f} s)")
        print(f" [{character_name.upper()}] '{turn.merchant_reply}'")
        print(
            f" [PROPOZYCJA]  action = {turn.proposed_action!r} "
            f"(JSON poprawny: {'TAK' if turn.json_valid else 'NIE'}, "
            f"Akcja dozwolona: {'TAK' if turn.action_valid else 'NIE'})"
        )
    print(sep)
    print(f" [SILNIK GRY]  {turn.system_message}")
    print(
        f" [STAN PO]     Gracz: {turn.state_after.coins} monet | "
        f"Magazyn kupca: {turn.state_after.potions} szt."
    )
    print(border + "\n")


def print_results_table(results: list[TurnResult]) -> None:
    """Wypisuje zbiorczą tabelę wyników testów gotową do analizy i sprawozdania."""
    header = (
        f"{'Nr':<3} | {'Scenariusz':<32} | {'Stan przed':<13} | "
        f"{'JSON':<4} | {'Propozycja LLM':<12} | {'Decyzja gry':<16} | {'Stan po':<13}"
    )
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(results, 1):
        s_before = f"{r.state_before.coins}m, {r.state_before.potions}szt"
        s_after = f"{r.state_after.coins}m, {r.state_after.potions}szt"
        json_ok = "TAK" if r.json_valid else "NIE"
        prop = str(r.proposed_action) if r.proposed_action is not None else "BRAK"
        if not r.json_valid or r.timed_out or not r.action_valid:
            decision = "BŁĄD/ODRZUCONO"
        elif r.proposed_action == "talk":
            decision = "ROZMOWA (brak zm)"
        elif r.tx_allowed:
            decision = "ZAKUP OK (-10m)"
        else:
            decision = "ZAKUP ODRZUCONY"
        name_short = (r.scenario_name[:29] + "...") if len(r.scenario_name) > 32 else r.scenario_name
        print(
            f"{idx:<3} | {name_short:<32} | {s_before:<13} | "
            f"{json_ok:<4} | {prop:<12} | {decision:<16} | {s_after:<13}"
        )


def print_markdown_table(results: list[TurnResult]) -> None:
    """Generuje tabelę w formacie Markdown do łatwego wklejenia w sprawozdanie."""
    print("| Nr | Scenariusz | Wypowiedź gracza | Stan początkowy | JSON OK | Proponowana akcja | Odpowiedź modelu (skrót) | Decyzja programu | Stan końcowy |")
    print("|---:|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(results, 1):
        s_before = f"{r.state_before.coins} monet, {r.state_before.potions} mikstur"
        s_after = f"{r.state_after.coins} monet, {r.state_after.potions} mikstur"
        json_ok = "Tak" if r.json_valid else "Nie"
        reply_clean = r.merchant_reply.replace("\n", " ").replace("|", "/")
        if len(reply_clean) > 60:
            reply_clean = reply_clean[:57] + "..."
        print(
            f"| {idx} | {r.scenario_name} | '{r.player_input}' | {s_before} | "
            f"{json_ok} | `{r.proposed_action}` | '{reply_clean}' | {r.system_message} | {s_after} |"
        )


# %%
# Konfiguracja klucza API OpenRouter (Google Colab Secrets / zmienna środowiskowa / input)
OPENROUTER_API_KEY = ""  # @param {type:"string"}


def configure_openrouter_api(manual_key: str = "") -> OpenAI:
    """Konfiguruje klienta OpenAI dla OpenRouter z formularza Colab, zmiennych środowiskowych, Colab Secrets lub promptu."""
    api_key = manual_key.strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        try:
            from google.colab import userdata

            api_key = (userdata.get("OPENROUTER_API_KEY") or "").strip()
        except Exception:
            pass

    if not api_key:
        api_key = getpass("Wklej swój klucz OPENROUTER_API_KEY z OpenRouter: ").strip()

    if not api_key:
        raise ValueError(
            "Brak klucza OPENROUTER_API_KEY! Dodaj go w Google Colab Secrets (nazwa: OPENROUTER_API_KEY) "
            "lub wklej po uruchomieniu komórki."
        )

    os.environ["OPENROUTER_API_KEY"] = api_key
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


client = configure_openrouter_api(OPENROUTER_API_KEY)

MODEL_NAME = DEFAULT_MODEL
TEMPERATURE = 0.2
TIMEOUT_SEC = 20.0

initial_state = GameState(coins=15, potions=2, potion_price=10)
print(f"Endpoint API: {OPENROUTER_BASE_URL}")
print(f"Wybrany model na OpenRouter: {MODEL_NAME}")
print(f"Dozwolone akcje: {ALLOWED_ACTIONS}")
print(f"Stan początkowy gry: {initial_state.format_short()}")


# %%
# Szybki test połączenia z OpenRouter (OpenAI Responses API)
raw_ping, ping_time, is_timeout, err = call_llm(
    system_prompt='Jesteś systemem testowym. Zwróć dokładnie taki JSON: {"status": "ok"}',
    user_message="Test połączenia",
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    timeout_sec=TIMEOUT_SEC,
)
print(f"Odpowiedź testowa z chmury ({ping_time:.2f} s, timeout={is_timeout}, błąd={err}):\n{raw_ping}")


# %% [markdown]
# ### 2. Instrukcja dla kupca i schemat odpowiedzi JSON (10–30 min)
#
# Konfigurujemy model na OpenRouter przez OpenAI Responses API, aby zwracał odpowiedź w formacie JSON (`text={"format": {"type": "json_schema", ...}}`). Definiujemy schemat za pomocą klasy `Pydantic`, a następnie budujemy instrukcję systemową (`system_prompt`), która przekazuje modelowi:
# - kim jest (kupiec Barnaba sprzedający wyłącznie mikstury uzdrawiające),
# - jaki jest aktualny stan gry (monety gracza, liczba dostępnych mikstur w magazynie, cena jednej mikstury),
# - jakie są dozwolone akcje (`"talk"` oraz `"buy_potion"`) i kiedy należy użyć każdej z nich,
# - wymóg odpowiadania po polsku w formacie JSON zgodnym ze schematem z polami `"reply"` i `"action"`.

# %%
class MerchantResponse(BaseModel):
    """Schemat odpowiedzi wymagany od modelu językowego."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(description="Wypowiedź kupca skierowana do gracza po polsku.")
    action: Literal["talk", "buy_potion"] = Field(
        description="Rozpoznana intencja/akcja: 'talk' dla rozmowy lub odmowy, 'buy_potion' gdy gracz chce kupić miksturę."
    )


RESPONSE_SCHEMA = MerchantResponse.model_json_schema()
print("Wygenerowany schemat JSON dla OpenAI Responses API (OpenRouter):")
print(json.dumps(RESPONSE_SCHEMA, indent=2, ensure_ascii=False))


# %%
def build_merchant_prompt(state: GameState) -> str:
    """Buduje instrukcję systemową dla kupca uwzględniającą aktualny stan gry."""
    prompt = f"""Jesteś kupcem Barnabą prowadzącym sklep z miksturami w grze RPG.
Odpowiadasz krótko (1-2 zdania), w klimacie fantasy, wyłącznie po polsku.

AKTUALNY STAN GRY:
- Monety gracza: {state.coins}
- Dostępne mikstury w Twoim magazynie: {state.potions} szt.
- Cena jednej mikstury: {state.potion_price} monet

ZASADY SKLEPU I WYBORU AKCJI (pole "action"):
1. Masz w ofercie wyłącznie mikstury uzdrawiające w cenie {state.potion_price} monet za sztukę. Jedna transakcja to zakup dokładnie 1 mikstury.
2. Nie skupujesz przedmiotów od gracza, nie udzielasz kredytu, nie dajesz zniżek i nie oferujesz innych towarów.
3. Ustaw pole "action" na "buy_potion" TYLKO WTEDY, gdy gracz wyraźnie prosi o zakup mikstury.
   Uwaga: jeśli gracz ma mniej niż {state.potion_price} monet lub w magazynie jest 0 mikstur, grzecznie poinformuj w "reply", że zakup jest niemożliwy (brak złota lub towaru).
4. Ustaw pole "action" na "talk", gdy gracz zadaje pytanie o ofertę/cenę, wita się, wypowiada się niejasno lub żąda nieobsługiwanej czynności (np. sprzedaży miecza, targowania się).
"""
    return prompt


# %%
# TODO: Sprawdź działanie modelu na zwykłym pytaniu o ofertę oraz na prośbie o zakup mikstury.
test_prompt = build_merchant_prompt(initial_state)

for sample_input in [
    "Dzień dobry! Co sprzedajesz i ile to kosztuje?",
    "Chcę kupić jedną miksturę uzdrawiającą.",
]:
    raw_out, elapsed, timed_out, error = call_llm(
        system_prompt=test_prompt,
        user_message=sample_input,
        model=MODEL_NAME,
        schema=RESPONSE_SCHEMA,
        temperature=TEMPERATURE,
        timeout_sec=TIMEOUT_SEC,
    )
    print(f"Gracz: {sample_input}")
    print(f"Surowy JSON z modelu ({elapsed:.2f} s): {raw_out}\n")


# %% [markdown]
# ### 3. Walidacja odpowiedzi i deterministyczna logika stanu gry (30–55 min)
#
# Nawet gdy model językowy ma wymuszony JSON i otrzymuje dokładny stan gry w prompcie, **nie wolno ufać jego ocenie sytuacji**:
# - Model może przekroczyć limit czasu (`timeout`).
# - Model może zignorować schemat i zwrócić zniekształcony string, albo pominąć pole `action`.
# - Model może zaproponować `"buy_potion"` (i w polu `"reply"` napisać *„Proszę bardzo, oto Twoja mikstura!”*), podczas gdy gracz ma tylko 5 monet albo magazyn jest pusty!
# - I odwrotnie: model może sam zablokować zakup i zwrócić `"talk"`, gdy widzi w prompcie brak monet — albo zwrócić `"buy_potion"` jako rozpoznaną intencję gracza, którą **dopiero silnik gry** odrzuca po sprawdzeniu salda.
#
# Dlatego implementujemy dwie funkcje:
# 1. `parse_and_validate_response(raw_text)` — parsuje JSON, sprawdza obecność i typ pól `"reply"` (niepusty napis) oraz `"action"` (czy należy do `ALLOWED_ACTIONS`).
# 2. `process_turn(...)` — wywołuje model w chmurze, waliduje wynik i **w kodzie Pythona** rozstrzyga transakcję na kopii obiektu `GameState`. Komunikat systemowy o wyniku transakcji (`system_message`) jest generowany przez program na podstawie rzeczywistego rozstrzygnięcia.

# %%
def parse_and_validate_response(raw_text: str) -> tuple[bool, bool, str | None, str, str | None]:
    """Waliduje surową odpowiedź modelu i bezpiecznie ekstrahuje JSON."""
    if not raw_text or not raw_text.strip():
        return False, False, None, "", "Pusta odpowiedź modelu."

    clean_text = raw_text.strip()

    # Niezawodna ekstrakcja JSON-a za pomocą stosu klamer
    # Zapobiega błędom "Extra data", gdy model zwraca podwójny JSON lub znaczniki markdown
    start_idx = clean_text.find('{')
    if start_idx != -1:
        stack = 0
        for i in range(start_idx, len(clean_text)):
            if clean_text[i] == '{':
                stack += 1
            elif clean_text[i] == '}':
                stack -= 1
                if stack == 0:
                    clean_text = clean_text[start_idx:i+1]
                    break

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        return False, False, None, "", f"Niepoprawny format JSON: {exc}"

    if not isinstance(data, dict):
        return False, False, None, "", "Odpowiedź JSON nie jest słownikiem (obiektem)."

    reply = data.get("reply")
    action = data.get("action")

    if not isinstance(reply, str) or not reply.strip() or not isinstance(action, str):
        return False, False, None, "", "Brak wymaganych pól 'reply' lub 'action' bądź niepoprawny typ danych."

    reply_clean = reply.strip()
    action_clean = action.strip()

    if action_clean not in ALLOWED_ACTIONS:
        return (
            True,
            False,
            action_clean,
            reply_clean,
            f"Nieznana akcja spoza listy dozwolonych: {action_clean!r} (dozwolone: {ALLOWED_ACTIONS}).",
        )

    return True, True, action_clean, reply_clean, None



# %%
def process_turn(
    player_input: str,
    state: GameState,
    scenario_name: str = "Rozmowa",
    prompt_builder=build_merchant_prompt,
    model: str = MODEL_NAME,
    schema: dict | None = RESPONSE_SCHEMA,
    temperature: float = TEMPERATURE,
    timeout_sec: float = TIMEOUT_SEC,
    mock_raw_output: str | None = None,
) -> TurnResult:
    """Przeprowadza pojedynczą turę rozmowy: wywołuje LLM, waliduje JSON i rozstrzyga zmianę stanu gry."""
    state_before = state.copy()
    state_after = state.copy()  # Pracujemy na kopii; w razie błędu stan pozostaje bez zmian

    if mock_raw_output is not None:
        raw_output, elapsed, timed_out, llm_error = mock_raw_output, 0.0, False, None
    else:
        system_prompt = prompt_builder(state_before)
        raw_output, elapsed, timed_out, llm_error = call_llm(
            system_prompt=system_prompt,
            user_message=player_input,
            model=model,
            schema=schema,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    # 1. Obsługa timeoutu lub błędu komunikacji (stan gry NIE ulega zmianie)
    if timed_out or llm_error is not None:
        sys_msg = f"[BŁĄD SYSTEMU] {llm_error}. Stan gry pozostaje bez zmian."
        return TurnResult(
            scenario_name=scenario_name,
            player_input=player_input,
            state_before=state_before,
            state_after=state_after,
            raw_output=raw_output,
            json_valid=False,
            proposed_action=None,
            merchant_reply="",
            action_valid=False,
            tx_allowed=False,
            system_message=sys_msg,
            elapsed_sec=elapsed,
            timed_out=timed_out,
            error=llm_error,
        )

    # 2. Walidacja JSON i dozwolonych akcji
    json_valid, action_valid, proposed_action, merchant_reply, val_error = parse_and_validate_response(raw_output)

    if not json_valid or not action_valid:
        sys_msg = f"[ODRZUCONO ODPOWIEDŹ MODELU] {val_error} Stan gry bez zmian."
        return TurnResult(
            scenario_name=scenario_name,
            player_input=player_input,
            state_before=state_before,
            state_after=state_after,
            raw_output=raw_output,
            json_valid=json_valid,
            proposed_action=proposed_action,
            merchant_reply=merchant_reply,
            action_valid=action_valid,
            tx_allowed=False,
            system_message=sys_msg,
            elapsed_sec=elapsed,
            timed_out=False,
            error=val_error,
        )

    # 3. Rozstrzygnięcie akcji przez deterministyczną logikę gry
    tx_allowed = False

    if proposed_action == "talk":
        tx_allowed = False
        sys_msg = "Brak transakcji (tryb rozmowy 'talk'). Stan gry bez zmian."
    elif proposed_action == "buy_potion":
        if state_before.potions <= 0:
            tx_allowed = False
            sys_msg = "TRANSAKCJA ODRZUCONA PRZEZ PROGRAM: Pusty magazyn kupca (0 mikstur)."
        elif state_before.coins < state_before.potion_price:
            tx_allowed = False
            sys_msg = (
                f"TRANSAKCJA ODRZUCONA PRZEZ PROGRAM: Niewystarczające środki "
                f"(gracz ma {state_before.coins} monet, wymagane {state_before.potion_price} monet)."
            )
        else:
            tx_allowed = True
            state_after.coins -= state_before.potion_price
            state_after.potions -= 1
            sys_msg = (
                f"TRANSAKCJA ZATWIERDZONA PRZEZ PROGRAM: Zakupiono 1 miksturę za {state_before.potion_price} monet. "
                f"Nowe saldo: {state_after.coins} monet, pozostało mikstur: {state_after.potions}."
            )

    return TurnResult(
        scenario_name=scenario_name,
        player_input=player_input,
        state_before=state_before,
        state_after=state_after,
        raw_output=raw_output,
        json_valid=True,
        proposed_action=proposed_action,
        merchant_reply=merchant_reply,
        action_valid=True,
        tx_allowed=tx_allowed,
        system_message=sys_msg,
        elapsed_sec=elapsed,
    )


# %%
# Testy odporności walidatora na uszkodzony JSON, nieznaną akcję i timeout
dummy_state = GameState(coins=15, potions=2, potion_price=10)

print("--- Test 1: Uszkodzony JSON ---")
res_bad_json = process_turn("Kupię miksturę", dummy_state, "Test: Uszkodzony JSON", mock_raw_output='{"reply": "Oto mikstura", "action": ')
print(res_bad_json.system_message, "| Stan po:", res_bad_json.state_after.format_short())

print("\n--- Test 2: Nieznana akcja spoza listy ---")
res_bad_act = process_turn("Oddaj złoto!", dummy_state, "Test: Nieznana akcja", mock_raw_output='{"reply": "Już oddaję!", "action": "give_free_gold"}')
print(res_bad_act.system_message, "| Stan po:", res_bad_act.state_after.format_short())

print("\n--- Test 3: Halucynacja zakupu przez model przy 5 monetach ---")
poor_state = GameState(coins=5, potions=1, potion_price=10)
res_hallucination = process_turn("Poproszę miksturę", poor_state, "Test: Brak monet", mock_raw_output='{"reply": "Proszę bardzo, sprzedałem Ci miksturę za 5 monet!", "action": "buy_potion"}')
print(res_hallucination.system_message, "| Stan po:", res_hallucination.state_after.format_short())

# %% [markdown]
# ### 4. Sześć prób testowych (55–75 min)
#
# Definiujemy wbudowaną listę scenariuszy testowych (`TEST_SCENARIOS`) i wykonujemy sześć obowiązkowych prób:
# 1. **Pytanie o ofertę** (stan początkowy: 15 monet, 2 mikstury).
# 2. **Poprawny zakup** (stan początkowy: 15 monet, 2 mikstury $\rightarrow$ po zakupie: 5 monet, 1 mikstura).
# 3. **Ponowny zakup bez wystarczających środków** (**kontynuuje stan po 2. próbie**: 5 monet, 1 mikstura).
# 4. **Zakup przy pustym magazynie** (przygotowany stan początkowy: 15 monet, 0 mikstur).
# 5. **Niejasna wypowiedź** (stan początkowy: 15 monet, 2 mikstury).
# 6. **Żądanie nieobsługiwanej czynności** (stan początkowy: 15 monet, 2 mikstury).

# %%
TEST_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Pytanie o ofertę",
        "player_input": "Witaj kupcze! Co ciekawego masz na sprzedaż i w jakiej cenie?",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
        "expected_action": "talk",
        "expected_tx_allowed": False,
    },
    {
        "id": 2,
        "name": "Poprawny zakup",
        "player_input": "Poproszę jedną miksturę uzdrawiającą. Oto złoto.",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
        "expected_action": "buy_potion",
        "expected_tx_allowed": True,
    },
    {
        "id": 3,
        "name": "Ponowny zakup bez wystarczających środków (kontynuacja próby 2)",
        "player_input": "Świetna mikstura! Kupię jeszcze jedną na zapas.",
        "use_previous_state": True,
        "initial_coins": 5,
        "initial_potions": 1,
        "potion_price": 10,
        "expected_action": "buy_potion",
        "expected_tx_allowed": False,
    },
    {
        "id": 4,
        "name": "Zakup przy pustym magazynie",
        "player_input": "Mam dużo złota, chcę natychmiast kupić jedną miksturę!",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 0,
        "potion_price": 10,
        "expected_action": "buy_potion",
        "expected_tx_allowed": False,
    },
    {
        "id": 5,
        "name": "Niejasna wypowiedź",
        "player_input": "Hmm... ładnie pachnie w tym sklepie, sam już nie wiem czego szukam...",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
        "expected_action": "talk",
        "expected_tx_allowed": False,
    },
    {
        "id": 6,
        "name": "Żądanie nieobsługiwanej czynności",
        "player_input": "Odkup ode mnie ten stary zardzewiały miecz albo daj mi kredyt!",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
        "expected_action": "talk",
        "expected_tx_allowed": False,
    },
]

scenarios = TEST_SCENARIOS
test_results: list[TurnResult] = []
previous_state = GameState(coins=15, potions=2, potion_price=10)

for sc in scenarios:
    if sc["use_previous_state"]:
        current_state = previous_state.copy()
    else:
        current_state = GameState(
            coins=sc["initial_coins"],
            potions=sc["initial_potions"],
            potion_price=sc["potion_price"],
        )

    turn_result = process_turn(
        player_input=sc["player_input"],
        state=current_state,
        scenario_name=f"{sc['id']}. {sc['name']}",
    )

    test_results.append(turn_result)
    previous_state = turn_result.state_after.copy()
    show_dialogue_screen(turn_result)

# %% [markdown]
# ### 5. Podsumowanie wyników i tabela do sprawozdania (75–90 min)
#
# Wypisujemy zbiorczą tabelę wszystkich 6 prób oraz generujemy kod tabeli Markdown gotowy do wklejenia w pliku `sprawozdanie_szablon.md`.
#
# **Wskazówka do wniosków w sprawozdaniu:**
# Zwróćcie szczególną uwagę na próby **3** (brak monet) oraz **4** (pusty magazyn):
# - Czy model zwrócił w nich `action = "buy_potion"` (rozpoznając samą intencję gracza), a transakcję zatrzymał dopiero Wasz kod w Pythonie?
# - Czy może model już na poziomie promptu odmówił sprzedaży i zwrócił `action = "talk"`?
# - Czy zdarzyło się, że w polu `reply` kupiec obiecał wydanie mikstury mimo braku środków, a program uratował spójność stanu gry?

# %%
print("=== ZBIORCZA TABELA WYNIKÓW (6 PRÓB) ===\n")
print_results_table(test_results)

print("\n=== TABELA MARKDOWN DO WKLEJENIA W SPRAWOZDANIE ===\n")
print_markdown_table(test_results)


# %% [markdown]
# ---
# ### 6. Zakres dodatkowy — Druga postać o odmiennym stylu wypowiedzi (dla zespołów, które skończą wcześniej)
#
# Dodajcie drugą postać kupca o zupełnie innym sposobie wypowiadania się — np. **Goblińskiego Alchemika Griza** (gburek mówiący łamaną polszczyzną, podejrzliwy i chciwy) albo **Elfią Mistrzynię Zielarstwa Elenwę** (wzniosły, poetycki język) — **zachowując identyczne reguły transakcji, cenę oraz schemat JSON**.
#
# Sprawdźcie na tych samych 6 scenariuszach, czy narzucenie silnego stylu językowego (roleplay) wpływa na:
# 1. Poprawność formatu JSON.
# 2. Trafność klasyfikacji pola `action` (`"talk"` vs `"buy_potion"`).

# %%
def build_goblin_prompt(state: GameState) -> str:
    """Buduje instrukcję dla drugiej postaci (Goblin Griz) przy zachowaniu tych samych reguł gry."""
    prompt = f"""Jesteś zrzędliwym, chciwym goblińskim alchemikiem o imieniu Griz!
Mówisz krótko, chaotycznie, używasz wykrzykników i wtrąceń typu 'Hehe!', 'Złotko!', 'Griz nie głupi!', wyłącznie po polsku.

AKTUALNY STAN GRY:
- Monety gracza: {state.coins}
- Mikstury w kotle Griza: {state.potions} szt.
- Cena jednej mikstury: {state.potion_price} monet

ZASADY SKLEPU I WYBORU AKCJI (pole "action"):
1. Sprzedajesz TYLKO mikstury po {state.potion_price} monet (1 zakup = 1 mikstura).
2. Nie kupujesz złomu od gracza i nie dajesz na krechę!
3. Ustaw pole "action" na "buy_potion" TYLKO WTEDY, gdy gracz chce kupić miksturę.
4. Ustaw pole "action" na "talk" przy pytaniach o towar, niejasnych gadkach lub próbach sprzedaży przedmiotów.
"""
    return prompt


goblin_results: list[TurnResult] = []
prev_goblin_state = GameState(coins=15, potions=2, potion_price=10)

for sc in scenarios:
    curr_state = prev_goblin_state.copy() if sc["use_previous_state"] else GameState(
        coins=sc["initial_coins"], potions=sc["initial_potions"], potion_price=sc["potion_price"]
    )
    t_res = process_turn(
        player_input=sc["player_input"],
        state=curr_state,
        scenario_name=f"{sc['id']}. {sc['name']} (Goblin Griz)",
        prompt_builder=build_goblin_prompt,
    )
    goblin_results.append(t_res)
    prev_goblin_state = t_res.state_after.copy()
    show_dialogue_screen(t_res, character_name="Goblin Griz")

print("=== PORÓWNANIE WYNIKÓW: GOBLIN GRIZ ===\n")
print_results_table(goblin_results)
