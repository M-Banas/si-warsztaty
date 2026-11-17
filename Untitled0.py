# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Warsztat 4: Rozmowa z kupcem w grze
#
# W tym ćwiczeniu zbudujemy prosty dialog tekstowy z kupcem w grze RPG. Gracz może pisać do niego zupełnie swobodnie: zapytać o towary, poprosić o miksturę, rzucić jakąś luźną uwagę albo zażądać czegoś, czego sklep w ogóle nie oferuje.
#
# Model językowy wcieli się w kupca i rozpozna, czego gracz chce w danej chwili, natomiast sam kod w Pythonie będzie pilnował twardych zasad gry: sprawdzi liczbę monet, stan magazynu i zdecyduje, czy zakup w ogóle może dojść do skutku. Chodzi o to, żeby zobaczyć, jak połączyć naturalną rozmowę z modelem ze ścisłym stanem aplikacji, którego model nie może przypadkiem zepsuć.
#
# Zanim zaczniecie pisać kod (pierwsze 10 minut warsztatu), ustalcie w zespole, za co odpowiada model, a za co logika programu. Otwórzcie też szablon sprawozdania i zanotujcie w nim nazwę wybranego modelu.

# %% [markdown] id="v9kAiVqfN1XV"
# ## Połączenie z modelem językowym

# %% [markdown] id="SwdCDk8wNPCR"
# ### Konfiguracja połączenia
#
# Do połączenia z modelem użyjemy biblioteki `openai` oraz dowolnego dostawcy zgodnego z jej API (domyślnie OpenRoutera, ale adres `PROVIDER_BASE_URL` możesz zmienić w polu formularza obok komórki). Żeby nie wklejać klucza API wprost do kodu, skorzystamy z wbudowanych sekretów Google Colab:
#
# 1. Wygeneruj klucz API u wybranego dostawcy, np. na stronie [OpenRouter](https://openrouter.ai/keys) (albo użyj klucza udostępnionego na warsztatach).
# 2. W Google Colab kliknij ikonę klucza (**Sekrety** / *Secrets*) na pasku po lewej stronie.
# 3. Dodaj nowy wpis o nazwie `PROVIDER_API_KEY` i wklej swój klucz w polu wartości.
# 4. Włącz przełącznik dostępu dla tego notatnika (*Dostęp z notatnika* / *Notebook access*).
#
# Dlaczego tak, a nie `api_key="sk-..."` w komórce? Notatniki bardzo często krążą potem na Teamsach, lądują na GitHubie albo widać je na rzutniku podczas prezentacji. Gdy klucz zostanie zapisany w kodzie, łatwo go przypadkiem udostępnić innym osobom, które będą mogły korzystać z Twojego konta.

# %% id="hvO456J1aOw2"
from google.colab import userdata
from openai import OpenAI

PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"  # @param {type:"string"}

client = OpenAI(
    base_url=PROVIDER_BASE_URL,
    api_key=userdata.get("PROVIDER_API_KEY")
)

# %% [markdown] id="YVbYeHOFNhLK"
# ### Wybór modelu
#
# Poniższa komórka pobiera z API listę darmowych modeli obsługujących tekst (z końcówką `:free`). Domyślnie zaznaczony jest `nvidia/nemotron-3.5-super-vl:free`. Wybierz model z listy i kliknij **Testuj połączenie**, żeby upewnić się, że wszystko działa.
#
# Darmowe modele bywają czasem przeciążone lub limitowane. Jeśli trafisz na błąd albo bardzo długi czas odpowiedzi, wróć tutaj, wybierz inny model z listy i ponownie przetestuj połączenie. Pamiętaj też, żeby zapisać ostateczną nazwę modelu w sprawozdaniu.

# %% id="fBlLPA9BaBb_"
import ipywidgets as widgets
from IPython.display import display

free_models = [
    x.id
    for x in client.models.list()
    if x.id.endswith(":free")
    and "text" in x.architecture["input_modalities"]
    and "text" in x.architecture["output_modalities"]
]

ping_prompt = "Halo, halo. Czy nas słyszysz?"

model_label = widgets.Label("Model")
model = widgets.Select(
    options=free_models,
    value='nvidia/nemotron-3.5-super-vl:free',
    rows=8,
    layout=widgets.Layout(width='100%'),
)
test_button = widgets.Button(
    description="Testuj połączenie",
    button_style="primary",
    layout=widgets.Layout(width='100%'),
)
test_output = widgets.Output(
    layout=widgets.Layout(
        width='100%',
        padding='8px',
        border='1px solid #e0e0e0',
    )
)


def on_test_click(_):
    test_button.disabled = True
    try:
        with test_output:
            test_output.clear_output()
            print(f"> {ping_prompt}")
            try:
                response = client.responses.create(
                    model=model.value,
                    input=ping_prompt,
                    reasoning={"effort": "none"},
                    max_output_tokens=64,
                )
                print(f"> {response.output_text}")
            except Exception as e:
                print(f"Błąd: {e}")
    finally:
        test_button.disabled = False


test_button.on_click(on_test_click)
form = widgets.VBox(
    [model_label, model, test_button, test_output],
    layout=widgets.Layout(width='640px', padding='12px', border='1px solid #d0d7de'),
)
display(form)

# %% [markdown]
# ### Domyślne parametry generowania
#
# W całym notatniku korzystamy ze stałych ustawień początkowych (`DEFAULT_REASONING_EFFORT`, `DEFAULT_MAX_OUTPUT_TOKENS`, `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`). Pozostaną one aktywne aż do momentu, gdy w sekcji z zadaniami dodatkowymi na końcu notatnika zaznaczysz pole wyboru przełączające źródło parametrów na kontrolki zaawansowane.

# %%
from typing import Any

USE_ADVANCED_SETTINGS = False


def get_generation_params() -> dict[str, Any]:
    if USE_ADVANCED_SETTINGS:
        return {
            "reasoning_effort": adv_reasoning_effort.value,
            "max_output_tokens": int(adv_max_output_tokens.value),
            "temperature": float(adv_temperature.value),
            "top_p": float(adv_top_p.value),
        }
    return {
        "reasoning_effort": "none",
        "max_output_tokens": 256,
        "temperature": 0.2,
        "top_p": 1.0,
    }

# %% [markdown] id="Z7VFFaiybLS7"
# ## Stan gry
#
# Na początku rozgrywki zasady są proste i dokładnie określone:
# - gracz ma w sakiewce 15 monet,
# - kupiec ma na stanie 2 mikstury uzdrawiające,
# - jedna mikstura kosztuje 10 monet, a pojedynczy zakup dotyczy zawsze jednej sztuki.
#
# Wszystkie te liczby przechowujemy w zwykłej klasie w Pythonie (`GameState`), a nie w samej historii rozmowy z modelem. Gdybyśmy pozwolili modelowi pamiętać i liczyć pieniądze samodzielnie, gracz mógłby w rozmowie wytargować rabat, poprosić o towar na kredyt albo po prostu wykorzystać błąd rachunkowy modelu.
#
# Oprócz klasy stanu definiujemy też krotkę `ALLOWED_ACTIONS`, czyli zamkniętą listę akcji, jakie nasz program w ogóle potrafi obsłużyć (`"talk"` oraz `"buy_potion"`). Metoda `copy()` przyda nam się za chwilę, żeby podczas przetwarzania tury pracować na bezpiecznej kopii stanu gry.

# %%
from dataclasses import dataclass

ALLOWED_ACTIONS = ("talk", "buy_potion")


@dataclass
class GameState:
    coins: int = 15
    potions: int = 2
    potion_price: int = 10

    def copy(self) -> "GameState":
        return GameState(
            coins=self.coins,
            potions=self.potions,
            potion_price=self.potion_price,
        )

    def format_short(self) -> str:
        return f"monety={self.coins}, mikstury={self.potions} (cena={self.potion_price})"


# %% [markdown]
# Utwórzmy obiekt ze stanem początkowym, z którego będziemy korzystać w kolejnych krokach.

# %%
initial_state = GameState()
print(f"Dozwolone akcje: {ALLOWED_ACTIONS}")
print(f"Stan początkowy: {initial_state.format_short()}")

# %% [markdown] id="A5Mf6cThN8YT"
# ## Odpowiedzi ustrukturyzowane
#
# Zwykły czat z modelem zwraca po prostu kawałek tekstu. W grze to za mało: potrzebujemy oddzielnie wypowiedzi kupca, którą pokażemy graczowi na ekranie (`reply`), i jednoznacznego sygnału dla kodu, co właściwie ma się wydarzyć (`action`: `"talk"` albo `"buy_potion"`).
#
# Zamiast pisać kruche wyrażenia regularne i zgadywać na podstawie słów kupca, czy doszło do sprzedaży, każemy modelowi odpowiadać w formacie JSON pasującym do z góry opisanego schematu (JSON Schema).

# %% [markdown] id="mxUFLhZHOZtW"
# ### Generowanie opisu struktury odpowiedzi z użyciem biblioteki Pydantic
#
# Zamiast ręcznie pisać słownik ze schematem JSON, możemy zdefiniować zwykłą klasę w Pythonie dziedziczącą po `BaseModel`. Typy pól mówią, jakiego rodzaju danych oczekujemy, a teksty w `Field(description=...)` trafiają bezpośrednio do schematu i podpowiadają modelowi, co powinien wpisać w każde pole.
#
# Ustawiamy też `ConfigDict(extra="forbid")`. Dzięki temu w wygenerowanym schemacie pojawi się `"additionalProperties": false`, którego wymaga tryb ścisły (`strict=True`) w OpenAI Responses API.

# %% id="EB6mL_63t3KF"
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MerchantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(description="Wypowiedź kupca skierowana do gracza po polsku.")
    action: Literal["talk", "buy_potion"] = Field(
        description="Rozpoznana intencja: 'talk' dla rozmowy, pytań lub odmowy, 'buy_potion' gdy gracz chce kupić miksturę."
    )


# %% [markdown] id="jm_2G_7OOgcV"
# Sprawdźmy uzyskany opis struktury danych (JSON Schema)

# %% id="TSVFyreKud0z"
import json

RESPONSE_SCHEMA = MerchantResponse.model_json_schema()
print(json.dumps(RESPONSE_SCHEMA, indent=2, ensure_ascii=False))

# %% [markdown] id="vBlC4EysPfEX"
# ### Żądania z ustrukturyzowanymi odpowiedziami
#
# Sam schemat JSON pilnuje tylko kształtu danych. Żeby model wiedział, jak się zachować, musimy napisać mu porządną instrukcję systemową. Kupiec powinien wiedzieć:
# - kim jest i w jakim tonie ma odpowiadać (np. kupiec Barnaba, krótko i po polsku, w klimacie fantasy),
# - jaki jest aktualny stan gry (ile monet ma gracz, ile mikstur zostało w sklepie i po ile są),
# - co dokładnie sprzedaje i czego nie robi (np. nie skupuje przedmiotów i nie sprzedaje na kredyt),
# - kiedy w polu `action` wpisać `"buy_potion"`, a kiedy zostawić `"talk"`.
#
# Funkcja `build_merchant_prompt(state)` wstawia bieżące wartości ze stanu gry bezpośrednio do treści instrukcji.

# %%
def build_merchant_prompt(state: GameState) -> str:
    return f"""Jesteś kupcem Barnabą prowadzącym sklep z miksturami w grze RPG.
Odpowiadasz krótko (1-2 zdania), w klimacie fantasy i wyłącznie po polsku.

Aktualny stan gry:
- Monety gracza: {state.coins}
- Mikstury w magazynie: {state.potions} szt.
- Cena jednej mikstury: {state.potion_price} monet

Zasady sklepu i wyboru pola "action":
1. Sprzedajesz wyłącznie mikstury uzdrawiające po {state.potion_price} monet za sztukę. Jeden zakup to zawsze jedna mikstura.
2. Nie skupujesz innych przedmiotów, nie dajesz zniżek i nie sprzedajesz na kredyt.
3. Wybierz action="buy_potion" tylko wtedy, gdy gracz wyraźnie prosi o zakup mikstury.
   Jeśli gracz ma za mało monet lub w magazynie jest 0 mikstur, uprzedź go w polu "reply", że nie możesz sprzedać towaru.
4. Wybierz action="talk", gdy gracz wita się, pyta o ofertę lub cenę, mówi niejasno albo prosi o coś nieobsługiwanego (np. sprzedaż miecza czy pożyczkę).
"""


# %% [markdown]
# Poniższy formularz pozwala wypróbować instrukcję systemową i sprawdzić, jaki JSON zwraca model dla różnych wypowiedzi gracza. Schemat `MerchantResponse.model_json_schema()` przekazujemy w parametrze `text`.

# %% id="1Dmf5XML-ALi"
import ipywidgets as widgets
from IPython.display import display

structured_style = {"description_width": "140px"}

structured_system_prompt = widgets.Textarea(
    value=build_merchant_prompt(initial_state),
    placeholder="Wpisz instrukcję systemową...",
    description="Instrukcja systemowa:",
    rows=10,
    style=structured_style,
    layout=widgets.Layout(width="100%"),
)
structured_test_prompt = widgets.Textarea(
    value="Dzień dobry! Co masz na sprzedaż i ile to kosztuje?",
    placeholder="Wpisz zapytanie testowe...",
    description="Zapytanie testowe:",
    rows=2,
    style=structured_style,
    layout=widgets.Layout(width="100%"),
)
structured_send_button = widgets.Button(
    description="Wyślij",
    button_style="primary",
    layout=widgets.Layout(width="100%"),
)
structured_output = widgets.Output(
    layout=widgets.Layout(
        width="100%",
        padding="8px",
        border="1px solid #e0e0e0",
    )
)


def on_structured_send_click(_):
    structured_send_button.disabled = True
    try:
        with structured_output:
            structured_output.clear_output()
            print(f"Model: {model.value}")
            print(f"Gracz: {structured_test_prompt.value}\n")
            try:
                params = get_generation_params()
                response = client.responses.create(
                    model=model.value,
                    instructions=structured_system_prompt.value,
                    input=structured_test_prompt.value,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_output_tokens=params["max_output_tokens"],
                    reasoning={"effort": params["reasoning_effort"]},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "MerchantResponse",
                            "strict": True,
                            "schema": MerchantResponse.model_json_schema(),
                        }
                    },
                    timeout=20.0,
                )
                print("Surowa odpowiedź:")
                print(response.output_text)
            except Exception as exc:
                print(f"Błąd: {exc}")
    finally:
        structured_send_button.disabled = False


structured_send_button.on_click(on_structured_send_click)

structured_form = widgets.VBox(
    [structured_system_prompt, structured_test_prompt, structured_send_button, structured_output],
    layout=widgets.Layout(width="640px", padding="12px", border="1px solid #d0d7de"),
)
display(structured_form)

# %% [markdown]
# Sprawdźmy teraz automatycznie dwa podstawowe przypadki z opisu zadania: zwykłe pytanie o ofertę (oczekujemy `"action": "talk"`) oraz prośbę o zakup jednej mikstury (oczekujemy `"action": "buy_potion"`).

# %%
prompt_text = build_merchant_prompt(initial_state)

for sample_input in [
    "Dzień dobry! Co sprzedajesz i ile to kosztuje?",
    "Chcę kupić jedną miksturę uzdrawiającą.",
]:
    print(f"Gracz: {sample_input}")
    try:
        params = get_generation_params()
        response = client.responses.create(
            model=model.value,
            instructions=prompt_text,
            input=sample_input,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_output_tokens=params["max_output_tokens"],
            reasoning={"effort": params["reasoning_effort"]},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "MerchantResponse",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                }
            },
            timeout=20.0,
        )
        print(f"Odpowiedź JSON: {response.output_text}\n")
    except Exception as exc:
        print(f"Błąd wywołania: {exc}\n")

# %% [markdown] id="X2z6hj9xbzEg"
# ### Wykorzystanie odpowiedzi modelu do zmiany stanu gry
#
# Sam schemat JSON nie załatwia jeszcze wszystkiego. W praktyce trzeba się liczyć z kilkoma problemami:
# - połączenie z API może przekroczyć limit czasu (`timeout`) albo zwrócić błąd,
# - niektóre modele potrafią mimo schematu dopisać coś poza nawiasami klamrowymi albo uciąć odpowiedź w połowie,
# - nawet poprawny JSON może zawierać złą decyzję biznesową, na przykład `action="buy_potion"` i tekst „Oto Twoja mikstura!”, mimo że gracz ma tylko 5 monet albo w sklepie nic już nie zostało.
#
# Podzielimy więc obsługę tury na kilka małych funkcji. Zaczniemy od `call_merchant_llm`, która wysyła zapytanie do API z pilnowaniem limitu czasu (`timeout_sec`) i zwraca surowy tekst odpowiedzi, czas trwania, informację o ewentualnym przekroczeniu czasu oraz treść błędu.

# %%
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from openai import APITimeoutError


def call_merchant_llm(
    system_prompt: str,
    user_message: str,
    model_name: str | None = None,
    schema: dict[str, Any] | None = RESPONSE_SCHEMA,
    temperature: float | None = None,
    timeout_sec: float = 20.0,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, float, bool, str | None]:
    target_model = model_name or model.value
    params = get_generation_params()
    eff_temperature = params["temperature"] if temperature is None else temperature
    eff_top_p = params["top_p"] if top_p is None else top_p
    eff_max_output_tokens = params["max_output_tokens"] if max_output_tokens is None else max_output_tokens
    eff_reasoning_effort = params["reasoning_effort"] if reasoning_effort is None else reasoning_effort

    def _invoke() -> str:
        request_kwargs: dict[str, Any] = {
            "model": target_model,
            "instructions": system_prompt,
            "input": user_message,
            "temperature": eff_temperature,
            "top_p": eff_top_p,
            "max_output_tokens": eff_max_output_tokens,
            "reasoning": {"effort": eff_reasoning_effort},
            "timeout": timeout_sec,
        }

        if schema:
            schema_payload = dict(schema)
            if schema_payload.get("type") == "object" and "additionalProperties" not in schema_payload:
                schema_payload["additionalProperties"] = False
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_payload.get("title", "MerchantResponse"),
                    "strict": True,
                    "schema": schema_payload,
                }
            }
        else:
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        response = client.responses.create(**request_kwargs)
        return response.output_text or ""

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            raw_text = future.result(timeout=timeout_sec)
            return raw_text, time.perf_counter() - start, False, None
        except (FuturesTimeoutError, APITimeoutError):
            return "", time.perf_counter() - start, True, f"Przekroczono limit czasu ({timeout_sec} s)"
        except Exception as exc:
            return "", time.perf_counter() - start, False, str(exc)


# %% [markdown]
# Niektóre modele potrafią owinąć odpowiedź w znaczniki ```` ```json ... ``` ```` albo dopisać komentarz przed lub po właściwym obiekcie. Pomocnicza funkcja `_extract_first_json_object` wyszukuje w tekście pierwszy kompletny fragment ujęty w nawiasy klamrowe `{...}`, uwzględniając przy tym cudzysłowy i znaki ucieczki.

# %%
def _extract_first_json_object(text: str) -> str:
    start_idx = text.find("{")
    if start_idx == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return text[start_idx:]


# %% [markdown]
# Kolejny krok to `parse_and_validate_response`. Funkcja parsuje tekst jako JSON i sprawdza, czy zawiera niepuste pole tekstowe `reply` oraz pole `action` należące do `ALLOWED_ACTIONS`. Zwraca osobno informację o poprawności samego JSON-a i o poprawności akcji, co przyda się później przy wypełnianiu tabeli w sprawozdaniu.

# %%
def parse_and_validate_response(raw_text: str) -> tuple[bool, bool, str | None, str, str | None]:
    if not raw_text or not raw_text.strip():
        return False, False, None, "", "Pusta odpowiedź modelu."

    clean_text = raw_text.strip()
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        candidate = _extract_first_json_object(clean_text)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return False, False, None, "", f"Niepoprawny format JSON: {exc}"

    if not isinstance(data, dict):
        return False, False, None, "", "Odpowiedź JSON nie jest obiektem."

    reply = data.get("reply")
    action = data.get("action")

    if not isinstance(reply, str) or not reply.strip() or not isinstance(action, str):
        return False, False, None, "", "Brak wymaganych pól 'reply' lub 'action' albo zły typ danych."

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


# %% [markdown]
# Żeby wygodnie analizować przebieg poszczególnych prób, zapiszemy wszystkie informacje o pojedynczej turze (stan przed i po, surową odpowiedź modelu, wynik walidacji i komunikat programu) w obiekcie `TurnResult`.

# %%
@dataclass
class TurnResult:
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


# %% [markdown]
# Funkcja `process_turn` spina całość logiki gry dla pojedynczej wypowiedzi gracza. Najpierw tworzy kopię stanu gry, odpytuje model (lub przyjmuje sztuczną odpowiedź `mock_raw_output` na potrzeby testów), a następnie uruchamia walidację.
#
# Jeśli wszystko jest poprawne, a model zaproponował `"buy_potion"`, program sam sprawdza, czy w magazynie jest przynajmniej jedna mikstura i czy gracz ma wystarczająco dużo monet. Dopiero wtedy odejmuje cenę od salda gracza i zmniejsza stan magazynu. W każdym innym przypadku stan gry pozostaje nietknięty.

# %%
from collections.abc import Callable


def process_turn(
    player_input: str,
    state: GameState,
    scenario_name: str = "Rozmowa",
    prompt_builder: Callable[[GameState], str] = build_merchant_prompt,
    model_name: str | None = None,
    schema: dict[str, Any] | None = RESPONSE_SCHEMA,
    temperature: float | None = None,
    timeout_sec: float = 20.0,
    mock_raw_output: str | None = None,
) -> TurnResult:
    state_before = state.copy()
    state_after = state.copy()

    if mock_raw_output is not None:
        raw_output, elapsed, timed_out, llm_error = mock_raw_output, 0.0, False, None
    else:
        raw_output, elapsed, timed_out, llm_error = call_merchant_llm(
            system_prompt=prompt_builder(state_before),
            user_message=player_input,
            model_name=model_name,
            schema=schema,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

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

    json_valid, action_valid, proposed_action, merchant_reply, val_error = parse_and_validate_response(raw_output)

    if not json_valid or not action_valid:
        sys_msg = f"[ODRZUCONO ODPOWIEDŹ] {val_error} Stan gry bez zmian."
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

    tx_allowed = False
    if proposed_action == "talk":
        sys_msg = "Brak transakcji (akcja 'talk'). Stan gry bez zmian."
    elif proposed_action == "buy_potion":
        if state_before.potions <= 0:
            sys_msg = "ZAKUP ODRZUCONY PRZEZ PROGRAM: pusty magazyn kupca (0 mikstur)."
        elif state_before.coins < state_before.potion_price:
            sys_msg = (
                f"ZAKUP ODRZUCONY PRZEZ PROGRAM: za mało monet "
                f"(masz {state_before.coins}, potrzeba {state_before.potion_price})."
            )
        else:
            tx_allowed = True
            state_after.coins -= state_before.potion_price
            state_after.potions -= 1
            sys_msg = (
                f"ZAKUP ZATWIERDZONY PRZEZ PROGRAM: kupiono 1 miksturę za {state_before.potion_price} monet. "
                f"Nowy stan: {state_after.coins} monet, {state_after.potions} mikstur."
            )
    else:
        sys_msg = f"[ODRZUCONO ODPOWIEDŹ] Nieobsługiwana akcja: {proposed_action!r}."

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


# %% [markdown]
# Dodajmy jeszcze prostą funkcję `print_turn_summary`, która w czytelny sposób wypisze na ekranie przebieg pojedynczej tury: stan początkowy, wypowiedź gracza, propozycję modelu oraz ostateczną decyzję programu.

# %%
def print_turn_summary(turn: TurnResult, character_name: str = "Kupiec Barnaba") -> None:
    border = "=" * 78
    sep = "-" * 78
    print(border)
    print(f" SCENARIUSZ: {turn.scenario_name}")
    print(sep)
    print(
        f" [STAN PRZED]  Gracz: {turn.state_before.coins} monet | "
        f"Magazyn: {turn.state_before.potions} szt. (cena: {turn.state_before.potion_price} monet)"
    )
    print(f" [GRACZ]       {turn.player_input!r}")
    print(sep)
    if turn.timed_out:
        print(f" [MODEL]       TIMEOUT ({turn.elapsed_sec:.2f} s) - {turn.error}")
    elif turn.error and not turn.raw_output:
        print(f" [MODEL]       BŁĄD ({turn.elapsed_sec:.2f} s) - {turn.error}")
    else:
        print(f" [SUROWY JSON] {turn.raw_output.strip()} ({turn.elapsed_sec:.2f} s)")
        print(f" [{character_name.upper()}] {turn.merchant_reply!r}")
        print(
            f" [PROPOZYCJA]  action={turn.proposed_action!r} | "
            f"JSON OK: {'TAK' if turn.json_valid else 'NIE'} | "
            f"Akcja OK: {'TAK' if turn.action_valid else 'NIE'}"
        )
    print(sep)
    print(f" [PROGRAM]     {turn.system_message}")
    print(
        f" [STAN PO]     Gracz: {turn.state_after.coins} monet | "
        f"Magazyn: {turn.state_after.potions} szt."
    )
    print(border + "\n")


# %% [markdown]
# Zanim przejdziemy do rozmowy z prawdziwym modelem, sprawdźmy zachowanie walidatora na trzech sztucznie spreparowanych odpowiedziach (`mock_raw_output`).
#
# Pierwszy test symuluje sytuację, w której model urwał odpowiedź w połowie i zwrócił niepoprawny składniowo JSON.

# %%
dummy_state = GameState(coins=15, potions=2, potion_price=10)

res_bad_json = process_turn(
    "Kupię miksturę",
    dummy_state,
    "Test: Ucięty JSON",
    mock_raw_output='{"reply": "Oto mikstura", "action": ',
)
print(res_bad_json.system_message, "| Stan po:", res_bad_json.state_after.format_short())
assert res_bad_json.state_after == dummy_state

# %% [markdown]
# Drugi test sprawdza poprawny JSON, w którym model wymyślił własną nazwę akcji spoza `ALLOWED_ACTIONS` (na przykład `"give_free_gold"`).

# %%
res_bad_action = process_turn(
    "Oddaj złoto!",
    dummy_state,
    "Test: Nieznana akcja",
    mock_raw_output='{"reply": "Już oddaję!", "action": "give_free_gold"}',
)
print(res_bad_action.system_message, "| Stan po:", res_bad_action.state_after.format_short())
assert res_bad_action.state_after == dummy_state

# %% [markdown]
# Trzeci test pokazuje najważniejszy przypadek biznesowy: gracz ma tylko 5 monet, a model w polu `reply` radośnie zgadza się na rabat i zwraca `"action": "buy_potion"`. Nasz kod w Pythonie powinien zignorować obietnicę z tekstu i zablokować transakcję po sprawdzeniu salda.

# %%
poor_state = GameState(coins=5, potions=1, potion_price=10)
res_poor = process_turn(
    "Poproszę miksturę za 5 monet",
    poor_state,
    "Test: Za mało monet",
    mock_raw_output='{"reply": "Niech będzie, oto Twoja mikstura!", "action": "buy_potion"}',
)
print(res_poor.system_message, "| Stan po:", res_poor.state_after.format_short())
assert res_poor.state_after == poor_state

# %% [markdown]
# Poniższy widżet pozwala porozmawiać z kupcem „na żywo” z zachowaniem ciągłości stanu gry (`live_state`). Każde kliknięcie **Wyślij** uruchamia `process_turn` i aktualizuje licznik monet oraz mikstur u góry formularza, a przycisk **Resetuj stan gry** przywraca wartości początkowe.

# %%
live_state = initial_state.copy()


def _format_state_html(st: GameState) -> str:
    return (
        f"<b>Aktualny stan gry:</b> {st.coins} monet | "
        f"{st.potions} mikstur w magazynie | cena: {st.potion_price} monet"
    )


state_banner = widgets.HTML(value=_format_state_html(live_state))
chat_input = widgets.Textarea(
    value="Poproszę jedną miksturę uzdrawiającą.",
    placeholder="Napisz coś do kupca...",
    description="Wypowiedź gracza:",
    rows=2,
    style={"description_width": "140px"},
    layout=widgets.Layout(width="100%"),
)
chat_send_btn = widgets.Button(
    description="Wyślij",
    button_style="primary",
    layout=widgets.Layout(width="50%"),
)
chat_reset_btn = widgets.Button(
    description="Resetuj stan gry",
    button_style="warning",
    layout=widgets.Layout(width="50%"),
)
chat_output = widgets.Output(
    layout=widgets.Layout(width="100%", padding="8px", border="1px solid #e0e0e0")
)


def on_chat_send(_):
    global live_state
    chat_send_btn.disabled = True
    try:
        player_msg = chat_input.value.strip()
        if not player_msg:
            return
        turn = process_turn(
            player_input=player_msg,
            state=live_state,
            scenario_name="Rozmowa na żywo",
        )
        live_state = turn.state_after.copy()
        state_banner.value = _format_state_html(live_state)
        with chat_output:
            print_turn_summary(turn)
    finally:
        chat_send_btn.disabled = False


def on_chat_reset(_):
    global live_state
    live_state = initial_state.copy()
    state_banner.value = _format_state_html(live_state)
    with chat_output:
        chat_output.clear_output()
        print(f"Zresetowano stan gry: {live_state.format_short()}")


chat_send_btn.on_click(on_chat_send)
chat_reset_btn.on_click(on_chat_reset)

chat_box = widgets.VBox(
    [
        state_banner,
        chat_input,
        widgets.HBox([chat_send_btn, chat_reset_btn]),
        chat_output,
    ],
    layout=widgets.Layout(width="640px", padding="12px", border="1px solid #d0d7de"),
)
display(chat_box)

# %% [markdown]
# ## Testy w sześciu scenariuszach
#
# Żeby sprawdzić działanie całego rozwiązania w powtarzalny sposób, przeprowadzimy sześć konkretnych prób:
# 1. Pytanie o ofertę (start: 15 monet, 2 mikstury).
# 2. Poprawny zakup jednej mikstury (start: 15 monet, 2 mikstury; po zakupie powinno zostać 5 monet i 1 mikstura).
# 3. Ponowny zakup bez wystarczających środków (kontynuuje stan bezpośrednio po próbie 2: 5 monet, 1 mikstura).
# 4. Zakup przy pustym magazynie (start: 15 monet, 0 mikstur).
# 5. Niejasna wypowiedź gracza (start: 15 monet, 2 mikstury).
# 6. Żądanie nieobsługiwanej czynności, na przykład sprzedaży starego miecza albo prośby o kredyt (start: 15 monet, 2 mikstury).
#
# Zapiszmy najpierw parametry wszystkich sześciu prób na liście `TEST_SCENARIOS`. Flaga `use_previous_state` w trzecim scenariuszu mówi programowi, żeby zamiast tworzyć nowy stan początkowy, przejął stan gry powstały po drugim scenariuszu.

# %%
TEST_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Pytanie o ofertę",
        "player_input": "Witaj kupcze! Co masz na sprzedaż i ile kosztuje?",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
    },
    {
        "id": 2,
        "name": "Poprawny zakup",
        "player_input": "Poproszę jedną miksturę uzdrawiającą. Oto złoto.",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
    },
    {
        "id": 3,
        "name": "Ponowny zakup (brak monet)",
        "player_input": "Świetna mikstura! Kupię jeszcze jedną na zapas.",
        "use_previous_state": True,
        "initial_coins": 5,
        "initial_potions": 1,
        "potion_price": 10,
    },
    {
        "id": 4,
        "name": "Zakup przy pustym magazynie",
        "player_input": "Mam dużo złota, chcę od razu kupić jedną miksturę!",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 0,
        "potion_price": 10,
    },
    {
        "id": 5,
        "name": "Niejasna wypowiedź",
        "player_input": "Hmm... ładnie tu pachnie ziołami, sam już nie wiem po co przyszedłem...",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
    },
    {
        "id": 6,
        "name": "Nieobsługiwana czynność",
        "player_input": "Odkup ode mnie ten stary zardzewiały miecz albo sprzedaj miksturę na kredyt!",
        "use_previous_state": False,
        "initial_coins": 15,
        "initial_potions": 2,
        "potion_price": 10,
    },
]

# %% [markdown]
# Funkcja `run_scenarios` przechodzi po kolei przez listę scenariuszy, wywołuje `process_turn`, wypisuje podsumowanie każdej próby i zwraca listę wyników. Uruchommy ją dla naszego domyślnego kupca Barnaby.

# %%
def run_scenarios(
    scenarios: list[dict[str, Any]] = TEST_SCENARIOS,
    prompt_builder: Callable[[GameState], str] = build_merchant_prompt,
    character_name: str = "Kupiec Barnaba",
) -> list[TurnResult]:
    results: list[TurnResult] = []
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

        turn = process_turn(
            player_input=sc["player_input"],
            state=current_state,
            scenario_name=f"{sc['id']}. {sc['name']}",
            prompt_builder=prompt_builder,
        )
        results.append(turn)
        previous_state = turn.state_after.copy()
        print_turn_summary(turn, character_name=character_name)

    return results


test_results = run_scenarios()

# %% [markdown]
# ### Podsumowanie wyników i przygotowanie do dema
#
# W opisie wyników w sprawozdaniu (oraz podczas krótkiego dema na koniec zajęć) warto wyraźnie rozdzielić trzy warstwy:
# 1. Poprawność formatu: czy model za każdym razem zwrócił poprawny JSON zgodny ze schematem.
# 2. Rozpoznanie intencji: czy pole `action` pasowało do słów gracza. Zwróć uwagę na próby 3 i 4, gdzie gracz chce kupić miksturę, ale nie pozwalają na to zasady gry. Nawet jeśli model zwróci `buy_potion` (bo rozpoznał chęć zakupu), transakcję powinien zatrzymać kod w Pythonie.
# 3. Zmianę stanu gry: czy monety i mikstury zmieniły się wyłącznie w próbie 2 i czy program nie uległ ewentualnym obietnicom modelu w polu `reply`.
#
# Najpierw wypiszmy zwięzłą tabelę tekstową podsumowującą wszystkie sześć prób.

# %%
def print_results_table(results: list[TurnResult]) -> None:
    header = (
        f"{'Nr':<3} | {'Scenariusz':<30} | {'Stan przed':<13} | "
        f"{'JSON':<4} | {'Akcja LLM':<12} | {'Decyzja programu':<17} | {'Stan po':<13}"
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
        name_short = (r.scenario_name[:27] + "...") if len(r.scenario_name) > 30 else r.scenario_name
        print(
            f"{idx:<3} | {name_short:<30} | {s_before:<13} | "
            f"{json_ok:<4} | {prop:<12} | {decision:<17} | {s_after:<13}"
        )


print_results_table(test_results)

# %% [markdown]
# Poniższa funkcja formatuje te same wyniki jako tabelę Markdown, którą można bezpośrednio skopiować i wkleić do pliku sprawozdania.

# %%
def print_markdown_table(results: list[TurnResult]) -> None:
    print(
        "| Nr | Scenariusz | Wypowiedź gracza | Stan początkowy | JSON OK | "
        "Proponowana akcja | Odpowiedź modelu (skrót) | Decyzja programu | Stan końcowy |"
    )
    print("|---:|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(results, 1):
        s_before = f"{r.state_before.coins} monet, {r.state_before.potions} mikstur"
        s_after = f"{r.state_after.coins} monet, {r.state_after.potions} mikstur"
        json_ok = "Tak" if r.json_valid else "Nie"
        reply_clean = r.merchant_reply.replace("\n", " ").replace("|", "/")
        if len(reply_clean) > 60:
            reply_clean = reply_clean[:57] + "..."
        player_clean = r.player_input.replace("\n", " ").replace("|", "/")
        sys_clean = r.system_message.replace("\n", " ").replace("|", "/")
        print(
            f"| {idx} | {r.scenario_name} | '{player_clean}' | {s_before} | "
            f"{json_ok} | `{r.proposed_action}` | '{reply_clean}' | {sys_clean} | {s_after} |"
        )


print_markdown_table(test_results)

# %% [markdown]
# ## Zadania dodatkowe (dla chętnych)
#
# ### 1. Druga postać o innym stylu wypowiedzi
#
# Spróbuj zmienić sam charakter sprzedawcy, zostawiając dokładnie te same zasady gry, cenę i format JSON. Poniżej przykład funkcji `build_goblin_prompt`, która tworzy instrukcję dla gburowatego goblińskiego alchemika Griza mówiącego krótko i chaotycznie.

# %%
def build_goblin_prompt(state: GameState) -> str:
    return f"""Jesteś zrzędliwym, chciwym goblińskim alchemikiem o imieniu Griz.
Mówisz krótko, szorstko i chaotycznie, używasz wykrzykników i wtrąceń typu 'Hehe!', 'Złotko!', 'Griz nie głupi!', wyłącznie po polsku.

Aktualny stan gry:
- Monety gracza: {state.coins}
- Mikstury w kotle Griza: {state.potions} szt.
- Cena jednej mikstury: {state.potion_price} monet

Zasady sklepu i wyboru pola "action":
1. Sprzedajesz tylko mikstury po {state.potion_price} monet (1 zakup = 1 mikstura).
2. Nie kupujesz złomu od gracza, nie dajesz zniżek i nie sprzedajesz na krechę.
3. Ustaw action="buy_potion" tylko wtedy, gdy gracz wyraźnie chce kupić miksturę.
4. Ustaw action="talk" przy pytaniach o towar, niejasnym gadaniu albo próbach sprzedaży innych rzeczy.
"""


# %% [markdown]
# Uruchommy teraz zestaw sześciu scenariuszy dla Goblina Griza i zobaczmy w tabeli, czy mocniejsze „wczucie się w rolę” wpłynęło na poprawność formatu JSON lub wybór pola `action`.

# %%
goblin_results = run_scenarios(
    prompt_builder=build_goblin_prompt,
    character_name="Goblin Griz",
)

print_results_table(goblin_results)

# %% [markdown]
# ### 2. Eksperymenty z parametrami generowania
#
# Na to, co i w jaki sposób odpowie model, wpływa nie tylko treść promptu, ale też parametry wywołania API. W poniższym formularzu możesz pobawić się kilkoma z nich i sprawdzić przy okazji, czy nasz walidator wyłapuje problemy. Jeśli zaznaczysz pole wyboru na górze formularza, ustawione tutaj wartości zastąpią domyślne parametry globalne w całym notatniku.
#
# - `temperature` (0.0 do 2.0): steruje losowością doboru kolejnych słów (tokenów). Przy niskich wartościach (np. `0.2`) odpowiedzi są przewidywalne i powtarzalne, co dobrze sprawdza się przy klasyfikacji intencji. Wysoka temperatura (powyżej `1.0`) daje bardziej barwne wypowiedzi kupca, ale łatwo wtedy o zmyślone fakty albo zepsucie formatu JSON.
# - `top_p` (0.0 do 1.0, tzw. *nucleus sampling*): zamiast rozważać cały słownik, model wybiera tylko spośród najbardziej prawdopodobnych tokenów, których łączne prawdopodobieństwo sumuje się do `top_p`. Zwykle zmienia się albo `temperature`, albo `top_p`, a drugi parametr zostawia się domyślny.
# - `max_output_tokens`: twardy limit długości odpowiedzi w tokenach. Jeśli ustawisz tu małą liczbę (np. `16` albo `32`), model urwie wypowiedź w pół słowa i zwróci uszkodzony JSON, co od razu przetestuje działanie walidatora.
# - `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`): dotyczy modeli z trybem rozumowania i określa, jak długo model „zastanawia się” w ukrytych krokach przed napisaniem właściwej odpowiedzi. Większy wysiłek pomaga przy trudnych poleceniach, ale wydłuża czas oczekiwania i zjada pulę `max_output_tokens` (część modeli nie obsługuje wszystkich poziomów i może wtedy zwrócić błąd z API).

# %%
adv_style = {"description_width": "140px"}

adv_use_custom_params = widgets.Checkbox(
    value=False,
    description="Użyj ustawień zaawansowanych w całym notatniku",
    indent=False,
    layout=widgets.Layout(width="100%"),
)


def on_adv_toggle_change(change):
    global USE_ADVANCED_SETTINGS
    USE_ADVANCED_SETTINGS = bool(change["new"])


adv_use_custom_params.observe(on_adv_toggle_change, names="value")

adv_system_prompt = widgets.Textarea(
    value=build_merchant_prompt(initial_state),
    placeholder="Wpisz instrukcję systemową...",
    description="Instrukcja systemowa:",
    rows=8,
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_test_prompt = widgets.Textarea(
    value="Chcę kupić jedną miksturę uzdrawiającą.",
    placeholder="Wpisz zapytanie testowe...",
    description="Zapytanie testowe:",
    rows=2,
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_max_output_tokens = widgets.IntText(
    value=256,
    description="max_output_tokens:",
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_top_p = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=1.0,
    step=0.05,
    description="top_p:",
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_temperature = widgets.FloatSlider(
    value=0.7,
    min=0.0,
    max=2.0,
    step=0.05,
    description="temperature:",
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_reasoning_effort = widgets.Dropdown(
    options=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    value="none",
    description="reasoning_effort:",
    style=adv_style,
    layout=widgets.Layout(width="100%"),
)
adv_send_button = widgets.Button(
    description="Wyślij",
    button_style="primary",
    layout=widgets.Layout(width="100%"),
)
adv_output = widgets.Output(
    layout=widgets.Layout(width="100%", padding="8px", border="1px solid #e0e0e0")
)


def on_adv_send_click(_):
    adv_send_button.disabled = True
    try:
        with adv_output:
            adv_output.clear_output()
            raw_out, elapsed, timed_out, err = call_merchant_llm(
                system_prompt=adv_system_prompt.value,
                user_message=adv_test_prompt.value,
                model_name=model.value,
                schema=RESPONSE_SCHEMA,
                temperature=float(adv_temperature.value),
                top_p=float(adv_top_p.value),
                max_output_tokens=int(adv_max_output_tokens.value),
                reasoning_effort=adv_reasoning_effort.value,
                timeout_sec=20.0,
            )
            print(f"Czas: {elapsed:.2f} s | Timeout: {timed_out} | Błąd API: {err}")
            print(f"Surowa odpowiedź:\n{raw_out}\n")
            json_ok, action_ok, action, reply, val_err = parse_and_validate_response(raw_out)
            print(
                f"Walidacja -> JSON OK: {json_ok}, Akcja OK: {action_ok}, "
                f"action={action!r}, reply={reply!r}, błąd={val_err}"
            )
    finally:
        adv_send_button.disabled = False


adv_send_button.on_click(on_adv_send_click)

adv_form = widgets.VBox(
    [
        adv_use_custom_params,
        adv_system_prompt,
        adv_test_prompt,
        adv_max_output_tokens,
        adv_top_p,
        adv_temperature,
        adv_reasoning_effort,
        adv_send_button,
        adv_output,
    ],
    layout=widgets.Layout(width="640px", padding="12px", border="1px solid #d0d7de"),
)
display(adv_form)
