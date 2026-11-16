"""Narzędzia pomocnicze do warsztatu 4: Rozmowa z kupcem sterowanym przez LLM (Google AI Studio)."""

import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

DEFAULT_MODEL = "gemma-2-2b-it"
ALLOWED_ACTIONS = ("talk", "buy_potion")


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


def load_scenarios(path: str | Path = "data/test_scenarios.json") -> list[dict[str, Any]]:
    """Wczytuje scenariusze testowe z pliku JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.2,
    seed: int = 42,
    timeout_sec: float = 20.0,
) -> tuple[str, float, bool, str | None]:
    """Wywołuje model Google Gemma przez API Google AI Studio z obsługą formatu JSON."""
    
    def _invoke() -> str:
        # Konfiguracja wymuszająca odpowiedź w formacie JSON
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        )
        
        model_instance = genai.GenerativeModel(model_name=model)
        
        schema_instruction = ""
        if schema:
            schema_instruction = f"\n\nMUSISZ ODPOWIEDZIEĆ W ŚCISŁYM FORMACIE JSON ZGODNYM Z PONIŻSZYM SCHEMATEM:\n{json.dumps(schema, ensure_ascii=False)}"

        full_prompt = f"{system_prompt}{schema_instruction}\n\nWYPOWIEDŹ GRACZA: {user_message}"
        
        response = model_instance.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return response.text

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            raw_text = future.result(timeout=timeout_sec)
            elapsed = time.perf_counter() - start
            return raw_text, elapsed, False, None
        except FuturesTimeoutError:
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
    print(f" [STAN PRZED]  Gracz: {turn.state_before.coins} monet | "
          f"Magazyn kupca: {turn.state_before.potions} szt. (cena: {turn.state_before.potion_price} monet)")
    print(f" [GRACZ]       '{turn.player_input}'")
    print(sep)
    if turn.timed_out:
        print(f" [MODEL LLM]   TIMEOUT ({turn.elapsed_sec:.2f} s) - {turn.error}")
    elif turn.error and not turn.raw_output:
        print(f" [MODEL LLM]   BŁĄD ({turn.elapsed_sec:.2f} s) - {turn.error}")
    else:
        print(f" [SUROWY JSON] {turn.raw_output.strip()} ({turn.elapsed_sec:.2f} s)")
        print(f" [{character_name.upper()}] '{turn.merchant_reply}'")
        print(f" [PROPOZYCJA]  action = {turn.proposed_action!r} "
              f"(JSON poprawny: {'TAK' if turn.json_valid else 'NIE'}, "
              f"Akcja dozwolona: {'TAK' if turn.action_valid else 'NIE'})")
    print(sep)
    print(f" [SILNIK GRY]  {turn.system_message}")
    print(f" [STAN PO]     Gracz: {turn.state_after.coins} monet | "
          f"Magazyn kupca: {turn.state_after.potions} szt.")
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
    """Generuje tabelę w formacie Markdown do łatwego wklejenia w sprawozdanie_szablon.md."""
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
