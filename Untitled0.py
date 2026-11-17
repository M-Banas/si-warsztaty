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

# %% [markdown] id="v9kAiVqfN1XV"
# ## Połączenie z modelem językowym

# %% [markdown] id="SwdCDk8wNPCR"
# ### Konfiguracja połączenia
# <!-- TODO: opis komórki, instrukcja jak dodać user data do colab, notka odnośnie bezpiecznego postępowania z kluczami -->

# %% id="hvO456J1aOw2"
from openai import OpenAI
from google.colab import userdata

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=userdata.get("OPENROUTER_API_KEY")
)

# %% [markdown] id="YVbYeHOFNhLK"
# ### Wybór modelu
# <!-- TODO: opis komórki -->

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

test_prompt = "Halo, halo. Czy nas słyszysz?"

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
            print(f"> {test_prompt}")
            try:
                response = client.responses.create(
                    model=model.value,
                    input=test_prompt,
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

# %% [markdown] id="Z7VFFaiybLS7"
# ## Stan gry

# %% [markdown] id="A5Mf6cThN8YT"
# ## Odpowiedzi ustrukturyzowane
#
#
#

# %% [markdown] id="mxUFLhZHOZtW"
# ### Generowanie opisu struktury odpowiedzi z użyciem biblioteki Pydantic

# %% id="EB6mL_63t3KF"
from pydantic import BaseModel, Field
from typing import Literal

class MerchantResponse(BaseModel):
    reply: str = Field(description="Wypowiedź kupca skierowana do gracza po polsku.")
    action: Literal["talk", "buy_potion"] = Field(
        description="Rozpoznana intencja/akcja: 'talk' dla rozmowy lub odmowy, 'buy_potion' gdy gracz chce kupić miksturę."
    )


# %% [markdown] id="jm_2G_7OOgcV"
# Sprawdźmy uzyskany opis struktury danych (JSON Schema)

# %% id="TSVFyreKud0z"
import json
MerchantResponse.model_json_schema()

# %% [markdown] id="vBlC4EysPfEX"
# ### Żądania z ustrukturyzowanymi odpowiedziami

# %% id="1Dmf5XML-ALi"
import ipywidgets as widgets
from IPython.display import display

style = {"description_width": "140px"}

system_prompt = widgets.Textarea(
    value="",
    placeholder="Wpisz instrukcję systemową...",
    description="Instrukcja systemowa:",
    rows=8,
    style=style,
    layout=widgets.Layout(width="100%"),
)
test_prompt = widgets.Textarea(
    value="",
    placeholder="Wpisz zapytanie testowe...",
    description="Zapytanie testowe:",
    rows=1,
    style=style,
    layout=widgets.Layout(width="100%"),
)
send_button = widgets.Button(
    description="Wyślij",
    button_style="primary",
    layout=widgets.Layout(width='100%'),
)

params_form = widgets.VBox(
    [system_prompt, test_prompt, send_button],
    layout=widgets.Layout(width="640px", padding="12px", border="1px solid #d0d7de"),
)
display(params_form)

# %% [markdown] id="X2z6hj9xbzEg"
# ### Wykorzystanie odpoeidzi modelu do zmiany stanu gry
