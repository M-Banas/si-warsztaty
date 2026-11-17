Warsztat 4 Rozmowa z kupcem sterowanym przez LLM
Cel i zadanie
Proszę zbudować tekstowy dialog z kupcem w grze. Gracz może pytać o towar i poprosić o zakup mikstury.
Model językowy rozpoznaje intencję i proponuje odpowiedź, a program rozstrzyga, czy transakcja jest
dozwolona. Celem jest poznanie sposobu połączenia swobodnej rozmowy ze ściśle określonym stanem gry.
Narzędzia i materiały startowe
Python, Jupyter Notebook, biblioteka json i przygotowane połączenie z LLM, np. przez Ollama. Model i
środowisko muszą być uruchomione przed warsztatem. Zespół liderski zapewnia funkcję wywołującą model,
obsługę timeoutu oraz prosty ekran tekstowy. Można użyć lokalnego, niewielkiego modelu instrukcyjnego lub
uzgodnionej usługi. Zakres nie obejmuje treningu modelu ani budowy rozbudowanej gry.
Stan początkowy: gracz ma 15 monet, kupiec dwie mikstury, każda kosztuje 10 monet. Dozwolone propozycje to
„talk” i „buy_potion”; pojedyncza akcja zakupu dotyczy jednej mikstury. Model zwraca JSON z polami „reply” i
„action”. Saldo, cena i zapas są utrzymywane przez program.
Przebieg pracy przez 90 minut
0–10 min: proszę omówić podział odpowiedzialności między model a logikę gry. Proszę rozpocząć sprawozdanie
i zapisać nazwę modelu.
10–30 min: proszę uzupełnić instrukcję opisującą rolę kupca, aktualny stan i dozwolone akcje. Proszę
zdefiniować format JSON; jeśli wybrane narzędzie obsługuje odpowiedzi zgodne ze schematem, proszę
wykorzystać tę możliwość. Proszę sprawdzić zwykłe pytanie oraz prośbę o zakup.
30–55 min: proszę dodać walidację pól i listy dozwolonych akcji. Program sam sprawdza liczbę monet i zapas, a
następnie wykonuje zakup albo go odrzuca. Niepoprawny JSON, nieznana akcja lub timeout nie zmieniają stanu.
Komunikat potwierdzający transakcję powstaje na podstawie wyniku programu, nie samej wypowiedzi modelu.
55–75 min: proszę wykonać sześć prób: pytanie o ofertę; poprawny zakup; ponowny zakup bez wystarczających
środków; zakup przy pustym magazynie; niejasną wypowiedź; żądanie nieobsługiwanej czynności. Proszę użyć
przygotowanych stanów początkowych, z wyjątkiem drugiego zakupu, który kontynuuje wcześniejszą transakcję.
75–90 min: proszę zapisać odpowiedzi, proponowane akcje, decyzje programu i stany końcowe. Proszę
podsumować obserwacje i oddać materiały.
Demo na forum grupy po części roboczej
5–6 minut na zespół: proszę zademonstrować zakup dozwolony i odrzucony. Proszę pokazać propozycję modelu
oraz faktyczny stan gry i wyjaśnić rolę walidacji.
Co powinno znaleźć się w wyniku
Instrukcja dla modelu, kod walidacji, tabela sześciu testów oraz zwięzłe wnioski. Proszę oddzielić poprawność
formatu, rozpoznanie intencji i poprawność zmiany stanu gry. Nawet odpowiedź zgodna ze schematem może
zawierać błędną propozycję. Proszę podać ustawienia generowania; przy ponownym uruchomieniu tekst
odpowiedzi może się różnić.
Zakres dodatkowy
Proszę dodać drugą postać o odmiennym sposobie wypowiadania się, zachowując identyczne reguły transakcji.
Proszę sprawdzić, czy zmiana stylu wpływa na poprawność akcji.
