# Catene di Markov applicate a Shut the box

Questo repository contiene il codice Python utilizzato nella tesi di laurea
**"Catene di Markov applicate ai giochi da tavolo"**.

Il codice costruisce e analizza un modello markoviano del gioco *Shut the box*,
nella versione a un solo giocatore con tessere numerate da 1 a 12.

## Contenuto del repository

Il file principale è:

- `matrici_tesi_markov.py`

Il codice permette di:

- generare lo spazio degli stati del gioco;
- costruire la matrice di transizione;
- individuare le matrici associate alla forma canonica della catena;
- calcolare le probabilità di assorbimento;
- determinare la probabilità di vittoria e di perdita;
- calcolare quantità legate al tempo medio di assorbimento;
- analizzare alcune estensioni del modello, tra cui la scelta non uniforme tra la chiusura degli addendi e la chiusura della somma.

## Modello considerato

Lo stato del gioco è rappresentato dall'insieme delle tessere ancora aperte.
Gli stati assorbenti sono:

- `W`, corrispondente alla vittoria;
- `L`, corrispondente alla perdita.

La matrice di transizione viene costruita considerando tutti i possibili esiti
del lancio di due dadi e le mosse disponibili in ciascuna configurazione.

Nel modello principale, quando un lancio non doppio permette sia di chiudere
gli addendi sia di chiudere la somma, le due mosse vengono scelte con la stessa
probabilità. Nel codice è inoltre considerata una possibile estensione in cui
questa scelta viene sbilanciata mediante un parametro `p`.

## Esecuzione

Per eseguire il codice è necessario avere Python installato.

## Note

Il repository ha lo scopo di rendere riproducibili i risultati computazionali
presentati nella tesi.
