from __future__ import annotations

import sys
from functools import lru_cache
from fractions import Fraction
from itertools import combinations, product
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    print("ERRORE: numpy non installato. Esegui: pip install numpy")
    exit(1)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_DISPONIBILE = True
except ImportError:
    SCIPY_DISPONIBILE = False

# Forza l'output UTF-8, così simboli come "≈" vengono stampati correttamente.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ================================================================
# TIPO: Stato = tupla di interi oppure la stringa "L"
# ================================================================
Stato = Tuple[int, ...] | str
ConteggiTransizione = Tuple[Dict[Stato, int], Dict[Stato, int], Dict[Stato, int]]

NUMERI_CASELLE = tuple(range(1, 13))
ESITI_DADI = tuple(product(range(1, 7), repeat=2))
PROBABILITA_ESITO = Fraction(1, 36)


# ================================================================
# FUNZIONI DI UTILITÀ
# ================================================================

def genera_sottoinsiemi(elementi: Sequence[int]) -> List[Tuple[int, ...]]:
    risultato: List[Tuple[int, ...]] = []
    for r in range(len(elementi) + 1):
        for sottoinsieme in combinations(elementi, r):
            risultato.append(sottoinsieme)
    return risultato


@lru_cache(maxsize=1)
def stati_fissi() -> Tuple[Stato, ...]:
    return tuple(genera_sottoinsiemi(NUMERI_CASELLE)) + ("L",)


def genera_stati() -> List[Stato]:
    return list(stati_fissi())


def genera_stati_assorbenti() -> List[Stato]:
    return [(), "L"]


def rimuovi_numeri(stato: Tuple[int, ...], numeri_da_rimuovere: Sequence[int]) -> Tuple[int, ...]:
    numeri_da_togliere = set(numeri_da_rimuovere)
    return tuple(numero for numero in stato if numero not in numeri_da_togliere)


def formatta_frazione(valore: Fraction) -> str:
    return f"{valore.numerator}/{valore.denominator}"


def formatta_stato_come_insieme(stato: Stato) -> str:
    if stato == "L":
        return "L"
    if stato == ():
        return "{}"
    return "{" + ",".join(str(numero) for numero in stato) + "}"


def formatta_matrice(matrice: Sequence[Sequence[Fraction]]) -> str:
    if not matrice:
        return "[]"

    return "\n".join(
        "[" + ", ".join(formatta_frazione(valore) for valore in riga) + "]"
        for riga in matrice
    )


def matrice_identita(n: int) -> List[List[Fraction]]:
    return [
        [Fraction(1 if i == j else 0, 1) for j in range(n)]
        for i in range(n)
    ]


def sottrai_matrici(a: Sequence[Sequence[Fraction]], b: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    return [
        [a[i][j] - b[i][j] for j in range(len(a[0]))]
        for i in range(len(a))
    ]


def moltiplica_matrici(a: Sequence[Sequence[Fraction]], b: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    righe = len(a)
    colonne = len(b[0])
    interne = len(b)

    risultato = [[Fraction(0, 1) for _ in range(colonne)] for _ in range(righe)]

    for i in range(righe):
        for k in range(interne):
            for j in range(colonne):
                risultato[i][j] += a[i][k] * b[k][j]

    return risultato


def inverti_matrice(matrice: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    n = len(matrice)
    identita = matrice_identita(n)
    aumentata = [
        [matrice[i][j] for j in range(n)] + identita[i]
        for i in range(n)
    ]

    for colonna in range(n):
        pivot = None
        for riga in range(colonna, n):
            if aumentata[riga][colonna] != 0:
                pivot = riga
                break

        if pivot is None:
            raise ValueError("La matrice non è invertibile.")

        if pivot != colonna:
            aumentata[colonna], aumentata[pivot] = aumentata[pivot], aumentata[colonna]

        valore_pivot = aumentata[colonna][colonna]
        aumentata[colonna] = [valore / valore_pivot for valore in aumentata[colonna]]

        for riga in range(n):
            if riga == colonna:
                continue
            fattore = aumentata[riga][colonna]
            if fattore == 0:
                continue
            aumentata[riga] = [
                aumentata[riga][j] - fattore * aumentata[colonna][j]
                for j in range(2 * n)
            ]

    return [riga[n:] for riga in aumentata]


def mossa_da_lancio(stato: Tuple[int, ...], d1: int, d2: int, p: float = 0.5) -> Dict[Stato, Fraction]:
    # Questa funzione mantiene una versione leggibile delle regole del gioco.
    # Il calcolo effettivo delle transizioni usa qui sotto una versione ottimizzata.
    p_frazione = Fraction(str(p))
    q_frazione = Fraction(1, 1) - p_frazione
    aperti = set(stato)
    somma = d1 + d2

    if stato == ():
        return {(): Fraction(1, 1)}

    if d1 == d2:
        if somma in aperti:
            prossimo_stato = rimuovi_numeri(stato, [somma])
            return {prossimo_stato: Fraction(1, 1)}
        return {"L": Fraction(1, 1)}

    mosse_possibili = []

    if d1 in aperti and d2 in aperti:
        mosse_possibili.append(rimuovi_numeri(stato, [d1, d2]))

    if somma in aperti:
        mosse_possibili.append(rimuovi_numeri(stato, [somma]))

    if not mosse_possibili:
        return {"L": Fraction(1, 1)}

    if len(mosse_possibili) == 1:
        return {mosse_possibili[0]: Fraction(1, 1)}

    # Se entrambe le mosse sono disponibili, scegliamo gli addendi con probabilità p
    # e la somma con probabilità 1-p.
    return {
        mosse_possibili[0]: p_frazione,
        mosse_possibili[1]: q_frazione,
    }


def conta_esiti_per_stato(stato: Tuple[int, ...]) -> ConteggiTransizione:
    conteggi_forzati: Dict[Stato, int] = {}
    conteggi_addendi: Dict[Stato, int] = {}
    conteggi_somma: Dict[Stato, int] = {}
    aperti = set(stato)

    for d1, d2 in ESITI_DADI:
        somma = d1 + d2

        if d1 == d2:
            if somma in aperti:
                prossimo_stato = rimuovi_numeri(stato, [somma])
            else:
                prossimo_stato = "L"
            conteggi_forzati[prossimo_stato] = conteggi_forzati.get(prossimo_stato, 0) + 1
            continue

        puo_chiudere_addendi = d1 in aperti and d2 in aperti
        puo_chiudere_somma = somma in aperti

        if puo_chiudere_addendi and puo_chiudere_somma:
            stato_addendi = rimuovi_numeri(stato, [d1, d2])
            stato_somma = rimuovi_numeri(stato, [somma])
            conteggi_addendi[stato_addendi] = conteggi_addendi.get(stato_addendi, 0) + 1
            conteggi_somma[stato_somma] = conteggi_somma.get(stato_somma, 0) + 1
        elif puo_chiudere_addendi:
            prossimo_stato = rimuovi_numeri(stato, [d1, d2])
            conteggi_forzati[prossimo_stato] = conteggi_forzati.get(prossimo_stato, 0) + 1
        elif puo_chiudere_somma:
            prossimo_stato = rimuovi_numeri(stato, [somma])
            conteggi_forzati[prossimo_stato] = conteggi_forzati.get(prossimo_stato, 0) + 1
        else:
            conteggi_forzati["L"] = conteggi_forzati.get("L", 0) + 1

    return conteggi_forzati, conteggi_addendi, conteggi_somma


@lru_cache(maxsize=1)
def struttura_transizioni() -> Dict[Tuple[int, ...], ConteggiTransizione]:
    return {
        stato: conta_esiti_per_stato(stato)
        for stato in stati_fissi()
        if isinstance(stato, tuple) and stato != ()
    }


def genera_transizioni(stati: Sequence[Stato], p: float = 0.5) -> Dict[Stato, Dict[Stato, Fraction]]:
    transizioni: Dict[Stato, Dict[Stato, Fraction]] = {}
    p_frazione = Fraction(str(p))
    q_frazione = Fraction(1, 1) - p_frazione
    struttura = struttura_transizioni()

    for stato in stati:
        if stato == "L":
            transizioni[stato] = {"L": Fraction(1, 1)}
            continue

        if stato == ():
            transizioni[stato] = {(): Fraction(1, 1)}
            continue

        prob_totali: Dict[Stato, Fraction] = {}
        conteggi_forzati, conteggi_addendi, conteggi_somma = struttura[stato]

        for prossimo_stato, conteggio in conteggi_forzati.items():
            prob_totali[prossimo_stato] = prob_totali.get(prossimo_stato, Fraction(0, 1)) + PROBABILITA_ESITO * conteggio

        for prossimo_stato, conteggio in conteggi_addendi.items():
            prob_totali[prossimo_stato] = prob_totali.get(prossimo_stato, Fraction(0, 1)) + PROBABILITA_ESITO * conteggio * p_frazione

        for prossimo_stato, conteggio in conteggi_somma.items():
            prob_totali[prossimo_stato] = prob_totali.get(prossimo_stato, Fraction(0, 1)) + PROBABILITA_ESITO * conteggio * q_frazione

        transizioni[stato] = prob_totali

    return transizioni


# ================================================================
# CLASSE PRINCIPALE
# ================================================================

class CatenaShutTheBox:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.stati: List[Stato] = genera_stati()
        self.stati_assorbenti: List[Stato] = genera_stati_assorbenti()
        self.insieme_assorbente = set(self.stati_assorbenti)
        self.stati_transitori: List[Stato] = [
            stato for stato in self.stati if stato not in self.insieme_assorbente
        ]
        self.stati_ordinati: List[Stato] = self.stati_transitori + self.stati_assorbenti
        self.transizioni: Dict[Stato, Dict[Stato, Fraction]] = genera_transizioni(self.stati, p)

        self.dizionario_indice_transitorio: Dict[Stato, int] = {
            stato: indice for indice, stato in enumerate(self.stati_transitori)
        }

        self.memoria_durata: Dict[Stato, Fraction] = {}
        self.vettore_durata: List[Fraction] | None = None
        self.memoria_fine_turno: Dict[Stato, Dict[Stato, Fraction]] = {}
        self.memoria_completamento: Dict[Stato, Fraction] = {}
        self.memoria_turni_medi: Dict[Stato, Fraction | None] = {}
        
        self.matrice_A_densa: np.ndarray | None = None
        self.matrice_A_sparsa: csr_matrix | None = None

    # ----------------------------------------------------------------
    # METODI PER ACCEDERE ALLE MATRICI
    # ----------------------------------------------------------------

    def ottieni_indice_transitorio(self, stato: Stato) -> int:
        return self.dizionario_indice_transitorio[stato]

    def elemento_P(self, da: Stato, a: Stato) -> Fraction:
        return self.transizioni[da].get(a, Fraction(0, 1))

    def elemento_Q(self, da: Stato, a: Stato) -> Fraction:
        if da in self.insieme_assorbente or a in self.insieme_assorbente:
            return Fraction(0, 1)
        return self.elemento_P(da, a)

    def riga_Q(self, stato: Stato) -> Dict[Stato, Fraction]:
        if stato in self.insieme_assorbente:
            raise ValueError("Q è definita solo sugli stati transitori.")

        return {
            prossimo_stato: probabilita
            for prossimo_stato, probabilita in self.transizioni[stato].items()
            if prossimo_stato not in self.insieme_assorbente
        }

    def riga_R(self, stato: Stato) -> Dict[Stato, Fraction]:
        if stato in self.insieme_assorbente:
            raise ValueError("R è definita solo sugli stati transitori.")

        return {
            prossimo_stato: probabilita
            for prossimo_stato, probabilita in self.transizioni[stato].items()
            if prossimo_stato in self.insieme_assorbente
        }

    def verifica_somme_righe(self) -> bool:
        for stato in self.stati_ordinati:
            somma_riga = sum(self.transizioni[stato].values(), Fraction(0, 1))
            if somma_riga != Fraction(1, 1):
                return False
        return True

    # ----------------------------------------------------------------
    # COSTRUZIONE DELLE MATRICI
    # ----------------------------------------------------------------

    def costruisci_matrice_A(self) -> Tuple[np.ndarray | None, csr_matrix | None]:
        if self.matrice_A_densa is not None or self.matrice_A_sparsa is not None:
            return self.matrice_A_densa, self.matrice_A_sparsa

        n = len(self.stati_transitori)
        if n == 0:
            self.matrice_A_densa = np.array([])
            self.matrice_A_sparsa = None
            return self.matrice_A_densa, self.matrice_A_sparsa

        # Costruiamo A = I - Q sfruttando solo le voci non nulle di Q.
        # Questo evita il doppio ciclo su tutti i 4095 stati transitori.
        righe: List[int] = []
        colonne: List[int] = []
        dati: List[float] = []

        for i, s in enumerate(self.stati_transitori):
            righe.append(i)
            colonne.append(i)
            dati.append(1.0)

            for t, probabilita in self.riga_Q(s).items():
                j = self.dizionario_indice_transitorio[t]
                righe.append(i)
                colonne.append(j)
                dati.append(-float(probabilita))

        if SCIPY_DISPONIBILE:
            self.matrice_A_sparsa = csr_matrix((dati, (righe, colonne)), shape=(n, n))
            self.matrice_A_densa = None
        else:
            A = np.zeros((n, n), dtype=float)
            for i, j, valore in zip(righe, colonne, dati):
                A[i, j] += valore
            self.matrice_A_densa = A
            self.matrice_A_sparsa = None

        return self.matrice_A_densa, self.matrice_A_sparsa

    def risolvi_sistema(self, A: np.ndarray | None, b: np.ndarray, A_sparsa: csr_matrix | None = None) -> np.ndarray | None:
        if A_sparsa is not None:
            n = A_sparsa.shape[0]
        elif A is not None:
            n = A.shape[0]
        else:
            return None

        if n == 0:
            return None

        try:
            if SCIPY_DISPONIBILE and A_sparsa is not None:
                return spsolve(A_sparsa, b)
            else:
                if A is None:
                    return None
                return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

    # ----------------------------------------------------------------
    # CALCOLO DEL VETTORE DELLE DURATE
    # ----------------------------------------------------------------

    def calcola_vettore_durata(self) -> List[Fraction]:
        if self.vettore_durata is not None:
            return self.vettore_durata

        n = len(self.stati_transitori)
        if n == 0:
            self.vettore_durata = []
            return self.vettore_durata

        A, A_sparsa = self.costruisci_matrice_A()
        b = np.ones(n, dtype=float)

        d_float = self.risolvi_sistema(A, b, A_sparsa)
        if d_float is None:
            self.vettore_durata = []
            return self.vettore_durata

        from fractions import Fraction as F
        self.vettore_durata = [
            F(int(round(val * 10**12)), 10**12) for val in d_float
        ]
        return self.vettore_durata

    # ----------------------------------------------------------------
    # FUNZIONI PER IL CALCOLO DELLE PROBABILITÀ (modello a lanci)
    # ----------------------------------------------------------------

    def probabilita_vittoria_turno(self, stato: Stato, memo: Dict[Stato, Fraction] | None = None) -> Fraction:
        if memo is None:
            memo = {}

        if stato in memo:
            return memo[stato]

        if stato == ():
            memo[stato] = Fraction(1, 1)
            return memo[stato]

        if stato == "L":
            memo[stato] = Fraction(0, 1)
            return memo[stato]

        valore = Fraction(0, 1)
        for prossimo_stato, probabilita in self.transizioni[stato].items():
            valore += probabilita * self.probabilita_vittoria_turno(prossimo_stato, memo)

        memo[stato] = valore
        return valore

    # ----------------------------------------------------------------
    # FUNZIONI PER IL CALCOLO DELLE DURATE
    # ----------------------------------------------------------------

    def durata_media_turno(self, stato: Stato) -> Fraction | None:
        if stato == "L" or stato == ():
            return Fraction(0, 1)

        if stato in self.memoria_durata:
            return self.memoria_durata[stato]

        vettore = self.calcola_vettore_durata()
        if not vettore:
            return None

        idx = self.ottieni_indice_transitorio(stato)
        risultato = vettore[idx]
        self.memoria_durata[stato] = risultato
        return risultato

    # ----------------------------------------------------------------
    # FUNZIONI PER IL MODELLO A TURNI
    # ----------------------------------------------------------------

    def distribuzione_fine_turno(self, stato: Stato, memo: Dict[Stato, Dict[Stato, Fraction]] | None = None) -> Dict[Stato, Fraction]:
        if memo is None:
            memo = self.memoria_fine_turno

        if stato in memo:
            return memo[stato]

        if stato == ():
            memo[stato] = {(): Fraction(1, 1)}
            return memo[stato]

        if stato == "L":
            raise ValueError("Questa funzione va applicata solo a stati validi diversi da 'L'.")

        risultato: Dict[Stato, Fraction] = {}

        for prossimo_stato, probabilita in self.transizioni[stato].items():
            if prossimo_stato == "L":
                risultato[stato] = risultato.get(stato, Fraction(0, 1)) + probabilita
            elif prossimo_stato == ():
                risultato[()] = risultato.get((), Fraction(0, 1)) + probabilita
            else:
                distribuzione_coda = self.distribuzione_fine_turno(prossimo_stato, memo)
                for stato_finale, prob_coda in distribuzione_coda.items():
                    risultato[stato_finale] = risultato.get(stato_finale, Fraction(0, 1)) + probabilita * prob_coda

        memo[stato] = risultato
        return risultato

    def probabilita_completamento(self, stato: Stato) -> Fraction:
        memo = self.memoria_completamento
        memo_fine_turno = self.memoria_fine_turno

        def ricorsiva(s: Stato) -> Fraction:
            if s in memo:
                return memo[s]

            if s == ():
                memo[s] = Fraction(1, 1)
                return memo[s]

            if s == "L":
                raise ValueError("Questa funzione va applicata solo a stati validi diversi da 'L'.")

            distribuzione = self.distribuzione_fine_turno(s, memo_fine_turno)
            prob_restart = distribuzione.get(s, Fraction(0, 1))
            numeratore = distribuzione.get((), Fraction(0, 1))

            for stato_finale, probabilita in distribuzione.items():
                if stato_finale == () or stato_finale == s:
                    continue
                numeratore += probabilita * ricorsiva(stato_finale)

            denominatore = Fraction(1, 1) - prob_restart
            if denominatore == 0:
                memo[s] = Fraction(0, 1)
                return memo[s]

            valore = numeratore / denominatore
            memo[s] = valore
            return valore

        return ricorsiva(stato)

    def turni_medi_condizionati_vittoria(self, stato: Stato) -> Fraction | None:
        memo = self.memoria_turni_medi
        memo_completamento = self.memoria_completamento
        memo_fine_turno = self.memoria_fine_turno

        def ricorsiva(s: Stato) -> Fraction | None:
            if s in memo:
                return memo[s]

            if s == ():
                memo[s] = Fraction(0, 1)
                return memo[s]

            if s == "L":
                raise ValueError("Questa funzione va applicata solo a stati validi diversi da 'L'.")

            if s in memo_completamento:
                p_s = memo_completamento[s]
            else:
                p_s = self.probabilita_completamento(s)
            if p_s == 0:
                memo[s] = None
                return None

            distribuzione = self.distribuzione_fine_turno(s, memo_fine_turno)
            vittoria_in_un_turno = distribuzione.get((), Fraction(0, 1))
            prob_restart = distribuzione.get(s, Fraction(0, 1))
            altri_contributi = Fraction(0, 1)

            for stato_finale, probabilita in distribuzione.items():
                if stato_finale == () or stato_finale == s:
                    continue

                if stato_finale in memo_completamento:
                    p_finale = memo_completamento[stato_finale]
                else:
                    p_finale = self.probabilita_completamento(stato_finale)
                prossimo_atteso = ricorsiva(stato_finale)

                if prossimo_atteso is None or p_finale == 0:
                    continue

                altri_contributi += probabilita * p_finale * (Fraction(1, 1) + prossimo_atteso)

            numeratore = vittoria_in_un_turno + prob_restart * p_s + altri_contributi
            denominatore = p_s * (Fraction(1, 1) - prob_restart)

            if denominatore == 0:
                memo[s] = None
                return None

            valore = numeratore / denominatore
            memo[s] = valore
            return valore

        return ricorsiva(stato)

# ================================================================
# FUNZIONI DI STAMPA
# ================================================================


def stampa_probabilita_da_stato(catena: CatenaShutTheBox, stato: Stato) -> None:
    """
    Stampa solo le probabilità di vittoria a partire da uno stato scelto.
    È pensata per l'uso pratico nel file finale della tesi.
    """
    print("\n" + "=" * 70)
    print("ANALISI DI UNO STATO INIZIALE SCELTO")
    print("=" * 70)
    print(f"Stato analizzato: {stato}")

    p_lanci = catena.probabilita_vittoria_turno(stato)
    q_lanci = Fraction(1, 1) - p_lanci
    print("\nProbabilità di vittoria nel modello a lanci:")
    print(f"   {formatta_frazione(p_lanci)} ≈ {float(p_lanci):.10f}")
    print("\nProbabilità di perdita nel modello a lanci:")
    print(f"   {formatta_frazione(q_lanci)} ≈ {float(q_lanci):.10f}")

    p_turni = catena.probabilita_completamento(stato)
    q_turni = Fraction(1, 1) - p_turni
    print("\nProbabilità finale di vittoria nel modello a turni:")
    print(f"   {formatta_frazione(p_turni)} ≈ {float(p_turni):.10f}")
    print("\nProbabilità finale di perdita nel modello a turni:")
    print(f"   {formatta_frazione(q_turni)} ≈ {float(q_turni):.10f}")


def stampa_tabella_probabilita_vittoria_s0() -> None:
    # Estensione del modello: probabilità di vittoria partendo da s0 al variare di p.
    s0 = tuple(range(1, 13))
    valori_p = [0, 0.25, 0.5, 0.75, 1]

    print("\n" + "=" * 70)
    print("TABELLA FINALE - PROBABILITÀ DI VITTORIA AL VARIARE DI p")
    print("=" * 70)
    print("Stato iniziale: s0 = {1,2,3,4,5,6,7,8,9,10,11,12}")
    print("p = probabilità di scegliere la chiusura degli addendi {d1,d2}")
    print("1-p = probabilità di scegliere la chiusura della somma d1+d2\n")

    intestazione = (
        f"{'Prob. addendi (p)':<18} | "
        f"{'Prob. somma (1-p)':<18} | "
        f"{'Valore esatto B_s0_W(p)':<34} | "
        f"{'Valore decimale'}"
    )
    print(intestazione)
    print("-" * len(intestazione))

    for p in valori_p:
        catena = CatenaShutTheBox(p=p)
        probabilita = catena.probabilita_vittoria_turno(s0)
        p_testo = f"{p:.2f}".rstrip("0").rstrip(".")
        q_testo = f"{1 - p:.2f}".rstrip("0").rstrip(".")
        probabilita_esatta = formatta_frazione(probabilita)
        probabilita_decimale = f"{float(probabilita):.10f}"
        print(
            f"{p_testo:<18} | "
            f"{q_testo:<18} | "
            f"{probabilita_esatta:<34} | "
            f"{probabilita_decimale}"
        )


def stampa_sottocatena_sezione_211(catena: CatenaShutTheBox) -> None:
    """
    Sezione 2.11 della tesi.
    Stampa le matrici Q, R, N e B per la sottocatena scelta nel testo.
    Gli stati transitori scelti sono:
    s1 = {10,11,12}, s2 = {10,11}, s3 = {10,12}, s4 = {11,12},
    s5 = {10}, s6 = {11}, s7 = {12}
    mentre gli stati assorbenti sono:
    W = (), L = "L".
    """
    stati_transitori = [
        (10, 11, 12),
        (10, 11),
        (10, 12),
        (11, 12),
        (10,),
        (11,),
        (12,),
    ]
    stati_assorbenti = [(), "L"]

    etichette_transitorie = [
        f"s{i} = {formatta_stato_come_insieme(stato)}"
        for i, stato in enumerate(stati_transitori, start=1)
    ]
    etichette_assorbenti = [
        "W = {}" if stato == () else formatta_stato_come_insieme(stato)
        for stato in stati_assorbenti
    ]

    Q = [
        [catena.elemento_Q(da, a) for a in stati_transitori]
        for da in stati_transitori
    ]
    R = [
        [catena.elemento_P(da, a) for a in stati_assorbenti]
        for da in stati_transitori
    ]
    N = inverti_matrice(sottrai_matrici(matrice_identita(len(stati_transitori)), Q))
    B = moltiplica_matrici(N, R)

    print("\n" + "=" * 70)
    print("RISULTATI PER LA TESI - SEZIONE 2.11")
    print("=" * 70)
    print("\nOrdine degli stati transitori:")
    for etichetta in etichette_transitorie:
        print(f"   {etichetta}")

    print("\nOrdine degli stati assorbenti:")
    for etichetta in etichette_assorbenti:
        print(f"   {etichetta}")

    print("\nMatrice Q:")
    print(formatta_matrice(Q))

    print("\nMatrice R:")
    print(formatta_matrice(R))

    print("\nMatrice N = (I - Q)^(-1):")
    print(formatta_matrice(N))

    print("\nMatrice B = N R:")
    print(formatta_matrice(B))


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    catena = CatenaShutTheBox()
    s0 = tuple(range(1, 13))

    # ============================================================
    # 1. SEZIONE 2.11
    #    Matrici Q, R, N e B per la sottocatena scelta.
    # ============================================================
    stampa_sottocatena_sezione_211(catena)

    # ============================================================
    # 2. ANALISI DI UNO STATO SCELTO
    #    Modifica questo stato se vuoi studiare un'altra configurazione.
    # ============================================================
    stato_da_analizzare = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    stampa_probabilita_da_stato(catena, stato_da_analizzare)

    # ============================================================
    # 3. RISULTATI PRINCIPALI PER LA TESI
    #    Sezione 2.13:
    #    - 2.13.1 durata media di un turno nel modello a lanci
    #    - 2.13.2 probabilità di vittoria e numero medio di turni
    #      nel modello a turni
    # ============================================================
    print("\n" + "=" * 70)
    print("RISULTATI PER LA TESI - SEZIONE 2.13")
    print("=" * 70)

    d0 = catena.durata_media_turno(s0)
    print(f"\n1. Durata media di un turno da s0 (incondizionata):")
    print(f"   {float(d0):.4f} lanci" if d0 else "   Non definito")

    p = catena.probabilita_completamento(s0)
    print(f"\n2. Probabilità di vittoria (modello a turni):")
    print(f"   {float(p):.6f}")

    tau = catena.turni_medi_condizionati_vittoria(s0)
    print(f"\n3. Numero medio di turni (condizionato a vittoria):")
    print(f"   {float(tau):.4f} turni" if tau else "   Non definito")


    # ============================================================
    # 4. ESTENSIONE PARAMETRICA
    #    Questa parte è un'estensione del modello richiamata in 2.15:
    #    serve per studiare cosa cambia se la scelta tra le due mosse
    #    non è uniforme.
    # ============================================================
    stampa_tabella_probabilita_vittoria_s0()
