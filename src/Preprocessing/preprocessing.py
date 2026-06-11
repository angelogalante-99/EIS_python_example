"""
EEG Signal Pre-processing Pipeline
----------------------------------
Legge i dati raw, applica filtri (Notch + Band-pass),
esegue l'epoching con OVERLAPPING (finestre di 4s, step 1s)
e rimuove gli artefatti tramite TRIAL REJECTION (soglia alta).
"""

# Importazione delle librerie necessarie
import pandas as pd  # Per la manipolazione dei dati in formato tabellare (DataFrame)
import numpy as np  # Per i calcoli matematici e vettoriali sulle matrici
import mne  # Libreria standard per l'elaborazione di dati elettroencefalografici (EEG)
import os  # Per la gestione dei percorsi dei file nel sistema operativo

# Definizione dei percorsi (Paths) basati sulla posizione di questo script
# Risale di 3 livelli per trovare la cartella principale del progetto
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# Percorso del file CSV contenente i dati EEG grezzi (raw)
INPUT_CSV = os.path.join(BASE_DIR, "data", "eeg_dataset_raw.csv")
# Percorso in cui verrà salvato il dataset pulito e pre-elaborato
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")

# Parametri di Acquisizione e di Epoching
SFREQ = 256.0  # Frequenza di campionamento (Frequenza di sampling): 256 campioni al secondo
EPOCH_LEN_S = 4.0  # Lunghezza di ogni singola finestra temporale (epoca) in secondi
STEP_LEN_S = 1.0  # Passo di avanzamento della finestra: 1 secondo (crea overlapping)

# Conversione dei tempi (secondi) in numero effettivo di campioni (punti dati)
SAMPLES_EPOCH = int(SFREQ * EPOCH_LEN_S)  # Ogni epoca conterrà 1024 campioni (256 * 4)
SAMPLES_STEP = int(SFREQ * STEP_LEN_S)  # La finestra avanza di 256 campioni alla volta

# Parametri per i Filtri Frequenziali
NOTCH_FREQ = 50.0  # Frequenza del filtro Notch per rimuovere l'interferenza della rete elettrica europea
BANDPASS_LOW = 1.0  # Frequenza di taglio inferiore: rimuove le frequenze quasi continue (deriva lenta del segnale)
BANDPASS_HIGH = 40.0  # Frequenza di taglio superiore: taglia le alte frequenze (rumore muscolare/elettromiogramma)

# Parametro per la rimozione degli artefatti (Trial Rejection)
ARTEFACT_TH = 500e-6  # Soglia di rigetto espressa in Volt (corrisponde a 500 µV)

# Dizionari e liste di supporto per la decodifica dei canali e delle etichette (label)
LABEL_MAP = {0: "neutral", 1: "concentrating"}  # Mappa i valori numerici della label in stati cognitivi
EEG_CHANNELS = [
    "TP9",
    "AF7",
    "AF8",
    "TP10",
]  # Nomi dei 4 canali EEG utilizzati (tipici della configurazione standard, es. Muse)


# ────────────────────────────────────────────────────────────


def clean_short_sessions(
    df: pd.DataFrame, min_duration_s: int = 8
) -> pd.DataFrame:
    """Rimuove dal dataset le sessioni che durano meno della soglia specificata in secondi."""
    print(f"\n--- Fase 1: Pulizia Dataset ---")

    # Calcola il numero minimo di campioni richiesti affinché una sessione sia valida
    min_samples = min_duration_s * SFREQ

    # Conta quanti campioni (righe) ci sono per ogni combinazione unica di soggetto, sessione e label
    session_counts = (
        df.groupby(["subject", "session", "label"])
        .size()
        .reset_index(name="counts")
    )

    # Identifica quali sessioni non raggiungono il numero minimo di campioni richiesto
    short_sessions = session_counts[session_counts["counts"] < min_samples]

    # Se ci sono sessioni troppo corte, procede alla rimozione e stampa a schermo i dettagli
    if not short_sessions.empty:
        print(
            f"Rimozione di {len(short_sessions)} sessioni troppo corte (< {min_duration_s}s):"
        )
        for _, row in short_sessions.iterrows():
            # Stampa le informazioni della sessione scartata e la sua durata effettiva in secondi
            print(
                f"  Scartata: {row['subject']} | sessione {row['session']} | stato: {LABEL_MAP[row['label']]} ({row['counts']/SFREQ:.1f}s)"
            )

        # Seleziona solo le sessioni che superano o uguagliano la soglia minima
        valid_sessions = session_counts[session_counts["counts"] >= min_samples]

        # Esegue un'operazione di inner join per mantenere nel DataFrame originale solo le sessioni valide
        df_clean = pd.merge(
            df,
            valid_sessions[["subject", "session", "label"]],
            on=["subject", "session", "label"],
        )
        return df_clean
    else:
        # Se tutte le sessioni sono sufficientemente lunghe, restituisce il DataFrame iniziale senza modifiche
        print("Nessuna sessione anomala rilevata.")
        return df


def process_session(df_session: pd.DataFrame) -> pd.DataFrame:
    """Applica filtri, epoching con overlapping e rigetto artefatti a una singola sessione logica."""

    # 1. Costruzione dell'oggetto RawArray di MNE
    # Estrae i dati dei canali EEG, li traspone (.T) perché MNE vuole i canali sulle righe,
    # e moltiplica per 1e-6 perché il CSV è in microVolt (µV) ma MNE lavora nativamente in Volt (V).
    data = df_session[EEG_CHANNELS].values.T * 1e-6
    # Crea l'oggetto Info che contiene i metadati: nomi dei canali, frequenza di campionamento e tipo di canale
    info = mne.create_info(
        ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=["eeg"] * 4
    )
    # Inizializza l'array Raw di MNE con i dati e le informazioni create
    raw = mne.io.RawArray(data, info, verbose=False)

    # 2. Filtraggio Frequenziale
    # Applica il filtro Notch a 50Hz su tutti i canali EEG per eliminare il ronzio della rete elettrica
    raw.notch_filter(freqs=NOTCH_FREQ, picks="eeg", verbose=False)
    # Applica il filtro Passa-Banda (Band-pass) tra 1 e 40 Hz per isolare le bande d'interesse EEG (Delta, Theta, Alpha, Beta)
    raw.filter(
        l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, picks="eeg", verbose=False
    )

    # Ritorno a Pandas per l'Epoching manuale
    # Estrae la matrice dei dati filtrati, la traspone nuovamente (righe=campioni, colonne=canali)
    # e la moltiplica per 1e6 per riportare l'unità di misura in microVolt (µV)
    filtered_data = raw.get_data().T * 1e6

    n_samples = len(filtered_data)  # Numero totale di campioni della sessione attuale
    epochs = []  # Lista vuota che conterrà i DataFrame delle singole epoche valide
    epoch_counter = 0  # Contatore sequenziale per assegnare un ID univoco alle epoche della sessione

    # 3. Epoching con Overlapping
    # Ciclo for che scorre i dati partendo da 0 fino alla fine, avanzando ogni volta del passo impostato (SAMPLES_STEP = 1s)
    # La condizione garantisce che ci siano abbastanza campioni residui per completare l'ultima epoca inf n_samples - SAMPLES_EPOCH mi dice quante epoche comple riesco a fare
    for start_idx in range(
        0, n_samples - SAMPLES_EPOCH + 1, SAMPLES_STEP
    ):
        end_idx = (
            start_idx + SAMPLES_EPOCH
        )  # Calcola l'indice finale della finestra corrente

        # Estrae il blocco di dati relativo all'epoca corrente (lungo 4 secondi)
        epoch_data = filtered_data[start_idx:end_idx, :]

        # 4. Trial Rejection (Controllo della Soglia degli Artefatti)
        # Calcola il valore assoluto massimo all'interno dell'epoca. Se supera la soglia (500 µV), viene considerata corrotta
        # Nota: Nel codice c'è un'incongruenza logica tra il commento (250 µV) e la variabile reale ARTEFACT_TH (500 µV)
        if np.max(np.abs(epoch_data)) > (ARTEFACT_TH * 1e6):
            continue  # Salta l'epoca corrente e passa al ciclo successivo senza salvarla (l'epoca viene rigettata)

        # Se l'epoca supera il controllo di qualità, la trasforma in un DataFrame Pandas temporaneo
        df_epoch = pd.DataFrame(epoch_data, columns=EEG_CHANNELS)
        df_epoch[
            "epoch_id"
        ] = epoch_counter  # Associa l'ID dell'epoca corrente ai suoi campioni
        epochs.append(
            df_epoch
        )  # Aggiunge il DataFrame dell'epoca alla lista complessiva
        epoch_counter += (
            1  # Incrementa il contatore per la prossima epoca valida
        )

    # Se nessuna epoca di questa sessione è sopravvissuta al rigetto, restituisce un DataFrame vuoto
    if not epochs:
        return pd.DataFrame()

    # Concatena verticalmente tutte le epoche valide salvate nella lista in un unico DataFrame di sessione
    df_final_session = pd.concat(epochs, ignore_index=True)
    return df_final_session


def main():
    # Verifica preliminare: controlla se il file CSV di input esiste nel percorso specificato
    if not os.path.exists(INPUT_CSV):
        print(f"Errore: Il file {INPUT_CSV} non esiste.")
        return

    # Carica l'intero dataset EEG grezzo dal file CSV
    df_raw = pd.read_csv(INPUT_CSV)

    # 1. Fase di pulizia: Rimuove le sessioni che durano meno di 8 secondi
    df_clean = clean_short_sessions(df_raw, min_duration_s=8)

    print(
        f"\n--- Fase 2: Filtraggio, Overlapping e Trial Rejection ---"
    )
    print(
        f"Finestra: {EPOCH_LEN_S}s | Avanzamento: {STEP_LEN_S}s"
    )
    print(f"Soglia Rigetto: ±{ARTEFACT_TH*1e6:.0f} µV")
    print("-" * 50)

    processed_sessions = (
        []
    )  # Lista che conterrà i DataFrame pre-elaborati di ogni singola sessione

    # Raggruppa i dati in base alle sessioni logiche reali identificando univocamente soggetto, sessione e stato (label)
    groups = df_clean.groupby(["subject", "session", "label"])

    # Itera attraverso ogni singola sessione logica identificata dal groupby
    for (subject, session, label), df_group in groups:
        state_name = LABEL_MAP[
            label
        ]  # Converte la label numerica nella stringa descrittiva

        # Formula matematica per calcolare quante epoche teoriche si possono estrarre dalla sessione considerando l'overlapping
        teoric_windows = (len(df_group) - SAMPLES_EPOCH) // SAMPLES_STEP + 1

        # Invia i dati della sessione corrente alla funzione di elaborazione (filtri + epoching + rigetto)
        df_processed = process_session(df_group)

        # Se la funzione restituisce un DataFrame vuoto, significa che tutte le epoche contenevano artefatti
        if df_processed.empty:
            print(
                f"[{subject} - {state_name} - Ses {session}] SCARTATA: 100% epoche inquinate."
            )
            continue  # Salta l'aggiunta di metadati e passa alla sessione successiva

        # Calcola quante finestre/epoche reali sono sopravvissute al processo di Trial Rejection
        surviving_windows = df_processed["epoch_id"].nunique()

        # Stampa il report della sessione corrente indicando le epoche salvate rispetto a quelle teoriche iniziali
        print(
            f"[{subject} - {state_name} - Ses {session}] OK! Salvate {surviving_windows}/{teoric_windows} epoche"
        )

        # Ricostruisce e inietta i metadati nel DataFrame elaborato per non perdere le informazioni di origine
        df_processed["subject"] = subject
        df_processed["session"] = session
        df_processed["label"] = label

        # Aggiunge il DataFrame della sessione corrente alla lista dei risultati
        processed_sessions.append(df_processed)

    print("-" * 50)

    # Gestione dell'errore critico: se nessuna sessione ha superato le fasi di filtraggio e pulizia, blocca lo script
    if not processed_sessions:
        print(
            "Errore critico: Nessuna sessione è sopravvissuta al preprocessing."
        )
        return

    # Unisce tutti i DataFrame delle singole sessioni elaborate in un unico grande DataFrame finale
    final_df = pd.concat(processed_sessions, ignore_index=True)

    # Riordina l'ordine delle colonne per una pulizia formale (Metadati all'inizio, segnali al centro, etichetta alla fine)
    cols = ["subject", "session", "epoch_id"] + EEG_CHANNELS + ["label"]
    final_df = final_df[cols]

    # Salva il DataFrame finale pre-elaborato in un nuovo file CSV, omettendo l'indice automatico di Pandas
    final_df.to_csv(OUTPUT_CSV, index=False)
    # Stampa di conferma finale con il conteggio totale delle righe generate (ogni riga rappresenta un singolo campione)
    print(
        f"\nSalvataggio completato: {OUTPUT_CSV} ({len(final_df):,} righe totali)."
    )


# Punto di ingresso dello script Python
if __name__ == "__main__":
    # Configura MNE affinché non intasi l'output del terminale con messaggi informativi, mostrando solo i Warning gravi
    mne.set_log_level("WARNING")
    # Avvia l'esecuzione della funzione principale
    main()