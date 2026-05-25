import os
from glob import glob
import pickle as pkl
from pyedflib import highlevel

users_data = glob('BIDS/*')
users_data = [f for f in users_data if not (f.endswith('.json') or f.endswith('.md'))]
users_data.sort()

dataframe = {}  # dizionario vuoto da popolare con i dati delle sessioni utente
corrupted_data = []  # lista per tenere traccia dei dati non validi
for user_file in users_data:  # itero per ogni file utente
    print(user_file)
    user_session = glob(user_file + '/*')
    user_session.sort(key=os.path.getmtime)
    for session in user_session:
        print(session)
        path = session + '/eeg/'
        pat_session = glob(path + '/*.edf')

        if pat_session:
            signals, signal_headers, header = highlevel.read_edf(pat_session[0])
            markers = (0, 8) # calcola i marcatori
           # Ottiene solo il nome dell'ultima cartella (es. sub-ID000) indipendentemente dal sistema operativo
            user_name = os.path.basename(os.path.normpath(user_file))
            session_number = session.split('/')[-1]

            session_data = {"session_id": session_number,
                            "time_series": signals,
                            "fixation_cross_marker": markers[0],
                            "start_video_marker": markers[1]}  # dizionario con informazione della sessione corrente
            # create dataframe # dizionario con chiavi (nomi utenti) e valori (liste di dati delle sessioni)
            if user_name in dataframe:
                dataframe[user_name].append(session_data)  # aggiorna dizionario aggiungendo la sessione al nome utente
            else:
                dataframe.update({user_name: [session_data]})


with open('dataframe.pkl', 'wb') as handle:
    pkl.dump(dataframe, handle, protocol=pkl.HIGHEST_PROTOCOL)