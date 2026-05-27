"""
Connette il Muse 2 via Bluetooth e avvia lo stream LSL.
Esegui questo script in un terminale separato prima di main.py.
"""
from muselsl import stream, list_muses

muses = list_muses()

if not muses:
    print('Nessun Muse 2 trovato. Verifica che il dispositivo sia acceso e vicino al PC.')
else:
    print(f'Trovato: {muses[0]["name"]} — avvio stream LSL...')
    stream(muses[0]['address'], ppg_enabled=False, acc_enabled=False, gyro_enabled=False)
    print('Stream terminato.')
