import time
import math
import sys
from src.audio.soundscapes_player import soundscapesplayer

def test_mixer_sinusoidale():
    print("="*50)
    print("TEST MOTORE AUDIO: Oscillazione Sinusoidale")
    print("Premi Ctrl+C per fermare il test.")
    print("="*50)
    
    try:
        # 1. Inizializza il player (userà i due file "_loud" che abbiamo creato)
        player = soundscapesplayer()
        
        # 2. Avvia il playback infinito (con il fade-in iniziale)
        player.start()
        
        # 3. Parametri per l'onda sinusoidale
        frequenza = 0.05  # Velocità dell'onda (più è bassa, più il fade è lento e "Zen")
        ampiezza = 0.5    # Per portare il range da [-0.5, 0.5]
        offset = 0.5      # Per alzare il range a [0.0, 1.0]
        
        start_time = time.time()
        
        while True:
            # Calcola il tempo trascorso
            t = time.time() - start_time
            
            # Genera un valore da 0.0 a 1.0 che oscilla dolcemente nel tempo
            # Usa math.sin per creare un'onda curva e morbida
            prob_ansia = ampiezza * math.sin(2 * math.pi * frequenza * t) + offset
            
            # Formattiamo per evitare numeri troppo lunghi a schermo
            prob_ansia = round(prob_ansia, 3)
            
            # Invia la finta "probabilità" al mixer
            player.update_volume(prob_ansia)
            
            # Pulisce la riga del terminale e stampa l'output come una barra
            ansia_pct = int(prob_ansia * 100)
            calma_pct = 100 - ansia_pct
            barra = "█" * (ansia_pct // 2) + "-" * (50 - (ansia_pct // 2))
            
            sys.stdout.write(f"\r[MIXER] Ansia: {ansia_pct:3d}% | Calma: {calma_pct:3d}% | [{barra}]")
            sys.stdout.flush()
            
            # Aggiorna ogni 100 millisecondi (10 Hz) per una fluidità totale
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nTest interrotto dall'utente.")
    finally:
        # Spegne tutto dolcemente con il fade-out
        player.stop()
        print("Test concluso.")

if __name__ == "__main__":
    test_mixer_sinusoidale()