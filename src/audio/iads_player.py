import os
import pygame


class IADSPlayer:
    def __init__(self, folder_path):
        self.folder = folder_path
        pygame.mixer.init()
        # Creiamo una memoria per ricordarci cosa sta suonando in questo momento
        self.traccia_corrente = None

    def start(self):
        # Iniziamo in modalità neutra (non suona nulla finché non arriva il primo dato)
        pass

    def stop(self):
        pygame.mixer.music.stop()

    def set_anxiety_level(self, anxiety_prob: float):
        # 1. NUOVA SOGLIA: Siccome i dati vanno da 0.68 a 0.82,
        # mettiamo lo spartiacque a 0.75 per forzare il cambio di canzone!
        if anxiety_prob < 0.75:
            nuova_traccia = "pioggia"
            percorso_file = os.path.join(self.folder, "calma.wav")  # o calma.wav in base a come lo hai chiamato
        else:
            nuova_traccia = "ansia"
            percorso_file = os.path.join(self.folder, "ansia.wav")  # o ansia.wav

        # 3. IL CAMBIO DISCO
        if self.traccia_corrente != nuova_traccia:
            try:
                pygame.mixer.music.load(percorso_file)
                pygame.mixer.music.play(-1)
                self.traccia_corrente = nuova_traccia
            except Exception as e:
                print(f"Errore: non trovo il file {percorso_file}! Controlla la cartella.")

        # 4. VOLUME DRAMMATICO:
        # Se ansia è 0.80, il volume sarà 20% (bassissimo).
        # Se ansia è 0.68, il volume sarà 32% (più alto e udibile).
        # Creiamo uno stacco molto più evidente per le tue orecchie!
        volume_attuale = 1.0 - anxiety_prob

        # Questa riga assicura che il volume non vada in errore se supera i limiti matematici 0-1
        volume_attuale = max(0.0, min(1.0, volume_attuale))

        pygame.mixer.music.set_volume(volume_attuale)