import pygame
import os

class soundscapesplayer:
    def __init__(self):
        print("[AUDIO] Inizializzazione Mixer Dinamico Crossfade...")
        
        # FIX PER IL BUFFER: Riduciamo la latenza per evitare "click" ai riavvii dei loop
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        
        # Inizializziamo Pygame richiedendo esplicitamente canali multipli
        if not pygame.mixer.get_init():
            pygame.mixer.init(channels=2)
            
        self.chan_ansia = None
        self.chan_calma = None
        self.sound_ansia = None
        self.sound_calma = None
        self.is_playing = False
        
        soundscapes_dir = os.path.join("data", "soundscapes")
        if not os.path.exists(soundscapes_dir):
            print("[AUDIO ERRORE] Cartella data/soundscapes non trovata!")
            return
            
        # Peschiamo i file e li ordiniamo alfabeticamente
        files = sorted([f for f in os.listdir(soundscapes_dir) if f.endswith(('.wav', '.mp3', '.ogg'))])
        
        if len(files) < 2:
            print(f"[AUDIO ERRORE] Trovato solo {len(files)} file in data/soundscapes. Servono ESATTAMENTE 2 file (es. 1_ansia_long.wav e 2_calma_long.wav)!")
            return
            
        print(f"[AUDIO] Traccia ANSIA assegnata a: {files[0]}")
        print(f"[AUDIO] Traccia CALMA assegnata a: {files[1]}")
        
        try:
            self.sound_ansia = pygame.mixer.Sound(os.path.join(soundscapes_dir, files[0]))
            self.sound_calma = pygame.mixer.Sound(os.path.join(soundscapes_dir, files[1]))
            print("[AUDIO] File caricati correttamente.")
        except Exception as e:
            print(f"[AUDIO ERRORE] Impossibile caricare i file: {e}")

    def start(self):
        if self.sound_ansia and self.sound_calma and not self.is_playing:
            # Mandiamo in PLAY ENTRAMBI i suoni contemporaneamente in loop infinito con una sfumatura iniziale (fade_ms)
            self.chan_ansia = self.sound_ansia.play(loops=-1, fade_ms=1000)
            self.chan_calma = self.sound_calma.play(loops=-1, fade_ms=1000)
            self.is_playing = True
            print("[AUDIO] Motore Multi-Traccia AVVIATO!")
        else:
            print("[AUDIO ERRORE] Avvio annullato: i file audio non sono stati caricati correttamente.")

    def update_volume(self, anxiety_prob: float):
        if not self.chan_ansia or not self.chan_calma:
            return
            
        vol_ansia = anxiety_prob
        vol_calma = 1.0 - anxiety_prob
        
        self.chan_ansia.set_volume(vol_ansia)
        self.chan_calma.set_volume(vol_calma)
        
        # print(f"[MIXER] Ansia: {vol_ansia*100:.0f}% | Calma: {vol_calma*100:.0f}%")

    def stop(self):
        if self.chan_ansia: self.chan_ansia.fadeout(1500)
        if self.chan_calma: self.chan_calma.fadeout(1500)
        self.is_playing = False
        print("[AUDIO] Mixer spento (fade out).")