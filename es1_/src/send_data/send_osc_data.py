import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from sklearn.preprocessing import Normalizer

ip = "127.0.0.1"
port = 1338
client = SimpleUDPClient(ip, port, allow_broadcast=True)

class SendOscData:
    def __init__(self):
        pass

    def send_osc_data(self, focus, relaxation, stress, engagements, valence, ):
        if focus is not None:
            data = np.array([[focus, relaxation, stress, engagements]])
            scaler = Normalizer()
            data = scaler.fit_transform(data)

            client.send_message('/prediction', [
                float(data[0, 0]), float(data[0, 1]), float(data[0, 2]), float(data[0, 3])
            ])

            print(f'focus level: {focus:.2f}')
            print(f'relaxation: {relaxation:.2f}')
            print(f'stress level: {stress:.2f}')
            print(f'engagements index: {engagements:.2f}')

            client.send_message('/valence', [float(valence)])

            