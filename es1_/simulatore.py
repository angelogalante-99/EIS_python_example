from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np
import time

FS = 256
N_CH = 4
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def create_muse_outlet(name, source_id, manufacturer="SimulatedMuse"):
    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=N_CH,
        nominal_srate=FS,
        channel_format="float32",
        source_id=source_id
    )

    desc = info.desc()
    desc.append_child_value("manufacturer", manufacturer)
    channels = desc.append_child("channels")

    for ch in CH_NAMES:
        c = channels.append_child("channel")
        c.append_child_value("label", ch)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")

    return StreamOutlet(info)


# Due outlet distinti
outlet1 = create_muse_outlet("MuseSim_1", "muse_sim_001")

t0 = time.time()
next_t = t0

while True:
    t = time.time() - t0
    ts = local_clock()

    # Device 1
    sample1 = np.array([
        30 * np.sin(2 * np.pi * 10 * t) + np.random.randn() * 5,
        25 * np.sin(2 * np.pi * 10 * t + 0.3) + np.random.randn() * 5,
        20 * np.sin(2 * np.pi * 12 * t + 0.6) + np.random.randn() * 5,
        28 * np.sin(2 * np.pi * 9 * t + 0.9) + np.random.randn() * 5,
    ], dtype=np.float32)

    # Device 2: frequenze/fasi leggermente diverse
    sample2 = np.array([
        22 * np.sin(2 * np.pi * 8 * t + 0.2) + np.random.randn() * 6,
        27 * np.sin(2 * np.pi * 11 * t + 0.5) + np.random.randn() * 5,
        24 * np.sin(2 * np.pi * 10 * t + 1.0) + np.random.randn() * 4,
        26 * np.sin(2 * np.pi * 9 * t + 1.3) + np.random.randn() * 6,
    ], dtype=np.float32)

    # Blink artificiale device 1 ogni ~5 s
    if int(t) % 5 == 0 and (t % 5) < 0.08:
        sample1[1] += 120
        sample1[2] += 120

    # Blink artificiale device 2 ogni ~7 s
    if int(t) % 7 == 0 and (t % 7) < 0.08:
        sample2[1] += 100
        sample2[2] += 100

    outlet1.push_sample(sample1.tolist(), ts)

    next_t += 1.0 / FS
    sleep_time = next_t - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)