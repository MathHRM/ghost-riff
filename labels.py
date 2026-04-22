# keys must match class indices used during data collection
chord_labels = {
    0: {
        "name": "Idle", # 🫳 mão relaxada semi-aberta (estado neutro)
        "audio": None
    },
    1: {
        "name": "C", # ☝️ 1 dedo (indicador levantado)
        "audio": "sounds/chords/C.mp3"
    },
    2: {
        "name": "G", # ✌️ 2 dedos (indicador + médio)
        "audio": "sounds/chords/G.mp3"
    },
    3: {
        "name": "D", # 🤟 3 dedos
        "audio": "sounds/chords/D.mp3"
    },
    4: {
        "name": "A", # 🖐️ mão aberta
        "audio": "sounds/chords/A.mp3"
    },
    5: {
        "name": "E", # ✊ punho fechado
        "audio": "sounds/chords/E.mp3"
    },
    6: {
        "name": "Am", # 👍 joinha
        "audio": "sounds/chords/Am.mp3"
    },
    7: {
        "name": "Em", # 🤘 rock
        "audio": "sounds/chords/Em.mp3"
    },
}

stroke_labels = {
    0: {
        "name": "Idle", # 🫳 mão relaxada semi-aberta (estado neutro)
    },
    1: {
        "name": "Down", # ☝️ movimento rápido de cima para baixo
    },
    2: {
        "name": "Up", # 👇 movimento rápido de baixo para cima
    },
    3: {
        "name": "Mute", # 🖐️ mão parada ou bloqueando (baixa velocidade)
    },
}
