# midi_decomposer_app.py - VERSIONE RIVISTA E CORRETTA

import streamlit as st
import streamlit.components.v1 as components
import mido
import random
import numpy as np
import io
import base64
from collections import defaultdict

# --- Configurazione della Pagina ---
st.set_page_config(
    page_title="MIDI Decomposer by loop507",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Titolo Principale ---
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1> MIDI Decomposer <span style='font-size:0.6em; color: #666;'>by <span style='font-size:0.8em;'>loop507</span></span></h1>
    <p style='font-size: 1.2em; color: #888;'>Scomponi e Ricomponi File MIDI in Nuove Strutture Musicali</p>
    <p style='font-style: italic;'>Esplora il caos e l'ordine nella generazione MIDI</p>
</div>
""", unsafe_allow_html=True)

# --- Funzioni di Utilità ---
def get_key_offset(key_name):
    """Converte il nome della tonalità in offset semitonale, supportando maggiori e minori."""
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    base_note_char = key_name[0]
    sharp_flat_char = ''
    if len(key_name) > 1 and (key_name[1] == '#' or key_name[1] == 'b'):
        sharp_flat_char = key_name[1]
    
    base_note_name = base_note_char + sharp_flat_char
    
    offset = note_offsets.get(base_note_name, 0)
    return offset

def get_scale_notes(scale_name):
    """Restituisce gli intervalli (in semitoni) di una scala rispetto alla sua radice."""
    scales = {
        "Cromatica": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Maggiore": [0, 2, 4, 5, 7, 9, 11],
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10],
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Blues": [0, 3, 5, 6, 7, 10]
    }
    return scales.get(scale_name, scales["Cromatica"])

def extract_notes(track, ticks_per_beat=384):
    """Helper per estrarre note e il loro tempo assoluto da una traccia."""
    notes = []
    active_notes = {}
    current_abs_time = 0
    for msg in track:
        current_abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[(msg.note, msg.channel)] = {'start': current_abs_time, 'velocity': msg.velocity}
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.note, msg.channel)
            if key in active_notes:
                start_data = active_notes.pop(key)
                notes.append({'start': start_data['start'], 'end': current_abs_time, 'pitch': msg.note, 'velocity': start_data['velocity'], 'channel': key[1]})
    # Note rimaste aperte senza note_off — chiuse con durata stimata di 1 beat
    # invece di usare current_abs_time che creerebbe note lunghissime
    for key, start_data in active_notes.items():
        estimated_end = start_data['start'] + ticks_per_beat  # 1 beat di default
        notes.append({'start': start_data['start'], 'end': estimated_end, 'pitch': key[0], 'velocity': start_data['velocity'], 'channel': key[1]})
    return notes


def _extract_instrument_header(track):
    """
    Estrae i messaggi che definiscono lo strumento di una traccia originale
    (program_change ed eventuali control_change di bank select 0/32), letti
    prima del primo evento nota. Le funzioni che ricostruiscono una traccia
    da zero a partire dalle sole note (extract_notes) devono ri-applicare
    questi messaggi in testa alla nuova traccia — altrimenti la DAW (es.
    Logic Pro) assegna il proprio strumento di default (tipicamente
    "Steinway Grand Piano") a tutte le tracce, perdendo l'orchestrazione
    originale anche quando il numero e i nomi delle tracce sono corretti.
    """
    header = []
    for msg in track:
        if msg.type in ('note_on', 'note_off'):
            break
        if msg.type == 'program_change':
            header.append(msg.copy(time=0))
        elif msg.type == 'control_change' and msg.control in (0, 32):
            header.append(msg.copy(time=0))
    return header


def _get_track_default_channel(track):
    """Ritorna il canale MIDI dominante di una traccia (il primo trovato), o 0 se assente."""
    for msg in track:
        if hasattr(msg, 'channel'):
            return msg.channel
    return 0

def reconstruct_track(notes, ticks_per_beat):
    """Helper per ricostruire una traccia da una lista di note."""
    new_track = mido.MidiTrack()
    events = []
    for note in notes:
        events.append({'msg': mido.Message('note_on', note=note['pitch'], velocity=note['velocity'], channel=note['channel'], time=0), 'abs_time': note['start']})
        events.append({'msg': mido.Message('note_off', note=note['pitch'], velocity=0, channel=note['channel'], time=0), 'abs_time': note['end']})
    
    events.sort(key=lambda x: x['abs_time'])

    last_abs_time = 0
    for event in events:
        delta_time = event['abs_time'] - last_abs_time
        if delta_time < 0:
            delta_time = 0
        
        new_msg = event['msg'].copy(time=delta_time)
        new_track.append(new_msg)
        last_abs_time = event['abs_time']
    return new_track

# --- Costas Array Utilities (costruzione di Welch, GF(p)) ---
# Rif: J.P. Costas (1965); L. Welch construction via radice primitiva mod p.
# Scott Rickard ha usato la stessa costruzione per generare melodie prive di
# autocorrelazione ("la canzone piu' irritante mai composta").

def _costas_is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def _costas_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _costas_find_prime(min_order):
    """Trova il piu' piccolo primo p tale che p-1 >= min_order."""
    p = max(3, min_order + 1)
    while not _costas_is_prime(p):
        p += 1
    return p

def _costas_primitive_root(p):
    """Trova una radice primitiva di p (esiste sempre per p primo)."""
    if p == 2:
        return 1
    phi = p - 1
    factors = _costas_prime_factors(phi)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return 2  # fallback teorico, non dovrebbe mai accadere per p primo

def generate_costas_array(min_order):
    """
    Genera una matrice/sequenza di Costas tramite la costruzione di Welch:
    per un primo p con radice primitiva g, la permutazione
        perm[i] = (g^(i+1) mod p) - 1   per i = 0..p-2
    e' una permutazione di {0,...,p-2} = {0,...,n-1} con la proprieta' di Costas
    (tutti i vettori differenza tra coppie di punti sono distinti).

    Ritorna: (perm, n, p, g)
      perm: lista di lunghezza n, permutazione di 0..n-1 (perm[riga] = colonna)
      n:    ordine effettivo della matrice (n = p-1, >= min_order richiesto)
      p:    primo usato
      g:    radice primitiva usata
    """
    min_order = max(1, int(min_order))
    p = _costas_find_prime(min_order)
    g = _costas_primitive_root(p)
    n = p - 1
    perm = [(pow(g, i + 1, p) - 1) for i in range(n)]
    return perm, n, p, g


def midi_costas_pitch_permutation(original_midi, transpose_octave=0):
    """
    Modalita' 1: Permutazione Pitch (cromatica).
    Usa una matrice di Costas di ordine 12 (p=13, primo) come cifrario di
    sostituzione deterministico e privo di autocorrelazione per le classi di
    altezza: ogni classe di pitch (0-11) viene rimappata secondo perm[pitch_class],
    mantenendo l'ottava originale (+ eventuale trasposizione).
    A differenza del Random Pitch Transformer, la mappatura e' fissa e
    biunivoca: stesso pitch in ingresso -> sempre stesso pitch in uscita.
    """
    perm, n, p, g = generate_costas_array(12)  # p=13 -> n=12, mappa cromatica esatta
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(original_track, 'name') and original_track.name:
            new_track.name = original_track.name

        for msg in original_track:
            if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                pitch_class = msg.note % 12
                octave = msg.note // 12
                new_pitch_class = perm[pitch_class % n]
                new_pitch = (octave * 12) + new_pitch_class + (transpose_octave * 12)
                new_pitch = max(0, min(127, new_pitch))
                new_track.append(msg.copy(note=new_pitch))
            else:
                new_track.append(msg)

        new_midi.tracks.append(new_track)
    return new_midi, (n, p, g)


def midi_costas_rhythmic_grid(original_midi, min_order, block_notes=None):
    """
    Modalita' 2: Griglia Ritmica Costas.
    Raggruppa le note (per traccia, in ordine di apertura) in blocchi di n note
    (n = ordine effettivo della matrice) e ridistribuisce gli onset all'interno
    di ciascun blocco secondo la permutazione di Costas su una griglia di n slot
    che copre l'estensione temporale originale del blocco. Pitch e durate
    restano quelli originali: cambia solo *dove* cade ogni nota — uno shuffle
    algoritmico non ripetitivo, non casuale.
    """
    perm, n, p, g = generate_costas_array(min_order)
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        _name = original_track.name if hasattr(original_track, 'name') else ''
        _header = _extract_instrument_header(original_track)
        notes = extract_notes(original_track, original_midi.ticks_per_beat)

        if not notes:
            new_midi.tracks.append(original_track)
            continue

        notes_sorted = sorted(notes, key=lambda x: x['start'])
        final_events = []

        for block_start in range(0, len(notes_sorted), n):
            block = notes_sorted[block_start: block_start + n]
            if not block:
                continue
            block_begin_tick = block[0]['start']
            block_end_tick = max(nd['end'] for nd in block)
            block_span = max(1, block_end_tick - block_begin_tick)
            slot_size = block_span / n

            for idx, note_data in enumerate(block):
                slot = perm[idx % n]
                new_start = block_begin_tick + int(round(slot * slot_size))
                duration = max(1, note_data['end'] - note_data['start'])
                new_end = new_start + duration

                final_events.append({'msg': mido.Message('note_on', note=note_data['pitch'], velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': new_start})
                final_events.append({'msg': mido.Message('note_off', note=note_data['pitch'], velocity=0, channel=note_data['channel'], time=0), 'abs_time': new_end})

        final_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))

        new_track = mido.MidiTrack()
        if _name:
            new_track.name = _name
        for _h in _header:
            new_track.append(_h)
        last_abs_time = 0
        for event_data in final_events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            new_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(new_track)
    return new_midi, (n, p, g)


def midi_costas_generator(original_midi, min_order, base_pitch, pitch_range_semitones, step_beats, channel=0):
    """
    Modalita' 3: Generatore Costas (nuova melodia) — nello spirito della
    "canzone piu' irritante" di Scott Rickard. Genera una traccia MIDI
    autonoma che copre l'intera durata del brano originale, in cui ogni passo
    i (su una griglia ciclica di n passi) suona il pitch:
        base_pitch + round(perm[i] * pitch_range / (n-1))
    Nessuna ripetizione di intervallo tra le coppie di note e' presente
    all'interno di ciascun ciclo (proprieta' di Costas), quindi la melodia
    non presenta alcun pattern memorizzabile.
    Le tracce originali vengono mantenute; questa si aggiunge come nuova traccia.
    """
    perm, n, p, g = generate_costas_array(min_order)
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)

    if total_ticks == 0:
        st.warning("Il brano originale non contiene eventi validi. Il generatore Costas non verra' aggiunto.")
        return new_midi, (n, p, g)

    step_ticks = max(1, int(round(step_beats * original_midi.ticks_per_beat)))
    costas_track = mido.MidiTrack()
    costas_track.name = f"Costas Generator (n={n}, p={p}, g={g})"
    costas_track.append(mido.Message('program_change', program=0, channel=channel, time=0))  # Acoustic Grand Piano di default

    events = []
    t = 0
    i = 0
    while t < total_ticks:
        slot = perm[i % n]
        denom = max(1, n - 1)
        pitch = base_pitch + int(round(slot * pitch_range_semitones / denom))
        pitch = max(0, min(127, pitch))
        note_len = max(1, int(step_ticks * 0.9))
        events.append({'msg': mido.Message('note_on', note=pitch, velocity=95, channel=channel, time=0), 'abs_time': t})
        events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=channel, time=0), 'abs_time': t + note_len})
        t += step_ticks
        i += 1

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for event_data in events:
        delta = max(0, event_data['abs_time'] - last_abs_time)
        costas_track.append(event_data['msg'].copy(time=delta))
        last_abs_time = event_data['abs_time']

    new_midi.tracks.append(costas_track)
    return new_midi, (n, p, g)


# --- Compositori: Karlheinz Stockhausen / Boulez — Serialismo Integrale (Punktuelle Musik) ---
# Rif: Stockhausen "Kreuzspiel" (1951), Boulez "Structures Ia" (1952), radicati nel
# "Mode de valeurs et d'intensites" di Messiaen (1949) — il primo a serializzare
# non solo l'altezza ma anche durata, dinamica e attacco. Ogni nota diventa un
# "punto" sonoro isolato le cui 4 dimensioni (pitch/durata/dinamica/timbro) sono
# governate da 4 forme indipendenti della stessa fila a 12 elementi.

def _row_prime(row):
    return list(row)

def _row_retrograde(row):
    return list(reversed(row))

def _row_inversion(row):
    return [(12 - x) % 12 for x in row]

def _row_retrograde_inversion(row):
    return list(reversed(_row_inversion(row)))

def derive_twelve_tone_row(original_midi):
    """
    Deriva la fila dodecafonica direttamente dal materiale del brano: le prime
    12 classi di altezza distinte incontrate, nell'ordine di apparizione
    (prassi seriale classica: la fila nasce dal materiale stesso).
    Se il brano non contiene 12 classi distinte, completa con la permutazione
    di Costas di ordine 12 (stessa costruzione di Welch del Costas Sequencer),
    cosi' la fila resta comunque priva di ripetizioni banali.
    """
    seen = []
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pc = msg.note % 12
                if pc not in seen:
                    seen.append(pc)
                if len(seen) == 12:
                    break
        if len(seen) == 12:
            break
    if len(seen) < 12:
        costas_perm, _n, _p, _g = generate_costas_array(12)
        for pc in costas_perm:
            if pc not in seen:
                seen.append(pc)
            if len(seen) == 12:
                break
    return seen[:12]


def midi_stockhausen_punktuelle(original_midi, serialize_duration=True, serialize_dynamics=True,
                                 serialize_timbre=True, isolamento_punti=True):
    """
    Serialismo integrale multiparametrico (stile Stockhausen/Boulez).
    Estrae una fila a 12 elementi dal brano, poi applica 4 forme indipendenti
    della fila a 4 parametri scorrelati di ogni singola nota, trattata come un
    punto sonoro isolato:
      - Altezza  -> forma Prima (P)         (classe di pitch rimappata)
      - Durata   -> forma Retrograda (R)    (12 classi di durata fisse)
      - Dinamica -> forma Inversione (I)    (12 classi di velocity fisse)
      - Timbro   -> forma Retrograda-Inversa (RI) (la nota "salta" su un'altra
        traccia/strumento del brano, nello spirito di Kreuzspiel dove
        Stockhausen disperde una linea tra strumenti diversi)
    Le note vengono processate in ordine cronologico assoluto attraverso tutte
    le tracce (non per traccia separata), perche' nella musica puntillistica
    ogni punto e' indipendente dal contesto melodico originale. La struttura
    a piu' tracce/strumenti del brano di partenza viene pero' SEMPRE
    preservata (stesso numero di tracce, stessi nomi, stesso program_change):
    solo la destinazione di ciascuna nota puo' cambiare, non l'esistenza
    delle tracce — altrimenti DAW come Logic Pro perdono l'assegnazione
    degli strumenti e riproducono tutto con un patch di default.
    Se isolamento_punti=True, ogni nota viene accorciata e seguita da un
    micro-silenzio, per accentuare la natura di "punti" isolati nello spazio
    sonoro invece che di frasi legate.
    """
    row = derive_twelve_tone_row(original_midi)
    row_P = _row_prime(row)
    row_R = _row_retrograde(row)
    row_I = _row_inversion(row)
    row_RI = _row_retrograde_inversion(row)

    ticks_per_beat = original_midi.ticks_per_beat
    base_unit = max(1, ticks_per_beat // 8)
    DURATION_CLASSES = [base_unit * (k + 1) for k in range(12)]  # 12 durate crescenti, in ottavi di beat
    DYNAMICS_CLASSES = [int(v) for v in np.linspace(24, 127, 12)]  # ppp -> fff su 12 gradini

    num_tracks = len(original_midi.tracks)
    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]
    track_channels = [_get_track_default_channel(t) for t in original_midi.tracks]

    # Raccogli tutte le note come punti indipendenti, in ordine cronologico assoluto
    all_points = []
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        for nd in notes:
            all_points.append({
                'start': nd['start'],
                'orig_pitch': nd['pitch'],
                'orig_velocity': nd['velocity'],
                'track_idx': track_idx,
            })

    if not all_points:
        st.warning("Nessuna nota trovata nel brano. La tecnica Punktuelle non verra' applicata.")
        return original_midi, row

    all_points.sort(key=lambda x: x['start'])

    events_per_track = [[] for _ in range(num_tracks)]

    for i, point in enumerate(all_points):
        pitch_class = point['orig_pitch'] % 12
        octave = point['orig_pitch'] // 12
        new_pitch_class = row_P[pitch_class]
        new_pitch = max(0, min(127, octave * 12 + new_pitch_class))

        if serialize_duration:
            dur_idx = row_R[i % 12]
            duration = DURATION_CLASSES[dur_idx]
        else:
            duration = base_unit * 2

        if serialize_dynamics:
            dyn_idx = row_I[i % 12]
            velocity = DYNAMICS_CLASSES[dyn_idx]
        else:
            velocity = point['orig_velocity']

        if serialize_timbre and num_tracks > 1:
            target_track_idx = row_RI[i % 12] % num_tracks
        else:
            target_track_idx = point['track_idx']

        channel = track_channels[target_track_idx]

        note_start = point['start']
        if isolamento_punti:
            note_len = max(1, int(duration * 0.55))  # nota staccata: meno della meta' dello slot
        else:
            note_len = max(1, duration)

        events_per_track[target_track_idx].append({'msg': mido.Message('note_on', note=new_pitch, velocity=velocity, channel=channel, time=0), 'abs_time': note_start})
        events_per_track[target_track_idx].append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=channel, time=0), 'abs_time': note_start + note_len})

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx in range(num_tracks):
        new_track = mido.MidiTrack()
        new_track.name = track_names[track_idx]
        for _h in track_headers[track_idx]:
            new_track.append(_h)

        track_events = events_per_track[track_idx]
        track_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for event_data in track_events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            new_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(new_track)

    return new_midi, row


# --- Compositori: Pierre Boulez — Moltiplicazione d'Accordi (Blocs Sonores) ---
# Rif: Le Marteau sans maitre (1955), Structures II, Eclat. Tecnica di "pitch-class
# set multiplication" (Heinemann 1993; Koblyakov 1990): dato un insieme A e un
# insieme B di classi di altezza, si trasla A per ciascun intervallo generato da B
# rispetto a un pivot; l'unione delle trasposizioni forma un nuovo aggregato
# armonico. A differenza del pointillisme di Stockhausen (un punto = una nota),
# qui ogni evento del brano diventa un accordo/massa sonora verticale.

def boulez_multiply_sets(set_a, set_b, pivot=None):
    """
    Moltiplicazione semplice di due pitch-class set (Boulez/Heinemann):
    per ciascuna classe b in set_b, calcola l'intervallo rispetto al pivot e
    trasla set_a di quell'intervallo; l'unione (senza doppioni) e' il risultato.
    Es.: {0,4,7} x {0,2} con pivot=0 -> {0,4,7} unito a {2,6,9} = {0,2,4,6,7,9}.
    """
    if not set_a or not set_b:
        return []
    if pivot is None:
        pivot = set_b[0]
    result = set()
    for b in set_b:
        interval = (b - pivot) % 12
        for a in set_a:
            result.add((a + interval) % 12)
    return sorted(result)


def derive_boulez_sets(original_midi, set_size=4):
    """
    Deriva due pitch-class set dal brano stesso, nello spirito seriale in cui
    il materiale genera i propri operandi: insieme A = prime `set_size` classi
    distinte incontrate; insieme B = le `set_size` successive. La fila completa
    a 12 elementi (con fallback Costas) garantisce che A e B non si sovrappongano
    e siano sempre disponibili anche su brani poveri di materiale.
    """
    row = derive_twelve_tone_row(original_midi)
    set_a = row[:set_size]
    set_b = row[set_size:set_size * 2]
    return set_a, set_b


def midi_boulez_multiplication(original_midi, set_size=4, chord_density=0, register_spread=1):
    """
    Ogni nota del brano originale viene sostituita da un accordo costruito
    sull'aggregato risultante dalla moltiplicazione d'accordi di Boulez,
    trasformando la linea melodica in una sequenza di blocs sonores (masse
    armoniche verticali) invece che di punti isolati.
    chord_density=0 -> usa l'intero insieme moltiplicato; altrimenti limita
    l'accordo a `chord_density` classi scelte equidistanti nell'insieme.
    register_spread>1 -> distribuisce le voci dell'accordo su piu' ottave
    vicine invece di ammassarle tutte nella stessa ottava (evita cluster).
    """
    set_a, set_b = derive_boulez_sets(original_midi, set_size)
    pivot = set_b[0] if set_b else 0
    multiplied = boulez_multiply_sets(set_a, set_b, pivot)

    if not multiplied:
        st.warning("Materiale insufficiente per la moltiplicazione d'accordi. Restituito il MIDI originale.")
        return original_midi, (set_a, set_b, multiplied)

    if chord_density and 0 < chord_density < len(multiplied):
        step = len(multiplied) / chord_density
        chosen_idx = sorted(set(int(round(i * step)) % len(multiplied) for i in range(chord_density)))
        chord_pcs = [multiplied[i] for i in chosen_idx]
    else:
        chord_pcs = multiplied

    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for original_track in original_midi.tracks:
        _name = original_track.name if hasattr(original_track, 'name') else ''
        _header = _extract_instrument_header(original_track)
        notes = extract_notes(original_track, original_midi.ticks_per_beat)

        if not notes:
            new_midi.tracks.append(original_track)
            continue

        final_events = []
        for nd in notes:
            base_octave = nd['pitch'] // 12
            for offset_idx, pc in enumerate(chord_pcs):
                octave_shift = 0
                if register_spread > 1:
                    octave_shift = (offset_idx % register_spread) - (register_spread // 2)
                new_pitch = max(0, min(127, (base_octave + octave_shift) * 12 + pc))
                final_events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=nd['channel'], time=0), 'abs_time': nd['start']})
                final_events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=nd['channel'], time=0), 'abs_time': nd['end']})

        final_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        new_track = mido.MidiTrack()
        if _name:
            new_track.name = _name
        for _h in _header:
            new_track.append(_h)
        last_abs_time = 0
        for event_data in final_events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            new_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(new_track)

    return new_midi, (set_a, set_b, multiplied)


# --- Compositori: Iannis Xenakis — Musica Stocastica (Nuvole di Suoni) ---
# Rif: Pithoprakta (1955-56, distribuzione Gaussiana per i glissandi),
# Achorripsis (1956-57, processo di Poisson per la densita' degli eventi nel
# tempo, distribuzione esponenziale per gli intertempi), teoria dei crivelli
# (cribles: insiemi costruiti per unione di classi di resto modulari).
# A differenza di Stockhausen/Boulez (deterministici), qui i parametri sono
# governati da distribuzioni di probabilita' controllate — "il minimo di
# vincoli logici necessario" (Xenakis) — non dal random uniforme grezzo.

def generate_sieve(moduli_residues, universe=(0, 128)):
    """
    Crivello di Xenakis (crible): unione di classi di resto x = r (mod m).
    moduli_residues: lista di coppie (m, r). Es. [(3,0),(4,1)] = tutti gli
    interi congrui a 0 mod 3 UNITI a tutti quelli congrui a 1 mod 4.
    """
    lo, hi = universe
    sieve_set = set()
    for m, r in moduli_residues:
        if m <= 0:
            continue
        r = r % m
        for x in range(lo, hi):
            if (x - r) % m == 0:
                sieve_set.add(x)
    return sorted(sieve_set)


def parse_sieve_string(s):
    """Parsa una stringa tipo '3:0, 4:1, 7:3' in una lista di coppie (m, r)."""
    pairs = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk or ':' not in chunk:
            continue
        m_str, r_str = chunk.split(':', 1)
        try:
            m = int(m_str.strip())
            r = int(r_str.strip())
            if m > 0:
                pairs.append((m, r % m))
        except ValueError:
            continue
    return pairs


def _xenakis_snap_to_sieve(value, sieve):
    if not sieve:
        return value
    return min(sieve, key=lambda s: abs(s - value))


def midi_xenakis_stochastic(original_midi, sieve_moduli, mean_events_per_beat, pitch_center,
                             pitch_spread_semitones, duration_mean_beats, velocity_mean,
                             velocity_spread, seed=None):
    """
    Genera una "nuvola di suoni" stocastica (Pithoprakta/Achorripsis):
      - Tempi di attacco: processo di Poisson (intertempi con distribuzione
        esponenziale), tasso medio mean_events_per_beat eventi/beat.
      - Altezza: distribuzione Gaussiana attorno a pitch_center, quantizzata
        sul crivello (sieve) definito da sieve_moduli.
      - Durata: distribuzione esponenziale attorno a duration_mean_beats.
      - Dinamica: distribuzione Gaussiana attorno a velocity_mean.
    Copre l'intera durata del brano originale; le tracce originali restano
    intatte, la nuvola si aggiunge come nuova traccia.
    """
    rng = np.random.default_rng(seed)
    sieve = generate_sieve(sieve_moduli, universe=(0, 128))
    if not sieve:
        sieve = list(range(128))

    ticks_per_beat = original_midi.ticks_per_beat
    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)

    if total_ticks == 0:
        st.warning("Il brano originale non contiene eventi validi. La nuvola stocastica non verra' aggiunta.")
        return new_midi, sieve

    xenakis_track = mido.MidiTrack()
    xenakis_track.name = f"Xenakis Stochastic Cloud (sieve n={len(sieve)})"
    xenakis_track.append(mido.Message('program_change', program=0, channel=0, time=0))  # Acoustic Grand Piano di default

    total_beats = total_ticks / ticks_per_beat
    lam = max(0.05, mean_events_per_beat)

    events = []
    t = 0.0
    while t < total_beats:
        inter_arrival = rng.exponential(1.0 / lam)  # processo di Poisson
        t += inter_arrival
        if t >= total_beats:
            break

        raw_pitch = rng.normal(pitch_center, pitch_spread_semitones)
        pitch = int(round(_xenakis_snap_to_sieve(raw_pitch, sieve)))
        pitch = max(0, min(127, pitch))

        dur_beats = max(0.05, rng.exponential(duration_mean_beats))
        velocity = int(round(np.clip(rng.normal(velocity_mean, velocity_spread), 1, 127)))

        start_tick = int(round(t * ticks_per_beat))
        end_tick = int(round((t + dur_beats) * ticks_per_beat))

        events.append({'msg': mido.Message('note_on', note=pitch, velocity=velocity, channel=0, time=0), 'abs_time': start_tick})
        events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=0, time=0), 'abs_time': end_tick})

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for event_data in events:
        delta = max(0, event_data['abs_time'] - last_abs_time)
        xenakis_track.append(event_data['msg'].copy(time=delta))
        last_abs_time = event_data['abs_time']

    new_midi.tracks.append(xenakis_track)
    return new_midi, sieve


# --- Compositori: John Cage — Operazioni di Caso (I Ching / Music of Changes) ---
# Rif: Music of Changes (1951) — Cage costrui' delle "charts" (tabelle a 64
# caselle, una per esagramma) per altezza, durata e dinamica, e uso' il
# metodo classico delle tre monete dell'I Ching per scegliere, ad ogni
# passo, quale casella consultare. A differenza della statistica continua
# di Xenakis, qui il caso e' un'operazione discreta e procedurale — e il
# silenzio e' materiale musicale legittimo quanto il suono (stesso principio
# alla base di 4'33").

def _cage_toss_line(rng):
    """Simula il lancio di 3 monete (metodo classico dell'I Ching): testa=3, croce=2."""
    coins_total = sum(3 if rng.integers(0, 2) == 1 else 2 for _ in range(3))  # somma in {6,7,8,9}
    return 1 if coins_total in (7, 9) else 0  # linea intera (yang) = 1, spezzata (yin) = 0


def _cage_toss_hexagram(rng):
    """6 lanci di linea -> indice di esagramma 0..63 (metodo delle tre monete)."""
    idx = 0
    for _ in range(6):
        idx = (idx << 1) | _cage_toss_line(rng)
    return idx


def midi_cage_chance_operations(original_midi, silence_probability=0.15, duration_variety=True, seed=None):
    """
    Operazioni di caso in stile "Music of Changes": ogni nota del brano
    originale diventa un evento le cui proprieta' (altezza, durata,
    dinamica, presenza/assenza di suono) sono determinate da esagrammi
    indipendenti, generati con il metodo classico delle tre monete
    dell'I Ching — non pseudocasuale grezzo, ma la stessa procedura
    combinatoria (64 esiti equiprobabili) usata da Cage per costruire
    le proprie tabelle. Il silenzio ha silence_probability di sostituire
    ciascun evento: e' trattato come materiale, non come nota mancante.
    Ogni nota resta sulla propria traccia/strumento originale: la struttura
    a piu' tracce del brano di partenza (numero, nomi, program_change) e'
    sempre preservata, altrimenti DAW come Logic Pro perdono l'assegnazione
    degli strumenti e riproducono tutto con un patch di default.
    """
    rng = np.random.default_rng(seed)
    ticks_per_beat = original_midi.ticks_per_beat
    base_unit = max(1, ticks_per_beat // 4)

    num_tracks = len(original_midi.tracks)
    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]

    all_points = []
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        for nd in notes:
            nd['track_idx'] = track_idx
            all_points.append(nd)

    if not all_points:
        st.warning("Nessuna nota trovata. Le operazioni di caso non verranno applicate.")
        return original_midi, []

    all_points.sort(key=lambda x: x['start'])

    pitches = [p['pitch'] for p in all_points]
    pitch_lo, pitch_hi = min(pitches), max(pitches)
    if pitch_hi <= pitch_lo:
        pitch_hi = pitch_lo + 12

    # Tabelle a 64 caselle (una per ciascun esagramma), come nelle charts di Cage
    PITCH_CHART = [pitch_lo + (i % (pitch_hi - pitch_lo + 1)) for i in range(64)]
    DURATION_CHART = [base_unit * (1 + (i % 8)) for i in range(64)]
    DYNAMICS_CHART = [int(v) for v in np.linspace(20, 120, 64)]

    events_per_track = [[] for _ in range(num_tracks)]
    hexagram_log = []
    for point in all_points:
        hex_pitch = _cage_toss_hexagram(rng)
        hex_dur = _cage_toss_hexagram(rng) if duration_variety else hex_pitch
        hex_dyn = _cage_toss_hexagram(rng)
        hex_silence = _cage_toss_hexagram(rng)
        hexagram_log.append((hex_pitch, hex_dur, hex_dyn, hex_silence))

        is_silence = (hex_silence / 64.0) < silence_probability
        if is_silence:
            continue  # il silenzio e' l'esito legittimo: nessun evento sonoro

        pitch = max(0, min(127, PITCH_CHART[hex_pitch]))
        duration = DURATION_CHART[hex_dur]
        velocity = DYNAMICS_CHART[hex_dyn]

        target_track_idx = point['track_idx']
        note_start = point['start']
        events_per_track[target_track_idx].append({'msg': mido.Message('note_on', note=pitch, velocity=velocity, channel=point['channel'], time=0), 'abs_time': note_start})
        events_per_track[target_track_idx].append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=point['channel'], time=0), 'abs_time': note_start + max(1, duration)})

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx in range(num_tracks):
        new_track = mido.MidiTrack()
        new_track.name = track_names[track_idx]
        for _h in track_headers[track_idx]:
            new_track.append(_h)

        track_events = events_per_track[track_idx]
        track_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for event_data in track_events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            new_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(new_track)

    return new_midi, hexagram_log


# --- Compositori: Brian Eno — Musica Generativa (Cicli Asincroni) ---
# Rif: "Discreet Music" (1975), "Music for Airports" (1978) — sistemi
# costruiti da loop di nastro indipendenti, ciascuno contenente una singola
# nota, che ripetono al proprio periodo. Le lunghezze dei loop sono scelte
# "incommensurabili" tra loro (nell'album reale: ~23.5s, ~25.9s, ~29.2s...),
# cosi' che l'insieme impieghi un tempo lunghissimo (il MCM delle lunghezze)
# prima di ripetersi esattamente, pur restando gli elementi di base sempre
# gli stessi. "Non compongo la musica, compongo i sistemi che la generano"
# (Eno). Qui le lunghezze derivano da numeri primi distinti per garantire
# la stessa incommensurabilita' in modo deterministico e verificabile.

def _eno_prime_sequence(count, start_from=11):
    """Genera i primi `count` numeri primi a partire da start_from, per ottenere
    lunghezze di ciclo il piu' possibile incommensurabili tra loro."""
    primes = []
    candidate = start_from if start_from % 2 != 0 else start_from + 1
    while len(primes) < count:
        if _costas_is_prime(candidate):
            primes.append(candidate)
        candidate += 2
    return primes


def midi_eno_generative(original_midi, num_loops=6, min_loop_beats=8, max_loop_beats=32,
                         note_length_ratio=0.35, duration_multiplier=4, velocity_base=55,
                         seed=None):
    """
    Genera un sistema di loop asincroni in stile Music for Airports/Discreet
    Music: ogni loop ripete una singola nota (derivata dal materiale del
    brano originale) al proprio periodo indipendente. Le lunghezze dei loop
    sono multipli di numeri primi distinti, cosi' che l'intero sistema
    impieghi un tempo lunghissimo prima di ripetersi esattamente — la stessa
    logica dei nastri fisici di lunghezza diversa che Eno faceva girare in
    loop, sfasandosi continuamente l'uno rispetto all'altro.
    Le tracce originali restano intatte; il sistema generativo si aggiunge
    come nuove tracce indipendenti (una per loop), per poter regolare in DAW
    volume/timbro di ciascun loop separatamente.
    """
    rng = np.random.default_rng(seed)
    ticks_per_beat = original_midi.ticks_per_beat

    pitches_found = []
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note not in pitches_found:
                pitches_found.append(msg.note)
    if not pitches_found:
        st.warning("Nessuna nota trovata nel brano. Il sistema generativo non verra' aggiunto.")
        return original_midi, []
    pitches_found.sort()

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)
    if total_ticks == 0:
        total_ticks = ticks_per_beat * 4 * 8
    total_ticks = int(total_ticks * max(1, duration_multiplier))

    primes = _eno_prime_sequence(num_loops, start_from=11)
    min_ticks = int(min_loop_beats * ticks_per_beat)
    max_ticks = max(min_ticks + ticks_per_beat, int(max_loop_beats * ticks_per_beat))

    loops_info = []
    for i in range(num_loops):
        p = primes[i]
        scale = min_ticks + (p % max(1, (max_ticks - min_ticks)))
        loop_len_ticks = max(ticks_per_beat, scale)

        pitch = pitches_found[i % len(pitches_found)]
        note_len = max(1, int(loop_len_ticks * note_length_ratio))
        phase_offset = int(rng.uniform(0, loop_len_ticks))  # entrata sfalsata del loop

        loop_track = mido.MidiTrack()
        loop_track.name = f"Eno Loop {i + 1} (pitch={pitch}, ciclo={loop_len_ticks}t, primo={p})"
        loop_track.append(mido.Message('program_change', program=0, channel=0, time=0))

        events = []
        t = phase_offset
        while t < total_ticks:
            vel = int(np.clip(velocity_base + rng.normal(0, 6), 15, 90))
            events.append({'msg': mido.Message('note_on', note=pitch, velocity=vel, channel=0, time=0), 'abs_time': t})
            events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=0, time=0), 'abs_time': t + note_len})
            t += loop_len_ticks

        events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for event_data in events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            loop_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(loop_track)
        loops_info.append((pitch, loop_len_ticks, p))

    return new_midi, loops_info


# --- Compositori: Arvo Pärt — Tintinnabuli ---
# Rif: tecnica inventata nel 1976. Voce M (melodica): si muove per gradi
# congiunti in una scala diatonica. Voce T (tintinnabuli): vincolata alle
# sole note della triade centrale, scelta secondo una posizione e direzione
# fisse rispetto a ciascuna nota della voce M (es. "la prima nota della
# triade sopra la voce M").

def _part_snap_to_scale(pitch, scale_pcs):
    """Arrotonda un pitch alla nota di scala piu' vicina (spostamento minimo)."""
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    diff_up = min(((s - pc) % 12) for s in scale_pcs)
    diff_down = min(((pc - s) % 12) for s in scale_pcs)
    return pitch + diff_up if diff_up <= diff_down else pitch - diff_down


def _part_nearest_triad_tone(pitch, triad_pcs, direction, position):
    """Trova l'N-esima nota della triade sopra/sotto un dato pitch (posizione 1,2,3)."""
    candidates = []
    for octv in range(-3, 4):
        for pc in triad_pcs:
            candidates.append(octv * 12 + pc)
    if direction == "sopra":
        options = sorted(c for c in candidates if c > pitch)
    else:
        options = sorted((c for c in candidates if c < pitch), reverse=True)
    if not options:
        return pitch
    idx = min(position - 1, len(options) - 1)
    return options[idx]


def midi_part_tintinnabuli(original_midi, root_note_name="A", mode="Minore Naturale",
                            direction="sotto", position=1, t_voice_program=14):
    """
    Applica la tecnica tintinnabuli di Arvo Part. Per ciascuna traccia con
    note: la voce M e' la melodia originale quantizzata sulla scala scelta
    (movimento per gradi congiunti); una nuova voce T viene aggiunta come
    traccia parallela, restretta alle sole note della triade tonale, scelta
    secondo la posizione/direzione indicate rispetto a ciascuna nota M.
    Le tracce originali (M-voice) vengono quantizzate sul posto; le nuove
    T-voice si aggiungono come tracce indipendenti (patch di default:
    Tubular Bells, GM program 14 — coerente con l'origine del nome
    "tintinnabuli", dal latino per le campanelle).
    """
    root_pc = get_key_offset(root_note_name)
    scale_intervals = get_scale_notes(mode)
    scale_pcs_ordered = [(root_pc + iv) % 12 for iv in scale_intervals]  # ordine dei gradi dalla tonica
    scale_pcs = sorted(set(scale_pcs_ordered))  # per la quantizzazione (l'ordine non conta)
    triad_pcs = [scale_pcs_ordered[0], scale_pcs_ordered[2 % len(scale_pcs_ordered)], scale_pcs_ordered[4 % len(scale_pcs_ordered)]]

    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    t_voice_tracks = []

    for original_track in original_midi.tracks:
        notes = extract_notes(original_track, original_midi.ticks_per_beat)
        if not notes:
            new_midi.tracks.append(original_track)
            continue

        _header = _extract_instrument_header(original_track)
        _name = original_track.name if hasattr(original_track, 'name') else ''

        # --- Voce M: melodia originale quantizzata sulla scala ---
        m_track = mido.MidiTrack()
        if _name:
            m_track.name = f"{_name} (M-voice)"
        for _h in _header:
            m_track.append(_h)
        m_events = []
        for nd in notes:
            new_pitch = max(0, min(127, _part_snap_to_scale(nd['pitch'], scale_pcs)))
            m_events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=nd['channel'], time=0), 'abs_time': nd['start']})
            m_events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=nd['channel'], time=0), 'abs_time': nd['end']})
        m_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for ev in m_events:
            delta = max(0, ev['abs_time'] - last_abs_time)
            m_track.append(ev['msg'].copy(time=delta))
            last_abs_time = ev['abs_time']
        new_midi.tracks.append(m_track)

        # --- Voce T: triade tonale, stessa ritmica della voce M ---
        t_track = mido.MidiTrack()
        t_track.name = f"{_name} (T-voice)" if _name else "T-voice"
        t_track.append(mido.Message('program_change', program=t_voice_program, channel=1, time=0))
        t_events = []
        for nd in notes:
            m_pitch = max(0, min(127, _part_snap_to_scale(nd['pitch'], scale_pcs)))
            t_pitch = max(0, min(127, _part_nearest_triad_tone(m_pitch, triad_pcs, direction, position)))
            t_events.append({'msg': mido.Message('note_on', note=t_pitch, velocity=max(30, nd['velocity'] - 15), channel=1, time=0), 'abs_time': nd['start']})
            t_events.append({'msg': mido.Message('note_off', note=t_pitch, velocity=0, channel=1, time=0), 'abs_time': nd['end']})
        t_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for ev in t_events:
            delta = max(0, ev['abs_time'] - last_abs_time)
            t_track.append(ev['msg'].copy(time=delta))
            last_abs_time = ev['abs_time']
        t_voice_tracks.append(t_track)

    for t in t_voice_tracks:
        new_midi.tracks.append(t)

    return new_midi, (root_note_name, mode, triad_pcs)


# --- Compositori: Olivier Messiaen — Modi a Trasposizione Limitata + Ritmi Non Retrogradabili ---
# Rif: "Technique de mon langage musical" (1944). I 7 modi di Messiaen sono
# scale simmetriche che, dopo un numero limitato di trasposizioni, tornano a
# contenere le stesse note (es. Modo 2 = scala ottatonica, 3 trasposizioni).
# I ritmi non retrogradabili sono sequenze di durate palindrome: lette in
# avanti o all'indietro, l'ordine resta identico.

MESSIAEN_MODES = {
    "Modo 1 (esatonale)": [0, 2, 4, 6, 8, 10],
    "Modo 2 (ottatonico)": [0, 1, 3, 4, 6, 7, 9, 10],
    "Modo 3": [0, 2, 3, 4, 6, 7, 8, 10, 11],
    "Modo 4": [0, 1, 2, 5, 6, 7, 8, 11],
    "Modo 5": [0, 1, 5, 6, 7, 11],
    "Modo 6": [0, 2, 4, 5, 6, 8, 10, 11],
    "Modo 7": [0, 1, 2, 3, 5, 6, 7, 8, 9, 11],
}


def _messiaen_snap_to_mode(pitch, mode_pcs):
    pc = pitch % 12
    if pc in mode_pcs:
        return pitch
    diff_up = min(((m - pc) % 12) for m in mode_pcs)
    diff_down = min(((pc - m) % 12) for m in mode_pcs)
    return pitch + diff_up if diff_up <= diff_down else pitch - diff_down


def _messiaen_make_palindrome(durations):
    """Costruisce una sequenza di durate palindroma (ritmo non retrogradabile)
    a partire da una lista di durate: [d1,d2,...] -> [d1,d2,...,d2,d1]
    (con valore centrale unico se la lunghezza e' dispari)."""
    if not durations:
        return durations
    half = list(durations)
    return half + half[-2::-1] if len(half) > 1 else half + half


def midi_messiaen_modal(original_midi, mode_name="Modo 2 (ottatonico)", root_note_name="C",
                         palindrome_block_size=4):
    """
    Applica il linguaggio modale di Messiaen: ogni nota viene quantizzata sul
    modo a trasposizione limitata scelto (invece che su una scala diatonica
    convenzionale); le durate delle note vengono raggruppate in blocchi e
    trasformate in sequenze palindrome (ritmi non retrogradabili) — lo stesso
    ritmo letto in avanti o indietro resta identico.
    """
    root_pc = get_key_offset(root_note_name)
    mode_intervals = MESSIAEN_MODES.get(mode_name, MESSIAEN_MODES["Modo 2 (ottatonico)"])
    mode_pcs = sorted(set((root_pc + iv) % 12 for iv in mode_intervals))

    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for original_track in original_midi.tracks:
        notes = extract_notes(original_track, original_midi.ticks_per_beat)
        if not notes:
            new_midi.tracks.append(original_track)
            continue

        _header = _extract_instrument_header(original_track)
        _name = original_track.name if hasattr(original_track, 'name') else ''
        notes_sorted = sorted(notes, key=lambda x: x['start'])

        # Durate palindrome, applicate a blocchi
        original_durations = [max(1, nd['end'] - nd['start']) for nd in notes_sorted]
        new_durations = []
        for block_start in range(0, len(original_durations), palindrome_block_size):
            block = original_durations[block_start: block_start + palindrome_block_size]
            new_durations.extend(_messiaen_make_palindrome(block)[:len(block)])

        new_track = mido.MidiTrack()
        if _name:
            new_track.name = _name
        for _h in _header:
            new_track.append(_h)

        events = []
        for i, nd in enumerate(notes_sorted):
            new_pitch = max(0, min(127, _messiaen_snap_to_mode(nd['pitch'], mode_pcs)))
            duration = new_durations[i] if i < len(new_durations) else max(1, nd['end'] - nd['start'])
            events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=nd['channel'], time=0), 'abs_time': nd['start']})
            events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=nd['channel'], time=0), 'abs_time': nd['start'] + duration})

        events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for ev in events:
            delta = max(0, ev['abs_time'] - last_abs_time)
            new_track.append(ev['msg'].copy(time=delta))
            last_abs_time = ev['abs_time']

        new_midi.tracks.append(new_track)

    return new_midi, mode_pcs


# --- Compositori: Steve Reich — Phasing ---
# Rif: "Piano Phase" (1967), "Clapping Music" (1972). Una cellula ritmico-
# melodica breve viene suonata simultaneamente da due voci identiche; la
# seconda voce accumula uno scarto di fase crescente e discreto rispetto
# alla prima, ciclo dopo ciclo, finche' non ha percorso un'intera rotazione
# e torna in unisono — passando per tutte le relazioni di fase intermedie.

def midi_reich_phasing(original_midi, pattern_length=8, num_cycles=None, shift_fraction=1.0):
    """
    Deriva una cellula di `pattern_length` note dalla prima traccia con note
    del brano, poi genera due voci che la ripetono simultaneamente: la Voce A
    resta fissa, la Voce B accumula ad ogni ciclo uno scarto di fase pari a
    shift_fraction * (durata di un passo della cellula), fino a completare
    una rotazione intera dopo `num_cycles` ripetizioni (default = pattern_length,
    cosi' che l'ultimo ciclo coincida di nuovo con la Voce A).
    """
    ticks_per_beat = original_midi.ticks_per_beat
    seed_notes = None
    for track in original_midi.tracks:
        notes = extract_notes(track, ticks_per_beat)
        if notes:
            seed_notes = sorted(notes, key=lambda x: x['start'])[:pattern_length]
            break

    if not seed_notes:
        st.warning("Nessuna nota trovata nel brano. Il phasing non verra' applicato.")
        return original_midi, 0

    step_ticks = max(1, ticks_per_beat // 4)
    cell = [(nd['pitch'], nd['velocity'], nd['channel']) for nd in seed_notes]
    n = len(cell)
    if num_cycles is None:
        num_cycles = n

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    def build_voice(name, phase_offset_ticks_fn):
        voice_track = mido.MidiTrack()
        voice_track.name = name
        voice_track.append(mido.Message('program_change', program=0, channel=0, time=0))
        events = []
        for cycle in range(num_cycles):
            offset = phase_offset_ticks_fn(cycle)
            for idx, (pitch, vel, ch) in enumerate(cell):
                t_start = cycle * n * step_ticks + idx * step_ticks + offset
                events.append({'msg': mido.Message('note_on', note=pitch, velocity=vel, channel=0, time=0), 'abs_time': t_start})
                events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=0, time=0), 'abs_time': t_start + step_ticks})
        events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
        last_abs_time = 0
        for ev in events:
            delta = max(0, ev['abs_time'] - last_abs_time)
            voice_track.append(ev['msg'].copy(time=delta))
            last_abs_time = ev['abs_time']
        return voice_track

    voice_a = build_voice("Reich Phasing — Voce A (fissa)", lambda c: 0)
    shift_step = max(1, int(step_ticks * shift_fraction))
    voice_b = build_voice("Reich Phasing — Voce B (sfasata)", lambda c: c * shift_step)

    new_midi.tracks.append(voice_a)
    new_midi.tracks.append(voice_b)

    return new_midi, num_cycles


# --- Compositori: Philip Glass — Processo Additivo ---
# Rif: "Two Pages" (1968), "Music in Contrary Motion" (1969). Una breve
# cellula viene sottoposta a un processo additivo: si parte da 1 sola nota,
# ripetuta; poi si aggiunge una nota alla volta fino a raggiungere l'intera
# cellula, ripetendo ogni stadio intermedio piu' volte prima di procedere.

def midi_glass_additive(original_midi, cell_length=6, repeats_per_stage=3, contract_after=True):
    """
    Deriva una cellula di `cell_length` note dalla prima traccia con note del
    brano, poi costruisce una nuova traccia che la espone tramite processo
    additivo: stadio 1 = prima nota ripetuta `repeats_per_stage` volte,
    stadio 2 = prime 2 note ripetute, ..., fino alla cellula completa.
    Se contract_after=True, il processo si inverte simmetricamente
    (contrazione) dopo aver raggiunto la lunghezza massima.
    """
    ticks_per_beat = original_midi.ticks_per_beat
    seed_notes = None
    for track in original_midi.tracks:
        notes = extract_notes(track, ticks_per_beat)
        if notes:
            seed_notes = sorted(notes, key=lambda x: x['start'])[:cell_length]
            break

    if not seed_notes:
        st.warning("Nessuna nota trovata nel brano. Il processo additivo non verra' applicato.")
        return original_midi, 0

    step_ticks = max(1, ticks_per_beat // 4)
    cell = [(nd['pitch'], nd['velocity'], nd['channel']) for nd in seed_notes]
    n = len(cell)

    stage_lengths = list(range(1, n + 1))
    if contract_after:
        stage_lengths += list(range(n - 1, 0, -1))

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    glass_track = mido.MidiTrack()
    glass_track.name = f"Glass Additive Process (cella={n} note)"
    glass_track.append(mido.Message('program_change', program=0, channel=0, time=0))

    events = []
    t = 0
    for stage_len in stage_lengths:
        for _rep in range(repeats_per_stage):
            for idx in range(stage_len):
                pitch, vel, ch = cell[idx]
                events.append({'msg': mido.Message('note_on', note=pitch, velocity=vel, channel=0, time=0), 'abs_time': t})
                events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=0, time=0), 'abs_time': t + step_ticks})
                t += step_ticks

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for ev in events:
        delta = max(0, ev['abs_time'] - last_abs_time)
        glass_track.append(ev['msg'].copy(time=delta))
        last_abs_time = ev['abs_time']

    new_midi.tracks.append(glass_track)
    return new_midi, len(stage_lengths)


# --- Compositori: Johann Sebastian Bach — Canone Rigoroso ---
# Rif: tecnica contrappuntistica sistematizzata nell'Offerta Musicale e
# nell'Arte della Fuga. Un "comes" (voce che segue) viene derivato dal "dux"
# (voce guida, la melodia originale) secondo una regola fissa: retrogrado
# (canone cancrizans, "a specchio nel tempo"), inversione (specchio negli
# intervalli attorno a un asse), o trasposizione a un dato intervallo con
# ritardo (canone imitativo).

def midi_bach_canon(original_midi, canon_type="Cancrizans (Retrogrado)", interval_semitones=7,
                     delay_beats=2, axis_pitch=None):
    """
    Deriva un "comes" dalla prima traccia con note del brano (il "dux") e lo
    aggiunge come voce canonica indipendente, secondo la regola scelta:
      - Cancrizans: il comes e' l'esatto retrogrado del dux (stesse note,
        ordine invertito), eseguito simultaneamente — il "canone del granchio".
      - Per Inversione: il comes e' lo specchio del dux attorno a un pitch-asse
        (ogni intervallo dall'asse viene invertito).
      - All'Intervallo (imitativo): il comes e' il dux trasposto di
        interval_semitones ed eseguito con un ritardo di delay_beats.
    """
    ticks_per_beat = original_midi.ticks_per_beat
    dux_notes = None
    for track in original_midi.tracks:
        notes = extract_notes(track, ticks_per_beat)
        if notes:
            dux_notes = sorted(notes, key=lambda x: x['start'])
            break

    if not dux_notes:
        st.warning("Nessuna nota trovata nel brano. Il canone non verra' applicato.")
        return original_midi, canon_type

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    comes_track = mido.MidiTrack()
    comes_track.name = f"Bach Canon — Comes ({canon_type})"
    comes_track.append(mido.Message('program_change', program=0, channel=1, time=0))

    events = []
    total_span = max(nd['end'] for nd in dux_notes) - dux_notes[0]['start']

    if canon_type == "Cancrizans (Retrogrado)":
        for nd in dux_notes:
            rel_start = nd['start'] - dux_notes[0]['start']
            rel_end = nd['end'] - dux_notes[0]['start']
            new_start = total_span - rel_end
            new_end = total_span - rel_start
            events.append({'msg': mido.Message('note_on', note=nd['pitch'], velocity=nd['velocity'], channel=1, time=0), 'abs_time': dux_notes[0]['start'] + new_start})
            events.append({'msg': mido.Message('note_off', note=nd['pitch'], velocity=0, channel=1, time=0), 'abs_time': dux_notes[0]['start'] + new_end})

    elif canon_type == "Per Inversione":
        axis = axis_pitch if axis_pitch is not None else dux_notes[0]['pitch']
        for nd in dux_notes:
            new_pitch = max(0, min(127, 2 * axis - nd['pitch']))
            events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=1, time=0), 'abs_time': nd['start']})
            events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=1, time=0), 'abs_time': nd['end']})

    else:  # "All'Intervallo (imitativo)"
        delay_ticks = int(delay_beats * ticks_per_beat)
        for nd in dux_notes:
            new_pitch = max(0, min(127, nd['pitch'] + interval_semitones))
            events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=1, time=0), 'abs_time': nd['start'] + delay_ticks})
            events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=1, time=0), 'abs_time': nd['end'] + delay_ticks})

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for ev in events:
        delta = max(0, ev['abs_time'] - last_abs_time)
        comes_track.append(ev['msg'].copy(time=delta))
        last_abs_time = ev['abs_time']

    new_midi.tracks.append(comes_track)
    return new_midi, canon_type


# --- Stile: Geometria Frattale (ispirato a Wallin/Sharp/Posadas) ---
# NON e' la tecnica esatta di alcun compositore specifico: quei metodi non
# sono documentati pubblicamente in dettaglio riproducibile (vedi nota nella
# UI). Questo modulo usa due algoritmi frattali STANDARD e verificabili:
#   - Insieme di Cantor: genera lo scheletro RITMICO (rimozione ricorsiva del
#     terzo centrale di un intervallo -> pattern gerarchico presenza/assenza,
#     auto-simile a scale temporali diverse).
#   - IFS (Iterated Function System, tipo triangolo di Sierpinski): genera il
#     contorno MELODICO (3 trasformazioni affini verso 3 vertici, iterate N
#     volte -> profilo di altezza auto-simile).

def _cantor_intervals(start, end, depth, min_width):
    """Ricorsione dell'insieme di Cantor: ritorna la lista dei sotto-intervalli
    sopravvissuti (terzo centrale rimosso ad ogni livello) fino a `depth`
    livelli o larghezza minima `min_width`."""
    width = end - start
    if depth <= 0 or width < min_width:
        return [(start, end)]
    third = width / 3.0
    left = _cantor_intervals(start, start + third, depth - 1, min_width)
    right = _cantor_intervals(end - third, end, depth - 1, min_width)
    return left + right


def _ifs_sierpinski_points(num_points, seed=None):
    """Genera `num_points` punti (x,y) in [0,1]x[0,1] tramite il sistema di
    funzioni iterate del triangolo di Sierpinski: ad ogni passo il punto
    corrente viene spostato a meta' strada verso uno dei 3 vertici del
    triangolo, scelto a caso — il classico "chaos game"."""
    rng = np.random.default_rng(seed)
    vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    x, y = 0.5, 0.5
    points = []
    for i in range(num_points):
        vx, vy = vertices[rng.integers(0, 3)]
        x = (x + vx) / 2.0
        y = (y + vy) / 2.0
        if i >= 5:  # scarta i primi punti transitori, non ancora sull'attrattore
            points.append((x, y))
    return points


def midi_fractal_geometry(original_midi, cantor_depth=4, min_width_ticks=None,
                           pitch_range_semitones=24, seed=None):
    """
    Genera una nuova traccia il cui ritmo segue lo scheletro dell'insieme di
    Cantor (istanti ricavati dai sotto-intervalli sopravvissuti dopo
    `cantor_depth` rimozioni ricorsive del terzo centrale) e la cui melodia
    segue il profilo d'altezza generato da un IFS stile Sierpinski (chaos
    game), mappato sul centro/estensione di pitch del brano originale.
    Le tracce originali restano intatte; il sistema frattale si aggiunge
    come nuova traccia indipendente.
    """
    rng_seed = seed
    ticks_per_beat = original_midi.ticks_per_beat

    pitches_found = []
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches_found.append(msg.note)
    if not pitches_found:
        st.warning("Nessuna nota trovata nel brano. Il modulo frattale non verra' aggiunto.")
        return original_midi, (0, 0)
    pitch_center = int(np.mean(pitches_found))

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)
    if total_ticks == 0:
        total_ticks = ticks_per_beat * 4 * 8

    if min_width_ticks is None:
        min_width_ticks = max(1, ticks_per_beat // 8)

    # --- Ritmo: scheletro dell'insieme di Cantor ---
    cantor_segments = _cantor_intervals(0, total_ticks, cantor_depth, min_width_ticks)
    cantor_segments.sort(key=lambda seg: seg[0])

    # --- Melodia: profilo IFS (chaos game, triangolo di Sierpinski) ---
    ifs_points = _ifs_sierpinski_points(len(cantor_segments), seed=rng_seed)

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    fractal_track = mido.MidiTrack()
    fractal_track.name = f"Geometria Frattale (Cantor depth={cantor_depth} + IFS Sierpinski)"
    fractal_track.append(mido.Message('program_change', program=0, channel=0, time=0))

    events = []
    for i, (seg_start, seg_end) in enumerate(cantor_segments):
        y = ifs_points[i][1] if i < len(ifs_points) else 0.5
        pitch = int(round(pitch_center + (y - 0.5) * pitch_range_semitones))
        pitch = max(0, min(127, pitch))
        note_len = max(1, int((seg_end - seg_start) * 0.8))
        vel = int(np.clip(60 + (y - 0.5) * 40, 20, 110))
        start_tick = int(seg_start)
        events.append({'msg': mido.Message('note_on', note=pitch, velocity=vel, channel=0, time=0), 'abs_time': start_tick})
        events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=0, time=0), 'abs_time': start_tick + note_len})

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for ev in events:
        delta = max(0, ev['abs_time'] - last_abs_time)
        fractal_track.append(ev['msg'].copy(time=delta))
        last_abs_time = ev['abs_time']

    new_midi.tracks.append(fractal_track)
    return new_midi, (len(cantor_segments), pitch_center)


# --- Funzioni di Decomposizione ---

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    """
    Rimodella le note MIDI in base a una scala, tonalità e randomizzazione di pitch/velocity.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    
    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name)

    for i, track in enumerate(original_midi.tracks):
        new_track = mido.MidiTrack()
        if hasattr(track, 'name') and track.name:
            new_track.name = track.name
        for msg in track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                original_note = msg.note
                shifted_note = original_note
                if pitch_shift_range > 0:
                    shifted_note += random.randint(-pitch_shift_range, pitch_shift_range)
                shifted_note = max(0, min(127, shifted_note))

                note_in_octave = (shifted_note - key_offset) % 12
                if note_in_octave < 0:
                    note_in_octave += 12 
                
                closest_scale_interval = min(target_scale_intervals, key=lambda x: abs(note_in_octave - x))
                
                octave = (shifted_note - key_offset) // 12 
                new_note_pitch = octave * 12 + closest_scale_interval + key_offset
                new_note_pitch = max(0, min(127, new_note_pitch))
                
                new_velocity = msg.velocity
                if msg.type == 'note_on' and velocity_randomization > 0:
                    new_velocity_float = float(new_velocity) * (1 + random.uniform(-velocity_randomization/100, velocity_randomization/100))
                    new_velocity = int(round(new_velocity_float))
                    new_velocity = max(1, min(127, new_velocity))

                new_msg = msg.copy(note=new_note_pitch, velocity=new_velocity)
                new_track.append(new_msg)
            else:
                new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi

def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style):
    """Riorganizza le frasi MIDI."""
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats

    if ticks_per_phrase == 0:
        st.warning("La lunghezza della frase è zero. Nessuna riorganizzazione applicata.")
        return original_midi

    for original_track in original_midi.tracks:
        phrases = []
        current_phrase_events = []
        _track_name = original_track.name if hasattr(original_track, 'name') else ''
        _header = _extract_instrument_header(original_track)
        current_phrase_start_tick = 0

        events_with_abs_time = []
        time_since_last_event = 0
        for msg in original_track:
            time_since_last_event += msg.time
            if msg.type == 'program_change' or (msg.type == 'control_change' and msg.control in (0, 32)):
                continue  # gia' catturati in _header, verranno fissati all'inizio
            events_with_abs_time.append({'msg': msg, 'abs_time': time_since_last_event})

        for event_data in events_with_abs_time:
            msg = event_data['msg']
            abs_time = event_data['abs_time']
            while abs_time >= current_phrase_start_tick + ticks_per_phrase:
                if current_phrase_events:
                    phrases.append(current_phrase_events)
                current_phrase_events = []
                current_phrase_start_tick += ticks_per_phrase
            current_phrase_events.append(msg)
        if current_phrase_events:
            phrases.append(current_phrase_events)

        if not phrases:
            _empty_track = mido.MidiTrack()
            if _track_name:
                _empty_track.name = _track_name
            for _h in _header:
                _empty_track.append(_h)
            new_midi.tracks.append(_empty_track)
            continue

        reorganized_phrases = []
        if reassembly_style == "Casuale":
            reorganized_phrases = list(phrases)
            random.shuffle(reorganized_phrases)
        elif reassembly_style == "Inversione":
            reorganized_phrases = list(reversed(phrases))
        elif reassembly_style == "Ciclico A-B-A":
            if len(phrases) >= 3:
                a_phrase, b_phrase, c_phrase = phrases[0], phrases[1], (phrases[2] if len(phrases) > 2 else phrases[1])
                num_repetitions = max(1, len(phrases) // 3)
                for _ in range(num_repetitions):
                    reorganized_phrases.extend([a_phrase, b_phrase, a_phrase, c_phrase])
            else:
                st.warning(f"Troppo poche frasi ({len(phrases)}) per lo stile 'Ciclico A-B-A'. Verrà usata la riorganizzazione casuale.")
                reorganized_phrases = list(phrases)
                random.shuffle(reorganized_phrases)
        elif reassembly_style == "Dal Più Corto al Più Lungo":
            def get_phrase_duration_in_ticks(phrase_events_list):
                if not phrase_events_list: return 0
                return sum(msg.time for msg in phrase_events_list)
            reorganized_phrases = sorted(phrases, key=get_phrase_duration_in_ticks)
        else:
            reorganized_phrases = list(phrases)

        new_track = mido.MidiTrack()
        if _track_name:
            new_track.name = _track_name
        for _h in _header:
            new_track.append(_h)
        flat_events_for_reconstruction = []
        absolute_time_in_reorganized_seq = 0

        for phrase_block in reorganized_phrases:
            # Traccia note aperte in questa frase — chiudi quelle senza note_off
            open_notes = {}  # (pitch, channel) -> abs_time di apertura
            phrase_abs = absolute_time_in_reorganized_seq

            for msg_in_phrase in phrase_block:
                phrase_abs += msg_in_phrase.time
                if msg_in_phrase.type == 'note_on' and msg_in_phrase.velocity > 0:
                    open_notes[(msg_in_phrase.note, msg_in_phrase.channel)] = phrase_abs
                    flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': phrase_abs})
                elif msg_in_phrase.type == 'note_off' or (msg_in_phrase.type == 'note_on' and msg_in_phrase.velocity == 0):
                    key = (msg_in_phrase.note, msg_in_phrase.channel)
                    if key in open_notes:
                        open_notes.pop(key, None)
                        flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': phrase_abs})
                    # else: note_off "orfano" — il note_on apparteneva a una frase
                    # precedente (gia' chiusa sinteticamente al suo confine, vedi sotto).
                    # Scartato per evitare un doppio note_off sulla stessa nota.
                else:
                    flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': phrase_abs})

            # Chiudi note rimaste aperte alla fine della frase
            phrase_end = phrase_abs
            for (pitch, ch) in list(open_notes.keys()):
                flat_events_for_reconstruction.append({
                    'msg': mido.Message('note_off', note=pitch, velocity=0, channel=ch, time=0),
                    'abs_time': phrase_end
                })

            absolute_time_in_reorganized_seq = phrase_end

        # Ordina: note_off prima di note_on allo stesso tick
        flat_events_for_reconstruction.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))

        last_abs_time = 0
        for event_data in flat_events_for_reconstruction:
            msg = event_data['msg']
            abs_time = event_data['abs_time']
            delta_time = max(0, abs_time - last_abs_time)
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time = abs_time

        new_midi.tracks.append(new_track)
    return new_midi

def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    """
    Modifica il timing e la durata delle note MIDI.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_subdivision = original_midi.ticks_per_beat / 4
    if ticks_per_subdivision == 0:
        st.warning("Ticks per beat è zero o troppo basso. Restituito il MIDI originale.")
        return original_midi

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(original_track, 'name') and original_track.name:
            new_track.name = original_track.name
        events_with_abs_time = []
        current_abs_time_stretched = 0

        for msg in original_track:
            stretched_delta_time = int(round(msg.time * stretch_factor))
            current_abs_time_stretched += stretched_delta_time
            events_with_abs_time.append({'msg': msg.copy(), 'abs_time_mod': current_abs_time_stretched})

        if quantization_strength > 0:
            for event_data in events_with_abs_time:
                msg = event_data['msg']
                abs_time_before_quant = event_data['abs_time_mod']

                if msg.type in ['note_on', 'note_off']:
                    snapped_abs_time = round(abs_time_before_quant / ticks_per_subdivision) * ticks_per_subdivision
                    if swing_amount > 0 and int(round((snapped_abs_time % original_midi.ticks_per_beat) / ticks_per_subdivision)) % 2 == 1:
                        swing_shift_ticks = (ticks_per_subdivision / 2) * (swing_amount / 100.0)
                        snapped_abs_time += swing_shift_ticks
                    
                    quant_factor = quantization_strength / 100.0
                    event_data['abs_time_mod'] = int(round(abs_time_before_quant * (1 - quant_factor) + snapped_abs_time * quant_factor))
                    event_data['abs_time_mod'] = max(0, event_data['abs_time_mod'])
        
        events_with_abs_time.sort(key=lambda x: x['abs_time_mod'])

        last_abs_time_mod = 0
        for event_data in events_with_abs_time:
            msg = event_data['msg']
            abs_time_mod = event_data['abs_time_mod']
            delta_time = max(0, abs_time_mod - last_abs_time_mod)
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time_mod = abs_time_mod

        new_midi.tracks.append(new_track)
    return new_midi

def midi_density_transformer(original_midi, add_note_probability, remove_note_probability, polyphony_mode):
    """
    Aggiunge o rimuove note per alterare la densita' MIDI.
    Fix: tracce senza note vengono passate intatte.
    Fix: note aggiunte hanno durata esplicita uguale alla nota originale.
    Fix: note_off sempre dopo note_on — abs_time note_off = start + durata originale.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        _dens_name = original_track.name if hasattr(original_track, 'name') else ''
        _dens_header = _extract_instrument_header(original_track)
        notes = extract_notes(original_track, original_midi.ticks_per_beat)

        # Se la traccia non ha note (metadati, controller, ecc.) — passa intatta
        if not notes:
            new_midi.tracks.append(original_track)
            continue

        modified_notes = [note for note in notes if random.randint(0, 100) >= remove_note_probability]

        # Durata minima garantita: almeno 1 tick
        def safe_duration(note):
            return max(1, note['end'] - note['start'])

        final_events = []
        track_end_time = max(n['end'] for n in notes)

        if polyphony_mode == "Droni" and add_note_probability > 0 and random.randint(0, 100) < add_note_probability:
            drone_pitch = 36
            drone_velocity = 64
            final_events.append({'msg': mido.Message('note_on', note=drone_pitch, velocity=drone_velocity, channel=0, time=0), 'abs_time': 0})
            final_events.append({'msg': mido.Message('note_off', note=drone_pitch, velocity=0, channel=0, time=0), 'abs_time': track_end_time + original_midi.ticks_per_beat * 4})

        for note_data in modified_notes:
            dur = safe_duration(note_data)
            note_start = note_data['start']
            note_end   = note_start + dur  # durata esplicita, non dipende da note_off originale

            final_events.append({'msg': mido.Message('note_on',  note=note_data['pitch'], velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_start})
            final_events.append({'msg': mido.Message('note_off', note=note_data['pitch'], velocity=0,                    channel=note_data['channel'], time=0), 'abs_time': note_end})

            if random.randint(0, 100) < add_note_probability:
                if polyphony_mode == "Riempi Accordo (Triadi)":
                    intervals = [4, 7]
                elif polyphony_mode == "Aggiungi Contro-Melodia":
                    intervals = [random.choice([-5, -3, -2, 2, 3, 5])]
                else:
                    intervals = []

                for interval in intervals:
                    new_pitch = note_data['pitch'] + interval
                    if 0 <= new_pitch <= 127:
                        # Nota aggiunta: stessa durata della nota originale, note_off esplicito
                        final_events.append({'msg': mido.Message('note_on',  note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_start})
                        final_events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0,                    channel=note_data['channel'], time=0), 'abs_time': note_end})

        # Ordina per abs_time, note_off prima di note_on allo stesso tick (evita sovrapposizioni)
        final_events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))

        new_track = mido.MidiTrack()
        if _dens_name:
            new_track.name = _dens_name
        for _h in _dens_header:
            new_track.append(_h)
        last_abs_time = 0
        for event_data in final_events:
            msg      = event_data['msg']
            abs_time = event_data['abs_time']
            delta    = max(0, abs_time - last_abs_time)
            new_track.append(msg.copy(time=delta))
            last_abs_time = abs_time

        new_midi.tracks.append(new_track)
    return new_midi

def midi_random_pitch_transformer(original_midi, random_pitch_strength):
    """
    Randomizes the pitch of notes based on a given strength (probability).
    Usa (pitch, channel) come chiave e un contatore per gestire note duplicate
    sullo stesso pitch/canale — nessuna nota resta aperta nel DAW.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(original_track, 'name') and original_track.name:
            new_track.name = original_track.name

        # pitch_map: (pitch_orig, channel) -> lista di pitch nuovi (stack LIFO)
        # gestisce piu' note_on sullo stesso pitch prima del note_off
        pitch_map = defaultdict(list)

        for msg in original_track:
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.note, msg.channel)
                if random.randint(0, 100) < random_pitch_strength:
                    new_pitch = random.randint(0, 127)
                else:
                    new_pitch = msg.note
                pitch_map[key].append(new_pitch)
                new_track.append(msg.copy(note=new_pitch))

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if pitch_map[key]:
                    # LIFO: chiude l'ultima nota aperta su questo pitch/canale
                    mapped_pitch = pitch_map[key].pop()
                else:
                    mapped_pitch = msg.note
                new_track.append(msg.copy(note=mapped_pitch))

            else:
                new_track.append(msg)

        # Chiudi eventuali note rimaste aperte (note_on senza note_off)
        for (orig_pitch, ch), pitches in pitch_map.items():
            for p in pitches:
                new_track.append(mido.Message('note_off', note=p, velocity=0, channel=ch, time=0))

        new_midi.tracks.append(new_track)
    return new_midi


def midi_add_rhythmic_base(original_midi, kick, snare, hihat, time_signature, rhythmic_pattern_style):
    """
    Aggiunge una o più tracce con una base ritmica che dura esattamente quanto il brano originale.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)
    
    DRUM_MAP = {
        "kick": 36,     
        "snare": 38,    
        "hihat_closed": 42,
    }
    
    try:
        beats_per_measure, note_value = map(int, time_signature.split('/'))
        if beats_per_measure <= 0 or note_value <= 0:
            raise ValueError
    except (ValueError, IndexError):
        st.warning(f"Metrica non valida: '{time_signature}'. Verrà usata la metrica 4/4.")
        beats_per_measure, note_value = 4, 4
    
    ticks_per_beat = new_midi.ticks_per_beat
    ticks_per_measure = int(ticks_per_beat * beats_per_measure * 4 / note_value)

    if ticks_per_measure == 0:
        st.warning("Ticks per misura è zero. Non è possibile aggiungere la base ritmica.")
        return new_midi

    # Calcolo della durata totale del brano originale in ticks
    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)

    if total_ticks == 0:
        st.warning("Il brano originale non contiene eventi validi per calcolare la lunghezza. La base ritmica non verrà aggiunta.")
        return new_midi

    rhythmic_patterns_in_measure = {
        "kick": [],
        "snare": [],
        "hihat_closed": []
    }
    
    if rhythmic_pattern_style == "Pattern Fisso (Pop/Rock)":
        if kick:
            rhythmic_patterns_in_measure["kick"].append({'start_tick': 0, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
            if beats_per_measure >= 3:
                rhythmic_patterns_in_measure["kick"].append({'start_tick': ticks_per_beat * 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
        if snare:
            if beats_per_measure >= 2:
                rhythmic_patterns_in_measure["snare"].append({'start_tick': ticks_per_beat, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
            if beats_per_measure >= 4:
                rhythmic_patterns_in_measure["snare"].append({'start_tick': ticks_per_beat * 3, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
        if hihat:
            for i in range(beats_per_measure * 2):
                rhythmic_patterns_in_measure["hihat_closed"].append({'start_tick': i * ticks_per_beat // 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 80})

    elif rhythmic_pattern_style == "Pattern Casuale":
        kick_prob, snare_prob, hihat_prob = 0.2, 0.1, 0.4
        ticks_per_subdivision = ticks_per_beat // 4
        total_subdivisions_in_measure = beats_per_measure * 4
        
        for i in range(total_subdivisions_in_measure):
            start_tick = i * ticks_per_subdivision
            duration = ticks_per_subdivision // 2 
            if kick and random.random() < kick_prob: rhythmic_patterns_in_measure["kick"].append({'start_tick': start_tick, 'duration_ticks': duration, 'velocity': random.randint(80, 110)})
            if snare and random.random() < snare_prob: rhythmic_patterns_in_measure["snare"].append({'start_tick': start_tick, 'duration_ticks': duration, 'velocity': random.randint(80, 110)})
            if hihat and random.random() < hihat_prob: rhythmic_patterns_in_measure["hihat_closed"].append({'start_tick': start_tick, 'duration_ticks': duration, 'velocity': random.randint(60, 90)})

    elif rhythmic_pattern_style == "Pattern Adattivo":
        note_on_counts = defaultdict(int)
        for track in original_midi.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
                    tick_in_measure = abs_time % ticks_per_measure
                    subdivision_ticks = ticks_per_beat // 4
                    snapped_tick = round(tick_in_measure / subdivision_ticks) * subdivision_ticks
                    note_on_counts[snapped_tick] += 1
        
        if note_on_counts:
            most_common_ticks = sorted(note_on_counts, key=note_on_counts.get, reverse=True)
            
            kick_ticks = []
            if kick:
                for tick in most_common_ticks:
                    if len(kick_ticks) >= 3: break
                    if (beats_per_measure == 4 and (tick == 0 or tick == ticks_per_beat * 2)) or (len(kick_ticks) < 2 and note_on_counts[tick] > 1):
                         kick_ticks.append(tick)
                if not kick_ticks: kick_ticks.extend([0, ticks_per_beat*2] if beats_per_measure >= 4 else [0])
                for tick in kick_ticks: rhythmic_patterns_in_measure["kick"].append({'start_tick': tick, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
            
            snare_ticks = []
            if snare:
                for tick in most_common_ticks:
                    is_kick_tick = any(abs(tick - kt) < ticks_per_beat / 4 for kt in kick_ticks)
                    if not is_kick_tick and len(snare_ticks) < 2: snare_ticks.append(tick)
                if not snare_ticks: snare_ticks.extend([ticks_per_beat, ticks_per_beat*3] if beats_per_measure >= 4 else [ticks_per_beat])
                for tick in snare_ticks: rhythmic_patterns_in_measure["snare"].append({'start_tick': tick, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})

            if hihat:
                ticks_per_eighth = ticks_per_beat // 2
                for i in range(int(ticks_per_measure / ticks_per_eighth)):
                    rhythmic_patterns_in_measure["hihat_closed"].append({'start_tick': i * ticks_per_eighth, 'duration_ticks': ticks_per_eighth // 2, 'velocity': random.randint(60, 90)})
        else:
            st.warning("Nessuna nota trovata per un pattern adattivo. Verrà usato un pattern fisso.")
            if kick: rhythmic_patterns_in_measure["kick"].append({'start_tick': 0, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
            if snare: rhythmic_patterns_in_measure["snare"].append({'start_tick': ticks_per_beat, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
            if hihat: rhythmic_patterns_in_measure["hihat_closed"].append({'start_tick': 0, 'duration_ticks': ticks_per_beat // 2, 'velocity': 80})

    instrument_names = {"kick": "Cassa", "snare": "Rullante", "hihat_closed": "Hi-hat"}

    # Calcola il numero di misure necessarie per coprire l'intera durata del brano
    num_measures = int(np.ceil(total_ticks / ticks_per_measure))
    
    for drum_note_name, patterns_in_measure in rhythmic_patterns_in_measure.items():
        if not patterns_in_measure: continue
        
        new_drum_track = mido.MidiTrack()
        new_drum_track.name = f"Ritmica: {instrument_names[drum_note_name]}"
        new_drum_track.append(mido.Message('program_change', program=0, channel=9, time=0))
        
        all_drum_events = []
        for i in range(num_measures):
            current_measure_start_tick = i * ticks_per_measure
            for event in patterns_in_measure:
                start_abs_time = current_measure_start_tick + event['start_tick']
                end_abs_time = start_abs_time + event['duration_ticks']
                
                # Aggiungi solo eventi che rientrano nella durata totale del brano
                if start_abs_time < total_ticks:
                    all_drum_events.append({'msg': mido.Message('note_on', note=DRUM_MAP[drum_note_name], velocity=event['velocity'], channel=9), 'abs_time': start_abs_time})
                    if end_abs_time < total_ticks:
                        all_drum_events.append({'msg': mido.Message('note_off', note=DRUM_MAP[drum_note_name], velocity=0, channel=9), 'abs_time': end_abs_time})

        all_drum_events.sort(key=lambda x: x['abs_time'])
        last_abs_time = 0
        for event in all_drum_events:
            delta_time = max(0, event['abs_time'] - last_abs_time)
            new_msg = event['msg'].copy(time=delta_time)
            new_drum_track.append(new_msg)
            last_abs_time = event['abs_time']
        
        # Aggiungi un messaggio finale per garantire che la traccia abbia la lunghezza corretta
        new_drum_track.append(mido.Message('note_off', note=DRUM_MAP[drum_note_name], velocity=0, channel=9, time=max(0, total_ticks - last_abs_time)))
        
        new_midi.tracks.append(new_drum_track)

    return new_midi



# Mappatura General MIDI: numero programma → nome famiglia strumentale
_GM_FAMILY = [
    "Piano","Chromatic Perc","Organ","Guitar",
    "Bass","Strings","Ensemble","Brass",
    "Reed","Pipe","Synth Lead","Synth Pad",
    "Synth FX","Ethnic","Percussive","Sound FX",
]

def _gm_track_name(program, channel):
    """Restituisce il nome GM della famiglia strumentale dato il programma e il canale."""
    if channel == 9:
        return "Drums"
    family = _GM_FAMILY[min(program // 8, 15)]
    return family


def _split_type0_to_tracks(midi):
    """
    Converte un MIDI tipo 0 (1 traccia, N canali) in un MIDI tipo 1
    con una traccia per canale attivo (canali senza note vengono ignorati).
    Nomina ogni traccia con il nome GM reale (Bass, Drums, Guitar, ecc.)
    preservando il canale originale.
    """
    from collections import defaultdict
    tpb = midi.ticks_per_beat
    src_track = midi.tracks[0]

    # Tempo assoluto per ogni messaggio
    events = []
    abs_time = 0
    for msg in src_track:
        abs_time += msg.time
        events.append((abs_time, msg))

    # Leggi program_change per canale (primo trovato vince)
    ch_program = {}
    for _, msg in events:
        if msg.type == 'program_change' and msg.channel not in ch_program:
            ch_program[msg.channel] = msg.program

    # Separa meta e messaggi per canale
    meta_events = []
    ch_events = defaultdict(list)
    for abs_t, msg in events:
        if msg.is_meta:
            meta_events.append((abs_t, msg))
        elif hasattr(msg, 'channel'):
            ch_events[msg.channel].append((abs_t, msg))
        else:
            meta_events.append((abs_t, msg))

    # Tieni solo canali che hanno almeno una nota
    active_channels = {
        ch for ch, evs in ch_events.items()
        if any(m.type == 'note_on' and m.velocity > 0 for _, m in evs)
    }

    new_midi = mido.MidiFile(ticks_per_beat=tpb, type=1)

    # Traccia 0: solo meta
    meta_track = mido.MidiTrack()
    meta_track.name = "Meta"
    last_t = 0
    for abs_t, msg in sorted(meta_events, key=lambda x: x[0]):
        delta = abs_t - last_t
        meta_track.append(msg.copy(time=delta))
        last_t = abs_t
    new_midi.tracks.append(meta_track)

    # Conta quante volte compare ogni nome GM (per disambiguare duplicati)
    name_count = defaultdict(int)
    ch_names = {}
    for ch in sorted(active_channels):
        prog = ch_program.get(ch, 0)
        base_name = _gm_track_name(prog, ch)
        name_count[base_name] += 1
        ch_names[ch] = (base_name, prog)

    # Se un nome compare più volte, aggiungi suffisso numerico
    seen = defaultdict(int)
    final_names = {}
    for ch in sorted(active_channels):
        base_name, prog = ch_names[ch]
        if name_count[base_name] > 1:
            seen[base_name] += 1
            final_names[ch] = f"{base_name} {seen[base_name]}"
        else:
            final_names[ch] = base_name

    # Una traccia per canale attivo
    for ch in sorted(active_channels):
        ch_track = mido.MidiTrack()
        ch_track.name = final_names[ch]
        last_t = 0
        evs = sorted(ch_events[ch], key=lambda x: x[0])
        for abs_t, msg in evs:
            delta = abs_t - last_t
            ch_track.append(msg.copy(time=delta))
            last_t = abs_t
        new_midi.tracks.append(ch_track)

    return new_midi


def midi_recomposer(original_midi, style):
    """
    Ricompone TRACCIA PER TRACCIA il MIDI originale.
    Se il file è tipo 0 (1 traccia, N canali) lo esplode prima in N tracce.
    Per ogni traccia:
      1. Estrae il pool di pitch (con frequenza proporzionale)
      2. Rileva il canale dominante della traccia
      3. Costruisce una nuova melodia con ritmo e struttura completamente nuovi
         usando solo le note di quella traccia come vocabolario
    Output: stesso numero di tracce/canali dell'originale — brano irriconoscibile.
    """
    from collections import Counter

    # File tipo 0: esplodi canali in tracce separate prima di ricomporre
    if original_midi.type == 0 or (len(original_midi.tracks) == 1 and
            len({m.channel for t in original_midi.tracks for m in t if hasattr(m,'channel')}) > 1):
        original_midi = _split_type0_to_tracks(original_midi)

    tpb = original_midi.ticks_per_beat

    # Durata totale originale in ticks
    total_ticks = 0
    for track in original_midi.tracks:
        t = sum(msg.time for msg in track)
        total_ticks = max(total_ticks, t)
    if total_ticks == 0:
        total_ticks = tpb * 4 * 32  # fallback 32 battute

    # --- DEFINIZIONE STILI ---
    style_configs = {
        "ambient": {
            "note_dur_range": (tpb * 2, tpb * 6),
            "gap_range":      (tpb // 2, tpb * 2),
            "vel_factor":     0.6,
            "pitch_step":     4,
        },
        "drone": {
            "note_dur_range": (tpb * 4, tpb * 8),
            "gap_range":      (tpb, tpb * 3),
            "vel_factor":     0.5,
            "pitch_step":     7,
        },
        "minimal": {
            "note_dur_range": (tpb // 2, tpb * 2),
            "gap_range":      (tpb // 4, tpb),
            "vel_factor":     0.7,
            "pitch_step":     2,
        },
        "armonico": {
            "note_dur_range": (tpb // 2, tpb),
            "gap_range":      (tpb // 8, tpb // 4),
            "vel_factor":     0.85,
            "pitch_step":     3,
        },
        "elettronico": {
            "note_dur_range": (tpb // 4, tpb // 2),
            "gap_range":      (tpb // 8, tpb // 4),
            "vel_factor":     0.9,
            "pitch_step":     0,
        },
        "minimalismo_ritmico": {
            "note_dur_range": (tpb // 4, tpb // 2),
            "gap_range":      (tpb // 2, tpb * 2),
            "vel_factor":     0.8,
            "pitch_step":     5,
        },
        "sperimentale": {
            "note_dur_range": (tpb // 8, tpb * 3),
            "gap_range":      (0, tpb * 2),
            "vel_factor":     1.0,
            "pitch_step":     0,
        },
    }

    cfg = style_configs.get(style, style_configs["minimal"])

    def build_track_from_pool(weighted_pool, vel_min, vel_max, channel, track_name, instrument_header=None):
        """Costruisce una nuova traccia dal pool di pitch di una traccia originale."""
        if not weighted_pool:
            return None

        def pick_pitch(base=None):
            if base is None or cfg["pitch_step"] == 0:
                return random.choice(weighted_pool)
            direction = random.choice([-1, 1])
            candidate = base + direction * cfg["pitch_step"]
            return min(weighted_pool, key=lambda p: abs(p - candidate))

        def pick_vel():
            v = random.randint(vel_min, vel_max)
            return max(1, min(127, int(v * cfg["vel_factor"])))

        new_track = mido.MidiTrack()
        new_track.name = track_name
        for _h in (instrument_header or []):
            new_track.append(_h)

        events = []
        current_tick = 0
        last_pitch = random.choice(weighted_pool)

        while current_tick < total_ticks:
            pitch = pick_pitch(last_pitch)
            pitch = max(0, min(127, pitch))
            vel   = pick_vel()
            dur   = random.randint(*cfg["note_dur_range"])
            gap   = random.randint(*cfg["gap_range"])

            # Per elettronico: snappa sulla griglia
            if style == "elettronico":
                grid = tpb // 4
                current_tick = (current_tick // grid) * grid
                dur = (dur // grid) * grid or grid

            note_end = min(current_tick + dur, total_ticks)
            events.append(("on",  current_tick, pitch, vel, channel))
            events.append(("off", note_end,      pitch, 0,  channel))

            last_pitch = pitch
            current_tick += dur + gap

        # Ordina: note_off prima di note_on allo stesso tick
        events.sort(key=lambda e: (e[1], 0 if e[0] == "off" else 1))

        last_t = 0
        for ev in events:
            kind, tick, p, v2, ch = ev
            delta = max(0, tick - last_t)
            if kind == "on":
                new_track.append(mido.Message("note_on",  note=p, velocity=v2, channel=ch, time=delta))
            else:
                new_track.append(mido.Message("note_off", note=p, velocity=0,  channel=ch, time=delta))
            last_t = tick

        return new_track

    new_midi = mido.MidiFile(ticks_per_beat=tpb)

    for track_idx, orig_track in enumerate(original_midi.tracks):
        # --- Estrai nome traccia originale ---
        track_name = orig_track.name if hasattr(orig_track, 'name') and orig_track.name else f"Track {track_idx}"

        # --- Estrai pitches, velocities, canale dominante ---
        pitches    = []
        velocities = []
        channels   = []
        has_meta_only = True

        for msg in orig_track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)
                velocities.append(msg.velocity)
                channels.append(msg.channel)
                has_meta_only = False
            elif msg.type not in ('note_on', 'note_off'):
                pass  # meta / control — non nota

        # Traccia senza note (es. traccia metadati/tempo) → copiala intatta
        if has_meta_only or not pitches:
            meta_track = mido.MidiTrack()
            meta_track.name = track_name
            for msg in orig_track:
                meta_track.append(msg.copy())
            new_midi.tracks.append(meta_track)
            continue

        # --- Canale dominante della traccia ---
        channel_counts = Counter(channels)
        dominant_channel = channel_counts.most_common(1)[0][0]

        # --- Header strumento (program_change/bank select) da preservare ---
        _recomp_header = _extract_instrument_header(orig_track)

        # --- Pool di pitch pesato ---
        pitch_counts = Counter(pitches)
        weighted_pool = []
        for pitch, count in pitch_counts.items():
            weighted_pool.extend([pitch] * max(1, count))

        vel_min = min(velocities)
        vel_max = max(velocities)
        if vel_min > vel_max: vel_min, vel_max = vel_max, vel_min
        vel_min = max(1, vel_min)
        vel_max = min(127, vel_max)
        if vel_min == vel_max: vel_min = max(1, vel_max - 10)

        # --- Costruisci nuova traccia ---
        new_track = build_track_from_pool(
            weighted_pool, vel_min, vel_max,
            dominant_channel, track_name, instrument_header=_recomp_header
        )

        if new_track is not None:
            new_midi.tracks.append(new_track)
        else:
            # Fallback: traccia vuota con nome originale (e strumento preservato)
            empty = mido.MidiTrack()
            empty.name = track_name
            for _h in _recomp_header:
                empty.append(_h)
            new_midi.tracks.append(empty)

    return new_midi

# --- Session State ---
if 'midi_ready'   not in st.session_state: st.session_state.midi_ready   = False
if 'midi_bytes'   not in st.session_state: st.session_state.midi_bytes   = None
if 'midi_report'  not in st.session_state: st.session_state.midi_report  = ""
if 'midi_filename' not in st.session_state: st.session_state.midi_filename = ""

# --- Funzione Report ---
def build_report(original_file, original_midi, output_midi, selected_methods, parameters, midi_methods, stile=None):
    n_tracks_in  = len(original_midi.tracks)
    n_tracks_out = len(output_midi.tracks)
    duration     = round(original_midi.length, 2)
    tpb          = original_midi.ticks_per_beat

    method_lines = []
    for i, method_key in enumerate(selected_methods):
        params = parameters.get(method_key, [])
        label  = midi_methods[method_key]
        method_lines.append(f"{i+1}. {label}")

        if method_key == "MIDI Note Remapper":
            method_lines.append(f"   * Scala: {params[0]} | Tonalita': {params[1]}")
            method_lines.append(f"   * Pitch Shift: +/-{params[2]} semitoni | Velocity: {params[3]}%")

        elif method_key == "MIDI Phrase Reconstructor":
            method_lines.append(f"   * Lunghezza frase: {params[0]} battute | Stile: {params[1]}")

        elif method_key == "MIDI Time Scrambler":
            method_lines.append(f"   * Stretch: {params[0]}x | Quantizzazione: {params[1]}% | Swing: {params[2]}%")

        elif method_key == "MIDI Density Transformer":
            method_lines.append(f"   * Aggiungi note: {params[0]}% | Rimuovi note: {params[1]}% | Polifonia: {params[2]}")

        elif method_key == "MIDI Random Pitch Transformer":
            method_lines.append(f"   * Forza randomizzazione: {params[0]}%")

        elif method_key == "MIDI Rhythmic Base":
            drums = []
            if params[0]: drums.append("Cassa")
            if params[1]: drums.append("Rullante")
            if params[2]: drums.append("Hi-hat")
            method_lines.append(f"   * Elementi: {', '.join(drums) if drums else 'Nessuno'}")
            method_lines.append(f"   * Metrica: {params[3]} | Pattern: {params[4]}")

        elif method_key == "MIDI Costas Sequencer":
            costas_mode = params[0]
            method_lines.append(f"   * Modalità: {costas_mode} | Ordine richiesto: {params[1]}")
            if costas_mode == "Permutazione Pitch (Cromatica)":
                method_lines.append(f"   * Trasposizione: {params[2]} ottave | Costruzione di Welch, n=12, p=13")
            elif costas_mode == "Griglia Ritmica Costas":
                method_lines.append("   * Costruzione di Welch (Scott Rickard / J.P. Costas)")
            else:
                method_lines.append(f"   * Pitch base: {params[2]} | Estensione: {params[3]} semitoni | Passo: {params[4]} beat")

        elif method_key == "MIDI Stockhausen Punktuelle":
            dur_on, dyn_on, timbre_on, iso_on, row_used = params
            method_lines.append(f"   * Fila dodecafonica: {row_used}")
            flags = []
            if dur_on: flags.append("Durata")
            if dyn_on: flags.append("Dinamica")
            if timbre_on: flags.append("Timbro")
            method_lines.append(f"   * Parametri serializzati: Altezza (P), {', '.join(flags) if flags else 'solo Altezza'}")
            method_lines.append(f"   * Isolamento punti (staccato): {'Sì' if iso_on else 'No'}")
            method_lines.append("   * Serialismo integrale (Stockhausen/Boulez, rad. Messiaen 'Mode de valeurs')")

        elif method_key == "MIDI Boulez Multiplication":
            set_size, chord_density, register_spread, set_a, set_b = params
            method_lines.append(f"   * Dimensione insiemi A/B: {set_size} | Densità accordo: {'completa' if not chord_density else chord_density} | Ottave: {register_spread}")
            method_lines.append(f"   * Insieme A: {set_a} | Insieme B: {set_b}")
            method_lines.append("   * Moltiplicazione d'accordi (Boulez, 'Le Marteau sans maître' / Structures II)")

        elif method_key == "MIDI Xenakis Stochastic":
            sieve_str, mean_ev, pc, ps, dm, vm, vs = params
            method_lines.append(f"   * Crivello (sieve): {sieve_str} | Tasso eventi: {mean_ev}/beat (processo di Poisson)")
            method_lines.append(f"   * Altezza: Gauss(μ={pc}, σ={ps}) | Durata: Exp(μ={dm} beat) | Dinamica: Gauss(μ={vm}, σ={vs})")
            method_lines.append("   * Musica stocastica (Xenakis, Pithoprakta/Achorripsis, 1955-57)")

        elif method_key == "MIDI Cage Chance":
            silence_p, dur_var = params
            method_lines.append(f"   * Probabilità di silenzio per evento: {silence_p:.0%} | Varietà durata: {'Sì' if dur_var else 'No'}")
            method_lines.append("   * Operazioni di caso via I Ching, metodo delle tre monete (Cage, 'Music of Changes', 1951)")

        elif method_key == "MIDI Eno Generative":
            num_loops_r, min_lb, max_lb, nlr, dm, vb = params
            method_lines.append(f"   * Numero loop: {num_loops_r} | Lunghezza: {min_lb}-{max_lb} beat | Estensione durata: ×{dm}")
            method_lines.append(f"   * Rapporto durata nota/loop: {nlr} | Velocity base: {vb}")
            method_lines.append("   * Loop asincroni a lunghezze incommensurabili (Eno, 'Music for Airports'/'Discreet Music')")

        elif method_key == "MIDI Part Tintinnabuli":
            root_n, mode_n, triad = params
            method_lines.append(f"   * Tonalità: {root_n} {mode_n} | Triade tintinnabuli: {triad}")
            method_lines.append("   * Tecnica tintinnabuli — voce M (melodica) + voce T (triade) (Pärt, 1976)")

        elif method_key == "MIDI Messiaen Modal":
            mode_name_r, root_n2, block_sz = params
            method_lines.append(f"   * Modo: {mode_name_r} | Tonica: {root_n2} | Blocco palindromo: {block_sz} note")
            method_lines.append("   * Modi a trasposizione limitata + ritmi non retrogradabili (Messiaen, 1944)")

        elif method_key == "MIDI Reich Phasing":
            pat_len, n_cyc, shift_f = params
            method_lines.append(f"   * Lunghezza cellula: {pat_len} note | Cicli: {n_cyc} | Scarto di fase: {shift_f}")
            method_lines.append("   * Phasing — due voci identiche che si sfasano gradualmente (Reich, 'Piano Phase', 1967)")

        elif method_key == "MIDI Glass Additive":
            cell_len, reps, contr = params
            method_lines.append(f"   * Lunghezza cellula: {cell_len} note | Ripetizioni per stadio: {reps} | Contrazione: {'Sì' if contr else 'No'}")
            method_lines.append("   * Processo additivo (Glass, 'Two Pages', 1968)")

        elif method_key == "MIDI Bach Canon":
            canon_t, interval_s, delay_b = params
            method_lines.append(f"   * Tipo di canone: {canon_t} | Intervallo: {interval_s} semitoni | Ritardo: {delay_b} beat")
            method_lines.append("   * Canone rigoroso — dux/comes (Bach, Offerta Musicale/Arte della Fuga)")

        elif method_key == "MIDI Fractal Geometry":
            cantor_d, n_segs, pitch_c = params
            method_lines.append(f"   * Profondità Cantor: {cantor_d} | Segmenti generati: {n_segs} | Centro pitch: {pitch_c}")
            method_lines.append("   * Insieme di Cantor (ritmo) + IFS/chaos game Sierpinski (melodia) — algoritmi frattali standard")
            method_lines.append("   * NON è la tecnica documentata di un compositore specifico — vedi nota nell'interfaccia")

    report = "[MIDI_DECOMPOSER] // VOL_01 // MIDI // STRUCTURAL_DECOMPOSITION\n"
    report += ":: MOTORE: midi_decomposer [v1.0]\n"
    report += f":: FILE: {original_file}\n"
    if stile:
        report += f":: STILE: {stile}\n"
    report += f":: TRACCE: {n_tracks_in} | DURATA: {duration} sec | TICKS/BEAT: {tpb}\n"
    report += "\n"
    report += "\"Il file e' entrato come partitura. E' uscito come esperimento.\"\n"
    report += "\n"
    report += "> METODI APPLICATI (in ordine):\n"
    report += "\n".join(method_lines) + "\n"
    report += "\n"
    report += "> TECHNICAL LOG SHEET:\n"
    report += f"* Tracce originali: {n_tracks_in} -> Tracce output: {n_tracks_out}\n"
    report += f"* Metodi applicati: {len(selected_methods)}\n"
    report += "\n"
    report += "> Regia e Algoritmo: Loop507\n"
    report += "\n"
    report += "#loop507 #mididecomposer #generativemusic #midiprocessing\n"
    report += "#structuraldecomposition #algorithmicmusic #experimentalmusic"
    return report

# --- Player MIDI in-browser (web component html-midi-player, no dipendenze server) ---
def render_midi_player(midi_bytes, label, key_suffix=""):
    """
    Incorpora un lettore/visualizzatore MIDI direttamente nel browser dell'utente,
    usando la libreria 'html-midi-player' (Tone.js + soundfont via CDN).
    Nessuna sintesi lato server: funziona anche su Streamlit Cloud.
    """
    b64_midi = base64.b64encode(midi_bytes).decode("utf-8")
    data_uri = f"data:audio/midi;base64,{b64_midi}"
    html_code = f"""
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
    <div style="background:#111;border-radius:8px;padding:12px;font-family:sans-serif;">
        <p style="color:#ddd;margin:0 0 8px 0;font-size:14px;">🎧 {label}</p>
        <midi-player
            src="{data_uri}"
            sound-font
            visualizer="#midi-visualizer-{key_suffix}"
            style="width:100%;">
        </midi-player>
        <midi-visualizer
            id="midi-visualizer-{key_suffix}"
            src="{data_uri}"
            type="piano-roll"
            style="width:100%;display:block;margin-top:8px;">
        </midi-visualizer>
    </div>
    """
    components.html(html_code, height=260, scrolling=False)

# --- Sezione Upload File MIDI ---
st.subheader("🎵 Carica il tuo file MIDI (.mid o .midi)")
uploaded_midi_file = st.file_uploader(
    "Trascina qui il tuo file MIDI o clicca per sfogliare",
    type=["mid", "midi"],
    help="Carica un file MIDI per iniziare la decomposizione."
)

decomposed_midi_file = None
if uploaded_midi_file is not None:
    st.success("File MIDI caricato con successo!")

    try:
        midi_data = mido.MidiFile(file=uploaded_midi_file)
        st.subheader("File MIDI Caricato: Panoramica")
        st.write(f"Nome file: **{uploaded_midi_file.name}**")
        st.write(f"Numero di tracce: **{len(midi_data.tracks)}**")
        st.write(f"Durata (stimata): **{midi_data.length:.2f} secondi**")

        with st.expander("🎧 Ascolta il MIDI originale"):
            _orig_bytes_io = io.BytesIO()
            midi_data.save(file=_orig_bytes_io)
            render_midi_player(_orig_bytes_io.getvalue(), "MIDI originale", key_suffix="original")

        st.markdown("---")
        st.subheader("⚙️ Modalita' di Decomposizione")

        midi_methods = {
            "MIDI Note Remapper": "🎶 Remapping di Note (Verticale)",
            "MIDI Phrase Reconstructor": "🔄 Riorganizzazione Frasi (Orizzontale)",
            "MIDI Time Scrambler": "⏳ Manipolazione Ritmo/Durata (Orizzontale)",
            "MIDI Density Transformer": "🎲 Controllo Densità (Armonia/Contrappunto)",
            "MIDI Random Pitch Transformer": "❓ Randomizzazione Totale Pitch (Caos)",
            "MIDI Rhythmic Base": "🥁 Aggiungi Base Ritmica",
            "MIDI Recomposer": "🔁 Ricomposizione (nuovo brano dal materiale originale)",
            # --- Compositori (vedi modalita' dedicata "🎼 Compositori") ---
            "MIDI Costas Sequencer": "🧮 Scott Rickard — Costas Sequencer",
            "MIDI Stockhausen Punktuelle": "🎯 Karlheinz Stockhausen — Punktuelle Musik",
            "MIDI Boulez Multiplication": "🔷 Pierre Boulez — Moltiplicazione d'Accordi",
            "MIDI Xenakis Stochastic": "☁️ Iannis Xenakis — Musica Stocastica (Nuvole)",
            "MIDI Cage Chance": "☯️ John Cage — Operazioni di Caso (I Ching)",
            "MIDI Eno Generative": "🌫️ Brian Eno — Musica Generativa (Cicli Asincroni)",
            "MIDI Part Tintinnabuli": "🔔 Arvo Pärt — Tintinnabuli",
            "MIDI Messiaen Modal": "🕊️ Olivier Messiaen — Modi e Ritmi Non Retrogradabili",
            "MIDI Reich Phasing": "🌀 Steve Reich — Phasing",
            "MIDI Glass Additive": "➕ Philip Glass — Processo Additivo",
            "MIDI Bach Canon": "🎼 Johann Sebastian Bach — Canone Rigoroso",
            "MIDI Fractal Geometry": "🌿 Geometria Frattale (ispirato a Wallin/Sharp/Posadas)",
        }
        # Metodi disponibili nella modalita' "🔧 Avanzato" (i Compositori hanno la loro modalita' dedicata)
        ADVANCED_METHODS_KEYS = [
            "MIDI Note Remapper", "MIDI Phrase Reconstructor", "MIDI Time Scrambler",
            "MIDI Density Transformer", "MIDI Random Pitch Transformer",
            "MIDI Rhythmic Base", "MIDI Recomposer",
        ]
        # Ordine alfabetico per cognome del compositore. "Geometria Frattale" resta
        # in fondo e separata: non e' la tecnica documentata di un singolo autore
        # (vedi nota nella UI), ma un algoritmo frattale standard onestamente
        # etichettato come "ispirato a".
        COMPOSITORI = {
            "🎼 Johann Sebastian Bach — Canone Rigoroso": "MIDI Bach Canon",
            "🔷 Pierre Boulez — Moltiplicazione d'Accordi": "MIDI Boulez Multiplication",
            "☯️ John Cage — Operazioni di Caso (I Ching)": "MIDI Cage Chance",
            "🌫️ Brian Eno — Musica Generativa": "MIDI Eno Generative",
            "➕ Philip Glass — Processo Additivo": "MIDI Glass Additive",
            "🕊️ Olivier Messiaen — Modi e Ritmi Non Retrogradabili": "MIDI Messiaen Modal",
            "🔔 Arvo Pärt — Tintinnabuli": "MIDI Part Tintinnabuli",
            "🌀 Steve Reich — Phasing": "MIDI Reich Phasing",
            "🧮 Scott Rickard — Costas Sequencer": "MIDI Costas Sequencer",
            "🎯 Karlheinz Stockhausen — Punktuelle Musik": "MIDI Stockhausen Punktuelle",
            "☁️ Iannis Xenakis — Musica Stocastica": "MIDI Xenakis Stochastic",
            "🌿 Geometria Frattale (ispirato a Wallin/Sharp/Posadas)": "MIDI Fractal Geometry",
        }

        # --- STILI RICOMPOSIZIONE (usati dal pulsante Ricomponi) ---
        RECOMPOSE_STYLES = {
            "🔇 Minimal":             ("minimal",             "Ritmo scarno, pause ampie. Brano sparso e meditativo."),
            "🌊 Ambient":             ("ambient",             "Note lunghe e rarefatte. Paesaggio sonoro lento."),
            "🎼 Armonico":            ("armonico",            "Melodia per gradi stretti, fluida e cantabile."),
            "🤖 Elettronico":         ("elettronico",         "Griglia rigida, pattern meccanici e ripetitivi."),
            "🔔 Drone":               ("drone",               "Note lunghissime, statico e ipnotico."),
            "🥁 Minimalismo Ritmico": ("minimalismo_ritmico", "Sincopato, poche note sparse, ritmo nuovo."),
            "🎲 Sperimentale":        ("sperimentale",        "Pitch random + durate caotiche. Brano irriconoscibile."),
        }

        # --- PRESET DECOMPOSIZIONE (usati dal pulsante Decomponi in modalità Avanzato) ---
        PRESETS = {
            "🎸 Elettroacustico": {
                "desc": "Ritmo deformato, frasi rimescolate, groove organico con base ritmica adattiva.",
                "methods": ["MIDI Phrase Reconstructor","MIDI Time Scrambler","MIDI Rhythmic Base"],
                "params": {
                    "MIDI Phrase Reconstructor": (2, "Casuale"),
                    "MIDI Time Scrambler": (1.0, 30, 55),
                    "MIDI Rhythmic Base": (True, True, True, "4/4", "Pattern Adattivo"),
                }
            },
            "⚡ Glitch": {
                "desc": "Random Pitch aggressivo + frasi rimescolate + timing spezzato.",
                "methods": ["MIDI Phrase Reconstructor","MIDI Time Scrambler","MIDI Random Pitch Transformer"],
                "params": {
                    "MIDI Phrase Reconstructor": (2, "Inversione"),
                    "MIDI Time Scrambler": (0.8, 90, 70),
                    "MIDI Random Pitch Transformer": (80,),
                }
            },
            "🎬 Cinematico": {
                "desc": "Frasi riorganizzate + stretch lento + Triadi. Epico e atmosferico.",
                "methods": ["MIDI Phrase Reconstructor","MIDI Time Scrambler","MIDI Density Transformer"],
                "params": {
                    "MIDI Phrase Reconstructor": (8, "Ciclico A-B-A"),
                    "MIDI Time Scrambler": (2.0, 40, 0),
                    "MIDI Density Transformer": (15, 0, "Riempi Accordo (Triadi)"),
                }
            },
            "🎷 Jazz Decostruito": {
                "desc": "Swing alto + contro-melodia + frasi rimescolate. Libertà ritmica.",
                "methods": ["MIDI Phrase Reconstructor","MIDI Time Scrambler","MIDI Density Transformer"],
                "params": {
                    "MIDI Phrase Reconstructor": (4, "Casuale"),
                    "MIDI Time Scrambler": (1.0, 20, 75),
                    "MIDI Density Transformer": (25, 0, "Aggiungi Contro-Melodia"),
                }
            },
            "📢 Noise": {
                "desc": "Frasi invertite + Triadi dense + Random Pitch estremo. Muro di suono.",
                "methods": ["MIDI Phrase Reconstructor","MIDI Density Transformer","MIDI Random Pitch Transformer"],
                "params": {
                    "MIDI Phrase Reconstructor": (2, "Inversione"),
                    "MIDI Density Transformer": (50, 0, "Riempi Accordo (Triadi)"),
                    "MIDI Random Pitch Transformer": (95,),
                }
            },
        }

        # Modalita' Preset / Avanzato
        modalita = st.radio("Modalita':", ["🎨 Stile", "🎼 Compositori", "🔧 Avanzato"], horizontal=True)

        decomposed_midi_file = midi_data
        parameters = {}
        selected_methods_keys = []

        if modalita == "🎨 Stile":
            st.markdown("#### 🔁 Ricomponi l'intero MIDI")
            st.markdown(
                "Ogni traccia viene ricostruita dal suo pool di note originali: "
                "**stesso numero di tracce, stessi canali, stessi nomi** — ritmo e struttura completamente nuovi."
            )
            style_label = st.selectbox(
                "Scegli uno stile:",
                list(RECOMPOSE_STYLES.keys()),
                key="recompose_style_label"
            )
            style_key, style_desc = RECOMPOSE_STYLES[style_label]
            st.info(style_desc)

            if st.button("🔁 Ricomponi", type="primary", use_container_width=True, key="btn_recomponi"):
                with st.spinner("Ricomponendo traccia per traccia..."):
                    recomposed = midi_recomposer(midi_data, style_key)
                    midi_out_bytes = io.BytesIO()
                    recomposed.save(file=midi_out_bytes)
                    midi_out_bytes.seek(0)
                    st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                    st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Recomposed.mid"
                    st.session_state.midi_report   = build_report(
                        uploaded_midi_file.name, midi_data, recomposed,
                        ["MIDI Recomposer"], {"MIDI Recomposer": (style_key,)},
                        midi_methods, stile=style_label
                    )
                    st.session_state.midi_ready = True
                    st.success(
                        f"✅ Ricomposizione completata! "
                        f"{len(midi_data.tracks)} tracce originali → "
                        f"{len(recomposed.tracks)} tracce ricomposte."
                    )

        elif modalita == "🎼 Compositori":
            st.markdown("#### 🎼 Tecniche Compositive Algoritmiche")
            st.markdown(
                "Ricomposizioni che replicano le tecniche di compositori/matematici storici "
                "che hanno usato schemi combinatori — nessuna rete neurale, solo algoritmo puro."
            )
            compositore_label = st.selectbox("Scegli compositore/tecnica:", list(COMPOSITORI.keys()), key="compositore_select")
            compositore_key = COMPOSITORI[compositore_label]

            if compositore_key == "MIDI Stockhausen Punktuelle":
                st.info(
                    "**Serialismo integrale (Kreuzspiel, 1951 / Structures Ia, 1952)** — radicato nel "
                    "\"Mode de valeurs et d'intensités\" di Messiaen. Una fila a 12 elementi, derivata "
                    "dal brano stesso, viene applicata in 4 forme indipendenti (Prima, Retrograda, "
                    "Inversione, Retrograda-Inversa) a 4 parametri scorrelati: altezza, durata, "
                    "dinamica e timbro. Ogni nota diventa un punto sonoro isolato."
                )
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    serialize_duration = st.checkbox("Serializza Durata", value=True, key="stock_dur")
                    serialize_dynamics = st.checkbox("Serializza Dinamica", value=True, key="stock_dyn")
                with col_st2:
                    serialize_timbre = st.checkbox("Serializza Timbro (canale)", value=True, key="stock_timbre")
                    isolamento_punti = st.checkbox("Isolamento punti (note staccate)", value=True, key="stock_iso")

                if st.button("🎯 Applica Punktuelle Musik", type="primary", use_container_width=True, key="btn_stockhausen"):
                    with st.spinner("Serializzando i 4 parametri (Stockhausen/Boulez)..."):
                        result_midi, row_used = midi_stockhausen_punktuelle(
                            midi_data, serialize_duration, serialize_dynamics, serialize_timbre, isolamento_punti
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Stockhausen.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Stockhausen Punktuelle"],
                            {"MIDI Stockhausen Punktuelle": (serialize_duration, serialize_dynamics, serialize_timbre, isolamento_punti, row_used)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Punktuelle Musik applicata! Fila dodecafonica usata: {row_used}")

            elif compositore_key == "MIDI Boulez Multiplication":
                st.info(
                    "**Moltiplicazione d'accordi** (*Le Marteau sans maître*, 1955 / *Structures II*) — "
                    "due insiemi di classi di altezza A e B vengono derivati dal brano; A viene traslato "
                    "per ciascun intervallo generato da B, e l'unione delle trasposizioni forma un nuovo "
                    "aggregato armonico. Ogni nota del brano diventa un accordo costruito su questo "
                    "aggregato: la linea melodica si trasforma in *blocs sonores*, masse armoniche "
                    "verticali invece di punti isolati (l'opposto testurale della Punktuelle Musik)."
                )
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    set_size = st.slider("Dimensione insiemi A e B:", 2, 6, 4, key="boulez_set_size")
                    register_spread = st.slider("Spargimento su ottave:", 1, 3, 1, key="boulez_register_spread")
                with col_b2:
                    limita_densita = st.checkbox("Limita densità accordo", value=False, key="boulez_limit_density")
                    chord_density = st.slider("Note per accordo:", 2, 12, 6, key="boulez_chord_density") if limita_densita else 0

                _preview_a, _preview_b = derive_boulez_sets(midi_data, set_size)
                _preview_pivot = _preview_b[0] if _preview_b else 0
                _preview_mult = boulez_multiply_sets(_preview_a, _preview_b, _preview_pivot)
                st.caption(f"Anteprima — Insieme A: {_preview_a} | Insieme B: {_preview_b} | Aggregato risultante: {_preview_mult} ({len(_preview_mult)} classi)")

                if st.button("🔷 Applica Moltiplicazione d'Accordi", type="primary", use_container_width=True, key="btn_boulez"):
                    with st.spinner("Moltiplicando gli insiemi di classi di altezza..."):
                        result_midi, sets_info = midi_boulez_multiplication(midi_data, set_size, chord_density, register_spread)
                        set_a, set_b, multiplied = sets_info
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Boulez.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Boulez Multiplication"],
                            {"MIDI Boulez Multiplication": (set_size, chord_density, register_spread, set_a, set_b)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Moltiplicazione applicata! Aggregato: {multiplied} ({len(multiplied)} classi di altezza)")

            elif compositore_key == "MIDI Xenakis Stochastic":
                st.info(
                    "**Musica stocastica** (*Pithoprakta*, 1955-56 / *Achorripsis*, 1956-57) — i parametri "
                    "sonori sono governati da distribuzioni di probabilità anziché da regole fisse: i tempi "
                    "di attacco seguono un **processo di Poisson** (come la densità delle 'nuvole di suoni' "
                    "di Xenakis), l'altezza segue una **Gaussiana** quantizzata su un **crivello** (sieve, "
                    "teoria dei cribles) definito da classi di resto modulari."
                )
                sieve_input = st.text_input(
                    "Crivello (formula m:r, unione, es. '3:0, 4:1'):",
                    value="3:0, 4:1",
                    key="xenakis_sieve_input",
                    help="Ogni coppia m:r seleziona tutti i numeri congrui a r modulo m. Le coppie si uniscono."
                )
                sieve_pairs = parse_sieve_string(sieve_input)
                _sieve_preview = generate_sieve(sieve_pairs, universe=(0, 24)) if sieve_pairs else []
                st.caption(f"Anteprima crivello (0-24): {_sieve_preview if _sieve_preview else 'nessuna coppia valida — verrà usato il range cromatico completo'}")

                col_x1, col_x2 = st.columns(2)
                with col_x1:
                    mean_events_per_beat = st.slider("Densità eventi (per beat, Poisson):", 0.25, 8.0, 2.0, 0.25, key="xenakis_density")
                    pitch_center = st.slider("Centro altezza (MIDI):", 24, 96, 60, key="xenakis_pitch_center")
                    pitch_spread = st.slider("Dispersione altezza (σ, semitoni):", 1, 36, 12, key="xenakis_pitch_spread")
                with col_x2:
                    duration_mean = st.slider("Durata media evento (beat, esponenziale):", 0.1, 4.0, 0.5, 0.1, key="xenakis_dur_mean")
                    velocity_mean = st.slider("Dinamica media (velocity):", 20, 120, 75, key="xenakis_vel_mean")
                    velocity_spread = st.slider("Dispersione dinamica (σ):", 1, 40, 15, key="xenakis_vel_spread")
                xenakis_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="xenakis_seed")
                xenakis_seed = int(xenakis_seed_input) if xenakis_seed_input.strip().isdigit() else None

                if st.button("☁️ Applica Musica Stocastica", type="primary", use_container_width=True, key="btn_xenakis"):
                    with st.spinner("Generando la nuvola stocastica (Poisson + Gauss + crivello)..."):
                        result_midi, sieve_used = midi_xenakis_stochastic(
                            midi_data, sieve_pairs, mean_events_per_beat, pitch_center, pitch_spread,
                            duration_mean, velocity_mean, velocity_spread, seed=xenakis_seed
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Xenakis.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Xenakis Stochastic"],
                            {"MIDI Xenakis Stochastic": (sieve_input, mean_events_per_beat, pitch_center, pitch_spread, duration_mean, velocity_mean, velocity_spread)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Nuvola stocastica generata! Crivello effettivo: {len(sieve_used)} classi disponibili su 128")

            elif compositore_key == "MIDI Cage Chance":
                st.info(
                    "**Operazioni di caso** (*Music of Changes*, 1951) — ogni nota diventa un evento le cui "
                    "proprietà (altezza, durata, dinamica, presenza/assenza di suono) sono determinate da "
                    "**esagrammi indipendenti**, generati con il metodo classico delle tre monete dell'I "
                    "Ching (64 esiti equiprobabili). Il **silenzio** è materiale musicale legittimo quanto "
                    "il suono — stesso principio alla base di *4'33\"*."
                )
                col_cg1, col_cg2 = st.columns(2)
                with col_cg1:
                    silence_probability = st.slider("Probabilità di silenzio per evento:", 0.0, 0.8, 0.15, 0.05, key="cage_silence_prob")
                with col_cg2:
                    duration_variety = st.checkbox("Varietà indipendente della durata", value=True, key="cage_dur_variety")
                cage_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="cage_seed")
                cage_seed = int(cage_seed_input) if cage_seed_input.strip().isdigit() else None

                if st.button("☯️ Applica Operazioni di Caso", type="primary", use_container_width=True, key="btn_cage"):
                    with st.spinner("Lanciando le monete dell'I Ching (64 esagrammi per parametro)..."):
                        result_midi, hexagram_log = midi_cage_chance_operations(
                            midi_data, silence_probability, duration_variety, seed=cage_seed
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Cage.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Cage Chance"],
                            {"MIDI Cage Chance": (silence_probability, duration_variety)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        n_eventi_originali = len(hexagram_log)
                        n_silenzi = sum(1 for h in hexagram_log if (h[3] / 64.0) < silence_probability)
                        st.success(f"✅ Operazioni di caso applicate! {n_eventi_originali - n_silenzi} suoni, {n_silenzi} silenzi su {n_eventi_originali} esagrammi lanciati")

            elif compositore_key == "MIDI Eno Generative":
                st.info(
                    "**Musica generativa a cicli asincroni** (*Music for Airports*, 1978 / *Discreet Music*, "
                    "1975) — ogni loop ripete una singola nota (derivata dal brano) al proprio periodo "
                    "indipendente. Le lunghezze dei loop sono scelte 'incommensurabili' tra loro (via numeri "
                    "primi distinti), cosi' che l'intero sistema impieghi un tempo lunghissimo prima di "
                    "ripetersi esattamente, pur restando gli elementi di base sempre gli stessi. "
                    "*\"Non compongo la musica, compongo i sistemi che la generano\"* (Eno)."
                )
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    num_loops = st.slider("Numero di loop:", 2, 12, 6, key="eno_num_loops")
                    min_loop_beats = st.slider("Lunghezza minima loop (beat):", 2, 32, 8, key="eno_min_loop")
                    max_loop_beats = st.slider("Lunghezza massima loop (beat):", 8, 128, 32, key="eno_max_loop")
                with col_e2:
                    note_length_ratio = st.slider("Durata nota (frazione del loop):", 0.05, 0.9, 0.35, 0.05, key="eno_note_ratio")
                    duration_multiplier = st.slider("Estensione durata brano (×):", 1, 12, 4, key="eno_dur_mult")
                    velocity_base = st.slider("Velocity base:", 15, 90, 55, key="eno_vel_base")
                eno_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="eno_seed")
                eno_seed = int(eno_seed_input) if eno_seed_input.strip().isdigit() else None

                if st.button("🌫️ Applica Musica Generativa", type="primary", use_container_width=True, key="btn_eno"):
                    with st.spinner("Costruendo i cicli asincroni (lunghezze basate su numeri primi)..."):
                        result_midi, loops_info = midi_eno_generative(
                            midi_data, num_loops, min_loop_beats, max_loop_beats,
                            note_length_ratio, duration_multiplier, velocity_base, seed=eno_seed
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Eno.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Eno Generative"],
                            {"MIDI Eno Generative": (num_loops, min_loop_beats, max_loop_beats, note_length_ratio, duration_multiplier, velocity_base)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        loop_desc = ", ".join(f"{p}t" for _, p, _ in loops_info[:6])
                        st.success(f"✅ Sistema generativo creato! {len(loops_info)} loop asincroni, cicli: {loop_desc}{'...' if len(loops_info) > 6 else ''}")

            elif compositore_key == "MIDI Part Tintinnabuli":
                st.info(
                    "**Tintinnabuli** (1976) — la voce M (melodica) è la linea originale quantizzata su una "
                    "scala diatonica; una nuova voce T (campanellina, dal latino *tintinnabulum*) viene "
                    "aggiunta in parallelo, vincolata alle sole note della triade tonale, scelta secondo "
                    "posizione e direzione rispetto a ciascuna nota M."
                )
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    part_root = st.selectbox("Tonica:", ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"], index=9, key="part_root")
                    part_mode = st.selectbox("Modo:", ["Minore Naturale", "Maggiore"], key="part_mode")
                with col_p2:
                    part_direction = st.selectbox("Direzione voce T:", ["sotto", "sopra"], key="part_direction")
                    part_position = st.slider("Posizione nella triade:", 1, 3, 1, key="part_position")

                if st.button("🔔 Applica Tintinnabuli", type="primary", use_container_width=True, key="btn_part"):
                    with st.spinner("Costruendo voce M e voce T..."):
                        result_midi, part_info = midi_part_tintinnabuli(
                            midi_data, part_root, part_mode, part_direction, part_position
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Part.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Part Tintinnabuli"],
                            {"MIDI Part Tintinnabuli": (part_root, part_mode, part_info[2])},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Tintinnabuli applicato! Triade tonale: {part_info[2]}")

            elif compositore_key == "MIDI Messiaen Modal":
                st.info(
                    "**Linguaggio modale di Messiaen** (1944) — ogni nota viene quantizzata su uno dei 7 "
                    "modi a trasposizione limitata (scale simmetriche che tornano su se stesse dopo poche "
                    "trasposizioni); le durate vengono raggruppate a blocchi e trasformate in sequenze "
                    "palindrome (**ritmi non retrogradabili**: lette avanti o indietro, restano identiche)."
                )
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    messiaen_mode = st.selectbox("Modo:", list(MESSIAEN_MODES.keys()), index=1, key="messiaen_mode")
                    messiaen_root = st.selectbox("Tonica:", ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"], index=0, key="messiaen_root")
                with col_m2:
                    messiaen_block = st.slider("Blocco palindromo (note):", 2, 12, 4, key="messiaen_block")

                if st.button("🕊️ Applica Linguaggio Modale", type="primary", use_container_width=True, key="btn_messiaen"):
                    with st.spinner("Quantizzando sul modo e costruendo ritmi palindromi..."):
                        result_midi, mode_pcs_used = midi_messiaen_modal(midi_data, messiaen_mode, messiaen_root, messiaen_block)
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Messiaen.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Messiaen Modal"],
                            {"MIDI Messiaen Modal": (messiaen_mode, messiaen_root, messiaen_block)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Linguaggio modale applicato! Classi di altezza del modo: {mode_pcs_used}")

            elif compositore_key == "MIDI Reich Phasing":
                st.info(
                    "**Phasing** (*Piano Phase*, 1967) — una cellula breve viene derivata dal brano e "
                    "suonata simultaneamente da due voci identiche: la Voce A resta fissa, la Voce B "
                    "accumula ad ogni ciclo uno scarto di fase crescente, finché non completa una "
                    "rotazione intera e torna in unisono con la Voce A."
                )
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    reich_pattern_length = st.slider("Lunghezza cellula (note):", 3, 16, 8, key="reich_pattern_len")
                    reich_shift = st.slider("Scarto di fase per ciclo (frazione di passo):", 0.25, 2.0, 1.0, 0.25, key="reich_shift")
                with col_r2:
                    reich_num_cycles = st.slider("Numero di cicli:", 2, 32, 8, key="reich_num_cycles")

                if st.button("🌀 Applica Phasing", type="primary", use_container_width=True, key="btn_reich"):
                    with st.spinner("Costruendo le due voci sfasate..."):
                        result_midi, cycles_used = midi_reich_phasing(midi_data, reich_pattern_length, reich_num_cycles, reich_shift)
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Reich.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Reich Phasing"],
                            {"MIDI Reich Phasing": (reich_pattern_length, cycles_used, reich_shift)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Phasing applicato! {cycles_used} cicli, Voce A e Voce B aggiunte.")

            elif compositore_key == "MIDI Glass Additive":
                st.info(
                    "**Processo additivo** (*Two Pages*, 1968) — una breve cellula derivata dal brano viene "
                    "esposta gradualmente: prima 1 nota ripetuta, poi 2, poi 3... fino alla cellula completa, "
                    "eventualmente seguita da una contrazione simmetrica."
                )
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    glass_cell_length = st.slider("Lunghezza cellula (note):", 2, 16, 6, key="glass_cell_len")
                    glass_repeats = st.slider("Ripetizioni per stadio:", 1, 8, 3, key="glass_repeats")
                with col_g2:
                    glass_contract = st.checkbox("Contrazione dopo l'apice", value=True, key="glass_contract")

                if st.button("➕ Applica Processo Additivo", type="primary", use_container_width=True, key="btn_glass"):
                    with st.spinner("Costruendo gli stadi del processo additivo..."):
                        result_midi, n_stages = midi_glass_additive(midi_data, glass_cell_length, glass_repeats, glass_contract)
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Glass.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Glass Additive"],
                            {"MIDI Glass Additive": (glass_cell_length, glass_repeats, glass_contract)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Processo additivo applicato! {n_stages} stadi generati.")

            elif compositore_key == "MIDI Bach Canon":
                st.info(
                    "**Canone rigoroso** (Offerta Musicale / Arte della Fuga) — un \"comes\" (voce che segue) "
                    "viene derivato dal \"dux\" (la melodia originale, il \"leader\") secondo una regola fissa: "
                    "retrogrado (canone cancrizans, il celebre \"canone del granchio\"), inversione (specchio "
                    "attorno a un asse), o trasposizione a un intervallo con ritardo (canone imitativo)."
                )
                bach_canon_type = st.selectbox(
                    "Tipo di canone:",
                    ["Cancrizans (Retrogrado)", "Per Inversione", "All'Intervallo (imitativo)"],
                    key="bach_canon_type"
                )
                if bach_canon_type == "All'Intervallo (imitativo)":
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        bach_interval = st.slider("Intervallo di trasposizione (semitoni):", -12, 12, 7, key="bach_interval")
                    with col_b2:
                        bach_delay = st.slider("Ritardo di ingresso (beat):", 0.5, 8.0, 2.0, 0.5, key="bach_delay")
                else:
                    bach_interval, bach_delay = 0, 0

                if st.button("🎼 Applica Canone", type="primary", use_container_width=True, key="btn_bach"):
                    with st.spinner("Derivando il comes dal dux..."):
                        result_midi, canon_used = midi_bach_canon(midi_data, bach_canon_type, bach_interval, bach_delay)
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Bach.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Bach Canon"],
                            {"MIDI Bach Canon": (bach_canon_type, bach_interval, bach_delay)},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Canone applicato! Tipo: {canon_used}")

            elif compositore_key == "MIDI Fractal Geometry":
                st.warning(
                    "⚠️ **Non è la tecnica documentata di un singolo compositore.** Wallin, Sharp, Posadas e "
                    "altri citano genericamente \"algoritmi frattali\" nei loro programmi di sala, ma non hanno "
                    "mai pubblicato la formula esatta usata. Questo modulo applica due algoritmi frattali "
                    "**standard e verificabili**, nello spirito (non nella lettera) di quei lavori."
                )
                st.info(
                    "**Ritmo** — insieme di Cantor: rimozione ricorsiva del terzo centrale di un intervallo, "
                    "genera uno scheletro temporale gerarchico e auto-simile a scale diverse.  \n"
                    "**Melodia** — IFS (chaos game, triangolo di Sierpinski): 3 trasformazioni affini verso i "
                    "vertici di un triangolo, iterate ripetutamente, generano un profilo di altezza auto-simile."
                )
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    cantor_depth = st.slider("Profondità Cantor (livelli di ricorsione):", 1, 8, 4, key="fractal_cantor_depth")
                with col_f2:
                    fractal_pitch_range = st.slider("Estensione melodica (semitoni):", 6, 48, 24, key="fractal_pitch_range")
                fractal_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="fractal_seed")
                fractal_seed = int(fractal_seed_input) if fractal_seed_input.strip().isdigit() else None

                if st.button("🌿 Applica Geometria Frattale", type="primary", use_container_width=True, key="btn_fractal"):
                    with st.spinner("Generando l'insieme di Cantor e il chaos game IFS..."):
                        result_midi, fractal_info = midi_fractal_geometry(
                            midi_data, cantor_depth, None, fractal_pitch_range, seed=fractal_seed
                        )
                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Fractal.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Fractal Geometry"],
                            {"MIDI Fractal Geometry": (cantor_depth, fractal_info[0], fractal_info[1])},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        st.success(f"✅ Sistema frattale generato! {fractal_info[0]} segmenti Cantor, centro pitch {fractal_info[1]}")

            else:  # Costas Sequencer
                st.caption(
                    "Basato sulla costruzione di Welch (Scott Rickard / J.P. Costas): "
                    "una permutazione algoritmica in cui nessun vettore-differenza tra "
                    "coppie si ripete. Nessuna rete neurale, nessun random puro — pura DSP/combinatoria."
                )
                costas_mode = st.selectbox(
                    "Modalità Costas:",
                    ["Permutazione Pitch (Cromatica)", "Griglia Ritmica Costas", "Generatore Costas (Nuova Melodia)"],
                    key="costas_mode_compositori"
                )

                if costas_mode == "Permutazione Pitch (Cromatica)":
                    st.info("Usa una matrice di Costas di ordine 12 (p=13) come cifrario di sostituzione fisso per le 12 classi di altezza. Ritmo invariato.")
                    transpose_octave = st.slider("Trasposizione (ottave):", -2, 2, 0, key="costas_transpose_compositori")
                    costas_params = (costas_mode, 12, transpose_octave, 0, 1.0)

                elif costas_mode == "Griglia Ritmica Costas":
                    costas_order_req = st.slider("Ordine minimo della matrice (n):", 3, 24, 8, key="costas_order_grid_compositori")
                    _p_preview = _costas_find_prime(costas_order_req)
                    st.caption(f"Ordine effettivo: n = {_p_preview - 1} (primo p = {_p_preview})")
                    costas_params = (costas_mode, costas_order_req, 0, 0, 1.0)

                else:  # Generatore Costas (Nuova Melodia)
                    costas_order_req = st.slider("Ordine minimo della matrice (n):", 3, 24, 12, key="costas_order_gen_compositori")
                    _p_preview = _costas_find_prime(costas_order_req)
                    st.caption(f"Ordine effettivo: n = {_p_preview - 1} (primo p = {_p_preview})")
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        base_pitch = st.slider("Pitch base (MIDI):", 24, 96, 60, key="costas_base_pitch_compositori")
                    with col_c2:
                        pitch_range_semitones = st.slider("Estensione (semitoni):", 6, 48, 24, key="costas_pitch_range_compositori")
                    step_beats = st.slider("Durata di ogni passo (in battute/beat):", 0.125, 2.0, 0.25, 0.125, key="costas_step_beats_compositori")
                    costas_params = (costas_mode, costas_order_req, base_pitch, pitch_range_semitones, step_beats)

                if st.button("🧮 Applica Costas Sequencer", type="primary", use_container_width=True, key="btn_costas"):
                    with st.spinner("Generando la matrice di Costas (costruzione di Welch)..."):
                        cmode, corder, cp1, cp2, cp3 = costas_params
                        if cmode == "Permutazione Pitch (Cromatica)":
                            result_midi, costas_info = midi_costas_pitch_permutation(midi_data, transpose_octave=cp1)
                        elif cmode == "Griglia Ritmica Costas":
                            result_midi, costas_info = midi_costas_rhythmic_grid(midi_data, corder)
                        else:
                            result_midi, costas_info = midi_costas_generator(midi_data, corder, cp1, cp2, cp3)

                        midi_out_bytes = io.BytesIO()
                        result_midi.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Costas.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Costas Sequencer"], {"MIDI Costas Sequencer": costas_params},
                            midi_methods, stile=compositore_label
                        )
                        st.session_state.midi_ready = True
                        n_costas, p_costas, g_costas = costas_info
                        st.success(f"✅ Costas Sequencer applicato! Ordine effettivo n={n_costas} (p={p_costas}, g={g_costas})")

        else:  # 🔧 Avanzato
            st.markdown("#### Metodi di Decomposizione")
            selected_methods_keys = st.multiselect("Seleziona uno o piu' metodi:", ADVANCED_METHODS_KEYS, format_func=lambda x: midi_methods[x])
            st.markdown("#### Parametri per i Metodi Selezionati:")

            for selected_method in selected_methods_keys:
                st.markdown(f"**Parametri per: {midi_methods[selected_method]}**")

                if selected_method == "MIDI Note Remapper":
                    col1_remap, col2_remap = st.columns(2)
                    with col1_remap:
                        target_scale = st.selectbox("Scala Target:", ["Cromatica", "Maggiore", "Minore Naturale", "Pentatonica Maggiore", "Blues"], key=f"remap_scale_{selected_method}")
                    with col2_remap:
                        target_key = st.selectbox("Tonalità Target:", ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B','Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'], index=0, key=f"remap_key_{selected_method}")
                    pitch_shift_range = st.slider("Range Pitch Shift Randomico (semitoni):", 0, 12, 0, key=f"remap_pitch_shift_{selected_method}")
                    velocity_randomization = st.slider("Percentuale Randomizzazione Velocity:", 0, 100, 0, key=f"remap_velocity_{selected_method}")
                    parameters[selected_method] = (target_scale, target_key, int(pitch_shift_range), int(velocity_randomization))

                elif selected_method == "MIDI Phrase Reconstructor":
                    phrase_length_beats = st.slider("Lunghezza Frase (battute):", 1, 16, 4, key=f"phrase_length_{selected_method}")
                    reassembly_style = st.selectbox("Stile Riorganizzazione Frasi:", ["Casuale", "Inversione", "Ciclico A-B-A", "Dal Più Corto al Più Lungo"], index=0, key=f"phrase_style_{selected_method}")
                    parameters[selected_method] = (phrase_length_beats, reassembly_style)

                elif selected_method == "MIDI Time Scrambler":
                    keep_original_duration = st.checkbox("Mantieni Durata Originale", key=f"time_keep_duration_{selected_method}")
                    execution_speed_preset = st.selectbox("Velocità di Esecuzione:", ["Medio (Originale)", "Lento (Metà velocità)", "Molto Lento (Un quarto velocità)", "Veloce (Doppia velocità)", "Molto Veloce (Quattro volte velocità)"], index=0, key=f"time_speed_preset_{selected_method}")
                    default_stretch_factor = 1.0
                    if execution_speed_preset == "Lento (Metà velocità)": default_stretch_factor = 2.0
                    elif execution_speed_preset == "Molto Lento (Un quarto velocità)": default_stretch_factor = 4.0
                    elif execution_speed_preset == "Veloce (Doppia velocità)": default_stretch_factor = 0.5
                    elif execution_speed_preset == "Molto Veloce (Quattro volte velocità)": default_stretch_factor = 0.25
                    stretch_factor = st.slider("Fattore di Stiramento/Compressione (Time Warp):", 0.1, 5.0, default_stretch_factor, 0.1, key=f"time_stretch_factor_{selected_method}")
                    quantization_strength = st.slider("Forza Quantizzazione (0=libero, 100=rigido):", 0, 100, 50, key=f"time_quant_strength_{selected_method}")
                    swing_amount = st.slider("Quantità di Swing (%):", 0, 100, 0, key=f"time_swing_amount_{selected_method}")
                    if keep_original_duration: stretch_factor = 1.0
                    parameters[selected_method] = (stretch_factor, quantization_strength, swing_amount)

                elif selected_method == "MIDI Density Transformer":
                    add_note_probability = st.slider("Probabilità di Aggiungere Note (%):", 0, 50, 0, key=f"density_add_prob_{selected_method}")
                    remove_note_probability = st.slider("Probabilità di Rimuovere Note (%):", 0, 50, 0, key=f"density_remove_prob_{selected_method}")
                    polyphony_mode = st.selectbox("Modalità Polifonia Aggiuntiva:", ["Nessuna", "Riempi Accordo (Triadi)", "Aggiungi Contro-Melodia", "Droni"], key=f"density_poly_mode_{selected_method}")
                    parameters[selected_method] = (add_note_probability, remove_note_probability, polyphony_mode)

                elif selected_method == "MIDI Random Pitch Transformer":
                    random_pitch_strength = st.slider("Forza Randomizzazione Pitch (%):", 0, 100, 100, key=f"random_pitch_strength_{selected_method}")
                    parameters[selected_method] = (random_pitch_strength,)

                elif selected_method == "MIDI Rhythmic Base":
                    st.markdown("Seleziona gli elementi ritmici per costruire il tuo pattern:")
                    col_rhythm1, col_rhythm2 = st.columns(2)
                    with col_rhythm1:
                        kick_enabled = st.checkbox("Cassa", value=True, key="rhythm_kick")
                        snare_enabled = st.checkbox("Rullante", value=True, key="rhythm_snare")
                        hihat_enabled = st.checkbox("Hi-hat", value=True, key="rhythm_hihat")
                    with col_rhythm2:
                        time_signature = st.text_input("Metrica (es. '4/4', '3/4', '5/8'):", value="4/4", key=f"rhythm_time_sig_{selected_method}")
                        rhythmic_pattern_style = st.selectbox("Stile Pattern Ritmico:", ["Pattern Adattivo", "Pattern Fisso (Pop/Rock)", "Pattern Casuale"], key=f"rhythm_pattern_style_{selected_method}")
                    parameters[selected_method] = (kick_enabled, snare_enabled, hihat_enabled, time_signature, rhythmic_pattern_style)

                elif selected_method == "MIDI Recomposer":
                    recomposer_style_adv = st.selectbox(
                        "Stile Recomposer:",
                        ["minimal","ambient","armonico","elettronico","drone","minimalismo_ritmico","sperimentale"],
                        key="recomposer_style_adv"
                    )
                    parameters[selected_method] = (recomposer_style_adv,)

            if st.button("🎶 DECOMPONI MIDI", type="primary", use_container_width=True):
                with st.spinner("Applicando le decomposizioni..."):
                    current_midi = midi_data
                    for method_key in selected_methods_keys:
                        method_params = parameters.get(method_key, [])
                        if method_key == "MIDI Note Remapper":
                            current_midi = midi_note_remapper(current_midi, *method_params)
                        elif method_key == "MIDI Phrase Reconstructor":
                            current_midi = midi_phrase_reconstructor(current_midi, *method_params)
                        elif method_key == "MIDI Time Scrambler":
                            current_midi = midi_time_scrambler(current_midi, *method_params)
                        elif method_key == "MIDI Density Transformer":
                            current_midi = midi_density_transformer(current_midi, *method_params)
                        elif method_key == "MIDI Random Pitch Transformer":
                            current_midi = midi_random_pitch_transformer(current_midi, *method_params)
                        elif method_key == "MIDI Rhythmic Base":
                            current_midi = midi_add_rhythmic_base(current_midi, *method_params)
                        elif method_key == "MIDI Recomposer":
                            recompose_style = method_params[0] if method_params else "minimal"
                            current_midi = midi_recomposer(current_midi, recompose_style)
                    decomposed_midi_file = current_midi

                    if decomposed_midi_file:
                        st.success("Decomposizione MIDI completata!")
                        midi_out_bytes = io.BytesIO()
                        decomposed_midi_file.save(file=midi_out_bytes)
                        midi_out_bytes.seek(0)
                        st.session_state.midi_bytes    = midi_out_bytes.getvalue()
                        st.session_state.midi_filename = f"{uploaded_midi_file.name.split('.')[0]}_Decomposed.mid"
                        st.session_state.midi_report   = build_report(
                            uploaded_midi_file.name, midi_data, decomposed_midi_file,
                            selected_methods_keys, parameters, midi_methods, stile=None
                        )
                        st.session_state.midi_ready = True

                        def get_track_display_name(track, index):
                            track_name = next((msg.name for msg in track if msg.type == 'track_name'), None)
                            if not track_name and hasattr(track, 'name') and 'Ritmica:' in track.name:
                                return track.name
                            return f"Traccia {index}: {track_name if track_name else '(Senza Nome)'}"

                        if len(decomposed_midi_file.tracks) > 0:
                            st.markdown("---")
                            st.subheader("Scarica Singole Tracce del MIDI Decomposto")
                            track_options = [get_track_display_name(track, i) for i, track in enumerate(decomposed_midi_file.tracks)]
                            selected_tracks_indices = st.multiselect(
                                "Seleziona una o più tracce da scaricare singolarmente:",
                                options=list(range(len(decomposed_midi_file.tracks))),
                                format_func=lambda x: track_options[x],
                                default=None,
                                help="Seleziona le tracce che vuoi scaricare come file MIDI separati."
                            )
                            if selected_tracks_indices:
                                for track_index in selected_tracks_indices:
                                    single_track_midi = mido.MidiFile()
                                    single_track_midi.tracks.append(decomposed_midi_file.tracks[track_index])
                                    single_track_midi.ticks_per_beat = decomposed_midi_file.ticks_per_beat
                                    single_track_bytes = io.BytesIO()
                                    single_track_midi.save(file=single_track_bytes)
                                    single_track_bytes.seek(0)
                                    original_file_base_name = uploaded_midi_file.name.split('.')[0]
                                    track_name_for_file = get_track_display_name(decomposed_midi_file.tracks[track_index], track_index).replace(' ', '_').replace(':', '')
                                    st.download_button(
                                        label=f"💾 Scarica {track_options[track_index]}",
                                        data=single_track_bytes,
                                        file_name=f"{original_file_base_name}_{track_name_for_file}.mid",
                                        mime="audio/midi",
                                        key=f"download_track_{track_index}"
                                    )
                        else:
                            st.info("Il MIDI decomposto non contiene tracce valide da scaricare singolarmente.")
                    else:
                        st.error("Impossibile generare il MIDI decomposto. Controlla i messaggi di avviso.")

    except Exception as e:
        st.error(f"❌ Errore durante la lettura o l'elaborazione del file MIDI: {str(e)}")
        st.error("Assicurati che sia un file MIDI valido (.mid o .midi) e riprova.")
        st.exception(e)
else:
    st.info("👆 Carica un file MIDI (.mid o (.midi) per iniziare la decomposizione.")
    with st.expander("📖 Come usare MIDI Decomposer"):
        st.markdown("""
        ### Benvenuto in MIDI Decomposer!
        Qui potrai caricare i tuoi file MIDI e applicare diverse tecniche di decomposizione per creare nuove strutture musicali.
        **Come funziona:**
        1.  **Carica il tuo file MIDI** (con estensione `.mid` o `.midi`).
        2.  Scegli i **metodi di decomposizione** e imposta i loro **parametri**. Puoi sceglierne uno o più!
        3.  Clicca su **"DECOMPONI MIDI"**.
        4.  Scarica il **file MIDI completo** o seleziona le **singole tracce** da scaricare.
        5.  Apri il file MIDI scaricato nel tuo software musicale (DAW) preferito per ascoltare il risultato.
        **Metodi di Decomposizione Disponibili:**
        * **🎶 MIDI Note Remapper**: Rimodella le note del pentagramma (verticale) in base a scale, tonalità e randomizzazione.
        * **🔄 MIDI Phrase Reconstructor**: Riorganizza e ricompone blocchi o "frasi" musicali (orizzontale).
        * **⏳ MIDI Time Scrambler**: Modifica il timing e la durata delle note per creare nuovi groove.
        * **🎲 MIDI Density Transformer**: Aggiunge o rimuove note per alterare la densità armonica.
        * **❓ MIDI Random Pitch Transformer**: Randomizza completamente l'altezza di ogni nota (pitch) per un caos melodico.
        * **🥁 Aggiungi Base Ritmica**: Aggiunge una nuova traccia di batteria al tuo brano per creare un sound dance o pop!
        """)
# RISULTATI PERSISTENTI
if st.session_state.midi_ready and st.session_state.midi_bytes:
    st.markdown("---")
    st.subheader("🎧 Ascolta il risultato prima di scaricare")
    render_midi_player(st.session_state.midi_bytes, "MIDI decomposto/ricomposto", key_suffix="result")

    st.subheader("Scarica il tuo MIDI Decomposto")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.download_button(
            label="💾 Scarica MIDI Decomposto",
            data=st.session_state.midi_bytes,
            file_name=st.session_state.midi_filename,
            mime="audio/midi",
            use_container_width=True,
            key="down_midi"
        )
    with c_d2:
        st.download_button(
            label="📄 Scarica Report",
            data=st.session_state.midi_report,
            file_name="report_midi_decomposer.txt",
            key="down_report"
        )
    st.text_area("📄 REPORT", st.session_state.midi_report, height=300)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><em>MIDI Decomposer by loop507</em></p>
    <p>Sperimenta la destrutturazione MIDI</p>
    <p style='font-size: 0.8em;'>Powered by Streamlit & Mido</p>
</div>
""", unsafe_allow_html=True)
