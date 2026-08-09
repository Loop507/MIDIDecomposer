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
      - Timbro   -> forma Retrograda-Inversa (RI) (rotazione tra i canali presenti)
    Le note vengono processate in ordine cronologico assoluto attraverso tutte
    le tracce (non per traccia separata), perche' nella musica puntillistica
    ogni punto e' indipendente dal contesto melodico/timbrico originale.
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

    channels_present = sorted(set(
        msg.channel for track in original_midi.tracks for msg in track
        if hasattr(msg, 'channel')
    )) or [0]

    # Raccogli tutte le note come punti indipendenti, in ordine cronologico assoluto
    all_points = []
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        for nd in notes:
            all_points.append({
                'start': nd['start'],
                'orig_pitch': nd['pitch'],
                'orig_channel': nd['channel'],
                'orig_velocity': nd['velocity'],
                'track_idx': track_idx,
            })

    if not all_points:
        st.warning("Nessuna nota trovata nel brano. La tecnica Punktuelle non verra' applicata.")
        return original_midi, row

    all_points.sort(key=lambda x: x['start'])

    # Un'unica traccia d'uscita: la musica puntillistica di Stockhausen non
    # distingue "voci" — ogni punto e' un evento autonomo nello spazio sonoro.
    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    punkt_track = mido.MidiTrack()
    row_str = "-".join(str(x) for x in row_P)
    punkt_track.name = f"Punktuelle (fila: {row_str})"

    events = []
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

        if serialize_timbre:
            timbre_idx = row_RI[i % 12] % len(channels_present)
            channel = channels_present[timbre_idx]
        else:
            channel = point['orig_channel']

        note_start = point['start']
        if isolamento_punti:
            note_len = max(1, int(duration * 0.55))  # nota staccata: meno della meta' dello slot
        else:
            note_len = max(1, duration)

        events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=velocity, channel=channel, time=0), 'abs_time': note_start})
        events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=channel, time=0), 'abs_time': note_start + note_len})

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for event_data in events:
        delta = max(0, event_data['abs_time'] - last_abs_time)
        punkt_track.append(event_data['msg'].copy(time=delta))
        last_abs_time = event_data['abs_time']

    new_midi.tracks.append(punkt_track)
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
        last_abs_time = 0
        for event_data in final_events:
            delta = max(0, event_data['abs_time'] - last_abs_time)
            new_track.append(event_data['msg'].copy(time=delta))
            last_abs_time = event_data['abs_time']

        new_midi.tracks.append(new_track)

    return new_midi, (set_a, set_b, multiplied)


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
        current_phrase_start_tick = 0

        events_with_abs_time = []
        time_since_last_event = 0
        for msg in original_track:
            time_since_last_event += msg.time
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
            new_midi.tracks.append(mido.MidiTrack())
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
                elif msg_in_phrase.type == 'note_off' or (msg_in_phrase.type == 'note_on' and msg_in_phrase.velocity == 0):
                    open_notes.pop((msg_in_phrase.note, msg_in_phrase.channel), None)
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
        from collections import defaultdict
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

    def build_track_from_pool(weighted_pool, vel_min, vel_max, channel, track_name):
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
            dominant_channel, track_name
        )

        if new_track is not None:
            new_midi.tracks.append(new_track)
        else:
            # Fallback: traccia vuota con nome originale
            empty = mido.MidiTrack()
            empty.name = track_name
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
        }
        # Metodi disponibili nella modalita' "🔧 Avanzato" (i Compositori hanno la loro modalita' dedicata)
        ADVANCED_METHODS_KEYS = [
            "MIDI Note Remapper", "MIDI Phrase Reconstructor", "MIDI Time Scrambler",
            "MIDI Density Transformer", "MIDI Random Pitch Transformer",
            "MIDI Rhythmic Base", "MIDI Recomposer",
        ]
        COMPOSITORI = {
            "🎯 Karlheinz Stockhausen — Punktuelle Musik": "MIDI Stockhausen Punktuelle",
            "🔷 Pierre Boulez — Moltiplicazione d'Accordi": "MIDI Boulez Multiplication",
            "🧮 Scott Rickard — Costas Sequencer": "MIDI Costas Sequencer",
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
