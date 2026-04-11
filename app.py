# midi_decomposer_app.py - VERSIONE AGGIORNATA CON METODI ALGORITMICI

import streamlit as st
import mido
import random
import numpy as np
import io
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
    """Converte il nome della tonalità in offset semitonale, supportando ora anche le minori."""
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    # Pulizia per gestire Am, Cm, etc.
    clean_key = key_name.replace('m', '')
    return note_offsets.get(clean_key, 0)

def extract_notes(track):
    """Estrae tutte le note da una traccia e restituisce una lista di dizionari con tempi assoluti."""
    notes = []
    active_notes = {}
    current_time = 0
    for msg in track:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[(msg.note, msg.channel)] = (current_time, msg.velocity)
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
            key = (msg.note, msg.channel)
            if key in active_notes:
                start_time, velocity = active_notes.pop(key)
                notes.append({
                    'pitch': msg.note,
                    'start': start_time,
                    'end': current_time,
                    'velocity': velocity,
                    'channel': msg.channel
                })
    return notes

def reconstruct_track(notes, ticks_per_beat):
    """Ricostruisce una traccia MIDI da una lista di note con tempi assoluti."""
    events = []
    for n in notes:
        events.append({'time': n['start'], 'type': 'note_on', 'note': n['pitch'], 'velocity': n['velocity'], 'channel': n['channel']})
        events.append({'time': n['end'], 'type': 'note_off', 'note': n['pitch'], 'velocity': 0, 'channel': n['channel']})
    
    events.sort(key=lambda x: x['time'])
    new_track = mido.MidiTrack()
    last_time = 0
    for e in events:
        delta = e['time'] - last_time
        new_track.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta, channel=e['channel']))
        last_time = e['time']
    return new_track

# --- Moduli di Decomposizione Originali e Nuovi ---

def midi_note_remapper(mid, key_offset, scale_type, randomization_level):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    scales = {
        "Maggiore": [0, 2, 4, 5, 7, 9, 11],
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10],
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Pentatonica Minore": [0, 3, 5, 7, 10],
        "Cromatica": list(range(12))
    }
    allowed_steps = scales.get(scale_type, list(range(12)))
    
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off'] and msg.channel != 9:
                new_note = msg.note + key_offset
                if randomization_level > 0:
                    new_note += random.randint(-randomization_level, randomization_level)
                
                octave = new_note // 12
                pitch_in_octave = new_note % 12
                if pitch_in_octave not in allowed_steps:
                    pitch_in_octave = min(allowed_steps, key=lambda x: abs(x - pitch_in_octave))
                
                msg.note = max(0, min(127, octave * 12 + pitch_in_octave))
            new_track.append(msg)
        new_mid.tracks.append(new_track)
    return new_mid

def midi_genetic_shuffler(mid, complexity):
    """DNA Shuffler: Ricompone le note originali in una nuova griglia."""
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    all_notes = []
    for track in mid.tracks:
        all_notes.extend([msg.note for msg in track if msg.type == 'note_on' and msg.velocity > 0])
    
    if not all_notes: return mid
    
    step = mid.ticks_per_beat // 2
    for _ in range(64):
        if random.randint(1, 10) <= complexity:
            p = random.choice(all_notes)
            new_track.append(mido.Message('note_on', note=p, velocity=80, time=0))
            new_track.append(mido.Message('note_off', note=p, velocity=0, time=step))
        else:
            new_track.append(mido.Message('note_off', note=0, velocity=0, time=step))
    new_mid.tracks.append(new_track)
    return new_mid

def midi_stochastic_composer(mid, density, dur_range):
    """Crea nuvole sonore stocastiche dai pitch originali."""
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    all_p = [m.note for t in mid.tracks for m in t if m.type == 'note_on']
    if not all_p: return mid
    
    notes_gen = []
    curr = 0
    while curr < 10000:
        if random.randint(0, 100) < density:
            p = random.choice(all_p)
            d = random.randint(dur_range[0], dur_range[1])
            notes_gen.append({'pitch': p, 'start': curr, 'end': curr + d, 'velocity': 70, 'channel': 0})
        curr += random.randint(100, 400)
    new_mid.tracks.append(reconstruct_track(notes_gen, mid.ticks_per_beat))
    return new_mid

def midi_brownian_walker(mid, steps, max_jump):
    """Genera una melodia 'errante' basata sui pitch originali."""
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    all_p = [m.note for t in mid.tracks for m in t if m.type == 'note_on']
    if not all_p: return mid
    
    curr_p = random.choice(all_p)
    step_t = mid.ticks_per_beat // 2
    for _ in range(steps):
        curr_p = max(0, min(127, curr_p + random.randint(-max_jump, max_jump)))
        new_track.append(mido.Message('note_on', note=curr_p, velocity=70, time=0))
        new_track.append(mido.Message('note_off', note=curr_p, velocity=0, time=step_t))
    new_mid.tracks.append(new_track)
    return new_mid

def midi_cellular_automata(mid, generations):
    """Automa cellulare 1D basato sulle note presenti."""
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    pitches = sorted(list(set([m.note for t in mid.tracks for m in t if m.type == 'note_on'])))
    if not pitches: return mid
    
    state = [1 if i in pitches else 0 for i in range(128)]
    step_t = mid.ticks_per_beat // 4
    for _ in range(generations):
        new_s = [0]*128
        for i in range(1, 127):
            neigh = state[i-1] + state[i+1]
            if (state[i] == 1 and neigh == 1) or (state[i] == 0 and neigh == 1): new_s[i] = 1
        for p, active in enumerate(new_s):
            if active:
                new_track.append(mido.Message('note_on', note=p, velocity=60, time=0))
                new_track.append(mido.Message('note_off', note=p, velocity=0, time=step_t))
        state = new_s
    new_mid.tracks.append(new_track)
    return new_mid

def midi_fractal_generator(mid, iterations):
    """Generazione L-System basata sul primo motivo trovato."""
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    motif = []
    for t in mid.tracks:
        for m in t:
            if m.type == 'note_on' and m.velocity > 0: motif.append(m.note)
            if len(motif) >= 4: break
    if not motif: return mid

    def expand(seq, lev):
        if lev == 0: return seq
        res = []
        for p in seq:
            res.extend([m + (p - motif[0]) for m in motif])
        return expand(res, lev - 1)

    pitches = expand(motif, iterations)[:200]
    step_t = mid.ticks_per_beat // 4
    for p in pitches:
        p_safe = max(0, min(127, p))
        new_track.append(mido.Message('note_on', note=p_safe, velocity=70, time=0))
        new_track.append(mido.Message('note_off', note=p_safe, velocity=0, time=step_t))
    new_mid.tracks.append(new_track)
    return new_mid

# --- Sidebar ---
st.sidebar.header("⚙️ Impostazioni")
uploaded_file = st.sidebar.file_uploader("Carica il tuo MIDI", type=['mid', 'midi'])

st.sidebar.markdown("---")
midi_methods = {
    "Note Remapper": "🎶 MIDI Note Remapper",
    "Genetic Shuffler": "🧬 Genetic Shuffler (Nuovo Brano)",
    "Stochastic Cloud": "🎲 Stochastic Cloud (Stocastica)",
    "Brownian Walker": "🚶 Brownian Walker (Melodia Errante)",
    "Cellular Automata": "🦠 Automi Cellulari (Game of Life)",
    "Fractal L-System": "❄️ Frattali (L-System)"
}

selected_methods_names = st.sidebar.multiselect("Scegli i metodi:", list(midi_methods.values()))
selected_methods_keys = [k for k, v in midi_methods.items() if v in selected_methods_names]

parameters = {}
for method in selected_methods_keys:
    st.sidebar.subheader(f"Parametri {method}")
    if method == "Note Remapper":
        key = st.sidebar.selectbox("Tonalità:", ["C", "Cm", "C#", "D", "Dm", "Eb", "E", "Em", "F", "Fm", "F#", "G", "Gm", "Ab", "A", "Am", "Bb", "B"])
        scale = st.sidebar.selectbox("Scala:", ["Maggiore", "Minore Naturale", "Pentatonica Maggiore", "Pentatonica Minore", "Cromatica"])
        rand = st.sidebar.slider("Randomizzazione Pitch:", 0, 12, 0)
        parameters[method] = (get_key_offset(key), scale, rand)
    elif method == "Genetic Shuffler":
        comp = st.sidebar.slider("Complessità Ritmica:", 1, 10, 5)
        parameters[method] = (comp,)
    elif method == "Stochastic Cloud":
        dens = st.sidebar.slider("Densità Nuvola:", 10, 100, 50)
        dur = st.sidebar.slider("Range durata (ticks):", 120, 1920, (240, 960))
        parameters[method] = (dens, dur)
    elif method == "Brownian Walker":
        jump = st.sidebar.slider("Salto massimo:", 1, 12, 4)
        parameters[method] = (80, jump) # 80 passi fissi
    elif method == "Cellular Automata":
        gen = st.sidebar.slider("Generazioni:", 4, 32, 16)
        parameters[method] = (gen,)
    elif method == "Fractal L-System":
        it = st.sidebar.slider("Iterazioni:", 1, 3, 2)
        parameters[method] = (it,)

# --- Main Content ---
if not uploaded_file:
    st.info("👋 Benvenuto! Carica un file MIDI dalla sidebar per iniziare la scomposizione.")
else:
    if st.button("🎶 DECOMPONI MIDI"):
        with st.spinner("Elaborazione in corso..."):
            mid = mido.MidiFile(file=io.BytesIO(uploaded_file.getvalue()))
            current_midi = mid
            
            for m_key in selected_methods_keys:
                params_val = parameters[m_key]
                if m_key == "Note Remapper":
                    current_midi = midi_note_remapper(current_midi, *params_val)
                elif m_key == "Genetic Shuffler":
                    current_midi = midi_genetic_shuffler(current_midi, *params_val)
                elif m_key == "Stochastic Cloud":
                    current_midi = midi_stochastic_composer(current_midi, *params_val)
                elif m_key == "Brownian Walker":
                    current_midi = midi_brownian_walker(current_midi, *params_val)
                elif m_key == "Cellular Automata":
                    current_midi = midi_cellular_automata(current_midi, *params_val)
                elif m_key == "Fractal L-System":
                    current_midi = midi_fractal_generator(current_midi, *params_val)
            
            buf = io.BytesIO()
            current_midi.save(file=buf)
            st.success("✨ Decomposizione completata!")
            st.download_button("💾 Scarica MIDI Elaborato", buf.getvalue(), "decomposed.mid", "audio/midi")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>MIDI Decomposer by loop507 - Algorithmic Edition</p>", unsafe_allow_html=True)
