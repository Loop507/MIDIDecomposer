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
</div>
""", unsafe_allow_html=True)

# --- Funzioni di Utilità ---
def get_key_offset(key_name):
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    clean_key = key_name.replace('m', '')
    return note_offsets.get(clean_key, 0)

def get_scale_notes(scale_name):
    scales = {
        "Cromatica": list(range(12)),
        "Maggiore": [0, 2, 4, 5, 7, 9, 11],
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10],
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Blues": [0, 3, 5, 6, 7, 10]
    }
    return scales.get(scale_name, scales["Cromatica"])

def extract_notes(track):
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
    return notes

def reconstruct_track(notes, ticks_per_beat):
    new_track = mido.MidiTrack()
    events = []
    for note in notes:
        events.append({'msg': mido.Message('note_on', note=note['pitch'], velocity=note['velocity'], channel=note['channel'], time=0), 'abs_time': note['start']})
        events.append({'msg': mido.Message('note_off', note=note['pitch'], velocity=0, channel=note['channel'], time=0), 'abs_time': note['end']})
    events.sort(key=lambda x: x['abs_time'])
    last_abs_time = 0
    for event in events:
        delta_time = max(0, event['abs_time'] - last_abs_time)
        new_track.append(event['msg'].copy(time=delta_time))
        last_abs_time = event['abs_time']
    return new_track

# --- Metodi di Elaborazione ---

def midi_note_remapper(original_midi, scale_name, key_name, shift, vel_rand):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    intervals = get_scale_notes(scale_name)
    offset = get_key_offset(key_name)
    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                new_p = max(0, min(127, msg.note + random.randint(-shift, shift)))
                note_in_oct = (new_p - offset) % 12
                closest = min(intervals, key=lambda x: abs(note_in_oct - x))
                final_p = ((new_p - offset) // 12) * 12 + closest + offset
                v = msg.velocity
                if msg.type == 'note_on' and vel_rand > 0:
                    v = max(1, min(127, int(v * (1 + random.uniform(-vel_rand/100, vel_rand/100)))))
                new_track.append(msg.copy(note=max(0, min(127, final_p)), velocity=v))
            else: new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi

def midi_cellular_automata(original_midi, generations):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    all_n = [n for t in original_midi.tracks for n in extract_notes(t)]
    if not all_n: return original_midi
    max_t = max(n['end'] for n in all_n)
    pitches = sorted(list(set([n['pitch'] for n in all_n])))
    state = [1 if i in pitches else 0 for i in range(128)]
    step = original_midi.ticks_per_beat // 2
    new_notes, curr_t = [], 0
    for _ in range(generations):
        if curr_t >= max_t: break
        new_s = [0]*128
        for i in range(1, 127):
            neighbors = state[i-1] + state[i+1]
            if neighbors == 1: new_s[i] = 1
        for p, active in enumerate(new_s):
            if active: new_notes.append({'pitch': p, 'start': curr_t, 'end': curr_t + step, 'velocity': 70, 'channel': 0})
        state = new_s
        curr_t += step
    new_midi.tracks.append(reconstruct_track(new_notes, original_midi.ticks_per_beat))
    return new_midi

def midi_fractal_generator(original_midi, iterations):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    notes = [n for t in original_midi.tracks for n in extract_notes(t)][:4]
    if not notes: return original_midi
    motif = [n['pitch'] for n in notes]
    def expand(seq, lev):
        if lev == 0: return seq
        res = []
        for p in seq: res.extend([m + (p - motif[0]) for m in motif])
        return expand(res, lev - 1)
    f_pitches = expand(motif, iterations)[:200]
    step = original_midi.ticks_per_beat // 4
    new_notes = [{'pitch': max(0, min(127, p)), 'start': i*step, 'end': (i+1)*step, 'velocity': 70, 'channel': 0} for i, p in enumerate(f_pitches)]
    new_midi.tracks.append(reconstruct_track(new_notes, original_midi.ticks_per_beat))
    return new_midi

# --- Sidebar ---
st.sidebar.header("⚙️ Parametri")
uploaded = st.sidebar.file_uploader("Carica MIDI", type=['mid', 'midi'])

methods = {
    "Remapper": "🎶 Note Remapper",
    "Cellular": "🦠 Automi Cellulari",
    "Fractal": "❄️ L-System Frattali"
}
sel = st.sidebar.multiselect("Metodi:", list(methods.values()))

params = {}
for s in sel:
    if s == methods["Remapper"]:
        params["Remapper"] = (st.sidebar.selectbox("Scala:", ["Maggiore", "Blues"]), st.sidebar.selectbox("Chiave:", ["C", "G"]), st.sidebar.slider("Shift:", 0, 12, 0), st.sidebar.slider("Vel Rand:", 0, 100, 0))
    elif s == methods["Cellular"]:
        params["Cellular"] = (st.sidebar.slider("Generazioni:", 4, 64, 16),)
    elif s == methods["Fractal"]:
        params["Fractal"] = (st.sidebar.slider("Iterazioni:", 1, 3, 2),)

# --- Esecuzione ---
if uploaded and st.button("🎶 DECOMPONI"):
    mid = mido.MidiFile(file=io.BytesIO(uploaded.getvalue()))
    curr = mid
    if methods["Remapper"] in sel: curr = midi_note_remapper(curr, *params["Remapper"])
    if methods["Cellular"] in sel: curr = midi_cellular_automata(curr, *params["Cellular"])
    if methods["Fractal"] in sel: curr = midi_fractal_generator(curr, *params["Fractal"])
    
    buf = io.BytesIO()
    curr.save(file=buf)
    st.download_button("💾 Scarica", buf.getvalue(), "decomposed.mid")
