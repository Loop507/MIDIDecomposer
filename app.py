# midi_decomposer_app.py - VERSIONE INTEGRALE POTENZIATA

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
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    base_note_name = key_name.replace('m', '') # Supporto base per nomi con 'm'
    return note_offsets.get(base_note_name, 0)

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

# --- Moduli di Decomposizione Originali ---

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name)
    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                new_p = max(0, min(127, msg.note + random.randint(-pitch_shift_range, pitch_shift_range)))
                note_in_oct = (new_p - key_offset) % 12
                closest = min(target_scale_intervals, key=lambda x: abs(note_in_oct - x))
                final_p = max(0, min(127, ((new_p - key_offset) // 12) * 12 + closest + key_offset))
                v = msg.velocity
                if msg.type == 'note_on' and velocity_randomization > 0:
                    v = max(1, min(127, int(v * (1 + random.uniform(-velocity_randomization/100, velocity_randomization/100)))))
                new_track.append(msg.copy(note=final_p, velocity=v))
            else: new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi

def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats
    for original_track in original_midi.tracks:
        notes = extract_notes(original_track)
        if not notes:
            new_midi.tracks.append(original_track.copy())
            continue
        max_time = max(n['end'] for n in notes)
        phrases = []
        for t in range(0, max_time, ticks_per_phrase):
            p_notes = [n for n in notes if t <= n['start'] < t + ticks_per_phrase]
            if p_notes: phrases.append((t, p_notes))
        if reassembly_style == "Casuale": random.shuffle(phrases)
        elif reassembly_style == "Inversione": phrases.reverse()
        new_notes, curr_offset = [], 0
        for orig_start, p_n in phrases:
            for n in p_n:
                new_notes.append({'pitch': n['pitch'], 'start': n['start'] - orig_start + curr_offset, 'end': n['end'] - orig_start + curr_offset, 'velocity': n['velocity'], 'channel': n['channel']})
            curr_offset += ticks_per_phrase
        new_midi.tracks.append(reconstruct_track(new_notes, original_midi.ticks_per_beat))
    return new_midi

def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    tpb = original_midi.ticks_per_beat
    for track in original_midi.tracks:
        notes = extract_notes(track)
        if not notes:
            new_midi.tracks.append(track.copy())
            continue
        for n in notes:
            n['start'] = int(n['start'] * stretch_factor)
            n['end'] = int(n['end'] * stretch_factor)
            if quantization_strength > 0:
                grid = tpb / 4
                snap = round(n['start'] / grid) * grid
                n['start'] = int(n['start'] * (1 - quantization_strength/100) + snap * (quantization_strength/100))
        new_midi.tracks.append(reconstruct_track(notes, tpb))
    return new_midi

# --- NUOVI MODULI ALGORITMICI ---

def midi_genetic_shuffler(mid, complexity, total_ticks):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    all_p = [n['pitch'] for t in mid.tracks for n in extract_notes(t)]
    if not all_p: return mid
    new_notes = []
    step = mid.ticks_per_beat // 2
    for t in range(0, total_ticks, step):
        if random.randint(1, 10) <= complexity:
            new_notes.append({'pitch': random.choice(all_p), 'start': t, 'end': t + step, 'velocity': 80, 'channel': 0})
    new_mid.tracks.append(reconstruct_track(new_notes, mid.ticks_per_beat))
    return new_mid

def midi_stochastic_composer(mid, density, dur_range, total_ticks):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    all_p = [n['pitch'] for t in mid.tracks for n in extract_notes(t)]
    if not all_p: return mid
    new_notes, curr = [], 0
    while curr < total_ticks:
        if random.randint(0, 100) < density:
            d = random.randint(dur_range[0], dur_range[1])
            new_notes.append({'pitch': random.choice(all_p), 'start': curr, 'end': min(total_ticks, curr + d), 'velocity': 70, 'channel': 0})
        curr += random.randint(100, 500)
    new_mid.tracks.append(reconstruct_track(new_notes, mid.ticks_per_beat))
    return new_mid

def midi_brownian_walker(mid, max_jump, total_ticks):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    all_p = [n['pitch'] for t in mid.tracks for n in extract_notes(t)]
    if not all_p: return mid
    curr_p = random.choice(all_p)
    new_notes, step = [], mid.ticks_per_beat // 2
    for t in range(0, total_ticks, step):
        curr_p = max(0, min(127, curr_p + random.randint(-max_jump, max_jump)))
        new_notes.append({'pitch': curr_p, 'start': t, 'end': t + step, 'velocity': 75, 'channel': 0})
    new_mid.tracks.append(reconstruct_track(new_notes, mid.ticks_per_beat))
    return new_mid

def midi_cellular_automata(mid, speed, total_ticks):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    pitches = sorted(list(set([n['pitch'] for t in mid.tracks for n in extract_notes(t)])))
    if not pitches: return mid
    state = [1 if i in pitches else 0 for i in range(128)]
    new_notes, curr_t, step = [], 0, mid.ticks_per_beat // speed
    while curr_t < total_ticks:
        new_s = [0]*128
        for i in range(1, 127):
            n = state[i-1] + state[i+1]
            if n == 1: new_s[i] = 1
        for p, active in enumerate(new_s):
            if active: new_notes.append({'pitch': p, 'start': curr_t, 'end': curr_t + step, 'velocity': 60, 'channel': 0})
        state = new_s
        curr_t += step
    new_mid.tracks.append(reconstruct_track(new_notes, mid.ticks_per_beat))
    return new_mid

def midi_fractal_generator(mid, iterations, total_ticks):
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    all_n = [n['pitch'] for t in mid.tracks for n in extract_notes(t)]
    if len(all_n) < 4: return mid
    motif = all_n[:4]
    def expand(seq, lev):
        if lev == 0: return seq
        res = []
        for p in seq: res.extend([max(0, min(127, m + (p - motif[0]))) for m in motif])
        return expand(res, lev - 1)
    pitches = expand(motif, iterations)
    step = total_ticks // len(pitches) if len(pitches) > 0 else 480
    new_notes = [{'pitch': p, 'start': i*step, 'end': (i+1)*step, 'velocity': 70, 'channel': 0} for i, p in enumerate(pitches)]
    new_mid.tracks.append(reconstruct_track(new_notes, mid.ticks_per_beat))
    return new_mid

# --- Sezione UI e Logica Streamlit ---

st.subheader("🎵 Carica il tuo file MIDI")
uploaded_midi_file = st.file_uploader("Trascina qui il tuo file", type=["mid", "midi"])

if uploaded_midi_file:
    midi_data = mido.MidiFile(file=uploaded_midi_file)
    # Calcolo durata totale corretta
    total_ticks = 0
    for track in midi_data.tracks:
        t_time = 0
        for msg in track: t_time += msg.time
        total_ticks = max(total_ticks, t_time)

    st.write(f"Tracce: {len(midi_data.tracks)} | Durata stimata: {midi_data.length:.2f}s")
    
    midi_methods = {
        "Remapper": "🎶 MIDI Note Remapper",
        "Phrases": "🔄 MIDI Phrase Reconstructor",
        "Scrambler": "⏳ MIDI Time Scrambler",
        "Genetic": "🧬 Genetic Shuffler (DNA)",
        "Stochastic": "🎲 Stochastic Cloud (Nuvola)",
        "Brownian": "🚶 Brownian Walker (Errante)",
        "Cellular": "🦠 Automi Cellulari",
        "Fractal": "❄️ Frattali L-System"
    }
    
    selected_keys = st.multiselect("Seleziona Metodi:", list(midi_methods.keys()), format_func=lambda x: midi_methods[x])
    parameters = {}

    for k in selected_keys:
        st.markdown(f"**Parametri {midi_methods[k]}**")
        if k == "Remapper":
            parameters[k] = (st.selectbox("Scala:", ["Maggiore", "Minore Naturale", "Blues"]), st.selectbox("Tonalità:", ["C", "D", "E", "F", "G", "A", "B"]), st.slider("Pitch Shift:", 0, 12, 0), st.slider("Random Velocity:", 0, 100, 0))
        elif k == "Phrases":
            parameters[k] = (st.slider("Beats frase:", 1, 16, 4), st.selectbox("Stile:", ["Casuale", "Inversione"]))
        elif k == "Scrambler":
            parameters[k] = (st.slider("Time Stretch:", 0.1, 5.0, 1.0), st.slider("Quantizzazione:", 0, 100, 50), 0)
        elif k == "Genetic":
            parameters[k] = (st.slider("Complessità:", 1, 10, 5), total_ticks)
        elif k == "Stochastic":
            parameters[k] = (st.slider("Densità %:", 10, 100, 50), (120, 960), total_ticks)
        elif k == "Brownian":
            parameters[k] = (st.slider("Salto Max:", 1, 12, 4), total_ticks)
        elif k == "Cellular":
            parameters[k] = (st.slider("Velocità:", 1, 8, 4), total_ticks)
        elif k == "Fractal":
            parameters[k] = (st.slider("Iterazioni:", 1, 3, 2), total_ticks)

    if st.button("🎶 DECOMPONI MIDI", type="primary", use_container_width=True):
        current_midi = midi_data
        for k in selected_keys:
            p = parameters[k]
            if k == "Remapper": current_midi = midi_note_remapper(current_midi, *p)
            elif k == "Phrases": current_midi = midi_phrase_reconstructor(current_midi, *p)
            elif k == "Scrambler": current_midi = midi_time_scrambler(current_midi, *p)
            elif k == "Genetic": current_midi = midi_genetic_shuffler(current_midi, *p)
            elif k == "Stochastic": current_midi = midi_stochastic_composer(current_midi, *p)
            elif k == "Brownian": current_midi = midi_brownian_walker(current_midi, *p)
            elif k == "Cellular": current_midi = midi_cellular_automata(current_midi, *p)
            elif k == "Fractal": current_midi = midi_fractal_generator(current_midi, *p)
        
        buf = io.BytesIO()
        current_midi.save(file=buf)
        st.success("Decomposizione completata!")
        st.download_button("💾 Scarica Risultato Completo", buf.getvalue(), "decomposed.mid", use_container_width=True)

        # Gestione download singole tracce (Tua logica originale)
        st.markdown("---")
        st.subheader("Scarica Singole Tracce")
        for i, track in enumerate(current_midi.tracks):
            track_name = next((msg.name for msg in track if msg.type == 'track_name'), f"Traccia {i}")
            single_mid = mido.MidiFile(ticks_per_beat=current_midi.ticks_per_beat)
            single_mid.tracks.append(track)
            s_buf = io.BytesIO()
            single_mid.save(file=s_buf)
            st.download_button(f"Scarica {track_name}", s_buf.getvalue(), f"track_{i}.mid", key=f"btn_{i}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>MIDI Decomposer by loop507 - Algorithmic Edition</div>", unsafe_allow_html=True)
