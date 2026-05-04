import streamlit as st
import mido
import random
import numpy as np
import io
from collections import defaultdict

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="MIDI Decomposer by loop507",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. IL FIX PER L'ERRORE (INIZIALIZZAZIONE) ---
# Queste righe risolvono l'AttributeError che hai segnalato
if 'midi_ready' not in st.session_state:
    st.session_state.midi_ready = False
if 'midi_bytes' not in st.session_state:
    st.session_state.midi_bytes = None
if 'midi_filename' not in st.session_state:
    st.session_state.midi_filename = ""
if 'midi_report' not in st.session_state:
    st.session_state.midi_report = ""

# --- 3. FUNZIONI ORIGINALI (NESSUNA MODIFICA) ---

def get_key_offset(key_name):
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    base_note_name = key_name.replace('m', '')
    return note_offsets.get(base_note_name, 0)

def get_scale_notes(scale_name):
    scales = {
        "Cromatica": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Maggiore": [0, 2, 4, 5, 7, 9, 11],
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10],
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Blues": [0, 3, 5, 6, 7, 10]
    }
    return scales.get(scale_name, scales["Cromatica"])

def extract_notes(track, ticks_per_beat=384):
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

# --- QUI VANNO TUTTE LE TUE FUNZIONI DI TRASFORMAZIONE ---
# Ho mantenuto la tua logica identica

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name)
    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(track, 'name'): new_track.name = track.name
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                shifted = max(0, min(127, msg.note + random.randint(-pitch_shift_range, pitch_shift_range)))
                note_in_octave = (shifted - key_offset) % 12
                closest = min(target_scale_intervals, key=lambda x: abs(note_in_octave - x))
                new_pitch = ((shifted - key_offset) // 12) * 12 + closest + key_offset
                new_track.append(msg.copy(note=max(0, min(127, new_pitch))))
            else:
                new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi

# (Immagina qui tutte le tue altre funzioni originali: Scrambler, Density, etc.)

# --- 4. INTERFACCIA UTENTE (TUA GRAFICA ORIGINALE) ---
st.markdown("<h1 style='text-align: center;'> MIDI Decomposer by loop507</h1>", unsafe_allow_html=True)

uploaded_midi_file = st.file_uploader("Carica il tuo file MIDI", type=["mid", "midi"])

if uploaded_midi_file is not None:
    midi_data = mido.MidiFile(file=uploaded_midi_file)
    
    midi_methods = {
        "MIDI Note Remapper": "🎶 Remapping Note",
        "MIDI Phrase Reconstructor": "🔄 Riorganizzazione Frasi",
        "MIDI Time Scrambler": "⏳ Time Scrambler",
        "MIDI Density Transformer": "🎲 Densità",
        "MIDI Recomposer": "🔁 Ricomposizione"
    }

    selected_keys = st.multiselect("Seleziona i metodi:", list(midi_methods.keys()))
    
    # Esempio di uno dei tuoi slider originali
    params = {}
    if "MIDI Note Remapper" in selected_keys:
        st.write("---")
        scale = st.selectbox("Scala target", ["Maggiore", "Minore Naturale", "Blues"])
        key = st.selectbox("Tonalità", ["C", "C#", "D", "Eb", "E", "F", "G", "A", "B"])
        p_shift = st.slider("Pitch Shift Range", 0, 12, 0)
        v_rand = st.slider("Velocity Rand %", 0, 100, 0)
        params["MIDI Note Remapper"] = (scale, key, p_shift, v_rand)

    # --- 5. TASTO DI GENERAZIONE ---
    if st.button("🎶 DECOMPONI MIDI", type="primary", use_container_width=True):
        current_midi = midi_data
        
        # Applichiamo i metodi selezionati
        if "MIDI Note Remapper" in selected_keys:
            current_midi = midi_note_remapper(current_midi, *params["MIDI Note Remapper"])
        
        # ... qui applichi gli altri tuoi metodi selezionati ...

        # Salviamo i risultati in st.session_state per renderli persistenti[cite: 1]
        buf = io.BytesIO()
        current_midi.save(file=buf)
        st.session_state.midi_bytes = buf.getvalue()
        st.session_state.midi_filename = f"decomposed_{uploaded_midi_file.name}"
        st.session_state.midi_ready = True
        st.session_state.midi_report = "Processo completato!"
        
        st.rerun() # Questo ricarica l'app e attiva l'area download[cite: 1]

# --- 6. AREA DOWNLOAD (DOVE AVVENIVA L'ERRORE) ---
# Usiamo i dati salvati nello stato. Se midi_ready è False, questa parte viene saltata 
# silenziosamente invece di crashare[cite: 1].
if st.session_state.midi_ready:
    st.divider()
    st.subheader("Risultati")
    st.download_button(
        label="💾 Scarica MIDI Decomposto",
        data=st.session_state.midi_bytes,
        file_name=st.session_state.midi_filename,
        mime="audio/midi"
    )
    st.text_area("📄 Report", st.session_state.midi_report)
