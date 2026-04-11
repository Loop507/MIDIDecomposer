# midi_decomposer_app.py - VERSIONE INTEGRALE DEFINITIVA

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
    clean_key = key_name.replace('m', '')
    return note_offsets.get(clean_key, 0)

def get_scale_notes(scale_name):
    scales = {
        "Cromatica": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
        new_msg = event['msg'].copy(time=delta_time)
        new_track.append(new_msg)
        last_abs_time = event['abs_time']
    return new_track

# --- MODULI ORIGINALI ---

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name)
    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                new_note = max(0, min(127, msg.note + random.randint(-pitch_shift_range, pitch_shift_range)))
                note_in_octave = (new_note - key_offset) % 12
                closest_scale = min(target_scale_intervals, key=lambda x: abs(note_in_octave - x))
                new_pitch = ((new_note - key_offset) // 12) * 12 + closest_scale + key_offset
                new_vel = msg.velocity
                if msg.type == 'note_on' and velocity_randomization > 0:
                    new_vel = max(1, min(127, int(msg.velocity * (1 + random.uniform(-velocity_randomization/100, velocity_randomization/100)))))
                new_track.append(msg.copy(note=max(0, min(127, new_pitch)), velocity=new_vel))
            else: new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi

def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats
    for track in original_midi.tracks:
        notes = extract_notes(track)
        if not notes:
            new_midi.tracks.append(track)
            continue
        max_time = max(n['end'] for n in notes)
        phrases = []
        for t in range(0, max_time, ticks_per_phrase):
            p_notes = [n for n in notes if t <= n['start'] < t + ticks_per_phrase]
            if p_notes: phrases.append((t, p_notes))
        if reassembly_style == "Casuale": random.shuffle(phrases)
        elif reassembly_style == "Inversione": phrases.reverse()
        new_notes = []
        curr_offset = 0
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
            new_midi.tracks.append(track)
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

# --- NUOVI MODULI ALGORITMICI (INTEGRATI) ---

def midi_stochastic_composer(original_midi, density, dur_range):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.
