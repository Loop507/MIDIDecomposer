# midi_decomposer_app.py - VERSIONE AGGIORNATA CON CONTROLLO VELOCITÀ ESTESO NEL TIME SCRAMBLER

import streamlit as st
import mido
import random
import numpy as np
import io # Necessario per gestire i file in memoria

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
    # Mappa le note di base a un offset semitonale (C=0, C#=1, D=2...)
    note_offsets = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    # Estrai la nota base (es. 'C' da 'Cm')
    base_note_char = key_name[0]
    sharp_flat_char = ''
    if len(key_name) > 1 and (key_name[1] == '#' or key_name[1] == 'b'):
        sharp_flat_char = key_name[1]
    
    base_note_name = base_note_char + sharp_flat_char
    
    offset = note_offsets.get(base_note_name, 0) # Ottieni l'offset della nota base

    # Le tonalità minori non influenzano l'offset di base della radice per la trasposizione,
    # ma sono utili per la logica di scalatura (gestita nella funzione remapper)
    return offset

def get_scale_notes(scale_name):
    """Restituisce gli intervalli (in semitoni) di una scala rispetto alla sua radice."""
    scales = {
        "Cromatica": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Maggiore": [0, 2, 4, 5, 7, 9, 11], # W W H W W W H
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10], # W H W W H W W
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Blues": [0, 3, 5, 6, 7, 10]
    }
    return scales.get(scale_name, scales["Cromatica"]) # Default alla scala cromatica se non trovata

# --- Funzioni di Decomposizione ---

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    """
    Rimodella le note MIDI in base a una scala, tonalità e randomizzazione di pitch/velocity.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat) # Mantiene i ticks per beat originali
    
    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name) # Offset della tonalità di destinazione

    for i, track in enumerate(original_midi.tracks):
        new_track = mido.MidiTrack()
        current_time = 0 # Tempo corrente in ticks per la traccia
        
        for msg in track:
            # Aggiorna il tempo corrente
            current_time += msg.time
            
            if msg.type == 'note_on' or msg.type == 'note_off':
                original_note = msg.note
                
                # 1. Applica Pitch Shift randomico (se > 0)
                shifted_note = original_note
                if pitch_shift_range > 0:
                    shifted_note += random.randint(-pitch_shift_range, pitch_shift_range)
                
                # Clampa tra 0-127
                shifted_note = max(0, min(127, shifted_note))

                # 2. Adatta alla scala target e tonalità
                # Calcola la nota base nella scala (0-11)
                note_in_octave = (shifted_note - key_offset) % 12
                # Assicurati che il resto sia positivo
                if note_in_octave < 0:
                    note_in_octave += 12 
                
                # Trova la nota della scala più vicina all'interno dell'ottava
                closest_scale_interval = min(target_scale_intervals, key=lambda x: abs(note_in_octave - x))
                
                # Calcola la nuova altezza della nota
                # Trova l'ottava della nota originale + l'offset della tonalità
                octave = (shifted_note - key_offset) // 12 
                new_note_pitch = octave * 12 + closest_scale_interval + key_offset

                # Assicurati che la nuova nota sia all'interno del range MIDI (0-127)
                new_note_pitch = max(0, min(127, new_note_pitch))
                
                # 3. Applica randomizzazione velocity (se > 0)
                new_velocity = msg.velocity
                if msg.type == 'note_on' and velocity_randomization > 0:
                    # Applica una variazione percentuale
                    # (1 + random.uniform(-V/100, V/100))
                    new_velocity_float = float(new_velocity) * (1 + random.uniform(-velocity_randomization/100, velocity_randomization/100))
                    new_velocity = int(round(new_velocity_float))
                    new_velocity = max(1, min(127, new_velocity)) # Clampa tra 1-127 per note_on

                # Crea il nuovo messaggio con le modifiche
                new_msg = msg.copy(note=new_note_pitch, velocity=new_velocity)
                new_track.append(new_msg)
            else:
                # Copia gli altri messaggi (control change, program change, tempo, etc.) senza modificarli
                new_track.append(msg.copy())
        new_midi.tracks.append(new_track)
    return new_midi


def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style):
    """
    Riorganizza le frasi MIDI.
    Divide ogni traccia in segmenti (frasi) e li riassembla in un nuovo ordine.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    # Determinare il tempo in microsecondi per beat (per gestire set_tempo)
    # Assumiamo un tempo costante per semplicità. Se ci sono più messaggi set_tempo, prenderemo l'ultimo prima del beat.
    # Per una gestione più precisa, si dovrebbe costruire una mappa del tempo.
    tempo_bpm = 120 # Default
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)
                break # Prendiamo il primo, o l'ultimo per semplicità
    
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats

    if ticks_per_phrase == 0:
        st.warning("La lunghezza della frase è zero. Nessuna riorganizzazione applicata.")
        return original_midi

    for original_track in original_midi.tracks:
        phrases = []
        current_phrase_events = []
        current_phrase_start_tick = 0

        # Raccogli tutti gli eventi con tempo assoluto e poi segmenta
        events_with_abs_time = []
        time_since_last_event = 0
        for msg in original_track:
            time_since_last_event += msg.time
            events_with_abs_time.append({'msg': msg, 'abs_time': time_since_last_event})

        # Segmenta in frasi
        for event_data in events_with_abs_time:
            msg = event_data['msg']
            abs_time = event_data['abs_time']

            # Se l'evento supera il limite della frase corrente
            while abs_time >= current_phrase_start_tick + ticks_per_phrase:
                if current_phrase_events: # Aggiungi la frase completa se non è vuota
                    phrases.append(current_phrase_events)
                current_phrase_events = [] # Inizia una nuova frase
                current_phrase_start_tick += ticks_per_phrase
            
            current_phrase_events.append(msg)
        
        # Aggiungi l'ultima frase incompleta se esiste
        if current_phrase_events:
            phrases.append(current_phrase_events)

        if not phrases:
            new_midi.tracks.append(mido.MidiTrack()) # Aggiungi una traccia vuota se non ci sono frasi
            continue # Passa alla prossima traccia

        # Riorganizza le frasi
        reorganized_phrases = []
        if reassembly_style == "Casuale":
            reorganized_phrases = list(phrases) # Copia la lista
            random.shuffle(reorganized_phrases)
        elif reassembly_style == "Inversione":
            reorganized_phrases = list(reversed(phrases))
        elif reassembly_style == "Ciclico A-B-A":
            if len(phrases) >= 3:
                a_phrase = phrases[0]
                b_phrase = phrases[1]
                c_phrase = phrases[2] if len(phrases) > 2 else phrases[1] # Usa B se non c'è C
                
                # CORREZIONE QUI: Usare append() per aggiungere intere frasi alla lista
                # invece di extend() che aggiunge i singoli messaggi, causando l'errore.
                num_repetitions = max(1, len(phrases) // 3) # Ripeti per avere lunghezza simile all'originale
                for _ in range(num_repetitions):
                    reorganized_phrases.append(a_phrase)
                    reorganized_phrases.append(b_phrase)
                    reorganized_phrases.append(a_phrase)
                    reorganized_phrases.append(c_phrase)
            else:
                st.warning(f"Troppo poche frasi ({len(phrases)}) per lo stile 'Ciclico A-B-A'. Verrà usata la riorganizzazione casuale.")
                reorganized_phrases = list(phrases)
                random.shuffle(reorganized_phrases)
        elif reassembly_style == "Dal Più Corto al Più Lungo":
            # IMPLEMENTAZIONE QUI:
            def get_phrase_duration_in_ticks(phrase_events_list):
                """Calcola la durata totale di una frase in tick MIDI."""
                if not phrase_events_list:
                    return 0
                # La durata è la somma dei delta time degli eventi nella frase
                return sum(msg.time for msg in phrase_events_list)

            # Ordina le frasi in base alla loro durata (dal più corto al più lungo)
            reorganized_phrases = sorted(phrases, key=get_phrase_duration_in_ticks)
        else:
            reorganized_phrases = list(phrases) # Fallback

        new_track = mido.MidiTrack()
        current_tick = 0
        
        # Ricalcola i delta time per la nuova traccia
        # Unisci tutti gli eventi delle frasi riorganizzate in una lista piatta di (messaggio, tempo_assoluto_nella_nuova_sequenza)
        # Questo è il modo più affidabile per ricostruire i delta time
        
        flat_events_for_reconstruction = []
        absolute_time_in_reorganized_seq = 0
        
        for phrase_block in reorganized_phrases:
            for msg_in_phrase in phrase_block:
                absolute_time_in_reorganized_seq += msg_in_phrase.time
                flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': absolute_time_in_reorganized_seq})

        # Ora che abbiamo tutti i messaggi con i loro tempi assoluti nella sequenza riorganizzata,
        # possiamo creare la nuova traccia ricalcolando i delta time
        last_abs_time = 0
        for event_data in flat_events_for_reconstruction:
            msg = event_data['msg']
            abs_time = event_data['abs_time']
            
            delta_time = abs_time - last_abs_time
            if delta_time < 0: # Difensivo, non dovrebbe succedere se ordinato correttamente
                delta_time = 0
            
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time = abs_time
            
        new_midi.tracks.append(new_track)
    return new_midi


def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    """
    Modifica il timing e la durata delle note MIDI.
    - stretch_factor: Allunga o comprime il tempo complessivo.
    - quantization_strength: Forza le note a "snappare" a una griglia ritmica (0-100%).
    - swing_amount: Aggiunge un "groove" swing alle note (0-100%).
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    
    # Define the base subdivision for quantization (e.g., 16th notes)
    ticks_per_subdivision = original_midi.ticks_per_beat / 4 # By default, 16th notes are our grid
    if original_midi.ticks_per_beat == 0 or ticks_per_subdivision == 0:
        st.warning("Ticks per beat è zero o troppo basso per applicare Time Scrambler. Restituito il MIDI originale.")
        return original_midi

    for track_index, original_track in enumerate(original_midi.tracks):
        new_track = mido.MidiTrack()
        events_with_abs_time = []
        current_abs_time_stretched = 0

        # Step 1: Accumulate absolute times, applying stretch factor to delta times
        for msg in original_track:
            stretched_delta_time = int(round(msg.time * stretch_factor))
            current_abs_time_stretched += stretched_delta_time
            # Store a copy of the message and its stretched absolute time
            events_with_abs_time.append({'msg': msg.copy(), 'abs_time_mod': current_abs_time_stretched})

        # Step 2: Apply Quantization and Swing to the modified absolute times
        if quantization_strength > 0:
            for event_data in events_with_abs_time:
                msg = event_data['msg']
                abs_time_before_quant = event_data['abs_time_mod']

                # Only apply quantization/swing to note-related messages
                if msg.type == 'note_on' or msg.type == 'note_off':
                    # Calculate the nearest grid point for quantization
                    snapped_abs_time = round(abs_time_before_quant / ticks_per_subdivision) * ticks_per_subdivision

                    # Apply Swing if active (for 'off-beat' subdivisions)
                    if swing_amount > 0:
                        # Determine which subdivision this snapped time falls into
                        # e.g., for 16th notes (4 per beat), subdivision_index 0, 1, 2, 3 per beat
                        subdivision_index_in_beat = int(round((snapped_abs_time % original_midi.ticks_per_beat) / ticks_per_subdivision))
                        
                        # Swing typically shifts the "even" (or "off") subdivisions
                        # For a 16th note grid, this affects subdivision_index 1 and 3 within a beat (0-indexed)
                        if subdivision_index_in_beat % 2 == 1: # If it's the 2nd or 4th 16th note of a beat
                            # Swing shift amount: percentage of half a subdivision
                            swing_shift_ticks = (ticks_per_subdivision / 2) * (swing_amount / 100.0)
                            snapped_abs_time += swing_shift_ticks
                            
                    # Apply quantization strength: interpolate between original (stretched) and snapped time
                    quant_factor = quantization_strength / 100.0
                    event_data['abs_time_mod'] = int(round(abs_time_before_quant * (1 - quant_factor) + snapped_abs_time * quant_factor))
                    
                    # Ensure absolute time is non-negative
                    event_data['abs_time_mod'] = max(0, event_data['abs_time_mod'])
        
        # Step 3: Sort events by their new absolute times and rebuild the track with new delta times
        events_with_abs_time.sort(key=lambda x: x['abs_time_mod'])

        last_abs_time_mod = 0
        for event_data in events_with_abs_time:
            msg = event_data['msg']
            abs_time_mod = event_data['abs_time_mod']

            delta_time = abs_time_mod - last_abs_time_mod
            if delta_time < 0: # Defensive check, should not happen after sort and max(0)
                delta_time = 0 
            
            # Append the original message type, but with the new delta time
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time_mod = abs_time_mod

        new_midi.tracks.append(new_track)
    return new_midi

def midi_density_transformer(original_midi, add_note_probability, remove_note_probability, polyphony_mode):
    """
    Aggiunge o rimuove note per alterare la densità MIDI.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for track_index, original_track in enumerate(original_midi.tracks):
        # 1. Convert track MIDI messages into a list of (start_abs_time, end_abs_time, pitch, velocity, channel) notes
        extracted_notes = [] 
        active_notes_map = {} # { (pitch, channel): {'start_abs_time': X, 'velocity': Y} }
        
        current_abs_time_track = 0
        for msg in original_track:
            current_abs_time_track += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes_map[(msg.note, msg.channel)] = {'start_abs_time': current_abs_time_track, 'velocity': msg.velocity}
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active_notes_map:
                    start_data = active_notes_map.pop(key)
                    extracted_notes.append({
                        'start': start_data['start_abs_time'],
                        'end': current_abs_time_track,
                        'pitch': msg.note,
                        'velocity': start_data['velocity'],
                        'channel': msg.channel
                    })
        # Handle any hanging notes (notes still active at the end of the track)
        for key, start_data in active_notes_map.items():
             extracted_notes.append({
                'start': start_data['start_abs_time'],
                'end': current_abs_time_track, # Assume it ends at the last event's time for this track
                'pitch': key[0],
                'velocity': start_data['velocity'],
                'channel': key[1]
            })

        modified_notes_for_track = [] # List of notes that will be used to generate MIDI events

        # 2. Phase: Remove Notes based on remove_note_probability
        for note in extracted_notes:
            if random.randint(0, 100) >= remove_note_probability: # Keep if random number is NOT below probability
                modified_notes_for_track.append(note)

        # Initialize list of final MIDI events with absolute times for this track
        final_events_for_track = [] 

        # 3. Phase: Add Drone Notes (if Drones mode selected and probability met)
        # This is added once per track, at the beginning, for the track's duration.
        if polyphony_mode == "Droni" and add_note_probability > 0:
            if random.randint(0, 100) < add_note_probability: # Roll dice for the drone
                drone_pitch = 36 # C3 (a common low drone pitch)
                drone_velocity = 64
                
                # Determine the effective end time of the track for drone duration
                track_effective_end_time = current_abs_time_track # Use the total absolute time of the original track
                # Ensure it's at least a minimum length if the track was very short or empty
                if track_effective_end_time < original_midi.ticks_per_beat * 4: 
                    track_effective_end_time = original_midi.ticks_per_beat * 4 # Default to 4 beats if too short
                
                final_events_for_track.append({'msg': mido.Message('note_on', note=drone_pitch, velocity=drone_velocity, channel=0, time=0), 'abs_time': 0})
                final_events_for_track.append({'msg': mido.Message('note_off', note=drone_pitch, velocity=0, channel=0, time=0), 'abs_time': track_effective_end_time + original_midi.ticks_per_beat * 4}) # Drone extends a bit beyond

        # 4. Phase: Add other types of notes (Triads, Counter-Melody) based on existing notes
        for note_data in modified_notes_for_track:
            # Add original note_on and note_off for the processed note
            final_events_for_track.append({'msg': mido.Message('note_on', note=note_data['pitch'], velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
            final_events_for_track.append({'msg': mido.Message('note_off', note=note_data['pitch'], velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})

            if random.randint(0, 100) < add_note_probability:
                if polyphony_mode == "Riempi Accordo (Triadi)":
                    # Add notes for a major triad above the current note
                    for interval in [4, 7]: # Major 3rd, Perfect 5th (for a major triad)
                        new_pitch = note_data['pitch'] + interval
                        if 0 <= new_pitch <= 127: # Ensure pitch is within MIDI range
                            final_events_for_track.append({'msg': mido.Message('note_on', note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
                            final_events_for_track.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})
                elif polyphony_mode == "Aggiungi Contro-Melodia":
                    # Simple harmony/counter-melody: add a note a few semitones away from the original
                    interval_choice = random.choice([-5, -3, -2, 2, 3, 5]) # Example intervals: P4 down, m3 down, m2 down, M2 up, M3 up, P4 up
                    new_pitch = note_data['pitch'] + interval_choice
                    if 0 <= new_pitch <= 127: # Ensure pitch is within MIDI range
                        final_events_for_track.append({'msg': mido.Message('note_on', note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
                        final_events_for_track.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})
                # "Nessuna" implies no additional notes added here.

        # 5. Phase: Sort all final events by absolute time and rebuild the MIDI track
        final_events_for_track.sort(key=lambda x: x['abs_time'])

        last_abs_time = 0
        new_track = mido.MidiTrack()
        for event_data in final_events_for_track:
            msg = event_data['msg']
            abs_time = event_data['abs_time']

            delta_time = abs_time - last_abs_time
            if delta_time < 0: # Defensive check, should not happen if sorted and logic is correct
                delta_time = 0 
            
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time = abs_time
        
        new_midi.tracks.append(new_track)
    return new_midi

# --- NUOVA FUNZIONE: MIDI Random Pitch Transformer ---
def midi_random_pitch_transformer(original_midi, random_pitch_strength):
    """
    Randomizes the pitch of notes based on a given strength (probability).
    Each note has a 'random_pitch_strength' % chance of having its pitch completely randomized (0-127).
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in original_track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                if random.randint(0, 100) < random_pitch_strength:
                    # Randomize pitch for this note
                    new_pitch = random.randint(0, 127) # Full MIDI pitch range
                    new_msg = msg.copy(note=new_pitch)
                else:
                    new_msg = msg.copy() # Keep original pitch
            else:
                new_msg = msg.copy() # Copy other messages as is
            new_track.append(new_msg)
        new_midi.tracks.append(new_track)
    return new_midi


# --- Sezione Upload File MIDI ---
st.subheader("🎵 Carica il tuo file MIDI (.mid o .midi)")
uploaded_midi_file = st.file_uploader(
    "Trascina qui il tuo file MIDI o clicca per sfogliare",
    type=["mid", "midi"], # Aggiunto .midi per maggiore compatibilità
    help="Carica un file MIDI per iniziare la decomposizione."
)

decomposed_midi_file = None # Variabile per memorizzare il MIDI decomposto

if uploaded_midi_file is not None:
    st.success("File MIDI caricato con successo!")

    try:
        midi_data = mido.MidiFile(file=uploaded_midi_file)

        st.subheader("File MIDI Caricato: Panoramica")
        st.write(f"Nome file: **{uploaded_midi_file.name}**")
        st.write(f"Numero di tracce: **{len(midi_data.tracks)}**")
        # Per la durata MIDI è più complesso di un semplice .length, spesso è basato sui tick/tempo map
        # Per ora usiamo il length di mido che è una stima in secondi
        st.write(f"Durata (stimata): **{midi_data.length:.2f} secondi**")

        st.markdown("---")
        st.subheader("⚙️ Scegli il Metodo di Decomposizione MIDI")

        midi_methods = {
            "MIDI Note Remapper": "🎶 Remapping di Note (Verticale)",
            "MIDI Phrase Reconstructor": "🔄 Riorganizzazione Frasi (Orizzontale)",
            "MIDI Time Scrambler": "⏳ Manipolazione Ritmo/Durata (Orizzontale)",
            "MIDI Density Transformer": "🎲 Controllo Densità (Armonia/Contrappunto)",
            "MIDI Random Pitch Transformer": "❓ Randomizzazione Totale Pitch (Caos)"
        }

        selected_midi_method = st.selectbox(
            "Seleziona il Metodo di Decomposizione:",
            list(midi_methods.keys()),
            format_func=lambda x: midi_methods[x],
            help="Scegli come vuoi decomporre le note e la struttura del tuo MIDI."
        )

        # --- Controlli specifici per il metodo selezionato ---
        st.markdown("#### Parametri per il Metodo Selezionato:")
        if selected_midi_method == "MIDI Note Remapper":
            col1_remap, col2_remap = st.columns(2)
            with col1_remap:
                target_scale = st.selectbox(
                    "Scala Target:",
                    ["Cromatica", "Maggiore", "Minore Naturale", "Pentatonica Maggiore", "Blues"],
                    help="Le note verranno adattate a questa scala."
                )
            with col2_remap:
                # Mido supporta nomi di chiave come 'C', 'C#m', 'Db' ecc.
                # Questa lista è un esempio, si potrebbe espanderla
                target_key = st.selectbox(
                    "Tonalità Target:",
                    ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                     'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'],
                    index=0, # Default C
                    help="La tonalità di destinazione per la decomposizione."
                )
            pitch_shift_range = st.slider(
                "Range Pitch Shift Randomico (semitoni):", 0, 12, 0,
                help="Aggiunge un offset casuale alle note prima di adattarle alla scala."
            )
            velocity_randomization = st.slider(
                "Percentuale Randomizzazione Velocity:", 0, 100, 0,
                help="Varia casualmente il volume delle note."
            )

        elif selected_midi_method == "MIDI Phrase Reconstructor":
            phrase_length_beats = st.slider(
                "Lunghezza Frase (battute):", 1, 16, 4,
                help="Definisce la dimensione dei blocchi musicali (in battute) da riorganizzare."
            )
            reassembly_style = st.selectbox(
                "Stile Riorganizzazione Frasi:",
                ["Casuale", "Inversione", "Ciclico A-B-A", "Dal Più Corto al Più Lungo"],
                index=0, # Default a Casuale
                help="Come le frasi verranno riassemblate. Dal Più Corto al Più Lungo è ora implementato."
            )

        elif selected_midi_method == "MIDI Time Scrambler":
            # NUOVO CONTROLLO: Velocità di Esecuzione
            execution_speed_preset = st.selectbox(
                "Velocità di Esecuzione:",
                ["Medio (Originale)", "Lento (Metà velocità)", "Molto Lento (Un quarto velocità)", "Veloce (Doppia velocità)", "Molto Veloce (Quattro volte velocità)"],
                index=0, # Default a Medio
                help="Imposta un preset per lo stiramento/compressione del tempo."
            )
            
            # Mappa il preset al valore di default dello slider
            default_stretch_factor = 1.0
            if execution_speed_preset == "Lento (Metà velocità)":
                default_stretch_factor = 2.0
            elif execution_speed_preset == "Molto Lento (Un quarto velocità)":
                default_stretch_factor = 4.0
            elif execution_speed_preset == "Veloce (Doppia velocità)":
                default_stretch_factor = 0.5
            elif execution_speed_preset == "Molto Veloce (Quattro volte velocità)":
                default_stretch_factor = 0.25

            stretch_factor = st.slider(
                "Fattore di Stiramento/Compressione (Time Warp):", 0.1, 5.0, default_stretch_factor, 0.1,
                help="Allunga (valori > 1) o comprime (valori < 1) il tempo generale del MIDI. Regola per fine-tuning."
            )
            quantization_strength = st.slider(
                "Forza Quantizzazione (0=libero, 100=rigido):", 0, 100, 50,
                help="Quanto le note verranno 'allineate' a una griglia ritmica (es. a 16esimi)."
            )
            swing_amount = st.slider(
                "Quantità di Swing (%):", 0, 100, 0,
                help="Aggiunge un 'groove' swing ritardando alcune suddivisioni (solo con quantizzazione attiva)."
            )

        elif selected_midi_method == "MIDI Density Transformer":
            add_note_probability = st.slider(
                "Probabilità di Aggiungere Note (%):", 0, 50, 0,
                help="Probabilità di inserire nuove note basate su armonie esistenti."
            )
            remove_note_probability = st.slider(
                "Probabilità di Rimuovere Note (%):", 0, 50, 0,
                help="Probabilità di eliminare note esistenti, per un suono più spoglio."
            )
            polyphony_mode = st.selectbox(
                "Modalità Polifonia Aggiuntiva:",
                ["Nessuna", "Riempi Accordo (Triadi)", "Aggiungi Contro-Melodia", "Droni"],
                help="Come le nuove note verranno generate per influenzare la densità."
            )
        
        elif selected_midi_method == "MIDI Random Pitch Transformer":
            random_pitch_strength = st.slider(
                "Forza Randomizzazione Pitch (%):", 0, 100, 100,
                help="Probabilità che ogni nota (on/off) abbia il suo pitch completamente randomizzato (0-127)."
            )


        if st.button("🎶 DECOMPONI MIDI", type="primary", use_container_width=True):
            with st.spinner(f"Applicando {midi_methods[selected_midi_method]}..."):
                # Esegui la funzione di decomposizione appropriata
                if selected_midi_method == "MIDI Note Remapper":
                    decomposed_midi_file = midi_note_remapper(
                        midi_data, target_scale, target_key, int(pitch_shift_range), int(velocity_randomization)
                    )
                elif selected_midi_method == "MIDI Phrase Reconstructor":
                    decomposed_midi_file = midi_phrase_reconstructor(
                        midi_data, phrase_length_beats, reassembly_style
                    )
                elif selected_midi_method == "MIDI Time Scrambler":
                    decomposed_midi_file = midi_time_scrambler(
                        midi_data, stretch_factor, quantization_strength, swing_amount
                    )
                elif selected_midi_method == "MIDI Density Transformer":
                    decomposed_midi_file = midi_density_transformer(
                        midi_data, add_note_probability, remove_note_probability, polyphony_mode
                    )
                elif selected_midi_method == "MIDI Random Pitch Transformer":
                    decomposed_midi_file = midi_random_pitch_transformer(
                        midi_data, random_pitch_strength
                    )

                if decomposed_midi_file:
                    st.success("Decomposizione MIDI completata!")
                    st.subheader("Scarica il tuo MIDI Decomposto (Completo)")

                    # Salva il file MIDI decomposto completo in un buffer di memoria per il download
                    decomposed_midi_bytes = io.BytesIO()
                    decomposed_midi_file.save(file=decomposed_midi_bytes)
                    decomposed_midi_bytes.seek(0)

                    st.download_button(
                        label="💾 Scarica MIDI Decomposto Completo",
                        data=decomposed_midi_bytes,
                        file_name=f"{uploaded_midi_file.name.split('.')[0]}_{selected_midi_method.replace(' ', '_')}_Decomposed.mid",
                        mime="audio/midi",
                        use_container_width=True
                    )
                    st.info("Apri il file MIDI scaricato nel tuo software musicale preferito per ascoltare il risultato completo!")

                    # --- NUOVA SEZIONE: Download Singole Tracce ---
                    # Per ottenere i nomi delle tracce in modo più robusto
                    def get_track_display_name(track, index):
                        track_name = ""
                        for msg in track:
                            if msg.type == 'track_name':
                                track_name = msg.name
                                break
                        return f"Traccia {index}: {track_name if track_name else '(Senza Nome)'}"

                    if len(decomposed_midi_file.tracks) > 0: # Controlla se ci sono tracce
                        st.markdown("---")
                        st.subheader("Scarica Singole Tracce del MIDI Decomposto")
                        
                        track_options = [get_track_display_name(track, i) for i, track in enumerate(decomposed_midi_file.tracks)]
                        
                        selected_tracks_indices = st.multiselect(
                            "Seleziona una o più tracce da scaricare singolarmente:",
                            options=list(range(len(decomposed_midi_file.tracks))), # Usiamo gli indici come valori
                            format_func=lambda x: track_options[x], # Mostriamo i nomi delle tracce
                            default=None, # Nessuna selezione di default
                            help="Seleziona le tracce che vuoi scaricare come file MIDI separati."
                        )

                        if selected_tracks_indices:
                            for track_index in selected_tracks_indices:
                                single_track_midi = mido.MidiFile()
                                single_track_midi.tracks.append(decomposed_midi_file.tracks[track_index])

                                single_track_bytes = io.BytesIO()
                                single_track_midi.save(file=single_track_bytes)
                                single_track_bytes.seek(0)

                                # Genera un nome file sensato
                                original_file_base_name = uploaded_midi_file.name.split('.')[0]
                                method_name_for_file = selected_midi_method.replace(' ', '_')
                                track_name_for_file = get_track_display_name(decomposed_midi_file.tracks[track_index], track_index).replace(' ', '_').replace(':', '')

                                st.download_button(
                                    label=f"💾 Scarica {track_options[track_index]}",
                                    data=single_track_bytes,
                                    file_name=f"{original_file_base_name}_{method_name_for_file}_{track_name_for_file}.mid",
                                    mime="audio/midi",
                                    key=f"download_track_{track_index}" # Chiave unica per ogni pulsante
                                )
                    else:
                        st.info("Il MIDI decomposto non contiene tracce valide da scaricare singolarmente.")


                else:
                    st.error("Impossibile generare il MIDI decomposto. Controlla i messaggi di avviso.")

    except Exception as e:
        st.error(f"❌ Errore durante la lettura o l'elaborazione del file MIDI: {str(e)}")
        st.error("Assicurati che sia un file MIDI valido (.mid o .midi) e riprova.")
        st.exception(e) # Mostra i dettagli completi dell'errore per il debug

else:
    st.info("👆 Carica un file MIDI (.mid o (.midi) per iniziare la decomposizione.")

    with st.expander("📖 Come usare MIDI Decomposer"):
        st.markdown("""
        ### Benvenuto in MIDI Decomposer!

        Qui potrai caricare i tuoi file MIDI e applicare diverse tecniche di decomposizione per creare nuove strutture musicali.

        **Come funziona:**

        1.  **Carica il tuo file MIDI** (con estensione `.mid` o `.midi`).
        2.  Scegli il **metodo di decomposizione** e imposta i suoi **parametri**.
        3.  Clicca su **"DECOMPONI MIDI"**.
        4.  Scarica il **file MIDI completo** o seleziona le **singole tracce** da scaricare.
        5.  Apri il file MIDI scaricato nel tuo software musicale (DAW) preferito per ascoltare il risultato.

        **Combinare gli Effetti:**
        Se desideri applicare più effetti (es. prima un remapping di note e poi un cambio di velocità), dovrai:
        1. Applicare il primo effetto e scaricare il file MIDI risultante.
        2. Ricaricare questo nuovo file MIDI nel Decomposer.
        3. Applicare il secondo effetto (es. "MIDI Time Scrambler" per la velocità).

        **Metodi di Decomposizione Disponibili:**

        * **🎶 MIDI Note Remapper**: Rimodella le note del pentagramma (verticale) in base a scale, tonalità e randomizzazione.
            * _Parametri:_ Scala Target (es. Maggiore, Cromatica), Tonalità Target (es. C, Am), Range Pitch Shift Randomico, Percentuale Randomizzazione Velocity.
        * **🔄 MIDI Phrase Reconstructor**: Riorganizza e ricompone blocchi o "frasi" musicali (orizzontale).
            * _Parametri:_ Lunghezza Frase (battute), Stile Riorganizzazione Frasi (Casuale, Inversione, Ciclico A-B-A, Dal Più Corto al Più Lungo).
        * **⏳ MIDI Time Scrambler**: Modifica il timing e la durata delle note per creare nuovi groove.
            * _Parametri:_ **Velocità di Esecuzione (Lento, Medio, Veloce)**, Fattore di Stiramento/Compressione, Forza Quantizzazione, Quantità di Swing.
        * **🎲 MIDI Density Transformer**: Aggiunge o rimuove note per alterare la densità armonica.
        * **❓ MIDI Random Pitch Transformer**: Randomizza completamente l'altezza di ogni nota (pitch) per un caos melodico.
        
        **Risoluzione Problemi:**
        
        - Assicurati che il file sia un MIDI valido (.mid o .midi)
        - Controlla che il file non sia corrotto
        - Prova con un file MIDI più semplice se hai problemi
        """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><em>MIDI Decomposer by loop507</em></p>
    <p>Sperimenta la destrutturazione MIDI</p>
    <p style='font-size: 0.8em;'>Powered by Streamlit & Mido</p>
</div>
""", unsafe_allow_html=True)
