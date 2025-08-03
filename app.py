# midi_decomposer_app.py - VERSIONE AGGIORNATA CON TUTTE LE MODIFICHE RICHIESTE

import streamlit as st
import mido
import random
import numpy as np
import io

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
        current_time = 0
        
        for msg in track:
            current_time += msg.time
            
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
    """
    Riorganizza le frasi MIDI.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    tempo_bpm = 120
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)
                break
    
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats

    if ticks_per_phrase == 0:
        st.warning("La lunghezza della frase è zero. Nessuna riorganizzazione applicata.")
        return original_midi

    for original_track in original_midi.tracks:
        phrases = []
        current_phrase_events = []
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
                a_phrase = phrases[0]
                b_phrase = phrases[1]
                c_phrase = phrases[2] if len(phrases) > 2 else phrases[1]
                
                num_repetitions = max(1, len(phrases) // 3)
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
            def get_phrase_duration_in_ticks(phrase_events_list):
                if not phrase_events_list:
                    return 0
                return sum(msg.time for msg in phrase_events_list)
            reorganized_phrases = sorted(phrases, key=get_phrase_duration_in_ticks)
        else:
            reorganized_phrases = list(phrases)

        new_track = mido.MidiTrack()
        current_tick = 0
        
        flat_events_for_reconstruction = []
        absolute_time_in_reorganized_seq = 0
        
        for phrase_block in reorganized_phrases:
            for msg_in_phrase in phrase_block:
                absolute_time_in_reorganized_seq += msg_in_phrase.time
                flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': absolute_time_in_reorganized_seq})

        last_abs_time = 0
        for event_data in flat_events_for_reconstruction:
            msg = event_data['msg']
            abs_time = event_data['abs_time']
            
            delta_time = abs_time - last_abs_time
            if delta_time < 0:
                delta_time = 0
            
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
    if original_midi.ticks_per_beat == 0 or ticks_per_subdivision == 0:
        st.warning("Ticks per beat è zero o troppo basso per applicare Time Scrambler. Restituito il MIDI originale.")
        return original_midi

    for track_index, original_track in enumerate(original_midi.tracks):
        new_track = mido.MidiTrack()
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

                if msg.type == 'note_on' or msg.type == 'note_off':
                    snapped_abs_time = round(abs_time_before_quant / ticks_per_subdivision) * ticks_per_subdivision

                    if swing_amount > 0:
                        subdivision_index_in_beat = int(round((snapped_abs_time % original_midi.ticks_per_beat) / ticks_per_subdivision))
                        
                        if subdivision_index_in_beat % 2 == 1:
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

            delta_time = abs_time_mod - last_abs_time_mod
            if delta_time < 0:
                delta_time = 0 
            
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
        extracted_notes = [] 
        active_notes_map = {}
        
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
                        'channel': key[1]
                    })
        for key, start_data in active_notes_map.items():
             extracted_notes.append({
                'start': start_data['start_abs_time'],
                'end': current_abs_time_track,
                'pitch': key[0],
                'velocity': start_data['velocity'],
                'channel': key[1]
            })

        modified_notes_for_track = []

        for note in extracted_notes:
            if random.randint(0, 100) >= remove_note_probability:
                modified_notes_for_track.append(note)

        final_events_for_track = []

        if polyphony_mode == "Droni" and add_note_probability > 0:
            if random.randint(0, 100) < add_note_probability:
                drone_pitch = 36
                drone_velocity = 64
                
                track_effective_end_time = current_abs_time_track
                if track_effective_end_time < original_midi.ticks_per_beat * 4: 
                    track_effective_end_time = original_midi.ticks_per_beat * 4
                
                final_events_for_track.append({'msg': mido.Message('note_on', note=drone_pitch, velocity=drone_velocity, channel=0, time=0), 'abs_time': 0})
                final_events_for_track.append({'msg': mido.Message('note_off', note=drone_pitch, velocity=0, channel=0, time=0), 'abs_time': track_effective_end_time + original_midi.ticks_per_beat * 4})

        for note_data in modified_notes_for_track:
            final_events_for_track.append({'msg': mido.Message('note_on', note=note_data['pitch'], velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
            final_events_for_track.append({'msg': mido.Message('note_off', note=note_data['pitch'], velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})

            if random.randint(0, 100) < add_note_probability:
                if polyphony_mode == "Riempi Accordo (Triadi)":
                    for interval in [4, 7]:
                        new_pitch = note_data['pitch'] + interval
                        if 0 <= new_pitch <= 127:
                            final_events_for_track.append({'msg': mido.Message('note_on', note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
                            final_events_for_track.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})
                elif polyphony_mode == "Aggiungi Contro-Melodia":
                    interval_choice = random.choice([-5, -3, -2, 2, 3, 5])
                    new_pitch = note_data['pitch'] + interval_choice
                    if 0 <= new_pitch <= 127:
                        final_events_for_track.append({'msg': mido.Message('note_on', note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_data['start']})
                        final_events_for_track.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=note_data['channel'], time=0), 'abs_time': note_data['end']})

        final_events_for_track.sort(key=lambda x: x['abs_time'])

        last_abs_time = 0
        new_track = mido.MidiTrack()
        for event_data in final_events_for_track:
            msg = event_data['msg']
            abs_time = event_data['abs_time']

            delta_time = abs_time - last_abs_time
            if delta_time < 0:
                delta_time = 0
            
            new_msg = msg.copy(time=delta_time)
            new_track.append(new_msg)
            last_abs_time = abs_time
        
        new_midi.tracks.append(new_track)
    return new_midi

def midi_random_pitch_transformer(original_midi, random_pitch_strength):
    """
    Randomizes the pitch of notes based on a given strength (probability).
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in original_track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                if random.randint(0, 100) < random_pitch_strength:
                    new_pitch = random.randint(0, 127)
                    new_msg = msg.copy(note=new_pitch)
                else:
                    new_msg = msg.copy()
            else:
                new_msg = msg.copy()
            new_track.append(new_msg)
        new_midi.tracks.append(new_track)
    return new_midi

def midi_add_rhythmic_base(original_midi, kick, snare, hihat):
    """
    Aggiunge una nuova traccia con una base ritmica, generata come un brano normale.
    """
    new_midi = original_midi.copy()
    new_drum_track = mido.MidiTrack()
    
    # Mappa le note MIDI standard della batteria (GM Drum Map)
    DRUM_MAP = {
        "kick": 36,     # Cassa (C1)
        "snare": 38,    # Rullante (D1)
        "hihat_closed": 42, # Hi-hat chiuso (F#1)
    }
    
    ticks_per_beat = new_midi.ticks_per_beat
    
    rhythmic_pattern = []
    
    if kick:
        # Pattern cassa in 4/4
        rhythmic_pattern.append({'note': "kick", 'start_tick': 0, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
        rhythmic_pattern.append({'note': "kick", 'start_tick': ticks_per_beat * 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
    if snare:
        # Rullante su 2 e 4
        rhythmic_pattern.append({'note': "snare", 'start_tick': ticks_per_beat, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
        rhythmic_pattern.append({'note': "snare", 'start_tick': ticks_per_beat * 3, 'duration_ticks': ticks_per_beat // 8, 'velocity': 100})
    if hihat:
        # Hi-hat sugli ottavi
        rhythmic_pattern.append({'note': "hihat_closed", 'start_tick': ticks_per_beat // 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 80})
        rhythmic_pattern.append({'note': "hihat_closed", 'start_tick': ticks_per_beat + ticks_per_beat // 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 80})
        rhythmic_pattern.append({'note': "hihat_closed", 'start_tick': ticks_per_beat * 2 + ticks_per_beat // 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 80})
        rhythmic_pattern.append({'note': "hihat_closed", 'start_tick': ticks_per_beat * 3 + ticks_per_beat // 2, 'duration_ticks': ticks_per_beat // 8, 'velocity': 80})

    
    total_ticks = new_midi.length * new_midi.ticks_per_beat
    
    new_drum_track.append(mido.Message('program_change', program=0, channel=9, time=0))
    
    current_time_ticks = 0
    all_drum_events = []
    
    while current_time_ticks < total_ticks:
        for event in rhythmic_pattern:
            all_drum_events.append({
                'msg': mido.Message('note_on', note=DRUM_MAP[event['note']], velocity=event['velocity'], channel=9),
                'abs_time': current_time_ticks + event['start_tick']
            })
            all_drum_events.append({
                'msg': mido.Message('note_off', note=DRUM_MAP[event['note']], velocity=0, channel=9),
                'abs_time': current_time_ticks + event['start_tick'] + event['duration_ticks']
            })
        
        current_time_ticks += ticks_per_beat * 4 

    all_drum_events.sort(key=lambda x: x['abs_time'])
    
    last_abs_time = 0
    for event in all_drum_events:
        delta_time = event['abs_time'] - last_abs_time
        if delta_time < 0:
            delta_time = 0
        new_msg = event['msg'].copy(time=delta_time)
        new_drum_track.append(new_msg)
        last_abs_time = event['abs_time']

    new_midi.tracks.append(new_drum_track)
    return new_midi

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

        st.markdown("---")
        st.subheader("⚙️ Scegli i Metodi di Decomposizione MIDI")

        midi_methods = {
            "MIDI Note Remapper": "🎶 Remapping di Note (Verticale)",
            "MIDI Phrase Reconstructor": "🔄 Riorganizzazione Frasi (Orizzontale)",
            "MIDI Time Scrambler": "⏳ Manipolazione Ritmo/Durata (Orizzontale)",
            "MIDI Density Transformer": "🎲 Controllo Densità (Armonia/Contrappunto)",
            "MIDI Random Pitch Transformer": "❓ Randomizzazione Totale Pitch (Caos)",
            "MIDI Rhythmic Base": "🥁 Aggiungi Base Ritmica" # Nuovo nome
        }

        # MODIFICA: Utilizzo di st.multiselect invece di st.selectbox
        selected_methods_keys = st.multiselect(
            "Seleziona uno o più metodi:",
            list(midi_methods.keys()),
            format_func=lambda x: midi_methods[x],
            help="Scegli come vuoi decomporre le note e la struttura del tuo MIDI."
        )

        decomposed_midi_file = midi_data

        # --- Controlli specifici per ogni metodo selezionato ---
        st.markdown("#### Parametri per i Metodi Selezionati:")
        
        parameters = {}
        for selected_method in selected_methods_keys:
            st.markdown(f"**Parametri per: {midi_methods[selected_method]}**")

            if selected_method == "MIDI Note Remapper":
                col1_remap, col2_remap = st.columns(2)
                with col1_remap:
                    target_scale = st.selectbox("Scala Target:", ["Cromatica", "Maggiore", "Minore Naturale", "Pentatonica Maggiore", "Blues"], help="Le note verranno adattate a questa scala.", key=f"remap_scale_{selected_method}")
                with col2_remap:
                    target_key = st.selectbox("Tonalità Target:", ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B','Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'], index=0, help="La tonalità di destinazione per la decomposizione.", key=f"remap_key_{selected_method}")
                pitch_shift_range = st.slider("Range Pitch Shift Randomico (semitoni):", 0, 12, 0, help="Aggiunge un offset casuale alle note prima di adattarle alla scala.", key=f"remap_pitch_shift_{selected_method}")
                velocity_randomization = st.slider("Percentuale Randomizzazione Velocity:", 0, 100, 0, help="Varia casualmente il volume delle note.", key=f"remap_velocity_{selected_method}")
                parameters[selected_method] = (target_scale, target_key, int(pitch_shift_range), int(velocity_randomization))

            elif selected_method == "MIDI Phrase Reconstructor":
                phrase_length_beats = st.slider("Lunghezza Frase (battute):", 1, 16, 4, help="Definisce la dimensione dei blocchi musicali (in battute) da riorganizzare.", key=f"phrase_length_{selected_method}")
                reassembly_style = st.selectbox("Stile Riorganizzazione Frasi:", ["Casuale", "Inversione", "Ciclico A-B-A", "Dal Più Corto al Più Lungo"], index=0, help="Come le frasi verranno riassemblate.", key=f"phrase_style_{selected_method}")
                parameters[selected_method] = (phrase_length_beats, reassembly_style)

            elif selected_method == "MIDI Time Scrambler":
                keep_original_duration = st.checkbox("Mantieni Durata Originale", help="Seleziona per applicare quantizzazione/swing senza modificare la durata totale del brano.", key=f"time_keep_duration_{selected_method}")
                execution_speed_preset = st.selectbox("Velocità di Esecuzione:", ["Medio (Originale)", "Lento (Metà velocità)", "Molto Lento (Un quarto velocità)", "Veloce (Doppia velocità)", "Molto Veloce (Quattro volte velocità)"], index=0, help="Imposta un preset per lo stiramento/compressione del tempo.", key=f"time_speed_preset_{selected_method}")
                
                default_stretch_factor = 1.0
                if execution_speed_preset == "Lento (Metà velocità)": default_stretch_factor = 2.0
                elif execution_speed_preset == "Molto Lento (Un quarto velocità)": default_stretch_factor = 4.0
                elif execution_speed_preset == "Veloce (Doppia velocità)": default_stretch_factor = 0.5
                elif execution_speed_preset == "Molto Veloce (Quattro volte velocità)": default_stretch_factor = 0.25
                
                stretch_factor = st.slider("Fattore di Stiramento/Compressione (Time Warp):", 0.1, 5.0, default_stretch_factor, 0.1, help="Allunga (valori > 1) o comprime (valori < 1) il tempo generale del MIDI. Regola per fine-tuning.", key=f"time_stretch_factor_{selected_method}")
                
                quantization_strength = st.slider("Forza Quantizzazione (0=libero, 100=rigido):", 0, 100, 50, help="Quanto le note verranno 'allineate' a una griglia ritmica (es. a 16esimi).", key=f"time_quant_strength_{selected_method}")
                swing_amount = st.slider("Quantità di Swing (%):", 0, 100, 0, help="Aggiunge un 'groove' swing ritardando alcune suddivisioni (solo con quantizzazione attiva).", key=f"time_swing_amount_{selected_method}")

                if keep_original_duration:
                    stretch_factor = 1.0
                
                parameters[selected_method] = (stretch_factor, quantization_strength, swing_amount)

            elif selected_method == "MIDI Density Transformer":
                add_note_probability = st.slider("Probabilità di Aggiungere Note (%):", 0, 50, 0, help="Probabilità di inserire nuove note basate su armonie esistenti.", key=f"density_add_prob_{selected_method}")
                remove_note_probability = st.slider("Probabilità di Rimuovere Note (%):", 0, 50, 0, help="Probabilità di eliminare note esistenti, per un suono più spoglio.", key=f"density_remove_prob_{selected_method}")
                polyphony_mode = st.selectbox("Modalità Polifonia Aggiuntiva:", ["Nessuna", "Riempi Accordo (Triadi)", "Aggiungi Contro-Melodia", "Droni"], help="Come le nuove note verranno generate per influenzare la densità.", key=f"density_poly_mode_{selected_method}")
                parameters[selected_method] = (add_note_probability, remove_note_probability, polyphony_mode)
            
            elif selected_method == "MIDI Random Pitch Transformer":
                random_pitch_strength = st.slider("Forza Randomizzazione Pitch (%):", 0, 100, 100, help="Probabilità che ogni nota (on/off) abbia il suo pitch completamente randomizzato (0-127).", key=f"random_pitch_strength_{selected_method}")
                parameters[selected_method] = (random_pitch_strength,)

            elif selected_method == "MIDI Rhythmic Base":
                st.markdown("Seleziona gli elementi ritmici per costruire il tuo pattern:")
                kick_enabled = st.checkbox("Cassa (4/4)", value=True, key="rhythm_kick")
                snare_enabled = st.checkbox("Rullante (su 2 e 4)", value=True, key="rhythm_snare")
                hihat_enabled = st.checkbox("Hi-hat (su ottavi)", value=True, key="rhythm_hihat")
                parameters[selected_method] = (kick_enabled, snare_enabled, hihat_enabled)

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

                decomposed_midi_file = current_midi

                if decomposed_midi_file:
                    st.success("Decomposizione MIDI completata!")
                    st.subheader("Scarica il tuo MIDI Decomposto (Completo)")

                    decomposed_midi_bytes = io.BytesIO()
                    decomposed_midi_file.save(file=decomposed_midi_bytes)
                    decomposed_midi_bytes.seek(0)

                    st.download_button(
                        label="💾 Scarica MIDI Decomposto Completo",
                        data=decomposed_midi_bytes,
                        file_name=f"{uploaded_midi_file.name.split('.')[0]}_Decomposed.mid",
                        mime="audio/midi",
                        use_container_width=True
                    )

                    # --- SEZIONE: Download Singole Tracce ---
                    def get_track_display_name(track, index):
                        track_name = ""
                        for msg in track:
                            if msg.type == 'track_name':
                                track_name = msg.name
                                break
                        if not track_name and any(msg.type == 'note_on' and msg.channel == 9 for msg in track):
                            return f"Traccia {index}: Ritmica (Drums)"
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
        * **⏳ MIDI Time Scrambler**: Modifica il timing e la durata delle note per creare nuovi groove. Ora puoi scegliere se mantenere la durata originale del brano!
        * **🎲 MIDI Density Transformer**: Aggiunge o rimuove note per alterare la densità armonica.
        * **❓ MIDI Random Pitch Transformer**: Randomizza completamente l'altezza di ogni nota (pitch) per un caos melodico.
        * **🥁 Aggiungi Base Ritmica**: Aggiunge una nuova traccia di batteria al tuo brano per creare un sound dance o pop!
        
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
