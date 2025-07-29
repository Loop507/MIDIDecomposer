# midi_decomposer_app.py - VERSIONE AGGIORNATA CON CORREZIONE mido.tempo2bpm

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
                # CORREZIONE QUI: mido.tempo_to_bpm -> mido.tempo2bpm
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
        absolute_tick_counter = 0

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
                
                # Crea una sequenza ABAC... ripetuta per una lunghezza ragionevole
                # Per semplicità, possiamo ripeterla un paio di volte o per la lunghezza originale
                num_repetitions = max(1, len(phrases) // 3) # Ripeti per avere lunghezza simile all'originale
                for _ in range(num_repetitions):
                    reorganized_phrases.extend(a_phrase)
                    reorganized_phrases.extend(b_phrase)
                    reorganized_phrases.extend(a_phrase)
                    reorganized_phrases.extend(c_phrase)
            else:
                st.warning(f"Troppo poche frasi ({len(phrases)}) per lo stile 'Ciclico A-B-A'. Verrà usata la riorganizzazione casuale.")
                reorganized_phrases = list(phrases)
                random.shuffle(reorganized_phrases)
        elif reassembly_style == "Dal Più Corto al Più Lungo":
            st.warning("La modalità 'Dal Più Corto al Più Lungo' non è ancora implementata. Verrà usata la riorganizzazione casuale.")
            reorganized_phrases = list(phrases)
            random.shuffle(reorganized_phrases)
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
                # Il msg.time qui è il delta rispetto al messaggio precedente all'interno della sua frase originale.
                # Per ricostruire una sequenza piatta, sommiamo i delta per ottenere il tempo assoluto
                # all'interno della sequenza riorganizzata.
                absolute_time_in_reorganized_seq += msg_in_phrase.time
                flat_events_for_reconstruction.append({'msg': msg_in_phrase.copy(), 'abs_time': absolute_time_in_reorganized_seq})

        # Ora che abbiamo tutti i messaggi con i loro tempi assoluti nella sequenza riorganizzata,
        # possiamo creare la nuova traccia ricalcolando i delta time
        last_abs_time = 0
        for event_data in flat_events_for_reconstruction:
            msg = event_data['msg']
            abs_time = event_data['abs_time']
            
            delta_time = abs_time - last_abs_time
            new_track.append(msg.copy(time=delta_time))
            last_abs_time = abs_time
            
        new_midi.tracks.append(new_track)
    return new_midi


def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    """
    Modifica il timing e la durata delle note MIDI.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Time Scrambler con Stretch: {stretch_factor}, Quantizzazione: {quantization_strength}, Swing: {swing_amount}")
    # Questa è una placeholder: restituisce il MIDI originale
    return original_midi # Placeholder: restituisce l'originale

def midi_density_transformer(original_midi, add_note_probability, remove_note_probability, polyphony_mode):
    """
    Aggiunge o rimuove note per alterare la densità MIDI.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Density Transformer con Probabilità Aggiunta: {add_note_probability}, Rimozione: {remove_note_probability}, Modalità Polifonia: {polyphony_mode}")
    # Questa è una placeholder: restituisce il MIDI originale
    return original_midi # Placeholder: restituisce l'originale

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
            "MIDI Density Transformer": "🎲 Controllo Densità (Armonia/Contrappunto)"
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
                help="Come le frasi verranno riassemblate. 'Dal Più Corto al Più Lungo' non è ancora implementato."
            )

        elif selected_midi_method == "MIDI Time Scrambler":
            stretch_factor = st.slider(
                "Fattore di Stiramento/Compressione (Time Warp):", 0.5, 2.0, 1.0, 0.1,
                help="Allunga (valori > 1) o comprime (valori < 1) il tempo generale del MIDI."
            )
            quantization_strength = st.slider(
                "Forza Quantizzazione (0=libero, 100=rigido):", 0, 100, 50,
                help="Quanto le note verranno 'allineate' a una griglia ritmica."
            )
            swing_amount = st.slider(
                "Quantità di Swing (%):", 0, 100, 0,
                help="Aggiunge un 'groove' swing al ritmo (se la quantizzazione è attiva)."
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
    st.info("👆 Carica un file MIDI (.mid o .midi) per iniziare la decomposizione.")

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

        **Metodi di Decomposizione Disponibili:**

        * **🎶 MIDI Note Remapper**: Rimodella le note del pentagramma (verticale) in base a scale, tonalità e randomizzazione.
            * _Parametri:_ Scala Target (es. Maggiore, Cromatica), Tonalità Target (es. C, Am), Range Pitch Shift Randomico, Percentuale Randomizzazione Velocity.
        * **🔄 MIDI Phrase Reconstructor**: Riorganizza e ricompone blocchi o "frasi" musicali (orizzontale).
            * _Parametri:_ Lunghezza Frase (battute), Stile Riorganizzazione Frasi (Casuale, Inversione, Ciclico A-B-A).
        * **⏳ MIDI Time Scrambler**: Modifica il timing e la durata delle note per creare nuovi groove.
        * **🎲 MIDI Density Transformer**: Aggiunge o rimuove note per alterare la densità armonica.
        
        Questi metodi sono progettati per fornirti strumenti per la **composizione algoritmica** e la manipolazione strutturale delle tue idee musicali MIDI!

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
