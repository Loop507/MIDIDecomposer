# midi_decomposer_app.py

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

# --- Funzioni di Decomposizione (PLACEHOLDER per ora) ---

def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization):
    """
    Rimodella le note MIDI in base a una scala, tonalità e randomizzazione di pitch/velocity.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Note Remapper con Scala: {target_scale_name}, Tonalità: {target_key_name}, Pitch Shift: {pitch_shift_range}, Velocity Rand: {velocity_randomization}")
    new_midi = mido.MidiFile()
    
    # Mappa note per scala e tonalità (esempio molto semplificato)
    # Questa è una logica PLACEHOLDER. Dovrà essere molto più sofisticata.
    scales = {
        "Cromatica": list(range(12)),
        "Maggiore": [0, 2, 4, 5, 7, 9, 11],
        "Minore Naturale": [0, 2, 3, 5, 7, 8, 10],
        "Pentatonica Maggiore": [0, 2, 4, 7, 9],
        "Blues": [0, 3, 5, 6, 7, 10]
    }
    key_offset = mido.MidiFile()._key_to_note_number(target_key_name) if target_key_name else 0
    target_scale = scales.get(target_scale_name, scales["Cromatica"])
    
    for i, track in enumerate(original_midi.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                original_note = msg.note
                
                # Applica Pitch Shift randomico
                shifted_note = original_note + random.randint(-pitch_shift_range, pitch_shift_range)
                
                # Adatta alla scala target e tonalità
                # Trova la nota più vicina nella scala target, considerando l'offset della tonalità
                octave = shifted_note // 12
                note_in_octave = shifted_note % 12
                
                # Trova la nota più vicina nella scala selezionata
                closest_scale_note = min(target_scale, key=lambda x: min(abs(note_in_octave - x), abs(note_in_octave - x - 12), abs(note_in_octave - x + 12)))
                new_note_pitch = octave * 12 + closest_scale_note + key_offset
                
                new_note_pitch = max(0, min(127, new_note_pitch)) # Clampa tra 0-127

                # Applica randomizzazione velocity
                new_velocity = msg.velocity
                if velocity_randomization > 0:
                    new_velocity = int(new_velocity * (1 + random.uniform(-velocity_randomization/100, velocity_randomization/100)))
                    new_velocity = max(1, min(127, new_velocity)) # Clampa tra 1-127 per velocity

                new_track.append(msg.copy(note=new_note_pitch, velocity=new_velocity))
            else:
                new_track.append(msg.copy()) # Copia gli altri messaggi (control change, program change, etc.)
        new_midi.tracks.append(new_track)
    return new_midi

def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style):
    """
    Riorganizza le frasi MIDI.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Phrase Reconstructor con Lunghezza Frase: {phrase_length_beats} battute, Stile: {reassembly_style}")
    # Questa è una placeholder: restituisce il MIDI originale
    return mido.MidiFile() # Placeholder: restituisce l'originale

def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    """
    Modifica il timing e la durata delle note MIDI.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Time Scrambler con Stretch: {stretch_factor}, Quantizzazione: {quantization_strength}, Swing: {swing_amount}")
    # Questa è una placeholder: restituisce il MIDI originale
    return mido.MidiFile() # Placeholder: restituisce l'originale

def midi_density_transformer(original_midi, add_note_probability, remove_note_probability, polyphony_mode):
    """
    Aggiunge o rimuove note per alterare la densità MIDI.
    (Logica da implementare)
    """
    st.info(f"Applico MIDI Density Transformer con Probabilità Aggiunta: {add_note_probability}, Rimozione: {remove_note_probability}, Modalità Polifonia: {polyphony_mode}")
    # Questa è una placeholder: restituisce il MIDI originale
    return mido.MidiFile() # Placeholder: restituisce l'originale

# --- Sezione Upload File MIDI ---
st.subheader("🎵 Carica il tuo file MIDI (.mid)")
uploaded_midi_file = st.file_uploader(
    "Trascina qui il tuo file MIDI o clicca per sfogliare",
    type=["mid"],
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
                "Lunghezza Frase (battute):", 1, 8, 4,
                help="Definisce la dimensione dei blocchi musicali da riorganizzare."
            )
            reassembly_style = st.selectbox(
                "Stile Riorganizzazione Frasi:",
                ["Casuale", "Inversione", "Ciclico A-B-A", "Dal Più Corto al Più Lungo"],
                help="Come le frasi verranno riassemblate."
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
                    if len(decomposed_midi_file.tracks) > 1:
                        st.markdown("---")
                        st.subheader("Scarica Singole Tracce del MIDI Decomposto")
                        
                        # Creare un elenco di opzioni per le tracce
                        track_options = [f"Traccia {i}: {mido.midifiles.tracks.get_track_name(track)}" if mido.midifiles.tracks.get_track_name(track) else f"Traccia {i} (Senza Nome)" for i, track in enumerate(decomposed_midi_file.tracks)]
                        
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

                                track_name_for_file = mido.midifiles.tracks.get_track_name(decomposed_midi_file.tracks[track_index]).replace(' ', '_') if mido.midifiles.tracks.get_track_name(decomposed_midi_file.tracks[track_index]) else f"Track_{track_index}"

                                st.download_button(
                                    label=f"💾 Scarica {track_options[track_index]}",
                                    data=single_track_bytes,
                                    file_name=f"{uploaded_midi_file.name.split('.')[0]}_{selected_midi_method.replace(' ', '_')}_{track_name_for_file}.mid",
                                    mime="audio/midi",
                                    key=f"download_track_{track_index}" # Chiave unica per ogni pulsante
                                )
                    elif len(decomposed_midi_file.tracks) == 1:
                        st.info("Il MIDI decomposto contiene una sola traccia, scarica il file completo qui sopra.")
                    else:
                        st.warning("Il MIDI decomposto non contiene tracce.")

                else:
                    st.error("Impossibile generare il MIDI decomposto. Controlla i messaggi di avviso.")

    except Exception as e:
        st.error(f"❌ Errore durante la lettura o l'elaborazione del file MIDI: {str(e)}")
        st.error("Assicurati che sia un file MIDI valido e riprova.")
        st.exception(e) # Mostra i dettagli completi dell'errore per il debug

else:
    st.info("👆 Carica un file MIDI (.mid) per iniziare la decomposizione.")

    with st.expander("📖 Come usare MIDI Decomposer"):
        st.markdown("""
        ### Benvenuto in MIDI Decomposer!

        Qui potrai caricare i tuoi file MIDI e applicare diverse tecniche di decomposizione per creare nuove strutture musicali.

        **Come funziona:**

        1.  **Carica il tuo file MIDI** (con estensione `.mid`).
        2.  Scegli il **metodo di decomposizione** e imposta i suoi **parametri**.
        3.  Clicca su **"DECOMPONI MIDI"**.
        4.  Scarica il **file MIDI completo** o seleziona le **singole tracce** da scaricare.
        5.  Apri il file MIDI scaricato nel tuo software musicale (DAW) preferito per ascoltare il risultato.

        **Metodi di Decomposizione Disponibili:**

        * **🎶 MIDI Note Remapper**: Rimodella le note del pentagramma (verticale) in base a scale, tonalità e randomizzazione, agendo anche su registri specifici.
        * **🔄 MIDI Phrase Reconstructor**: Riorganizza e ricompone blocchi o "frasi" musicali (orizzontale), creando nuove strutture compositive.
        * **⏳ MIDI Time Scrambler**: Modifica il timing e la durata delle note, intervenendo su ritmo, swing e quantizzazione per creare nuovi groove.
        * **🎲 MIDI Density Transformer**: Aggiunge o rimuove note in modo intelligente per alterare la densità armonica e contrappuntistica, influenzando la "tessitura" del brano.
        
        Questi metodi sono progettati per fornirti strumenti per la **composizione algoritmica** e la manipolazione strutturale delle tue idee musicali MIDI!
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
