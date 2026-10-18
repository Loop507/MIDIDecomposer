# MidiDecomposer_12_0.py — MIDI Decomposer by loop507 (file unico)
# Struttura: [1] MOTORE (utilita' MIDI, seed, compositori, metodi, validazione, report)
#            [2] INTERFACCIA Streamlit.

import base64
import bisect
import copy as _copy
import functools
import io
import itertools
import logging
import math
import os
import random
import threading
import zlib
from collections import Counter, defaultdict, deque

import mido
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# [1] MOTORE
# =====================================================================

# --- Contesto per-thread: impostazioni utente e canale degli avvisi ---
# Ogni sessione Streamlit gira nel proprio thread: una configurazione per-thread evita
# che seed / batteria / strumento di un utente finiscano in quella di un altro.
_ctx = threading.local()
_log = logging.getLogger("midi_engine")


def configure(settings=None, warn=None):
    """Collega il motore all'interfaccia (da richiamare a ogni esecuzione dello script).
    settings: mapping con .get(chiave, default) — chiavi usate: 'preserve_drums',
              '_run_seed', 'gen_program_label' (es. st.session_state).
    warn:     funzione(str) per gli avvisi (es. st.warning); default: logging."""
    _ctx.settings = settings
    _ctx.warn = warn


def _setting(key, default=None):
    settings = getattr(_ctx, 'settings', None)
    if settings is None:
        return default
    try:
        value = settings.get(key, default)
    except Exception:
        return default
    return default if value is None else value



def _clone(msg, **changes):
    """Copia un messaggio mido applicando le modifiche. Equivale a msg.copy(**changes) ma
    senza ri-validare l'intero messaggio a ogni copia (mido.Message.copy e' ~3.5x piu' lento):
    i singoli valori modificati sono comunque validati dal setter pubblico di mido."""
    new = _copy.copy(msg)
    for key, value in changes.items():
        setattr(new, key, value)
    return new


def warn(message):
    handler = getattr(_ctx, 'warn', None)
    if handler is not None:
        handler(message)
    else:
        _log.warning(message)


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
    """Helper per estrarre note e il loro tempo assoluto da una traccia.
    Coda FIFO per (nota, canale): se la stessa altezza viene riattaccata prima del
    note_off, il note_off chiude la nota piu' VECCHIA (convenzione dei sequencer);
    nessuna nota viene persa e non nascono note annidate lunghissime.
    Note rimaste aperte senza note_off: chiuse con durata stimata di 1 beat."""
    notes = []
    active_notes = defaultdict(deque)
    current_abs_time = 0
    for msg in track:
        current_abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[(msg.note, msg.channel)].append((current_abs_time, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            queue = active_notes.get((msg.note, msg.channel))
            if queue:
                start, velocity = queue.popleft()
                notes.append({'start': start, 'end': current_abs_time, 'pitch': msg.note,
                              'velocity': velocity, 'channel': msg.channel})
    for (pitch, channel), queue in active_notes.items():
        for start, velocity in queue:
            notes.append({'start': start, 'end': start + ticks_per_beat, 'pitch': pitch,
                          'velocity': velocity, 'channel': channel})
    return notes


# Controller "di mix" letti prima della prima nota: volume (7), pan (10), expression (11),
# riverbero (91), chorus (93). Senza di essi una traccia ricostruita torna ai valori di default
# della DAW (volume pieno, pan al centro, niente effetti) e il bilanciamento originale va perso.
_MIX_CCS = (7, 10, 11, 91, 93)


def _is_instrument_header_msg(msg):
    """True per i messaggi che definiscono strumento e mix di una traccia."""
    if msg.type == 'program_change':
        return True
    return msg.type == 'control_change' and (msg.control in (0, 32) or msg.control in _MIX_CCS)


def _extract_instrument_header(track):
    """
    Estrae i messaggi che definiscono strumento e mix di una traccia originale
    (program_change, bank select 0/32, volume/pan/expression/riverbero/chorus),
    letti prima del primo evento nota. Le funzioni che ricostruiscono una traccia
    da zero a partire dalle sole note (extract_notes) devono ri-applicare
    questi messaggi in testa alla nuova traccia — altrimenti la DAW (es.
    Logic Pro) assegna il proprio strumento di default (tipicamente
    "Steinway Grand Piano") a tutte le tracce e azzera il bilanciamento, perdendo
    l'orchestrazione originale anche quando numero e nomi delle tracce sono corretti.
    """
    header = []
    for msg in track:
        if msg.type in ('note_on', 'note_off'):
            break
        if _is_instrument_header_msg(msg):
            header.append(_clone(msg, time=0))
    return header


def _get_track_default_channel(track):
    """Ritorna il canale MIDI dominante di una traccia (il primo trovato), o 0 se assente."""
    for msg in track:
        if hasattr(msg, 'channel'):
            return msg.channel
    return 0

# --- Helper globali: meta-messaggi, canali liberi, batteria ---
_GLOBAL_META_TYPES = ('set_tempo', 'time_signature', 'key_signature')
DRUM_CHANNEL = 9  # canale 10 General MIDI (indice 0-based)


def _drums_preserved(flag=None):
    """True se la batteria (canale 10) va lasciata intatta. Se flag e' None
    legge il checkbox 'preserve_drums' dell'interfaccia (default True)."""
    if flag is not None:
        return bool(flag)
    return bool(_setting('preserve_drums', True))


def _collect_global_meta(midi):
    """Ritorna [(abs_time, MetaMessage)] per tempo, metrica e tonalita' di tutte le tracce."""
    found = []
    for track in midi.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type in _GLOBAL_META_TYPES:
                found.append((t, _clone(msg, time=0)))
    found.sort(key=lambda x: x[0])
    return found


def _restore_global_meta(source_midi, result_midi):
    """Se result_midi ha perso set_tempo / time_signature / key_signature del brano
    sorgente, li reinserisce (stesso tempo assoluto) nella prima traccia.
    La prima traccia viene sostituita da una copia: i MIDI originali non sono mai modificati."""
    if source_midi is result_midi:
        return result_midi
    present = {msg.type for tr in result_midi.tracks for msg in tr
               if msg.is_meta and msg.type in _GLOBAL_META_TYPES}
    missing = [(t, m) for (t, m) in _collect_global_meta(source_midi) if m.type not in present]
    if not missing:
        return result_midi
    if not result_midi.tracks:
        result_midi.tracks.append(mido.MidiTrack())
    first = result_midi.tracks[0]
    events = []
    t = 0
    for msg in first:
        t += msg.time
        if msg.is_meta and msg.type == 'end_of_track':
            continue
        events.append((t, 1, msg))
    for mt, m in missing:
        events.append((mt, 0, m))
    events.sort(key=lambda e: (e[0], e[1]))  # stabile: i meta precedono gli altri eventi allo stesso tick
    new_first = mido.MidiTrack()
    last = 0
    for t, _prio, msg in events:
        new_first.append(_clone(msg, time=t - last))
        last = t
    result_midi.tracks[0] = new_first
    return result_midi


def _preserves_meta(func):
    """Decoratore: garantisce che tempo/metrica/tonalita' del brano originale
    sopravvivano a qualunque trasformazione (ricostruita da zero o no)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        original = args[0] if args else kwargs.get('original_midi')
        if original is not None:
            if isinstance(result, tuple) and result and isinstance(result[0], mido.MidiFile):
                _restore_global_meta(original, result[0])
            elif isinstance(result, mido.MidiFile):
                _restore_global_meta(original, result)
        return result
    return wrapper


def _tempo_pairs(midi):
    """[(tick_assoluto, tempo)] di tutti i set_tempo del file, in ordine di tick
    (a parita' di tick, ordine delle tracce: come la fusione di mido)."""
    return [(t, m.tempo) for t, m in _collect_global_meta(midi) if m.type == 'set_tempo']


def _sync_tempo_map(source_midi, result_midi):
    """Fa coincidere la mappa dei tempi del risultato con quella del sorgente.
    Serve ai metodi che ricostruiscono le tracce SENZA cambiare la scala dei tick: se il tempo
    del brano e' sparso su piu' tracce (es. una traccia 'tempo' a 60 BPM + un set_tempo a 190 BPM
    dentro una traccia di note), la ricostruzione ne perdeva una parte e la durata cambiava
    (es. 5 minuti -> 16). Se le due mappe sono gia' uguali non tocca nulla.
    Le tracce del risultato non vengono mai modificate sul posto (possono essere condivise)."""
    if source_midi is result_midi:
        return result_midi
    src = _tempo_pairs(source_midi)
    res = _tempo_pairs(result_midi)
    if dict(src) == dict(res):  # mappa effettiva uguale (l'ultimo tempo a ogni tick vince)
        return result_midi
    for i, track in enumerate(result_midi.tracks):
        if not any(msg.is_meta and msg.type == 'set_tempo' for msg in track):
            continue
        rebuilt = mido.MidiTrack()
        carry = 0
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                carry += msg.time
                continue
            rebuilt.append(_clone(msg, time=msg.time + carry) if carry else msg)
            carry = 0
        result_midi.tracks[i] = rebuilt
    if not result_midi.tracks:
        result_midi.tracks.append(mido.MidiTrack())
    first = result_midi.tracks[0]
    events = []
    t = 0
    for msg in first:
        t += msg.time
        if msg.is_meta and msg.type == 'end_of_track':
            continue
        events.append((t, 1, msg))
    for tick, tempo in src:
        events.append((tick, 0, mido.MetaMessage('set_tempo', tempo=tempo, time=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    new_first = mido.MidiTrack()
    last = 0
    for t, _prio, msg in events:
        new_first.append(_clone(msg, time=t - last))
        last = t
    result_midi.tracks[0] = new_first
    return result_midi


def _keeps_tempo_map(func):
    """Decoratore per i metodi che ricostruiscono le tracce mantenendo i tick:
    il tempo (e quindi la durata in secondi) del brano originale resta quello originale."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        original = args[0] if args else kwargs.get('original_midi')
        if original is not None:
            if isinstance(result, tuple) and result and isinstance(result[0], mido.MidiFile):
                _sync_tempo_map(original, result[0])
            elif isinstance(result, mido.MidiFile):
                _sync_tempo_map(original, result)
        return result
    return wrapper


def _clamp_tail(midi, end_tick):
    """Riporta a end_tick ogni evento che cade oltre (marker/end_of_track/tempi lontani):
    la durata del file coincide con la fine della musica. Le tracce toccate vengono sostituite
    da copie (mai modificate sul posto: possono essere condivise con il MIDI originale)."""
    for i, track in enumerate(midi.tracks):
        t = 0
        over = False
        for msg in track:
            t += msg.time
            if t > end_tick:
                over = True
                break
        if not over:
            continue
        rebuilt = mido.MidiTrack()
        t = 0
        last = 0
        for msg in track:
            t += msg.time
            at = min(t, end_tick)
            rebuilt.append(_clone(msg, time=at - last))
            last = at
        midi.tracks[i] = rebuilt
    return midi


def _trims_tail(func):
    """Decoratore (piu' esterno) per i metodi che generano la musica da zero: il risultato
    finisce all'ultima nota, non a un end_of_track/marker lontano ereditato dal file originale."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        out = result[0] if isinstance(result, tuple) and result else result
        if isinstance(out, mido.MidiFile):
            end = midi_last_note_tick(out)
            if end > 0:
                _clamp_tail(out, end)
        return result
    return wrapper


# --- Ottimizzazione dell'output per le DAW ---
_GLOBAL_META_RANK = {'set_tempo': 0, 'time_signature': 1, 'key_signature': 2}


def _build_track(items):
    """[(tick_assoluto, priorita', msg)] -> MidiTrack con delta-time (ordine stabile per (tick, priorita'))."""
    track = mido.MidiTrack()
    last = 0
    for at, _prio, msg in sorted(items, key=lambda x: (x[0], x[1])):
        track.append(_clone(msg, time=at - last))
        last = at
    return track


def _global_meta_with_defaults(midi):
    """{(tick, tipo): messaggio} di tempo/metrica/tonalita' di tutto il file (a parita' di tick vince
    l'ultimo, come la fusione di mido), con tempo 120 BPM e metrica 4/4 espliciti al tick 0 se
    mancano. Ritorna (dizionario, elenco dei default aggiunti)."""
    glob = {}
    for track in midi.tracks:
        at = 0
        for msg in track:
            at += msg.time
            if msg.is_meta and msg.type in _GLOBAL_META_TYPES:
                glob[(at, msg.type)] = _clone(msg, time=0)
    defaults = []
    if (0, 'set_tempo') not in glob:
        glob[(0, 'set_tempo')] = mido.MetaMessage('set_tempo', tempo=500000, time=0)
        defaults.append('tempo 120 BPM')
    if (0, 'time_signature') not in glob:
        glob[(0, 'time_signature')] = mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0)
        defaults.append('metrica 4/4')
    return glob, defaults


def track_to_midi_file(midi, index):
    """MIDI con la sola traccia `index`, completo di tempo/metrica/tonalita' del file di origine.
    Salvare una traccia da sola senza questi eventi fa importare il file a 120 BPM, 4/4
    (la DAW non trova la mappa dei tempi): la parte suonerebbe a velocita' sbagliata."""
    glob, _defaults = _global_meta_with_defaults(midi)
    items = []
    at = 0
    for msg in midi.tracks[index]:
        at += msg.time
        if msg.is_meta and msg.type in _GLOBAL_META_TYPES:
            continue
        items.append((at, 10, msg))
    items.extend((tick, _GLOBAL_META_RANK[typ], msg) for (tick, typ), msg in glob.items())
    single = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    single.tracks.append(_build_track(items))
    return single


def get_track_display_name(track, index):
    """Etichetta leggibile di una traccia per la UI (nome traccia o '(Senza Nome)')."""
    track_name = next((msg.name for msg in track if msg.type == 'track_name'), None)
    if not track_name and hasattr(track, 'name') and 'Ritmica:' in track.name:
        return track.name
    return f"Traccia {index}: {track_name if track_name else '(Senza Nome)'}"


def daw_optimize(midi, split_channels=True):
    """Rende il MIDI di output piu' robusto all'import nelle DAW. Non modifica `midi`.
      1. Tempo, metrica e tonalita' stanno SOLO nella prima traccia (come prevede lo standard
         MIDI: 'tempo information should always be stored in the first MTrk chunk').
      2. Tempo e metrica espliciti al tick 0 (default 120 BPM, 4/4): senza, alcune DAW importano
         valori propri (es. Ardour ricade su 120 BPM, 4/4).
      3. Una traccia = un canale (split_channels): Ableton Live ignora i canali MIDI e fonde in
         un'unica clip i canali di una traccia; FL Studio e MuseScore li trattano per canale.
    Ritorna (nuovo_midi, info). Note, tempi e durata restano identici."""
    info = {'tempo_moved': False, 'defaults': [], 'tracks_split': 0, 'tracks_added': 0}
    if midi.type == 2 or not midi.tracks:
        return midi, info

    # 1) eventi globali di tutte le tracce + 2) tempo e metrica espliciti all'inizio
    glob, info['defaults'] = _global_meta_with_defaults(midi)
    info['tempo_moved'] = any(msg.is_meta and msg.type in _GLOBAL_META_TYPES
                              for track in midi.tracks[1:] for msg in track)
    glob_items = [(at, _GLOBAL_META_RANK[typ], msg) for (at, typ), msg in glob.items()]

    # 3) ricostruzione: globali solo in traccia 0; poi una traccia per canale
    rebuilt = []
    for idx, track in enumerate(midi.tracks):
        items = []
        at = 0
        for msg in track:
            at += msg.time
            if msg.is_meta and msg.type in _GLOBAL_META_TYPES:
                continue
            items.append((at, 10, msg))
        if idx == 0:
            items.extend(glob_items)
        rebuilt.append((track, items))

    out_tracks = []
    for track, items in rebuilt:
        note_count = defaultdict(int)
        for _at, _p, msg in items:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count[msg.channel] += 1
        if not split_channels or len(note_count) < 2:
            out_tracks.append(_build_track(items))
            continue
        dominant = max(note_count, key=lambda c: (note_count[c], -c))
        base = track.name if getattr(track, 'name', '') else 'Traccia'
        pieces = {c: [] for c in note_count}
        for entry in items:
            msg = entry[2]
            ch = getattr(msg, 'channel', None)
            pieces[ch if ch in pieces else dominant].append(entry)
        info['tracks_split'] += 1
        out_tracks.append(_build_track(pieces[dominant]))
        for ch in sorted(c for c in pieces if c != dominant):
            program = next((m.program for _a, _p, m in pieces[ch] if m.type == 'program_change'), None)
            label = "Drums" if ch == DRUM_CHANNEL else (GM_PROGRAM_NAMES[program] if program is not None else f"ch {ch + 1}")
            piece = _build_track([(0, 0, mido.MetaMessage('track_name', name=f"{base} - {label}", time=0))] + pieces[ch])
            out_tracks.append(piece)
            info['tracks_added'] += 1

    new_midi = mido.MidiFile(type=1 if len(out_tracks) > 1 else midi.type, ticks_per_beat=midi.ticks_per_beat)
    new_midi.tracks.extend(out_tracks)
    return new_midi, info


# --- Variazioni di groove della batteria ---
# Ruoli GM del canale 10 (altezza = strumento del kit). Le variazioni usano SOLO altezze gia' presenti
# nella batteria del brano: nessun pezzo del kit viene inventato.
_DRUM_ROLES = {
    'kick': (36, 35),
    'snare': (38, 40, 37),
    'hat_closed': (42, 44),
    'tom': (47, 45, 43, 41, 48, 50),
    'crash': (49, 57, 55, 52),
}


def drum_variations(midi, amount=0.4, seed=None):
    """Varia il groove della batteria (canale 10) mantenendo le altezze del kit. Non modifica `midi`.
    amount 0..1 = intensita'. Tecniche (tutte con sorteggio riproducibile dal seed):
      - umanizzazione: dinamica con accenti sul battere / sui levare piu' leggeri, micro-spostamenti di tempo;
      - colpi mancanti: qualche colpo cade (mai la cassa sul battere ne' i crash);
      - ghost note del rullante: colpi leggerissimi sui sedicesimi liberi;
      - fill: nell'ultima battuta di ogni frase da 4, l'ultimo movimento diventa un rullo di sedicesimi
        su tom (o rullante), con crescendo, e un crash sul battere successivo se il kit lo ha.
    Le note non-batteria restano identiche; la durata del brano non cambia.
    Ritorna (nuovo_midi, info)."""
    info = {'ghost': 0, 'fills': 0, 'dropped': 0, 'crash': 0, 'tracks': 0}
    amount = max(0.0, min(1.0, float(amount)))
    if amount <= 0 or midi.type == 2:
        return midi, info
    rng = random.Random(_effective_seed(seed, "drumvar"))
    tpb = midi.ticks_per_beat
    ts = next((m for _t, m in _collect_global_meta(midi) if m.type == 'time_signature'), None)
    numer, denom = (ts.numerator, ts.denominator) if ts else (4, 4)
    bar = max(1, int(tpb * 4 * numer / max(1, denom)))
    step = max(1, tpb // 4)  # sedicesimo
    jitter = max(0, int(tpb / 48 * amount * 2))

    new_tracks = []
    for track in midi.tracks:
        notes = extract_notes(track, tpb)
        drum = sorted((n for n in notes if n['channel'] == DRUM_CHANNEL), key=lambda n: (n['start'], n['pitch']))
        if not drum:
            new_tracks.append(track)
            continue
        info['tracks'] += 1
        present = {n['pitch'] for n in drum}
        role = {r: next((p for p in pits if p in present), None) for r, pits in _DRUM_ROLES.items()}
        t_first = min(n['start'] for n in drum)
        t_last = max(n['end'] for n in drum)
        occupied = {(n['start'], n['pitch']) for n in drum}

        # fill: battute di fine frase (ogni 4 battute) e relativo ultimo movimento
        fill_bars = set()
        last_bar_index = (t_last - 1) // bar
        for b in range(3, last_bar_index + 1, 4):
            if (b + 1) * bar <= t_last and rng.random() < amount and (role['tom'] or role['snare']):
                fill_bars.add(b)
        fill_zones = [((b + 1) * bar - tpb, (b + 1) * bar) for b in sorted(fill_bars)]

        out_notes = []
        for n in drum:
            in_fill = any(a <= n['start'] < z for a, z in fill_zones)
            is_anchor = (n['start'] % tpb == 0 and n['pitch'] in _DRUM_ROLES['kick']) or n['pitch'] in _DRUM_ROLES['crash']
            if in_fill and not n['pitch'] in _DRUM_ROLES['crash']:
                continue  # sostituito dal rullo
            if not is_anchor and rng.random() < 0.10 * amount:
                info['dropped'] += 1
                continue
            vel = n['velocity']
            if n['start'] % bar == 0:
                vel += int(8 * amount)
            elif n['start'] % max(1, tpb // 2) != 0:
                vel -= int(6 * amount)
            vel += rng.randint(-int(12 * amount), int(12 * amount))
            shift = rng.randint(-jitter, jitter) if jitter and not is_anchor else 0
            start = max(0, n['start'] + shift)
            end = max(start + 1, n['end'] + shift)
            out_notes.append((start, end, n['pitch'], max(1, min(127, vel))))

        # ghost note del rullante sui sedicesimi liberi
        if role['snare'] is not None:
            per_bar = 1 + int(2 * amount)
            first_bar, last_b = t_first // bar, (t_last - 1) // bar
            for b in range(first_bar, last_b + 1):
                if b in fill_bars or rng.random() >= amount:
                    continue
                for _ in range(per_bar):
                    pos = b * bar + rng.randrange(0, max(1, bar // step)) * step
                    if (pos % tpb == 0 or not (t_first <= pos < t_last - step)
                            or (pos, role['snare']) in occupied
                            or any(a <= pos < z for a, z in fill_zones)):
                        continue
                    out_notes.append((pos, pos + max(1, step // 2), role['snare'], rng.randint(22, 44)))
                    occupied.add((pos, role['snare']))
                    info['ghost'] += 1

        # fill: rullo di sedicesimi con crescendo + crash sul battere successivo
        for (a, z) in fill_zones:
            pitches = [p for p in _DRUM_ROLES['tom'] if p in present] or [role['snare']]
            count = max(1, (z - a) // step)
            for i in range(count):
                pos = a + i * step
                vel = int(58 + (110 - 58) * i / max(1, count - 1))
                out_notes.append((pos, min(z, pos + step), pitches[i % len(pitches)], max(1, min(127, vel))))
            info['fills'] += 1
            if role['crash'] is not None and z < t_last and (z, role['crash']) not in occupied:
                out_notes.append((z, min(t_last, z + tpb), role['crash'], 100))
                info['crash'] += 1

        # ricostruzione: i messaggi non-batteria restano com'erano
        items = []
        at = 0
        for msg in track:
            at += msg.time
            if msg.type in ('note_on', 'note_off') and msg.channel == DRUM_CHANNEL:
                continue
            if msg.is_meta and msg.type == 'end_of_track':
                prio = 99
            elif msg.type in ('note_on', 'note_off'):
                prio = 7
            else:
                prio = 1
            items.append((at, prio, msg))
        for start, end, pitch, vel in out_notes:
            items.append((start, 6, mido.Message('note_on', note=pitch, velocity=vel, channel=DRUM_CHANNEL, time=0)))
            items.append((end, 5, mido.Message('note_off', note=pitch, velocity=0, channel=DRUM_CHANNEL, time=0)))
        new_tracks.append(_build_track(items))

    new_midi = mido.MidiFile(type=midi.type, ticks_per_beat=tpb)
    new_midi.tracks.extend(new_tracks)
    return new_midi, info


def _free_channels(midi):
    """Canali MIDI non usati dal brano (batteria esclusa), per le tracce generate.
    Se il brano li usa tutti, ripiega sui 15 canali melodici."""
    used = {msg.channel for tr in midi.tracks for msg in tr if hasattr(msg, 'channel')}
    melodic = [c for c in range(16) if c != DRUM_CHANNEL]
    free = [c for c in melodic if c not in used]
    return free if free else melodic


# --- Seed: riproducibilita' (protocollo Loop507) ---
_RUN_SEED_FALLBACK = {}


def _current_run_seed():
    """Seed dell'esecuzione corrente: lo imposta l'interfaccia (campo 'Seed globale',
    oppure casuale a ogni esecuzione e riportato nel report). Fuori dall'app ne genera uno una volta."""
    s = _setting('_run_seed')
    if s is None:
        s = _RUN_SEED_FALLBACK.setdefault('v', random.SystemRandom().randrange(2 ** 31))
    return int(s)


def _effective_seed(seed, salt):
    """Seed esplicito del metodo se presente (invariato, cosi' i vecchi seed continuano a valere);
    altrimenti un seed derivato in modo stabile da (seed globale, nome metodo)."""
    if seed is not None:
        return int(seed)
    return zlib.crc32(f"{_current_run_seed()}:{salt}".encode("utf-8"))


def _rng(seed, salt):
    """random.Random locale e riproducibile: nessun uso dello stato globale di `random`."""
    return random.Random(_effective_seed(seed, salt))


GM_PROGRAM_NAMES = [
    'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
    'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet',
    'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone',
    'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
    'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ',
    'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion',
    'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)', 'Electric Guitar (clean)',
    'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar Harmonics',
    'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass',
    'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2',
    'Violin', 'Viola', 'Cello', 'Contrabass',
    'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani',
    'String Ensemble 1', 'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2',
    'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit',
    'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet',
    'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2',
    'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax',
    'Oboe', 'English Horn', 'Bassoon', 'Clarinet',
    'Piccolo', 'Flute', 'Recorder', 'Pan Flute',
    'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
    'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)',
    'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)',
    'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)',
    'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)',
    'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)',
    'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)',
    'Sitar', 'Banjo', 'Shamisen', 'Koto',
    'Kalimba', 'Bag pipe', 'Fiddle', 'Shanai',
    'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock',
    'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal',
    'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet',
    'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot',
]
GM_PROGRAM_OPTIONS = ["Predefinito del metodo"] + [f"{i:03d} · {n}" for i, n in enumerate(GM_PROGRAM_NAMES)]
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Profili di tonalita' di Krumhansl-Kessler (stima della tonalita' dall'istogramma delle classi di altezza)
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _generated_program(default=0):
    """Programma GM per le tracce generate: scelta dell'utente ('gen_program_label') oppure il default del metodo."""
    label = _setting('gen_program_label')
    if isinstance(label, str) and label[:3].isdigit():
        return max(0, min(127, int(label[:3])))
    return default


def _generated_program_label():
    """'Violin (40)' se l'utente ha scelto uno strumento, altrimenti None (per il report)."""
    p = _generated_program(None)
    return None if p is None else f"{GM_PROGRAM_NAMES[p]} ({p})"


def estimate_key(midi, ignore_drums=True):
    """Stima la tonalita' (Krumhansl-Schmuckler) dalle durate delle classi di altezza.
    Ritorna (tonica 0-11, 'Maggiore'|'Minore', correlazione) oppure None se non ci sono note utili."""
    hist = np.zeros(12)
    for tr in midi.tracks:
        for n in extract_notes(tr, midi.ticks_per_beat):
            if ignore_drums and n['channel'] == DRUM_CHANNEL:
                continue
            hist[n['pitch'] % 12] += max(1, n['end'] - n['start'])
    if hist.sum() == 0 or np.std(hist) == 0:
        return None
    best = None
    for tonic in range(12):
        for mode, profile in (("Maggiore", _KS_MAJOR), ("Minore", _KS_MINOR)):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if not np.isnan(r) and (best is None or r > best[2]):
                best = (tonic, mode, float(r))
    return best


def midi_last_note_tick(midi):
    """Tick assoluto dell'ultima nota (note_on/note_off) di tutto il file: la fine della MUSICA,
    che puo' essere molto prima della fine 'grezza' del file (end_of_track o marker lontani)."""
    last = 0
    for track in midi.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if t > last and msg.type in ('note_on', 'note_off'):
                last = t
    return last


def midi_length_seconds(midi, end_tick=None):
    """Durata in secondi calcolata dai tempi assoluti (stesso risultato di mido.MidiFile.length,
    ma senza fondere e rileggere tutti i messaggi: ~15x piu' veloce sui file grandi).
    None per i file di tipo 2 (tracce indipendenti: la durata non e' definita).
    end_tick: se indicato, misura fino a quel tick invece che alla fine grezza del file."""
    if midi.type == 2:
        return None
    tpb = midi.ticks_per_beat
    tempi = [(0, 500000)]  # tempo di default MIDI: 120 BPM
    end = 0
    for track in midi.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'set_tempo':
                tempi.append((t, msg.tempo))
        if t > end:
            end = t
    if end_tick is not None:
        end = end_tick
    tempi.sort(key=lambda x: x[0])  # stabile: a parita' di tick vale l'ultimo
    seconds = 0.0
    for i, (tick, tempo) in enumerate(tempi):
        if tick >= end:
            break
        nxt = tempi[i + 1][0] if i + 1 < len(tempi) else end
        nxt = min(nxt, end)
        seconds += (nxt - tick) * tempo / (tpb * 1_000_000)
    return seconds


def validate_midi_output(source_midi, output_midi):
    """Controlli di coerenza sul risultato: note totali, note non bilanciate (appese o note_off orfani),
    durata prima/dopo, tempo/metrica/tonalita' preservati."""
    def count(m):
        notes, unbalanced = 0, 0
        for tr in m.tracks:
            c = defaultdict(int)
            for x in tr:
                if x.type == 'note_on' and x.velocity > 0:
                    notes += 1
                    c[(x.note, x.channel)] += 1
                elif x.type in ('note_off', 'note_on'):
                    c[(x.note, x.channel)] -= 1
            unbalanced += sum(1 for v in c.values() if v != 0)
        return notes, unbalanced

    def duration(m):
        secs = midi_length_seconds(m)
        return None if secs is None else round(secs, 2)

    def meta_types(m):
        return {x.type for tr in m.tracks for x in tr if x.is_meta and x.type in _GLOBAL_META_TYPES}

    notes_in, _ = count(source_midi)
    notes_out, unbalanced = count(output_midi)
    meta_ok = meta_types(source_midi) <= meta_types(output_midi)
    return {
        'notes_in': notes_in, 'notes_out': notes_out, 'unbalanced': unbalanced,
        'duration_in': duration(source_midi), 'duration_out': duration(output_midi),
        'tracks_in': len(source_midi.tracks), 'tracks_out': len(output_midi.tracks),
        'meta_ok': meta_ok, 'ok': unbalanced == 0 and meta_ok,
    }


def normalize_midi(midi):
    """Se il file ha UNA sola traccia con note su piu' canali (tipo 0), la divide in una traccia per canale,
    cosi' tutti i metodi lavorano per strumento. Ritorna (midi, e_stato_diviso)."""
    if len(midi.tracks) == 1:
        chans = {m.channel for m in midi.tracks[0] if m.type == 'note_on' and m.velocity > 0}
        if len(chans) > 1:
            return _split_type0_to_tracks(midi), True
    return midi, False


def append_events_to_track(track, events):
    """Accoda a `track` gli eventi [{'msg', 'abs_time'}] convertendoli in delta-time.
    Ordine stabile per tempo assoluto, con note_off prima di note_on allo stesso tick
    (evita sovrapposizioni e note appese). Sostituisce il blocco sort+delta ripetuto ovunque."""
    last = 0
    for ev in sorted(events, key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1)):
        track.append(_clone(ev['msg'], time=max(0, ev['abs_time'] - last)))
        last = ev['abs_time']
    return track


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


@_preserves_meta
def midi_costas_pitch_permutation(original_midi, transpose_octave=0, preserve_drums=None):
    """
    Modalita' 1: Permutazione Pitch (cromatica).
    Usa una matrice di Costas di ordine 12 (p=13, primo) come cifrario di
    sostituzione deterministico e privo di autocorrelazione per le classi di
    altezza: ogni classe di pitch (0-11) viene rimappata secondo perm[pitch_class],
    mantenendo l'ottava originale (+ eventuale trasposizione).
    A differenza del Random Pitch Transformer, la mappatura e' fissa e
    biunivoca: stesso pitch in ingresso -> sempre stesso pitch in uscita.
    """
    preserve = _drums_preserved(preserve_drums)
    perm, n, p, g = generate_costas_array(12)  # p=13 -> n=12, mappa cromatica esatta
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(original_track, 'name') and original_track.name:
            new_track.name = original_track.name

        for msg in original_track:
            if (msg.type in ('note_on', 'note_off') and hasattr(msg, 'note')
                    and not (preserve and msg.channel == DRUM_CHANNEL)):
                pitch_class = msg.note % 12
                octave = msg.note // 12
                new_pitch_class = perm[pitch_class % n]
                new_pitch = (octave * 12) + new_pitch_class + (transpose_octave * 12)
                new_pitch = max(0, min(127, new_pitch))
                new_track.append(_clone(msg, note=new_pitch))
            else:
                new_track.append(msg)

        new_midi.tracks.append(new_track)
    return new_midi, (n, p, g)


@_preserves_meta
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
        _header = _extract_instrument_header(original_track)
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

        new_track = mido.MidiTrack()
        if _name:
            new_track.name = _name
        for _h in _header:
            new_track.append(_h)
        append_events_to_track(new_track, final_events)

        new_midi.tracks.append(new_track)
    return new_midi, (n, p, g)


@_preserves_meta
def midi_costas_generator(original_midi, min_order, base_pitch, pitch_range_semitones, step_beats, channel=None):
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
    if channel is None:
        channel = _free_channels(original_midi)[0]
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
        warn("Il brano originale non contiene eventi validi. Il generatore Costas non verra' aggiunto.")
        return new_midi, (n, p, g)

    step_ticks = max(1, int(round(step_beats * original_midi.ticks_per_beat)))
    costas_track = mido.MidiTrack()
    costas_track.name = f"Costas Generator (n={n}, p={p}, g={g})"
    costas_track.append(mido.Message('program_change', program=_generated_program(0), channel=channel, time=0))  # Acoustic Grand Piano di default

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

    append_events_to_track(costas_track, events)

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
    preserve = _drums_preserved()
    seen = []
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and not (preserve and msg.channel == DRUM_CHANNEL):
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


@_preserves_meta
@_keeps_tempo_map
def midi_stockhausen_punktuelle(original_midi, serialize_duration=True, serialize_dynamics=True,
                                 serialize_timbre=True, isolamento_punti=True, preserve_drums=None):
    """
    Serialismo integrale multiparametrico (stile Stockhausen/Boulez).
    Estrae una fila a 12 elementi dal brano, poi applica 4 forme indipendenti
    della fila a 4 parametri scorrelati di ogni singola nota, trattata come un
    punto sonoro isolato:
      - Altezza  -> forma Prima (P)         (classe di pitch rimappata)
      - Durata   -> forma Retrograda (R)    (12 classi di durata fisse)
      - Dinamica -> forma Inversione (I)    (12 classi di velocity fisse)
      - Timbro   -> forma Retrograda-Inversa (RI) (la nota "salta" su un'altra
        traccia/strumento del brano, nello spirito di Kreuzspiel dove
        Stockhausen disperde una linea tra strumenti diversi)
    Le note vengono processate in ordine cronologico assoluto attraverso tutte
    le tracce (non per traccia separata), perche' nella musica puntillistica
    ogni punto e' indipendente dal contesto melodico originale. La struttura
    a piu' tracce/strumenti del brano di partenza viene pero' SEMPRE
    preservata (stesso numero di tracce, stessi nomi, stesso program_change):
    solo la destinazione di ciascuna nota puo' cambiare, non l'esistenza
    delle tracce — altrimenti DAW come Logic Pro perdono l'assegnazione
    degli strumenti e riproducono tutto con un patch di default.
    Se isolamento_punti=True, ogni nota viene accorciata e seguita da un
    micro-silenzio, per accentuare la natura di "punti" isolati nello spazio
    sonoro invece che di frasi legate.
    """
    preserve = _drums_preserved(preserve_drums)
    row = derive_twelve_tone_row(original_midi)
    row_P = _row_prime(row)
    row_R = _row_retrograde(row)
    row_I = _row_inversion(row)
    row_RI = _row_retrograde_inversion(row)

    ticks_per_beat = original_midi.ticks_per_beat
    base_unit = max(1, ticks_per_beat // 8)
    DURATION_CLASSES = [base_unit * (k + 1) for k in range(12)]  # 12 durate crescenti, in ottavi di beat
    DYNAMICS_CLASSES = [int(v) for v in np.linspace(24, 127, 12)]  # ppp -> fff su 12 gradini

    num_tracks = len(original_midi.tracks)
    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]
    track_channels = [_get_track_default_channel(t) for t in original_midi.tracks]
    # tracce verso cui il 'timbro' seriale puo' saltare (la batteria resta al suo posto)
    # (solo tracce con note: una traccia di metadati/tempo non deve ricevere note su un canale a caso)
    _has_notes = [any(m.type == 'note_on' and m.velocity > 0 for m in t) for t in original_midi.tracks]
    timbre_targets = [k for k in range(num_tracks)
                      if _has_notes[k] and not (preserve and track_channels[k] == DRUM_CHANNEL)]

    # Raccogli tutte le note come punti indipendenti, in ordine cronologico assoluto
    all_points = []
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        for nd in notes:
            all_points.append({
                'start': nd['start'],
                'orig_pitch': nd['pitch'],
                'orig_velocity': nd['velocity'],
                'track_idx': track_idx,
                'end': nd['end'],
                'channel': nd['channel'],
            })

    if not all_points:
        warn("Nessuna nota trovata nel brano. La tecnica Punktuelle non verra' applicata.")
        return original_midi, row

    all_points.sort(key=lambda x: x['start'])

    events_per_track = [[] for _ in range(num_tracks)]

    for i, point in enumerate(all_points):
        if preserve and point['channel'] == DRUM_CHANNEL:
            d_end = max(point['start'] + 1, point['end'])
            events_per_track[point['track_idx']].append({'msg': mido.Message('note_on', note=point['orig_pitch'], velocity=point['orig_velocity'], channel=point['channel'], time=0), 'abs_time': point['start']})
            events_per_track[point['track_idx']].append({'msg': mido.Message('note_off', note=point['orig_pitch'], velocity=0, channel=point['channel'], time=0), 'abs_time': d_end})
            continue
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

        if serialize_timbre and len(timbre_targets) > 1:
            target_track_idx = timbre_targets[row_RI[i % 12] % len(timbre_targets)]
        else:
            target_track_idx = point['track_idx']

        channel = track_channels[target_track_idx]

        note_start = point['start']
        if isolamento_punti:
            note_len = max(1, int(duration * 0.55))  # nota staccata: meno della meta' dello slot
        else:
            note_len = max(1, duration)

        events_per_track[target_track_idx].append({'msg': mido.Message('note_on', note=new_pitch, velocity=velocity, channel=channel, time=0), 'abs_time': note_start})
        events_per_track[target_track_idx].append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=channel, time=0), 'abs_time': note_start + note_len})

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx in range(num_tracks):
        new_track = mido.MidiTrack()
        new_track.name = track_names[track_idx]
        for _h in track_headers[track_idx]:
            new_track.append(_h)

        track_events = events_per_track[track_idx]
        append_events_to_track(new_track, track_events)

        new_midi.tracks.append(new_track)

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


@_preserves_meta
@_keeps_tempo_map
def midi_boulez_multiplication(original_midi, set_size=4, chord_density=0, register_spread=1, preserve_drums=None):
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
    preserve = _drums_preserved(preserve_drums)
    set_a, set_b = derive_boulez_sets(original_midi, set_size)
    pivot = set_b[0] if set_b else 0
    multiplied = boulez_multiply_sets(set_a, set_b, pivot)

    if not multiplied:
        warn("Materiale insufficiente per la moltiplicazione d'accordi. Restituito il MIDI originale.")
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
        _header = _extract_instrument_header(original_track)
        notes = extract_notes(original_track, original_midi.ticks_per_beat)

        if not notes:
            new_midi.tracks.append(original_track)
            continue

        final_events = []
        for nd in notes:
            if preserve and nd['channel'] == DRUM_CHANNEL:
                final_events.append({'msg': mido.Message('note_on', note=nd['pitch'], velocity=nd['velocity'], channel=nd['channel'], time=0), 'abs_time': nd['start']})
                final_events.append({'msg': mido.Message('note_off', note=nd['pitch'], velocity=0, channel=nd['channel'], time=0), 'abs_time': max(nd['start'] + 1, nd['end'])})
                continue
            base_octave = nd['pitch'] // 12
            for offset_idx, pc in enumerate(chord_pcs):
                octave_shift = 0
                if register_spread > 1:
                    octave_shift = (offset_idx % register_spread) - (register_spread // 2)
                new_pitch = max(0, min(127, (base_octave + octave_shift) * 12 + pc))
                final_events.append({'msg': mido.Message('note_on', note=new_pitch, velocity=nd['velocity'], channel=nd['channel'], time=0), 'abs_time': nd['start']})
                final_events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0, channel=nd['channel'], time=0), 'abs_time': nd['end']})

        new_track = mido.MidiTrack()
        if _name:
            new_track.name = _name
        for _h in _header:
            new_track.append(_h)
        append_events_to_track(new_track, final_events)

        new_midi.tracks.append(new_track)

    return new_midi, (set_a, set_b, multiplied)


# --- Compositori: Iannis Xenakis — Musica Stocastica (Nuvole di Suoni) ---
# Rif: Pithoprakta (1955-56, distribuzione Gaussiana per i glissandi),
# Achorripsis (1956-57, processo di Poisson per la densita' degli eventi nel
# tempo, distribuzione esponenziale per gli intertempi), teoria dei crivelli
# (cribles: insiemi costruiti per unione di classi di resto modulari).
# A differenza di Stockhausen/Boulez (deterministici), qui i parametri sono
# governati da distribuzioni di probabilita' controllate — "il minimo di
# vincoli logici necessario" (Xenakis) — non dal random uniforme grezzo.

def generate_sieve(moduli_residues, universe=(0, 128)):
    """
    Crivello di Xenakis (crible): unione di classi di resto x = r (mod m).
    moduli_residues: lista di coppie (m, r). Es. [(3,0),(4,1)] = tutti gli
    interi congrui a 0 mod 3 UNITI a tutti quelli congrui a 1 mod 4.
    """
    lo, hi = universe
    sieve_set = set()
    for m, r in moduli_residues:
        if m <= 0:
            continue
        r = r % m
        for x in range(lo, hi):
            if (x - r) % m == 0:
                sieve_set.add(x)
    return sorted(sieve_set)


def parse_sieve_string(s):
    """Parsa una stringa tipo '3:0, 4:1, 7:3' in una lista di coppie (m, r)."""
    pairs = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk or ':' not in chunk:
            continue
        m_str, r_str = chunk.split(':', 1)
        try:
            m = int(m_str.strip())
            r = int(r_str.strip())
            if m > 0:
                pairs.append((m, r % m))
        except ValueError:
            continue
    return pairs


def _xenakis_snap_to_sieve(value, sieve):
    if not sieve:
        return value
    return min(sieve, key=lambda s: abs(s - value))


@_preserves_meta
def midi_xenakis_stochastic(original_midi, sieve_moduli, mean_events_per_beat, pitch_center,
                             pitch_spread_semitones, duration_mean_beats, velocity_mean,
                             velocity_spread, seed=None):
    """
    Genera una "nuvola di suoni" stocastica (Pithoprakta/Achorripsis):
      - Tempi di attacco: processo di Poisson (intertempi con distribuzione
        esponenziale), tasso medio mean_events_per_beat eventi/beat.
      - Altezza: distribuzione Gaussiana attorno a pitch_center, quantizzata
        sul crivello (sieve) definito da sieve_moduli.
      - Durata: distribuzione esponenziale attorno a duration_mean_beats.
      - Dinamica: distribuzione Gaussiana attorno a velocity_mean.
    Copre l'intera durata del brano originale; le tracce originali restano
    intatte, la nuvola si aggiunge come nuova traccia.
    """
    rng = np.random.default_rng(_effective_seed(seed, "xenakis"))
    gen_channel = _free_channels(original_midi)[0]
    sieve = generate_sieve(sieve_moduli, universe=(0, 128))
    if not sieve:
        sieve = list(range(128))

    ticks_per_beat = original_midi.ticks_per_beat
    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)

    if total_ticks == 0:
        warn("Il brano originale non contiene eventi validi. La nuvola stocastica non verra' aggiunta.")
        return new_midi, sieve

    xenakis_track = mido.MidiTrack()
    xenakis_track.name = f"Xenakis Stochastic Cloud (sieve n={len(sieve)})"
    xenakis_track.append(mido.Message('program_change', program=_generated_program(0), channel=gen_channel, time=0))  # Acoustic Grand Piano di default

    total_beats = total_ticks / ticks_per_beat
    lam = max(0.05, mean_events_per_beat)

    events = []
    t = 0.0
    while t < total_beats:
        inter_arrival = rng.exponential(1.0 / lam)  # processo di Poisson
        t += inter_arrival
        if t >= total_beats:
            break

        raw_pitch = rng.normal(pitch_center, pitch_spread_semitones)
        pitch = int(round(_xenakis_snap_to_sieve(raw_pitch, sieve)))
        pitch = max(0, min(127, pitch))

        dur_beats = max(0.05, rng.exponential(duration_mean_beats))
        velocity = int(round(np.clip(rng.normal(velocity_mean, velocity_spread), 1, 127)))

        start_tick = int(round(t * ticks_per_beat))
        end_tick = int(round((t + dur_beats) * ticks_per_beat))

        events.append({'msg': mido.Message('note_on', note=pitch, velocity=velocity, channel=gen_channel, time=0), 'abs_time': start_tick})
        events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=gen_channel, time=0), 'abs_time': end_tick})

    append_events_to_track(xenakis_track, events)

    new_midi.tracks.append(xenakis_track)
    return new_midi, sieve


# --- Compositori: John Cage — Operazioni di Caso (I Ching / Music of Changes) ---
# Rif: Music of Changes (1951) — Cage costrui' delle "charts" (tabelle a 64
# caselle, una per esagramma) per altezza, durata e dinamica, e uso' il
# metodo classico delle tre monete dell'I Ching per scegliere, ad ogni
# passo, quale casella consultare. A differenza della statistica continua
# di Xenakis, qui il caso e' un'operazione discreta e procedurale — e il
# silenzio e' materiale musicale legittimo quanto il suono (stesso principio
# alla base di 4'33").

def _cage_toss_hexagram(rng):
    """6 lanci di linea (3 monete l'una) -> indice di esagramma 0..63 (metodo delle tre monete).
    Le 18 monete sono estratte in un'unica chiamata al generatore: la sequenza e' identica a
    18 chiamate singole, ma molto piu' veloce."""
    heads = rng.integers(0, 2, size=18).reshape(6, 3).sum(axis=1)  # teste per linea, 0..3
    idx = 0
    for h in heads:
        idx = (idx << 1) | (1 if int(h) in (1, 3) else 0)  # 7 o 9 -> teste dispari
    return idx


@_preserves_meta
@_keeps_tempo_map
def midi_cage_chance_operations(original_midi, silence_probability=0.15, duration_variety=True, seed=None):
    """
    Operazioni di caso in stile "Music of Changes": ogni nota del brano
    originale diventa un evento le cui proprieta' (altezza, durata,
    dinamica, presenza/assenza di suono) sono determinate da esagrammi
    indipendenti, generati con il metodo classico delle tre monete
    dell'I Ching — non pseudocasuale grezzo, ma la stessa procedura
    combinatoria (64 esiti equiprobabili) usata da Cage per costruire
    le proprie tabelle. Il silenzio ha silence_probability di sostituire
    ciascun evento: e' trattato come materiale, non come nota mancante.
    Ogni nota resta sulla propria traccia/strumento originale: la struttura
    a piu' tracce del brano di partenza (numero, nomi, program_change) e'
    sempre preservata, altrimenti DAW come Logic Pro perdono l'assegnazione
    degli strumenti e riproducono tutto con un patch di default.
    """
    rng = np.random.default_rng(_effective_seed(seed, "cage"))
    ticks_per_beat = original_midi.ticks_per_beat
    base_unit = max(1, ticks_per_beat // 4)

    num_tracks = len(original_midi.tracks)
    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]

    all_points = []
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        for nd in notes:
            nd['track_idx'] = track_idx
            all_points.append(nd)

    if not all_points:
        warn("Nessuna nota trovata. Le operazioni di caso non verranno applicate.")
        return original_midi, []

    all_points.sort(key=lambda x: x['start'])

    preserve = _drums_preserved()
    drum_points = [p for p in all_points if preserve and p['channel'] == DRUM_CHANNEL]
    melodic_points = [p for p in all_points if not (preserve and p['channel'] == DRUM_CHANNEL)]

    pitches = [p['pitch'] for p in (melodic_points or all_points)]
    pitch_lo, pitch_hi = min(pitches), max(pitches)
    if pitch_hi <= pitch_lo:
        pitch_hi = pitch_lo + 12

    # Tabelle a 64 caselle (una per ciascun esagramma), come nelle charts di Cage
    PITCH_CHART = [pitch_lo + (i % (pitch_hi - pitch_lo + 1)) for i in range(64)]
    DURATION_CHART = [base_unit * (1 + (i % 8)) for i in range(64)]
    DYNAMICS_CHART = [int(v) for v in np.linspace(20, 120, 64)]

    events_per_track = [[] for _ in range(num_tracks)]
    hexagram_log = []
    # Batteria preservata: i colpi restano com'erano (altezza = strumento del kit), niente caso
    for point in drum_points:
        events_per_track[point['track_idx']].append({'msg': mido.Message('note_on', note=point['pitch'], velocity=point['velocity'], channel=point['channel'], time=0), 'abs_time': point['start']})
        events_per_track[point['track_idx']].append({'msg': mido.Message('note_off', note=point['pitch'], velocity=0, channel=point['channel'], time=0), 'abs_time': max(point['end'], point['start'] + 1)})
    for point in melodic_points:
        hex_pitch = _cage_toss_hexagram(rng)
        hex_dur = _cage_toss_hexagram(rng) if duration_variety else hex_pitch
        hex_dyn = _cage_toss_hexagram(rng)
        hex_silence = _cage_toss_hexagram(rng)
        hexagram_log.append((hex_pitch, hex_dur, hex_dyn, hex_silence))

        is_silence = (hex_silence / 64.0) < silence_probability
        if is_silence:
            continue  # il silenzio e' l'esito legittimo: nessun evento sonoro

        pitch = max(0, min(127, PITCH_CHART[hex_pitch]))
        duration = DURATION_CHART[hex_dur]
        velocity = DYNAMICS_CHART[hex_dyn]

        target_track_idx = point['track_idx']
        note_start = point['start']
        events_per_track[target_track_idx].append({'msg': mido.Message('note_on', note=pitch, velocity=velocity, channel=point['channel'], time=0), 'abs_time': note_start})
        events_per_track[target_track_idx].append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=point['channel'], time=0), 'abs_time': note_start + max(1, duration)})

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx in range(num_tracks):
        new_track = mido.MidiTrack()
        new_track.name = track_names[track_idx]
        for _h in track_headers[track_idx]:
            new_track.append(_h)

        track_events = events_per_track[track_idx]
        append_events_to_track(new_track, track_events)

        new_midi.tracks.append(new_track)

    return new_midi, hexagram_log


# --- Compositori: Brian Eno — Musica Generativa (Cicli Asincroni) ---
# Rif: "Discreet Music" (1975), "Music for Airports" (1978) — sistemi
# costruiti da loop di nastro indipendenti, ciascuno contenente una singola
# nota, che ripetono al proprio periodo. Le lunghezze dei loop sono scelte
# "incommensurabili" tra loro (nell'album reale: ~23.5s, ~25.9s, ~29.2s...),
# cosi' che l'insieme impieghi un tempo lunghissimo (il MCM delle lunghezze)
# prima di ripetersi esattamente, pur restando gli elementi di base sempre
# gli stessi. "Non compongo la musica, compongo i sistemi che la generano"
# (Eno). Qui le lunghezze derivano da numeri primi distinti per garantire
# la stessa incommensurabilita' in modo deterministico e verificabile.

def _eno_prime_sequence(count, start_from=11):
    """Genera i primi `count` numeri primi a partire da start_from, per ottenere
    lunghezze di ciclo il piu' possibile incommensurabili tra loro."""
    primes = []
    candidate = start_from if start_from % 2 != 0 else start_from + 1
    while len(primes) < count:
        if _costas_is_prime(candidate):
            primes.append(candidate)
        candidate += 2
    return primes


@_preserves_meta
def midi_eno_generative(original_midi, num_loops=6, min_loop_beats=8, max_loop_beats=32,
                         note_length_ratio=0.35, duration_multiplier=4, velocity_base=55,
                         seed=None):
    """
    Genera un sistema di loop asincroni in stile Music for Airports/Discreet
    Music: ogni loop ripete una singola nota (derivata dal materiale del
    brano originale) al proprio periodo indipendente. Le lunghezze dei loop
    sono multipli di numeri primi distinti, cosi' che l'intero sistema
    impieghi un tempo lunghissimo prima di ripetersi esattamente — la stessa
    logica dei nastri fisici di lunghezza diversa che Eno faceva girare in
    loop, sfasandosi continuamente l'uno rispetto all'altro.
    Le tracce originali restano intatte; il sistema generativo si aggiunge
    come nuove tracce indipendenti (una per loop), per poter regolare in DAW
    volume/timbro di ciascun loop separatamente.
    """
    preserve = _drums_preserved()
    rng = np.random.default_rng(_effective_seed(seed, "eno"))
    gen_channel = _free_channels(original_midi)[0]
    ticks_per_beat = original_midi.ticks_per_beat

    pitches_found = []
    for track in original_midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note not in pitches_found and not (preserve and msg.channel == DRUM_CHANNEL):
                pitches_found.append(msg.note)
    if not pitches_found:
        warn("Nessuna nota trovata nel brano. Il sistema generativo non verra' aggiunto.")
        return original_midi, []
    pitches_found.sort()

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    total_ticks = 0
    for track in original_midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
        total_ticks = max(total_ticks, current_time)
    if total_ticks == 0:
        total_ticks = ticks_per_beat * 4 * 8
    total_ticks = int(total_ticks * max(1, duration_multiplier))

    primes = _eno_prime_sequence(num_loops, start_from=11)
    min_ticks = int(min_loop_beats * ticks_per_beat)
    max_ticks = max(min_ticks + ticks_per_beat, int(max_loop_beats * ticks_per_beat))

    loops_info = []
    for i in range(num_loops):
        p = primes[i]
        scale = min_ticks + (p % max(1, (max_ticks - min_ticks)))
        loop_len_ticks = max(ticks_per_beat, scale)

        pitch = pitches_found[i % len(pitches_found)]
        note_len = max(1, int(loop_len_ticks * note_length_ratio))
        phase_offset = int(rng.uniform(0, loop_len_ticks))  # entrata sfalsata del loop

        loop_track = mido.MidiTrack()
        loop_track.name = f"Eno Loop {i + 1} (pitch={pitch}, ciclo={loop_len_ticks}t, primo={p})"
        loop_track.append(mido.Message('program_change', program=_generated_program(0), channel=gen_channel, time=0))

        events = []
        t = phase_offset
        while t < total_ticks:
            vel = int(np.clip(velocity_base + rng.normal(0, 6), 15, 90))
            events.append({'msg': mido.Message('note_on', note=pitch, velocity=vel, channel=gen_channel, time=0), 'abs_time': t})
            events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=gen_channel, time=0), 'abs_time': t + note_len})
            t += loop_len_ticks

        append_events_to_track(loop_track, events)

        new_midi.tracks.append(loop_track)
        loops_info.append((pitch, loop_len_ticks, p))

    return new_midi, loops_info


# --- Compositori: Johann Sebastian Bach — Canone Rigoroso (Contrappunto Matematico) ---
# Rif: L'Arte della Fuga, Canoni enigmatici, Offerta Musicale — Bach costruiva
# canoni per trasformazione rigorosa e deterministica di un soggetto (dux):
# imitazione esatta a distanza di tempo fissa (comes), a un dato intervallo,
# talvolta con inversione (canone al rovescio), moto retrogrado (canone
# cancrizzante) o aumentazione ritmica (canone per aumentazione). La
# costruzione e' interamente deterministica: nessun elemento stocastico, la
# coerenza nasce dalla trasformazione esatta di un'unica linea generatrice.

def derive_bach_subject(original_midi, max_notes=24):
    """Estrae il soggetto (dux): le prime max_notes note della prima traccia
    con contenuto melodico, in ordine cronologico."""
    for track in original_midi.tracks:
        notes = extract_notes(track, original_midi.ticks_per_beat)
        if _drums_preserved():
            notes = [n for n in notes if n['channel'] != DRUM_CHANNEL]
        if notes:
            notes.sort(key=lambda n: n['start'])
            return notes[:max_notes]
    return []


@_preserves_meta
def midi_bach_canon(original_midi, num_voices=2, interval_semitones=7,
                     delay_beats=2, transformation="Nessuna (canone rigoroso)",
                     augmentation_factor=2):
    """
    Costruisce un canone rigoroso: il soggetto (dux) e' estratto dal brano,
    poi ogni voce successiva (comes) lo ripropone a distanza di delay_beats,
    trasposta di interval_semitones * indice-voce, ed eventualmente
    trasformata (inversione, retrogrado, aumentazione ritmica) — le tecniche
    classiche del contrappunto rigoroso bachiano. Ogni voce e' una traccia
    MIDI indipendente; le tracce originali restano intatte.
    """
    subject = derive_bach_subject(original_midi, max_notes=24)
    if not subject:
        warn("Nessun soggetto melodico trovato. Il canone non verra' generato.")
        return original_midi, []

    ticks_per_beat = original_midi.ticks_per_beat
    t0 = subject[0]['start']
    norm_subject = [{'pitch': n['pitch'], 'start': n['start'] - t0, 'end': n['end'] - t0, 'velocity': n['velocity']} for n in subject]
    axis_pitch = norm_subject[0]['pitch']  # asse di inversione = prima nota del soggetto (dux)

    if transformation == "Retrogrado (canone cancrizzante)":
        total_dur = max(n['end'] for n in norm_subject)
        norm_subject = [{'pitch': n['pitch'], 'start': total_dur - n['end'], 'end': total_dur - n['start'], 'velocity': n['velocity']} for n in norm_subject]
        norm_subject.sort(key=lambda n: n['start'])

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    delay_ticks = int(delay_beats * ticks_per_beat)
    bach_free = _free_channels(original_midi)
    voices_info = []
    for v in range(num_voices):
        voice_track = mido.MidiTrack()
        transpose = interval_semitones * v
        aug = augmentation_factor if (v > 0 and transformation == "Aumentazione ritmica (comes raddoppiato)") else 1
        channel = bach_free[v % len(bach_free)]
        voice_track.name = f"Bach Canone Voce {v + 1} ({'dux' if v == 0 else 'comes'}, +{transpose}st, delay={v * delay_beats}beat, aug x{aug})"
        voice_track.append(mido.Message('program_change', program=_generated_program(0), channel=channel, time=0))

        events = []
        voice_delay = delay_ticks * v
        for n in norm_subject:
            pitch = n['pitch'] + transpose
            if v > 0 and transformation == "Inversione (canone al rovescio)":
                pitch = (2 * axis_pitch - n['pitch']) + transpose
            pitch = max(0, min(127, pitch))
            start = int(n['start'] * aug) + voice_delay
            end = int(n['end'] * aug) + voice_delay
            events.append({'msg': mido.Message('note_on', note=pitch, velocity=n['velocity'], channel=channel, time=0), 'abs_time': start})
            events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=channel, time=0), 'abs_time': max(start + 1, end)})

        append_events_to_track(voice_track, events)

        new_midi.tracks.append(voice_track)
        voices_info.append((transpose, v * delay_beats, aug))

    return new_midi, voices_info


# --- Compositori: Philip Glass — Processo Additivo (Musica a Moduli) ---
# Rif: "Two Pages" (1968), "1+1" (1968), "Music in Twelve Parts" — Glass
# costruisce brani a partire da una cellula ritmico-melodica breve che si
# accresce (o si riduce) di una nota per volta ad ogni stadio: 1 nota, poi
# le prime 2, poi le prime 3... (processo additivo), talvolta seguito dal
# processo inverso (sottrattivo). La trasformazione e' un processo
# aritmetico esplicito applicato alla LUNGHEZZA della cellula, non alla sua
# sostanza melodica, che resta sempre quella derivata dal brano originale.

def derive_glass_cell(original_midi, cell_length=8):
    for track in original_midi.tracks:
        notes = extract_notes(track, original_midi.ticks_per_beat)
        if _drums_preserved():
            notes = [n for n in notes if n['channel'] != DRUM_CHANNEL]
        if notes:
            notes.sort(key=lambda n: n['start'])
            return notes[:cell_length]
    return []


@_preserves_meta
def midi_glass_additive(original_midi, cell_length_notes=8, direction="Additivo (solo crescita)",
                         repeats_per_stage=2):
    """
    Estrae una cellula di cell_length_notes note dal brano e la sottopone al
    processo additivo di Glass: ad ogni stadio la cellula viene troncata a
    1, 2, 3... note (fino alla lunghezza piena), ciascuno stadio ripetuto
    repeats_per_stage volte prima di passare allo stadio successivo. Se
    direction e' "additivo-sottrattivo", dopo aver raggiunto la lunghezza
    piena il processo si inverte, tornando a 1 nota. Il risultato si
    aggiunge come nuova traccia; le tracce originali restano intatte.
    """
    gen_channel = _free_channels(original_midi)[0]
    cell = derive_glass_cell(original_midi, cell_length_notes)
    if len(cell) < 2:
        warn("Materiale insufficiente per costruire la cellula. Il processo additivo non verra' generato.")
        return original_midi, []

    ticks_per_beat = original_midi.ticks_per_beat
    t0 = cell[0]['start']
    norm_cell = [{'pitch': n['pitch'], 'start': n['start'] - t0, 'end': n['end'] - t0, 'velocity': n['velocity']} for n in cell]

    stages = list(range(1, len(norm_cell) + 1))
    if direction == "Additivo-sottrattivo (cresce poi decresce)":
        stages = stages + list(range(len(norm_cell) - 1, 0, -1))

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    glass_track = mido.MidiTrack()
    glass_track.name = f"Glass Additive Process (cellula={len(norm_cell)} note, {len(stages)} stadi)"
    glass_track.append(mido.Message('program_change', program=_generated_program(0), channel=gen_channel, time=0))

    events = []
    cursor = 0
    for stage_len in stages:
        sub_cell = norm_cell[:stage_len]
        stage_dur = max(n['end'] for n in sub_cell)
        for _ in range(max(1, repeats_per_stage)):
            for n in sub_cell:
                start = cursor + n['start']
                end = cursor + n['end']
                events.append({'msg': mido.Message('note_on', note=n['pitch'], velocity=n['velocity'], channel=gen_channel, time=0), 'abs_time': start})
                events.append({'msg': mido.Message('note_off', note=n['pitch'], velocity=0, channel=gen_channel, time=0), 'abs_time': max(start + 1, end)})
            cursor += stage_dur

    append_events_to_track(glass_track, events)

    new_midi.tracks.append(glass_track)
    return new_midi, stages


# --- Compositori: Olivier Messiaen — Modi a Trasposizione Limitata + Ritmo Non Retrogradabile ---
# Rif: "Technique de mon langage musical" (1944) — i modi a trasposizione
# limitata sono scale che, per costruzione simmetrica, ammettono un numero
# di trasposizioni distinte minore delle 12 possibili (es. Modo 2 =
# ottatonico, 3 trasposizioni; Modo 3, 4 trasposizioni; Modo 6/7, 6
# trasposizioni). I ritmi non retrogradabili sono sequenze di durate
# palindrome: identiche lette avanti o indietro, per costruzione simmetrica
# attorno a un valore centrale — una proprieta' puramente aritmetica.

MESSIAEN_MODES = {
    2: [0, 1, 3, 4, 6, 7, 9, 10],       # Modo 2 (ottatonico) — 3 trasposizioni
    3: [0, 2, 3, 4, 6, 7, 8, 10, 11],   # Modo 3 — 4 trasposizioni
    4: [0, 1, 2, 5, 6, 7, 8, 11],       # Modo 4 — 6 trasposizioni
    6: [0, 2, 4, 5, 6, 8, 10, 11],      # Modo 6 — 6 trasposizioni
    7: [0, 1, 2, 3, 5, 6, 7, 8, 9, 11], # Modo 7 — 6 trasposizioni
}


def _messiaen_snap_to_mode(pitch, mode_intervals, transposition=0):
    pc = (pitch - transposition) % 12
    nearest = min(mode_intervals, key=lambda m: min(abs(m - pc), 12 - abs(m - pc)))
    delta = nearest - pc
    if delta > 6:
        delta -= 12
    if delta < -6:
        delta += 12
    return pitch + delta


def build_non_retrogradable_rhythm(cell_length, base_unit_ticks, rng):
    """Costruisce una sequenza di durate palindroma (ritmo non retrogradabile):
    identica letta avanti o indietro, per costruzione simmetrica attorno a un
    valore centrale libero."""
    half = cell_length // 2
    half_values = [base_unit_ticks * int(rng.integers(1, 4)) for _ in range(half)]
    center = [base_unit_ticks * int(rng.integers(1, 4))] if cell_length % 2 == 1 else []
    return half_values + center + list(reversed(half_values))


@_preserves_meta
@_keeps_tempo_map
def midi_messiaen_modes(original_midi, mode_number=2, transposition=0,
                         non_retrogradable_rhythm=True, rhythm_cell_notes=7, seed=None, preserve_drums=None):
    """
    Riquantizza ogni altezza del brano sulla classe piu' vicina del modo a
    trasposizione limitata scelto, e (se attivo) sostituisce il ritmo
    originale con una sequenza di durate non retrogradabile (palindroma),
    riapplicata ciclicamente. Struttura a piu' tracce sempre preservata.
    """
    preserve = _drums_preserved(preserve_drums)
    rng = np.random.default_rng(_effective_seed(seed, "messiaen"))
    mode_intervals = MESSIAEN_MODES.get(mode_number, MESSIAEN_MODES[2])
    ticks_per_beat = original_midi.ticks_per_beat
    base_unit = max(1, ticks_per_beat // 4)

    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]

    rhythm_cell = build_non_retrogradable_rhythm(rhythm_cell_notes, base_unit, rng) if non_retrogradable_rhythm else None

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    any_notes = False
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)
        new_track = mido.MidiTrack()
        new_track.name = track_names[track_idx]
        for h in track_headers[track_idx]:
            new_track.append(h)

        if not notes:
            new_midi.tracks.append(new_track)
            continue
        any_notes = True
        notes.sort(key=lambda n: n['start'])

        events = []
        cursor = notes[0]['start']
        rhythm_idx = 0
        for n in notes:
            if preserve and n['channel'] == DRUM_CHANNEL:
                events.append({'msg': mido.Message('note_on', note=n['pitch'], velocity=n['velocity'], channel=n['channel'], time=0), 'abs_time': n['start']})
                events.append({'msg': mido.Message('note_off', note=n['pitch'], velocity=0, channel=n['channel'], time=0), 'abs_time': max(n['start'] + 1, n['end'])})
                continue
            pitch = max(0, min(127, _messiaen_snap_to_mode(n['pitch'], mode_intervals, transposition)))
            if non_retrogradable_rhythm:
                dur = rhythm_cell[rhythm_idx % len(rhythm_cell)]
                rhythm_idx += 1
            else:
                dur = max(1, n['end'] - n['start'])
            start = cursor
            end = cursor + dur
            events.append({'msg': mido.Message('note_on', note=pitch, velocity=n['velocity'], channel=n['channel'], time=0), 'abs_time': start})
            events.append({'msg': mido.Message('note_off', note=pitch, velocity=0, channel=n['channel'], time=0), 'abs_time': end})
            cursor = end

        append_events_to_track(new_track, events)

        new_midi.tracks.append(new_track)

    if not any_notes:
        warn("Nessuna nota trovata. I modi di Messiaen non verranno applicati.")
        return original_midi, mode_intervals

    return new_midi, mode_intervals


# --- Compositori: Arvo Pärt — Tintinnabuli ---
# Rif: tecnica "tintinnabuli" (dal 1976: "Spiegel im Spiegel", "Für Alina")
# — due voci: la voce M (melodica) mantiene il materiale del brano
# originale; la voce T (tintinnabuli) suona SEMPRE una nota della triade di
# tonica, scelta secondo una regola fissa e deterministica (la piu' vicina
# sopra o sotto la nota M corrente). Il risultato nasce interamente dalla
# combinazione di queste due regole, senza libera scelta armonica —
# l'opposto della stocastica xenakisiana.

def _part_nearest_triad_tone(pitch, triad_pcs, position="T-1 (piu' vicina sotto)"):
    """Trova la nota della triade piu' vicina a `pitch`: sotto se la
    posizione contiene 'sotto', sopra se contiene 'sopra'; il numero indica
    quale delle note della triade, in ordine di vicinanza (1 = la piu'
    vicina, 2 = la seconda piu' vicina)."""
    rank = 2 if "2" in position else 1
    below = "sotto" in position

    candidates = sorted({octv * 12 + pc for octv in range(-1, 11) for pc in triad_pcs if 0 <= octv * 12 + pc <= 127})
    if below:
        pool = sorted([c for c in candidates if c <= pitch], reverse=True)
    else:
        pool = sorted([c for c in candidates if c >= pitch])

    if len(pool) >= rank:
        return pool[rank - 1]
    return pool[-1] if pool else pitch


@_preserves_meta
@_keeps_tempo_map
def midi_part_tintinnabuli(original_midi, tonic_key="C", triad_type="Minore",
                            t_voice_position="T-1 (piu' vicina sotto)", preserve_drums=None):
    """
    Trasforma ogni traccia in una coppia di voci: la voce M mantiene
    l'altezza originale, la voce T viene calcolata deterministicamente come
    la nota della triade di tonica piu' vicina alla nota M, nella posizione
    scelta. Struttura a piu' tracce raddoppiata (M + T per ogni traccia
    originale con contenuto melodico).
    """
    preserve = _drums_preserved(preserve_drums)
    part_free = _free_channels(original_midi)
    key_offset = get_key_offset(tonic_key)
    triad_intervals = [0, 3, 7] if triad_type == "Minore" else [0, 4, 7]
    triad_pcs = [(key_offset + iv) % 12 for iv in triad_intervals]

    ticks_per_beat = original_midi.ticks_per_beat
    track_headers = [_extract_instrument_header(t) for t in original_midi.tracks]
    track_names = [
        (t.name if hasattr(t, 'name') and t.name else f"Traccia {i + 1}")
        for i, t in enumerate(original_midi.tracks)
    ]

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    any_notes = False
    for track_idx, track in enumerate(original_midi.tracks):
        notes = extract_notes(track, ticks_per_beat)

        m_track = mido.MidiTrack()
        m_track.name = f"{track_names[track_idx]} (M-voice, Pärt)"
        for h in track_headers[track_idx]:
            m_track.append(h)

        t_track = mido.MidiTrack()
        t_channel = part_free[track_idx % len(part_free)]
        t_track.name = f"{track_names[track_idx]} (T-voice tintinnabuli)"
        t_track.append(mido.Message('program_change', program=_generated_program(8), channel=t_channel, time=0))  # celesta di default

        if not notes:
            # traccia di metadati/tempo: passa intatta, senza creare due tracce vuote (M/T) in DAW
            new_midi.tracks.append(mido.MidiTrack([_clone(m) for m in track]))
            continue
        any_notes = True
        notes.sort(key=lambda n: n['start'])

        m_events, t_events = [], []
        for n in notes:
            m_events.append({'msg': mido.Message('note_on', note=n['pitch'], velocity=n['velocity'], channel=n['channel'], time=0), 'abs_time': n['start']})
            m_events.append({'msg': mido.Message('note_off', note=n['pitch'], velocity=0, channel=n['channel'], time=0), 'abs_time': max(n['start'] + 1, n['end'])})

            if preserve and n['channel'] == DRUM_CHANNEL:
                continue  # la batteria non ha voce T

            t_pitch = max(0, min(127, _part_nearest_triad_tone(n['pitch'], triad_pcs, t_voice_position)))
            t_events.append({'msg': mido.Message('note_on', note=t_pitch, velocity=max(10, n['velocity'] - 15), channel=t_channel, time=0), 'abs_time': n['start']})
            t_events.append({'msg': mido.Message('note_off', note=t_pitch, velocity=0, channel=t_channel, time=0), 'abs_time': max(n['start'] + 1, n['end'])})

        for evs, trk in ((m_events, m_track), (t_events, t_track)):
            append_events_to_track(trk, evs)

        new_midi.tracks.append(m_track)
        if t_events:  # niente traccia T vuota (es. per la batteria preservata): in DAW sarebbe solo rumore
            new_midi.tracks.append(t_track)

    if not any_notes:
        warn("Nessuna nota trovata. Il tintinnabuli non verra' applicato.")
        return original_midi, triad_pcs

    return new_midi, triad_pcs


# --- Compositori: Steve Reich — Phasing (Sfasamento Processuale) ---
# Rif: "Piano Phase" (1967), "Clapping Music" (1972) — due voci suonano lo
# stesso pattern in loop; la voce B si sposta gradualmente rispetto alla
# voce A di un incremento fisso ogni N cicli, restando poi ferma nella
# nuova posizione di fase finche' non avviene lo sfasamento successivo. Il
# processo attraversa deterministicamente una sequenza di relazioni di fase
# — a differenza del ciclo asincrono "infinito" di Eno, qui lo sfasamento e'
# un processo discreto e udibile passo per passo.

def derive_reich_cell(original_midi, cell_length=8):
    for track in original_midi.tracks:
        notes = extract_notes(track, original_midi.ticks_per_beat)
        if _drums_preserved():
            notes = [n for n in notes if n['channel'] != DRUM_CHANNEL]
        if notes:
            notes.sort(key=lambda n: n['start'])
            return notes[:cell_length]
    return []


@_preserves_meta
def midi_reich_phasing(original_midi, cell_length_notes=8, num_cycles=32,
                        phase_shift_units=1, shift_every_n_cycles=4):
    """
    Estrae una cellula di cell_length_notes note e la fa suonare in loop su
    due voci identiche: la voce A resta fissa, la voce B si sposta di
    phase_shift_units "unita' di sfasamento" (in frazioni della durata media
    di una nota della cellula) ogni shift_every_n_cycles cicli, in stile
    Piano Phase/Clapping Music. Le tracce originali restano intatte.
    """
    _free = _free_channels(original_midi)
    ch_a = _free[0]
    ch_b = _free[1 % len(_free)]
    cell = derive_reich_cell(original_midi, cell_length_notes)
    if len(cell) < 2:
        warn("Materiale insufficiente per costruire la cellula. Il phasing non verra' generato.")
        return original_midi, 0

    ticks_per_beat = original_midi.ticks_per_beat
    t0 = cell[0]['start']
    norm_cell = [{'pitch': n['pitch'], 'start': n['start'] - t0, 'end': n['end'] - t0, 'velocity': n['velocity']} for n in cell]
    cell_span = max(n['end'] for n in norm_cell)
    unit = cell_span / max(1, len(norm_cell))  # granularita' dello sfasamento

    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    voice_a = mido.MidiTrack()
    voice_a.name = f"Reich Phasing Voce A (fissa, cellula={len(norm_cell)} note)"
    voice_a.append(mido.Message('program_change', program=_generated_program(0), channel=ch_a, time=0))

    voice_b = mido.MidiTrack()
    voice_b.name = f"Reich Phasing Voce B (sfasa +{phase_shift_units} ogni {shift_every_n_cycles} cicli)"
    voice_b.append(mido.Message('program_change', program=_generated_program(0), channel=ch_b, time=0))

    events_a, events_b = [], []
    current_phase_ticks = 0
    for cyc in range(max(1, num_cycles)):
        base_a = cyc * cell_span
        base_b = cyc * cell_span + current_phase_ticks
        for n in norm_cell:
            sa, ea = base_a + n['start'], base_a + n['end']
            events_a.append({'msg': mido.Message('note_on', note=n['pitch'], velocity=n['velocity'], channel=ch_a, time=0), 'abs_time': sa})
            events_a.append({'msg': mido.Message('note_off', note=n['pitch'], velocity=0, channel=ch_a, time=0), 'abs_time': max(sa + 1, ea)})

            sb, eb = base_b + n['start'], base_b + n['end']
            events_b.append({'msg': mido.Message('note_on', note=n['pitch'], velocity=n['velocity'], channel=ch_b, time=0), 'abs_time': sb})
            events_b.append({'msg': mido.Message('note_off', note=n['pitch'], velocity=0, channel=ch_b, time=0), 'abs_time': max(sb + 1, eb)})

        if shift_every_n_cycles > 0 and (cyc + 1) % shift_every_n_cycles == 0:
            current_phase_ticks += int(unit * phase_shift_units)

    for evs, trk in ((events_a, voice_a), (events_b, voice_b)):
        append_events_to_track(trk, evs)

    new_midi.tracks.append(voice_a)
    new_midi.tracks.append(voice_b)
    return new_midi, current_phase_ticks


# =====================================================================
# --- MATEMATICA SPERIMENTALE ---
# Strutture matematiche ispirate a "Godel, Escher, Bach" (D. Hofstadter, 1979) e a
# "Blueprints" (M. du Sautoy, 2025): canone eterno ascendente, scala impossibile di
# Shepard, metamorfosi, sistema formale MIU, simmetrie del fregio, cicli hamiltoniani e
# serie infinita di Norgard. Pura combinatoria/DSP: nessuna rete neurale. Le tracce
# originali restano intatte; il risultato si aggiunge come nuove tracce.
# =====================================================================

def _math_fold_pitch(p, lo=12, hi=115):
    """Riporta un pitch in [lo, hi] spostandolo per ottave (mai clamp: niente note identiche accumulate)."""
    p = int(p)
    while p > hi:
        p -= 12
    while p < lo:
        p += 12
    return max(0, min(127, p))


def _math_new_midi(original_midi):
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)
    return new_midi


def _math_voice_channels(original_midi, count):
    free = _free_channels(original_midi)
    return [free[i % len(free)] for i in range(count)]


def _math_pool_notes(original_midi):
    """Tutte le note del brano (batteria esclusa se preservata)."""
    notes = []
    for tr in original_midi.tracks:
        for n in extract_notes(tr, original_midi.ticks_per_beat):
            if _drums_preserved() and n['channel'] == DRUM_CHANNEL:
                continue
            notes.append(n)
    return notes


def _math_subject(original_midi, max_notes):
    """Prime max_notes note della prima traccia melodica, con il tempo azzerato sulla prima nota."""
    subject = derive_bach_subject(original_midi, max_notes=max_notes)
    if not subject:
        return []
    t0 = subject[0]['start']
    return [{'pitch': n['pitch'], 'start': n['start'] - t0,
             'end': max(n['end'] - t0, n['start'] - t0 + 1), 'velocity': n['velocity']} for n in subject]


def _math_scale_pcs(pool, scale_source):
    """Classi di altezza della scala: quelle del brano (se ce ne sono almeno 3) oppure cromatica."""
    if scale_source == "Dal brano":
        pcs = sorted({n['pitch'] % 12 for n in pool})
        if len(pcs) >= 3:
            return pcs, "Dal brano"
        return list(range(12)), "Cromatica (il brano ha meno di 3 classi di altezza)"
    return list(range(12)), "Cromatica"


def _math_add_track(new_midi, name, notes, channel, program):
    """Aggiunge una traccia da note [{'pitch','start','end','velocity'}] in tick assoluti.
    Tronca ogni nota all'attacco successivo dello stesso pitch (evita note_off ambigui)."""
    by_pitch = defaultdict(list)
    for n in notes:
        p = max(0, min(127, int(n['pitch'])))
        s = max(0, int(n['start']))
        by_pitch[p].append({'pitch': p, 'start': s, 'end': max(s + 1, int(n['end'])),
                            'velocity': max(1, min(127, int(n['velocity'])))})
    events = []
    for p, lst in by_pitch.items():
        lst.sort(key=lambda n: n['start'])
        for i, n in enumerate(lst):
            end = n['end']
            if i + 1 < len(lst):
                if lst[i + 1]['start'] <= n['start']:
                    continue  # attacco doppio nello stesso tick: tengo l'ultimo
                end = min(end, lst[i + 1]['start'])
            end = max(n['start'] + 1, end)
            events.append({'msg': mido.Message('note_on', note=p, velocity=n['velocity'], channel=channel, time=0), 'abs_time': n['start']})
            events.append({'msg': mido.Message('note_off', note=p, velocity=0, channel=channel, time=0), 'abs_time': end})
    track = mido.MidiTrack()
    track.name = name
    track.append(mido.Message('program_change', program=program, channel=channel, time=0))
    append_events_to_track(track, events)
    new_midi.tracks.append(track)
    return track


def math_melodic_tracks(midi):
    """[(indice traccia, nome, numero note)] delle tracce con note (batteria esclusa se preservata)."""
    out = []
    for i, tr in enumerate(midi.tracks):
        notes = [n for n in extract_notes(tr, midi.ticks_per_beat)
                 if not (_drums_preserved() and n['channel'] == DRUM_CHANNEL)]
        if notes:
            name = tr.name if getattr(tr, 'name', '') else f"Traccia {i}"
            out.append((i, name, len(notes)))
    return out


# --- 1. Canon per Tonos (Bach, Offerta Musicale, 1747; discusso da Hofstadter in GEB) ---
# Il canone ricomincia a ogni giro un tono piu' in alto; dopo sei giri e' tornato nella
# tonalita' d'origine, un'ottava sopra: sale per sempre senza uscire dalla tonalita'.

@_preserves_meta
def midi_canon_per_tonos(original_midi, num_voices=2, step_semitones=2, laps=1, delay_beats=2.0,
                         voice_interval=7, subject_notes=12, close_on_octave=True):
    subject = _math_subject(original_midi, subject_notes)
    if not subject:
        warn("Nessun soggetto melodico trovato. Il Canon per Tonos non verra' generato.")
        return original_midi, {'ok': False}
    tpb = original_midi.ticks_per_beat
    step = min((1, 2, 3, 4, 6), key=lambda d: abs(d - int(step_semitones)))  # solo divisori di 12: il giro si chiude
    cycles = (12 // step) * max(1, int(laps))
    total_cycles = cycles + (1 if close_on_octave else 0)
    cycle_ticks = max(tpb, int(math.ceil(max(n['end'] for n in subject) / tpb)) * tpb)
    delay_ticks = int(float(delay_beats) * tpb)
    num_voices = max(1, int(num_voices))

    # Se il giro finale uscirebbe dal registro, abbassa tutto per ottave intere.
    top = max(n['pitch'] for n in subject) + voice_interval * (num_voices - 1) + step * cycles
    lower = 0
    if top > 115:
        lower = -12 * int(math.ceil((top - 115) / 12))

    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, num_voices)
    program = _generated_program(48)
    for v in range(num_voices):
        notes = []
        for k in range(total_cycles):
            shift = k * step + voice_interval * v + lower
            for n in subject:
                notes.append({'pitch': _math_fold_pitch(n['pitch'] + shift, 0, 127),
                              'start': k * cycle_ticks + v * delay_ticks + n['start'],
                              'end': k * cycle_ticks + v * delay_ticks + n['end'],
                              'velocity': int(n['velocity'] * (1.0 if v == 0 else 0.85))})
        _math_add_track(new_midi, f"Canon per Tonos Voce {v + 1} (+{step}st a giro, {total_cycles} giri)", notes, channels[v], program)
    return new_midi, {'ok': True, 'cycles': total_cycles, 'step': step, 'voices': num_voices,
                      'total_shift': cycles * step}


# --- 2. Scala di Shepard-Penrose (Shepard 1964; "Salendo e scendendo" di Escher, 1960) ---
# Strati di ottave con peso a campana sul registro: la scala sale (o scende) senza fine.

@_preserves_meta
def midi_shepard_penrose(original_midi, direction="Ascendente", num_steps=48, step_beats=0.5,
                         layers=6, scale_source="Dal brano", peak_velocity=100):
    pool = _math_pool_notes(original_midi)
    pcs, scale_used = _math_scale_pcs(pool, scale_source)
    order = list(pcs) if str(direction).startswith("Asc") else list(reversed(pcs))
    layers = max(3, min(8, int(layers)))
    span = 12 * layers
    low = max(0, (60 - span // 2) // 12 * 12)
    center, sigma = low + span / 2.0, span / 4.0
    tpb = original_midi.ticks_per_beat
    step_ticks = max(1, int(float(step_beats) * tpb))
    notes = []
    for s in range(max(1, int(num_steps))):
        pc = order[s % len(order)]
        for i in range(layers):
            p = low + 12 * i + pc
            w = math.exp(-0.5 * ((p - center) / sigma) ** 2)
            if w < 0.03:
                continue
            notes.append({'pitch': p, 'start': s * step_ticks, 'end': (s + 1) * step_ticks,
                          'velocity': max(1, int(round(peak_velocity * w)))})
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    _math_add_track(new_midi, f"Shepard-Penrose {direction} ({layers} strati)", notes, ch, _generated_program(89))
    return new_midi, {'ok': True, 'scale': scale_used, 'layers': layers, 'steps': int(num_steps), 'notes': len(notes)}


# --- 3. Metamorfosi (Escher, serie "Metamorphosis", 1937-1968) ---
# Una traccia si trasforma gradualmente in un'altra: altezza, attacco, durata e velocity
# vengono interpolati nota per nota da A a B.

@_preserves_meta
def midi_escher_metamorphosis(original_midi, track_a=None, track_b=None, max_notes=128, snap_to_pool=True):
    # File tipo 0 (una traccia, piu' canali): esplodi i canali in tracce, come fa il Recomposer.
    if original_midi.type == 0 or (len(original_midi.tracks) == 1 and
            len({m.channel for t in original_midi.tracks for m in t if hasattr(m, 'channel')}) > 1):
        original_midi = _split_type0_to_tracks(original_midi)
    tracks = math_melodic_tracks(original_midi)
    if len(tracks) < 2:
        warn("La Metamorfosi richiede almeno due tracce con note. Non verra' generata.")
        return original_midi, {'ok': False}
    valid = {i for i, _, _ in tracks}
    if track_a not in valid or track_b not in valid or track_a == track_b:
        track_a, track_b = tracks[0][0], tracks[1][0]
    names = {i: nm for i, nm, _ in tracks}
    tpb = original_midi.ticks_per_beat

    def track_notes(idx):
        ns = [n for n in extract_notes(original_midi.tracks[idx], tpb)
              if not (_drums_preserved() and n['channel'] == DRUM_CHANNEL)]
        ns.sort(key=lambda n: (n['start'], n['pitch']))
        return ns
    A, B = track_notes(track_a), track_notes(track_b)
    n_out = max(2, min(max(len(A), len(B)), int(max_notes)))
    union = sorted({n['pitch'] for n in A} | {n['pitch'] for n in B})

    def snap(p):
        k = bisect.bisect_left(union, p)
        if k == 0:
            return union[0]
        if k == len(union):
            return union[-1]
        lo, hi = union[k - 1], union[k]
        return lo if (p - lo) <= (hi - p) else hi

    out = []
    for i in range(n_out):
        t = i / (n_out - 1)
        a = A[min(len(A) - 1, int(i * len(A) / n_out))]
        b = B[min(len(B) - 1, int(i * len(B) / n_out))]
        mix = lambda x, y: (1 - t) * x + t * y
        pitch = mix(a['pitch'], b['pitch'])
        pitch = snap(pitch) if snap_to_pool else int(round(pitch))
        start = int(round(mix(a['start'], b['start'])))
        dur = max(1, int(round(mix(a['end'] - a['start'], b['end'] - b['start']))))
        out.append({'pitch': pitch, 'start': start, 'end': start + dur,
                    'velocity': int(round(mix(a['velocity'], b['velocity'])))})
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    _math_add_track(new_midi, f"Escher Metamorfosi: {names[track_a]} -> {names[track_b]}", out, ch, _generated_program(0))
    return new_midi, {'ok': True, 'a': track_a, 'b': track_b, 'a_name': names[track_a], 'b_name': names[track_b], 'notes': n_out}


# --- 4. Sistema formale MIU (Hofstadter, "Godel, Escher, Bach", cap. 1) ---
# Alfabeto {M,I,U}, assioma MI, quattro regole: (1) xI -> xIU; (2) Mx -> Mxx;
# (3) III -> U; (4) UU -> (nulla). Ogni passo applica una regola scelta a caso (seed) fra
# quelle applicabili. Invariante: il numero di I non e' mai multiplo di 3, quindi MU non
# e' derivabile. Ogni stringa diventa una sezione: I = cellula, U = cellula invertita, M = pedale.

def _miu_applications(s, allow_grow):
    """Elenco [(regola, posizione)] delle regole applicabili a s."""
    apps = []
    if s.endswith("I") and allow_grow:
        apps.append((1, len(s) - 1))
    if allow_grow:
        apps.append((2, 0))
    for j in range(len(s) - 2):
        if s[j:j + 3] == "III":
            apps.append((3, j))
    for j in range(len(s) - 1):
        if s[j:j + 2] == "UU":
            apps.append((4, j))
    return apps


def _miu_apply(s, rule, pos):
    if rule == 1:
        return s + "U"
    if rule == 2:
        return s + s[1:]
    if rule == 3:
        return s[:pos] + "U" + s[pos + 3:]
    return s[:pos] + s[pos + 2:]


def miu_derive(rng, steps, max_len, budget=20000):
    """Derivazione MIU: lista di stringhe da 'MI'. Ricerca casuale con ritorno indietro: le regole che
    allungano rispettano max_len e, se una derivazione finisce in un vicolo cieco (nessuna regola piu'
    applicabile), si torna indietro e si prova un'altra mossa, cosi' si raggiunge il numero di passi
    richiesto quando esiste un percorso. Entro il budget; altrimenti la derivazione piu' lunga trovata."""
    def legal(s):
        grow_ok = {1: len(s) + 1 <= max_len, 2: 1 + 2 * (len(s) - 1) <= max_len}
        return [(r, p) for (r, p) in _miu_applications(s, True) if grow_ok.get(r, True)]

    steps = max(0, int(steps))
    state = {'best': ["MI"], 'nodes': 0}
    hist = ["MI"]

    def dfs(s):
        if len(hist) > len(state['best']):
            state['best'] = hist[:]
        if len(hist) - 1 >= steps:
            return True
        state['nodes'] += 1
        if state['nodes'] > budget:
            return False
        by_kind = defaultdict(list)
        for r, p in legal(s):
            by_kind[r].append(p)
        kinds = sorted(by_kind)
        rng.shuffle(kinds)  # il primo tipo di regola e' scelto uniformemente fra quelli applicabili
        for r in kinds:
            positions = by_kind[r]
            rng.shuffle(positions)
            for p in positions:
                t = _miu_apply(s, r, p)
                hist.append(t)
                if dfs(t):
                    return True
                hist.pop()
                if state['nodes'] > budget:
                    return False
        return False

    dfs("MI")
    return state['best']


@_preserves_meta
def midi_miu_system(original_midi, steps=8, max_len=16, cell_notes=3, cell_beats=2.0, seed=None):
    cell = _math_subject(original_midi, max(2, int(cell_notes)))
    if len(cell) < 2:
        warn("Servono almeno 2 note per la cellula del sistema MIU. Non verra' generato.")
        return original_midi, {'ok': False}
    rng = _rng(seed, "miu")
    history = miu_derive(rng, steps, max(4, int(max_len)))
    tpb = original_midi.ticks_per_beat
    cell_ticks = max(2, int(float(cell_beats) * tpb))
    p0 = cell[0]['pitch']
    pitches_i = [n['pitch'] for n in cell]
    pitches_u = [2 * p0 - p for p in pitches_i]
    vels = [n['velocity'] for n in cell]
    k = len(cell)
    notes, t = [], 0
    for s in history:
        for sym in s:
            if sym == "M":
                notes.append({'pitch': _math_fold_pitch(p0 - 12, 0, 127), 'start': t, 'end': t + cell_ticks, 'velocity': 80})
            else:
                src = pitches_i if sym == "I" else pitches_u
                for j in range(k):
                    a = t + (j * cell_ticks) // k
                    b = t + ((j + 1) * cell_ticks) // k
                    notes.append({'pitch': _math_fold_pitch(src[j], 0, 127), 'start': a,
                                  'end': max(a + 1, a + int((b - a) * 0.9)), 'velocity': vels[j]})
            t += cell_ticks
        t += cell_ticks  # pausa fra una derivazione e la successiva
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    _math_add_track(new_midi, f"MIU {len(history) - 1} passi (da MI a {history[-1][:24]})", notes, ch, _generated_program(0))
    return new_midi, {'ok': True, 'history': history, 'final': history[-1], 'cells': sum(len(s) + 1 for s in history)}


# --- 5. Simmetrie del fregio (Escher, divisione regolare del piano) ---
# Le isometrie del piano tempo-altezza applicate a un motivo: e = identita',
# I = riflessione dell'altezza (inversione), R = riflessione del tempo (retrogrado),
# RI = rotazione di 180 gradi (retrogrado-inversione). Le celle del fregio si susseguono
# secondo lo schema scelto; le voci impilate in una cella vanno su tracce separate.

FRIEZE_PATTERNS = {
    "Traslazione (p1)":           [["e"]],
    "Riflessione con scorrimento (p11g)": [["e"], ["I"]],
    "Specchio verticale (p1m1)":  [["e"], ["R"]],
    "Rotazione di 180° (p2)":     [["e"], ["RI"]],
    "Specchio orizzontale (p11m)": [["e", "I"]],
    "Doppio specchio (p2mm)":     [["e", "I"], ["R", "RI"]],
}


def _frieze_transform(notes, op, pmin, pmax, D):
    out = []
    for n in notes:
        p, s, e = n['pitch'], n['start'], n['end']
        if op in ("I", "RI"):
            p = (pmin + pmax) - p
        if op in ("R", "RI"):
            s, e = D - n['end'], D - n['start']
        out.append({'pitch': p, 'start': s, 'end': e, 'velocity': n['velocity']})
    return out


@_preserves_meta
def midi_escher_symmetry(original_midi, pattern="Traslazione (p1)", motif_notes=8, cells=16):
    motif = _math_subject(original_midi, max(2, int(motif_notes)))
    if len(motif) < 2:
        warn("Servono almeno 2 note per il motivo del fregio. Non verra' generato.")
        return original_midi, {'ok': False}
    spec = FRIEZE_PATTERNS.get(pattern, FRIEZE_PATTERNS["Traslazione (p1)"])
    tpb = original_midi.ticks_per_beat
    D = max(tpb, int(math.ceil(max(n['end'] for n in motif) / tpb)) * tpb)
    pmin, pmax = min(n['pitch'] for n in motif), max(n['pitch'] for n in motif)
    slots = max(len(c) for c in spec)
    slot_notes = [[] for _ in range(slots)]
    for c in range(max(1, int(cells))):
        cell = spec[c % len(spec)]
        for j, op in enumerate(cell):
            for n in _frieze_transform(motif, op, pmin, pmax, D):
                slot_notes[j].append({'pitch': n['pitch'], 'start': n['start'] + c * D,
                                      'end': n['end'] + c * D, 'velocity': n['velocity']})
    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, slots)
    for j in range(slots):
        _math_add_track(new_midi, f"Escher Fregio {pattern} - voce {j + 1}", slot_notes[j], channels[j], _generated_program(0))
    return new_midi, {'ok': True, 'cells': int(cells), 'cell_ticks': D, 'slots': slots}


# --- 6. Ciclo hamiltoniano (W. R. Hamilton, Icosian game, 1857; du Sautoy, "Blueprints") ---
# Grafo: nodi = pitch del brano, archi fra pitch a distanza <= max_interval semitoni.
# Un ciclo hamiltoniano visita ogni nodo una sola volta e richiude il cerchio: una
# melodia senza ripetizioni che riparte da capo senza salti. Ricerca con backtracking
# (euristica di Warnsdorff, parita' risolte dal seed) e budget massimo di passi.

def hamiltonian_search(adj, rng, budget=200000):
    """Ritorna (percorso, e_un_ciclo_hamiltoniano_completo, espansioni). Lavora sulla componente connessa piu' grande.
    Se il ciclo non esiste (o non si trova nel budget) il percorso e' il piu' lungo trovato; se la
    componente ha un nodo di grado < 2 un ciclo e' impossibile e ci si ferma al primo percorso completo."""
    # componente connessa piu' grande (a parita' vince quella con l'indice minimo)
    seen, comps = set(), []
    for i in range(len(adj)):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    comp = max(comps, key=lambda c: (len(c), -c[0]))
    if len(comp) < 2:
        return [comp[0]], False, 0
    local = {g: i for i, g in enumerate(comp)}
    ladj = [[local[v] for v in adj[g]] for g in comp]
    n = len(ladj)
    # Un ciclo hamiltoniano richiede un grafo 2-connesso: senza nodi di articolazione e con grado >= 2.
    def _connected_without(skip):
        rest = [i for i in range(n) if i != skip]
        seen_r, stack_r = {rest[0]}, [rest[0]]
        while stack_r:
            u = stack_r.pop()
            for v in ladj[u]:
                if v != skip and v not in seen_r:
                    seen_r.add(v)
                    stack_r.append(v)
        return len(seen_r) == len(rest)
    cycle_possible = n >= 3 and min(len(a) for a in ladj) >= 2 and all(_connected_without(v) for v in range(n))
    start = min(range(n), key=lambda i: (len(ladj[i]), i))
    state = {'exp': 0, 'best': [start], 'cycle': None}
    path, visited = [start], {start}

    def dfs(u):
        if len(path) > len(state['best']):
            state['best'] = path[:]
        if len(path) == n:
            if start in ladj[u]:
                state['cycle'] = path[:]
                return True
            return not cycle_possible  # percorso completo ma ciclo impossibile: basta cosi'
        state['exp'] += 1
        if state['exp'] > budget:
            return False
        cands = [v for v in ladj[u] if v not in visited]
        cands.sort(key=lambda v: (sum(1 for w in ladj[v] if w not in visited), rng.random()))
        for v in cands:
            path.append(v)
            visited.add(v)
            if dfs(v):
                return True
            path.pop()
            visited.discard(v)
            if state['exp'] > budget:
                return False
        return False

    dfs(start)
    if state['cycle'] is not None:
        # True solo se il ciclo copre TUTTO il grafo; su una componente piu' piccola resta un anello parziale
        return [comp[i] for i in state['cycle']], len(comp) == len(adj), state['exp']
    return [comp[i] for i in state['best']], False, state['exp']


@_preserves_meta
def midi_hamiltonian_cycle(original_midi, max_interval=7, max_nodes=16, repeats=4, seed=None):
    pool = _math_pool_notes(original_midi)
    stats = defaultdict(list)
    for n in pool:
        stats[n['pitch']].append(n)
    ranked = sorted(stats, key=lambda p: (-len(stats[p]), p))[:max(3, int(max_nodes))]
    nodes = sorted(ranked)
    if len(nodes) < 3:
        warn("Servono almeno 3 pitch distinti per il grafo. Il ciclo hamiltoniano non verra' generato.")
        return original_midi, {'ok': False}
    rng = _rng(seed, "hamiltonian")
    maxi = max(1, int(max_interval))
    adj = [[j for j in range(len(nodes)) if j != i and abs(nodes[i] - nodes[j]) <= maxi] for i in range(len(nodes))]
    path, is_cycle, expansions = hamiltonian_search(adj, rng)
    closed = len(path) >= 3 and path[0] in adj[path[-1]]
    status = 'full' if is_cycle else ('partial' if closed else 'path')
    tpb = original_midi.ticks_per_beat

    def node_note(idx):
        ns = stats[nodes[idx]]
        durs = sorted(n['end'] - n['start'] for n in ns)
        dur = max(tpb // 8, durs[len(durs) // 2])
        vel = int(sum(n['velocity'] for n in ns) / len(ns))
        return nodes[idx], dur, vel

    notes, t = [], 0
    for _ in range(max(1, int(repeats))):
        for idx in path:
            p, dur, vel = node_note(idx)
            notes.append({'pitch': p, 'start': t, 'end': t + dur, 'velocity': vel})
            t += dur
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    kind = {'full': "ciclo", 'partial': "ciclo parziale", 'path': "percorso piu' lungo"}[status]
    _math_add_track(new_midi, f"Hamilton {kind} su {len(path)}/{len(nodes)} pitch (salto max {maxi}st)", notes, ch, _generated_program(0))
    return new_midi, {'ok': True, 'nodes': len(nodes), 'status': status, 'covered': len(path),
                      'expansions': expansions, 'pitches': [nodes[i] for i in path]}


# --- 7. Serie infinita di Norgard (Per Norgard, anni '60) ---
# a(0)=0, a(2n) = -a(n), a(2n+1) = a(n)+1: 0, 1, -1, 2, 1, 0, -2, 3, ...
# E' autosimile: prendendo un elemento ogni 2^k si riottiene la serie (invertita se k e'
# dispari). Gli strati lenti sono quindi lo "scheletro" dello strato veloce.

def norgard_value(n):
    n = int(n)
    if n == 0:
        return 0
    return -norgard_value(n >> 1) if n % 2 == 0 else norgard_value(n >> 1) + 1


@_preserves_meta
def midi_norgard_infinity(original_midi, num_notes=64, step_beats=0.5, layers=3, scale_source="Dal brano"):
    pool = _math_pool_notes(original_midi)
    if not pool:
        warn("Nessuna nota trovata per la serie di Norgard. Non verra' generata.")
        return original_midi, {'ok': False}
    pcs, scale_used = _math_scale_pcs(pool, scale_source)
    n_deg = len(pcs)

    def step_pitch(k):
        octave, deg = divmod(k, n_deg)
        return 12 * octave + pcs[deg]

    median = sorted(n['pitch'] for n in pool)[len(pool) // 2]
    k_guess = (median // 12) * n_deg
    k0 = min(range(k_guess - 2 * n_deg, k_guess + 3 * n_deg), key=lambda k: (abs(step_pitch(k) - median), k))
    tpb = original_midi.ticks_per_beat
    step_ticks = max(1, int(float(step_beats) * tpb))
    total_ticks = max(1, int(num_notes)) * step_ticks
    layers = max(1, min(4, int(layers)))
    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, layers)
    for k in range(layers):
        notes, j = [], 0
        stride = 2 ** k
        while j * stride * step_ticks < total_ticks:
            p = step_pitch(k0 + norgard_value(j * stride)) - 12 * k
            notes.append({'pitch': _math_fold_pitch(p), 'start': j * stride * step_ticks,
                          'end': (j + 1) * stride * step_ticks, 'velocity': min(120, 85 + 10 * k)})
            j += 1
        _math_add_track(new_midi, f"Norgard Strato {k + 1} (1 nota ogni {stride} passi)", notes, channels[k], _generated_program(0))
    return new_midi, {'ok': True, 'scale': scale_used, 'layers': layers,
                      'series_head': [norgard_value(i) for i in range(16)]}



# =====================================================================
# --- TEORIA MATEMATICA DELLA MUSICA (Mathematical Music Theory) ---
# Cinque metodi dal filone di ricerca della Society for Mathematics and Computation in Music:
# ritmi euclidei (Toussaint), trasformazioni neo-riemanniane (Lewin) con voice leading
# geometrico (Tymoczko), Spiral Array (Chew), insiemi massimamente uniformi (Clough-Douthett)
# e DFT degli insiemi di altezze (Lewin, Quinn, Amiot). Stesso schema dei metodi precedenti: pura
# matematica/numpy, nessuna rete neurale; le tracce originali restano intatte.
# =====================================================================

# --- 8. Ritmi euclidei (G. Toussaint, "The Euclidean Algorithm Generates Traditional Musical Rhythms", 2005) ---
# L'algoritmo di Bjorklund (1999) distribuisce k attacchi su n passi nel modo piu' uniforme
# possibile, con la stessa ricorsione dell'algoritmo di Euclide sul massimo comun divisore.
# E(3,8) = x..x..x.   E(5,8) = x.xx.xx.   E(7,12) = x.xx.x.xx.x.
# Le altezze sono quelle del soggetto del brano, ciclate sugli attacchi.

def bjorklund(pulses, steps):
    """Pattern euclideo E(pulses, steps) come lista di 0/1 (1 = attacco)."""
    steps = max(1, int(steps))
    pulses = max(0, min(int(pulses), steps))
    if pulses == 0:
        return [0] * steps
    head = [[1] for _ in range(pulses)]
    tail = [[0] for _ in range(steps - pulses)]
    while len(tail) > 1:
        n = min(len(head), len(tail))
        merged = [head[i] + tail[i] for i in range(n)]
        left = head[n:] if len(head) > len(tail) else tail[n:]
        head, tail = merged, left
    return [x for group in head + tail for x in group]


def _pattern_string(pattern):
    return "".join("x" if v else "." for v in pattern)


@_preserves_meta
def midi_euclidean_rhythm(original_midi, steps=16, pulses=5, rotation=0, step_beats=0.25,
                          layers=1, cycles=4, subject_notes=8):
    subject = _math_subject(original_midi, max(1, int(subject_notes)))
    if not subject:
        warn("Nessun soggetto melodico trovato. Il ritmo euclideo non verra' generato.")
        return original_midi, {'ok': False}
    tpb = original_midi.ticks_per_beat
    steps = max(2, int(steps))
    step_ticks = max(1, int(float(step_beats) * tpb))
    layers = max(1, min(3, int(layers)))
    cycles = max(1, int(cycles))
    rot = int(rotation) % steps
    pitches = [n['pitch'] for n in subject]
    vels = [n['velocity'] for n in subject]
    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, layers)
    program = _generated_program(0)
    patterns = []
    for j in range(layers):
        k = max(1, min(steps, int(pulses) - 2 * j))  # ogni strato successivo ha 2 attacchi in meno
        pat = bjorklund(k, steps)
        pat = pat[rot:] + pat[:rot]
        patterns.append((k, _pattern_string(pat)))
        notes, hit = [], 0
        for c in range(cycles):
            first = True
            for i, on in enumerate(pat):
                if not on:
                    continue
                idx = hit % len(pitches)
                start = (c * steps + i) * step_ticks
                vel = vels[idx] + (12 if first else 0)
                notes.append({'pitch': _math_fold_pitch(pitches[idx] - 12 * j),
                              'start': start, 'end': start + max(1, int(step_ticks * 0.9)),
                              'velocity': min(127, vel)})
                first = False
                hit += 1
        _math_add_track(new_midi, f"Euclideo E({k},{steps}) strato {j + 1}", notes, channels[j], program)
    return new_midi, {'ok': True, 'steps': steps, 'layers': layers, 'patterns': patterns, 'rotation': rot}


# --- 9. Trasformazioni neo-riemanniane (D. Lewin, "Generalized Musical Intervals and Transformations", 1987) ---
# Una triade e' un oggetto (fondamentale, modo); P, L, R sono funzioni che la trasformano:
#   P (parallela):        Do maggiore <-> Do minore
#   L (nota sensibile):   Do maggiore <-> Mi minore
#   R (relativa):         Do maggiore <-> La minore
# Sono involuzioni e muovono una sola voce (di 1 o 2 semitoni). Le sequenze alternate danno cicli
# chiusi: P-L = ciclo esatonico (6 accordi), P-R = ottatonico (8), L-R = tutte le 24 triadi.
# Il voicing e' scelto con il voice leading a distanza minima (D. Tymoczko, Science, 2006):
# gli accordi sono punti di uno spazio, il voice leading e' la distanza fra due punti.

NR_PATTERNS = {
    "Ciclo esatonico (P-L)": ('P', 'L'),
    "Ciclo ottatonico (P-R)": ('P', 'R'),
    "Ciclo diatonico (L-R)": ('L', 'R'),
    "Passeggiata casuale (seed)": None,
}


def nr_transform(triad, op):
    """Applica P, L o R a una triade (fondamentale 0-11, 'M'|'m')."""
    root, q = triad
    if q == 'M':
        return {'P': (root, 'm'), 'L': ((root + 4) % 12, 'm'), 'R': ((root + 9) % 12, 'm')}[op]
    return {'P': (root, 'M'), 'L': ((root + 8) % 12, 'M'), 'R': ((root + 3) % 12, 'M')}[op]


def _nr_pcs(triad):
    root, q = triad
    return [root % 12, (root + (4 if q == 'M' else 3)) % 12, (root + 7) % 12]


def _nr_name(triad):
    return _NOTE_NAMES[triad[0] % 12] + ('' if triad[1] == 'M' else 'm')


def _vl_next(prev, pcs):
    """Voice leading di distanza minima da `prev` (lista di pitch, una per voce) alle classi di
    altezza `pcs`: prova ogni assegnazione voce -> classe e prende, per ciascuna voce, il pitch piu'
    vicino. Ritorna (distanza totale in semitoni, nuovi pitch)."""
    best = None
    for perm in itertools.permutations(pcs):
        cand, total = [], 0
        for p, pc in zip(prev, perm):
            d = (pc - p) % 12
            if d > 6:
                d -= 12
            cand.append(p + d)
            total += abs(d)
        if len(set(cand)) < len(cand):
            continue  # due voci sullo stesso pitch: scartata
        if best is None or total < best[0]:
            best = (total, cand)
    if best is None:  # impossibile con 3 classi distinte, ma per sicurezza
        best = (0, list(prev))
    total, cand = best
    while min(cand) < 40:
        cand = [p + 12 for p in cand]
    while max(cand) > 88:
        cand = [p - 12 for p in cand]
    return total, cand


def _vl_first(triad, base=52):
    """Voicing iniziale in posizione fondamentale, fondamentale in [base, base+11]."""
    root = triad[0]
    r0 = base + (root - base) % 12
    return [r0, r0 + (4 if triad[1] == 'M' else 3), r0 + 7]


@_preserves_meta
def midi_neo_riemannian(original_midi, pattern="Ciclo esatonico (P-L)", steps=12, chord_beats=2.0,
                        start="Dal brano", tie_common_tones=True, seed=None):
    if not isinstance(start, str) or start == "Dal brano" or not start:
        est = estimate_key(original_midi)
        triad = (est[0] % 12, 'M' if est[1] == "Maggiore" else 'm') if est else (0, 'M')
    else:
        triad = (get_key_offset(start) % 12, 'm' if start.endswith('m') else 'M')
    steps = max(2, int(steps))
    if pattern in NR_PATTERNS and NR_PATTERNS[pattern] is not None:
        cycle = NR_PATTERNS[pattern]
        ops = [cycle[i % len(cycle)] for i in range(steps - 1)]
    else:
        rng = _rng(seed, "neo-riemannian")
        ops, last = [], None
        for _ in range(steps - 1):
            op = rng.choice([o for o in "PLR" if o != last])  # P, L, R sono involuzioni: evita di annullarsi
            ops.append(op)
            last = op
    triads = [triad]
    for op in ops:
        triads.append(nr_transform(triads[-1], op))
    period = next((i for i in range(1, len(triads)) if triads[i] == triads[0]), None)

    voicings = [_vl_first(triads[0])]
    distances = []
    for t in triads[1:]:
        d, v = _vl_next(voicings[-1], _nr_pcs(t))
        voicings.append(v)
        distances.append(d)

    tpb = original_midi.ticks_per_beat
    chord_ticks = max(2, int(float(chord_beats) * tpb))
    notes = []
    for voice in range(3):
        i = 0
        while i < len(voicings):
            j = i
            if tie_common_tones:
                while j + 1 < len(voicings) and voicings[j + 1][voice] == voicings[i][voice]:
                    j += 1
            notes.append({'pitch': voicings[i][voice], 'start': i * chord_ticks,
                          'end': (j + 1) * chord_ticks - (0 if tie_common_tones else max(1, chord_ticks // 16)),
                          'velocity': 80})
            i = j + 1
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    _math_add_track(new_midi, f"Neo-Riemanniano {pattern} ({steps} accordi da {_nr_name(triads[0])})",
                    notes, ch, _generated_program(48))
    return new_midi, {'ok': True, 'start': _nr_name(triads[0]), 'chords': [_nr_name(t) for t in triads],
                      'ops': ops, 'period': period, 'vl_total': sum(distances), 'vl_steps': distances}


# --- 10. Spiral Array (E. Chew, "Towards a Mathematical Model of Tonality", 2000) ---
# Le altezze stanno su un'elica 3D lungo la linea delle quinte: P(k) = (r sin(k*pi/2), r cos(k*pi/2), k*h).
# Con h = r*sqrt(2/15) la distanza di quinta (k, k+1) e quella di terza maggiore (k, k+4) coincidono.
# Un accordo e' il baricentro pesato delle sue note, una tonalita' quello dei suoi accordi I, V, IV.
# Il "centro d'effetto" (CE) di un frammento di musica e' il baricentro delle sue note, pesato
# sulle durate: l'accordo piu' vicino al CE e' l'armonia del frammento; la distanza del CE dal centro
# della tonalita' e' una misura di tensione. I pesi sono parametri liberi del modello: qui valori
# semplici che rispettano w1 >= w2 >= w3 con somma 1.

_SP_H = math.sqrt(2.0 / 15.0)
_SP_W = (0.6, 0.3, 0.1)      # pesi: fondamentale, quinta, terza (accordo); I, V, IV (tonalita')
_SP_ALPHA = 0.75             # terza minore: armonica (k-3) contro melodica (k+3)
_SP_BETA = 0.75              # tonalita' minore: V maggiore contro v minore


def _sp_pos(k):
    a = k * math.pi / 2.0
    return np.array([math.sin(a), math.cos(a), k * _SP_H])


def _sp_k(pc, c):
    """Indice sulla linea delle quinte della classe di altezza pc, riportato in [c-5, c+6].
    Il MIDI non distingue Do# da Reb: la finestra centrata sulla tonalita' sceglie l'ortografia piu' vicina."""
    k = (7 * pc) % 12
    while k < c - 5:
        k += 12
    while k > c + 6:
        k -= 12
    return k


def _sp_chord(k, q):
    w1, w2, w3 = _SP_W
    if q == 'M':
        return w1 * _sp_pos(k) + w2 * _sp_pos(k + 1) + w3 * _sp_pos(k + 4)
    third = _SP_ALPHA * _sp_pos(k - 3) + (1 - _SP_ALPHA) * _sp_pos(k + 3)
    return w1 * _sp_pos(k) + w2 * _sp_pos(k + 1) + w3 * third


def _sp_key(k, q):
    v1, v2, v3 = _SP_W
    if q == 'M':
        return v1 * _sp_chord(k, 'M') + v2 * _sp_chord(k + 1, 'M') + v3 * _sp_chord(k - 1, 'M')
    dom = _SP_BETA * _sp_chord(k + 1, 'M') + (1 - _SP_BETA) * _sp_chord(k + 1, 'm')
    sub = _SP_ALPHA * _sp_chord(k - 1, 'm') + (1 - _SP_ALPHA) * _sp_chord(k - 1, 'M')
    return v1 * _sp_chord(k, 'm') + v2 * dom + v3 * sub


def _sp_ce(hist, c):
    tot = float(sum(hist))
    if tot <= 0:
        return None
    v = np.zeros(3)
    for pc in range(12):
        if hist[pc] > 0:
            v += hist[pc] * _sp_pos(_sp_k(pc, c))
    return v / tot


def _sp_spelling_center(hist):
    """Centro cs della finestra [cs-5, cs+6] sulla linea delle quinte che rende l'insieme di note piu'
    compatto (varianza pesata minima): la stessa ortografia vale per CE, accordi e tonalita'."""
    best = None
    for cs in range(12):
        ks = [(_sp_k(pc, cs), hist[pc]) for pc in range(12) if hist[pc] > 0]
        tot = sum(w for _, w in ks)
        if tot <= 0:
            return 0
        mean = sum(k * w for k, w in ks) / tot
        var = sum(w * (k - mean) ** 2 for k, w in ks) / tot
        if best is None or var < best[0] - 1e-12:
            best = (var, cs)
    return best[1]


def spiral_best_key(hist):
    """(tonica 0-11, 'M'|'m', distanza CE-tonalita', cs) della tonalita' il cui centro e' piu' vicino al
    CE dell'istogramma; cs e' il centro dell'ortografia usata (vedi _sp_spelling_center)."""
    cs = _sp_spelling_center(hist)
    ce = _sp_ce(hist, cs)
    if ce is None:
        return None
    best = None
    for t in range(12):
        kt = _sp_k(t, cs)
        for q in ('M', 'm'):
            d = float(np.linalg.norm(ce - _sp_key(kt, q)))
            if best is None or d < best[2] - 1e-12:
                best = (t, q, d, cs)
    return best


@_preserves_meta
def midi_spiral_array(original_midi, window_beats=4.0, add_bass=True):
    pool = _math_pool_notes(original_midi)
    if not pool:
        warn("Nessuna nota trovata per lo Spiral Array. L'analisi non verra' generata.")
        return original_midi, {'ok': False}
    tpb = original_midi.ticks_per_beat
    win = max(1, int(float(window_beats) * tpb))
    total = max(n['end'] for n in pool)
    n_win = max(1, int(math.ceil(total / win)))
    hists = [[0.0] * 12 for _ in range(n_win)]
    for n in pool:
        w0, w1 = n['start'] // win, max(n['start'], n['end'] - 1) // win
        for w in range(w0, min(w1, n_win - 1) + 1):
            ov = min(n['end'], (w + 1) * win) - max(n['start'], w * win)
            if ov > 0:
                hists[w][n['pitch'] % 12] += ov
    global_hist = [sum(h[pc] for h in hists) for pc in range(12)]
    key = spiral_best_key(global_hist)
    if key is None:
        return original_midi, {'ok': False}
    tonic, kq, _, c = key
    key_center = _sp_key(_sp_k(tonic, c), kq)
    key_name = f"{_NOTE_NAMES[tonic]} {'maggiore' if kq == 'M' else 'minore'}"

    chord_centers = {(r, q): _sp_chord(_sp_k(r, c), q) for r in range(12) for q in ('M', 'm')}
    segs, tensions = [], []   # segs: [chord, start_win, end_win, tension]
    for w in range(n_win):
        ce = _sp_ce(hists[w], c)
        if ce is None:
            continue
        chord = min(chord_centers, key=lambda ch_: (float(np.linalg.norm(ce - chord_centers[ch_])), ch_))
        tension = float(np.linalg.norm(ce - key_center))
        tensions.append(tension)
        if segs and segs[-1][0] == chord and segs[-1][2] == w - 1:
            segs[-1][2] = w
            segs[-1][3] = max(segs[-1][3], tension)
        else:
            segs.append([chord, w, w, tension])
    max_t = max(tensions) if tensions else 0.0

    voicing = None
    notes, bass = [], []
    for chord, w0, w1, tension in segs:
        if voicing is None:
            voicing = _vl_first(chord)
        else:
            _, voicing = _vl_next(voicing, _nr_pcs(chord))
        vel = 55 + int(round(60 * tension / max_t)) if max_t > 0 else 80
        start, end = w0 * win, (w1 + 1) * win
        for p in voicing:
            notes.append({'pitch': p, 'start': start, 'end': end, 'velocity': vel})
        bass.append({'pitch': 36 + chord[0] % 12, 'start': start, 'end': end, 'velocity': min(127, vel + 10)})
    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, 2)
    _math_add_track(new_midi, f"Spiral Array accordi ({key_name}, finestra {window_beats} beat)",
                    notes, channels[0], _generated_program(48))
    if add_bass:
        _math_add_track(new_midi, "Spiral Array basso", bass, channels[1], _generated_program(32))
    prog = [_nr_name(s[0]) for s in segs]
    return new_midi, {'ok': True, 'key': key_name, 'windows': n_win, 'progression': prog,
                      'mean_tension': (sum(tensions) / len(tensions)) if tensions else 0.0,
                      'max_tension': max_t}


# --- 11. Insiemi massimamente uniformi (J. Clough, J. Douthett, "Maximally Even Sets", Journal of Music Theory, 1991) ---
# I c elementi dell'insieme ME(c, d) sono J(k) = floor(k*d/c), k = 0..c-1: i c punti piu' regolari
# possibili sul cerchio delle d altezze. Con d = 12: c=7 scala diatonica, c=5 pentatonica, c=6 scala
# per toni interi, c=4 settima diminuita, c=3 triade aumentata. Le rotazioni sono i modi.
# Sono gli stessi oggetti dei ritmi euclidei, sul cerchio delle altezze invece che dei passi.
# La struttura e' annidata: un insieme ME dentro un insieme ME (triade ME(3,7) dentro la scala ME(7,12)
# = terze diatoniche). Il brano viene quantizzato sull'insieme e armonizzato con il sotto-insieme.

def me_set(c, d=12):
    """Insieme massimamente uniforme ME(c, d) in forma nativa: lista ordinata di c interi in [0, d)."""
    d = max(1, int(d))
    c = max(1, min(int(c), d))
    return [(k * d) // c for k in range(c)]


def _me_mode(elems, mode, d=12):
    """Rotazione: il modo che parte dall'elemento `mode`, con lo 0 come tonica."""
    c = len(elems)
    m = int(mode) % c
    rot = elems[m:] + elems[:m]
    return sorted((e - rot[0]) % d for e in rot)


def _intervals(pcs, d=12):
    return [(pcs[(i + 1) % len(pcs)] - pcs[i]) % d or d for i in range(len(pcs))]


def _snap_to_pcs(p, pcs_set):
    """Pitch piu' vicino a p con classe di altezza in pcs_set (a parita' vince il piu' basso)."""
    for d in range(0, 7):
        for cand in (p - d, p + d):
            if 0 <= cand <= 127 and cand % 12 in pcs_set:
                return cand
    return int(p)


@_preserves_meta
def midi_maximally_even(original_midi, cardinality=7, mode=1, tonic="Dal brano", sub_cardinality=3,
                        chord_beats=4.0, subject_notes=32):
    subject = _math_subject(original_midi, max(2, int(subject_notes)))
    if not subject:
        warn("Nessun soggetto melodico trovato. L'insieme massimamente uniforme non verra' generato.")
        return original_midi, {'ok': False}
    if not isinstance(tonic, str) or tonic == "Dal brano" or not tonic:
        est = estimate_key(original_midi)
        tonic_pc = est[0] % 12 if est else 0
    else:
        tonic_pc = get_key_offset(tonic) % 12
    c = max(2, min(11, int(cardinality)))
    native = me_set(c, 12)
    offsets = _me_mode(native, mode, 12)
    scale = [(tonic_pc + e) % 12 for e in offsets]          # classi di altezza in ordine di grado
    scale_set = set(scale)
    # equivalenza con il ritmo euclideo E(c,12): verificata, non assunta
    ind = [1 if n in set(native) else 0 for n in range(12)]
    euc = bjorklund(c, 12)
    euclid_equiv = any(euc[r:] + euc[:r] == ind for r in range(12))

    tpb = original_midi.ticks_per_beat
    quant, moved = [], 0
    for n in subject:
        p = _snap_to_pcs(n['pitch'], scale_set)
        moved += int(p != n['pitch'])
        quant.append({'pitch': p, 'start': n['start'], 'end': n['end'], 'velocity': n['velocity']})
    new_midi = _math_new_midi(original_midi)
    channels = _math_voice_channels(original_midi, 2)
    _math_add_track(new_midi, f"ME({c},12) melodia ({_NOTE_NAMES[tonic_pc]}, modo {int(mode) % c})",
                    quant, channels[0], _generated_program(0))

    sub = int(sub_cardinality)
    sub_idx = []
    if 2 <= sub < c:
        sub_idx = [(j * c) // sub for j in range(sub)]       # ME(sub, c) dentro la scala
        win = max(1, int(float(chord_beats) * tpb))
        total = max(n['end'] for n in quant)
        chords = []
        for w in range(max(1, int(math.ceil(total / win)))):
            weight = defaultdict(float)
            for n in quant:
                ov = min(n['end'], (w + 1) * win) - max(n['start'], w * win)
                if ov > 0 and n['pitch'] % 12 in scale_set:
                    weight[n['pitch'] % 12] += ov
            if not weight:
                continue
            top_pc = max(sorted(weight), key=lambda pc: (weight[pc], -scale.index(pc)))
            deg = scale.index(top_pc)
            pcs = [scale[(deg + i) % c] for i in sub_idx]
            pitches = [48 + (pcs[0] - 48) % 12]
            for pc in pcs[1:]:
                step = (pc - pitches[-1]) % 12
                pitches.append(pitches[-1] + (step if step else 12))
            chords.append((w, pitches))
        chord_notes = [{'pitch': p, 'start': w * win, 'end': (w + 1) * win, 'velocity': 70}
                       for w, pitches in chords for p in pitches]
        if chord_notes:
            _math_add_track(new_midi, f"ME({sub},{c}) accordi", chord_notes, channels[1], _generated_program(48))
    return new_midi, {'ok': True, 'tonic': _NOTE_NAMES[tonic_pc], 'pcs': scale,
                      'names': [_NOTE_NAMES[pc] for pc in scale], 'intervals': _intervals(offsets),
                      'sub_idx': sub_idx, 'moved': moved, 'notes': len(subject),
                      'euclid_equiv': euclid_equiv, 'cardinality': c, 'mode': int(mode) % c}


# --- 12. DFT degli insiemi di altezze (D. Lewin 1959; J. Quinn 2006; E. Amiot, "Music through Fourier Space", 2016) ---
# Un insieme di altezze e' un segnale sui 12 punti del cerchio: F_k = sum_n x(n) exp(-2*pi*i*k*n/12).
# |F_k| misura quanto l'insieme somiglia alla divisione regolare dell'ottava in k parti:
# k=1 cluster cromatico, k=2 tritoni, k=3 triadi aumentate, k=4 settime diminuite (ottatonico),
# k=5 diatonico (5 e' il moltiplicatore che scambia cromatico e quinte), k=6 toni interi.
# La fase di F_k dice dove sta l'insieme sul cerchio: sfasare tutte le componenti = trasporre.
# Il metodo filtra l'istogramma del brano tenendo solo le componenti scelte, prende le c classi di
# altezza piu' forti del profilo ricostruito e quantizza il soggetto su quell'insieme. Opzionale:
# moltiplicazione M5/M7 dell'insieme (F'_k = F_{5k}): il diatonico diventa un cluster.

def pc_dft(hist):
    """Coefficienti F_0..F_11 di un istogramma/indicatore di classi di altezza."""
    return np.fft.fft(np.asarray(hist, dtype=float))


def pc_dft_magnitudes(hist):
    """|F_1|..|F_6| normalizzati su F_0 (peso totale): 1 = insieme perfettamente regolare di quel tipo."""
    F = pc_dft(hist)
    f0 = float(abs(F[0]))
    if f0 <= 0:
        return [0.0] * 6
    return [float(abs(F[k])) / f0 for k in range(1, 7)]


DFT_COMPONENT_LABELS = {1: "1 · cluster cromatico", 2: "2 · tritoni", 3: "3 · triadi aumentate",
                        4: "4 · settime diminuite (ottatonico)", 5: "5 · diatonico", 6: "6 · toni interi"}


@_preserves_meta
def midi_pc_dft_filter(original_midi, components=(5,), cardinality=7, shift=0, multiplier="Nessuno",
                       subject_notes=32):
    pool = _math_pool_notes(original_midi)
    hist = [0.0] * 12
    for n in pool:
        hist[n['pitch'] % 12] += max(1, n['end'] - n['start'])
    if sum(hist) <= 0:
        warn("Nessuna nota trovata per la DFT dei pc-set. Il filtro non verra' generato.")
        return original_midi, {'ok': False}
    subject = _math_subject(original_midi, max(2, int(subject_notes)))
    if not subject:
        warn("Nessun soggetto melodico trovato. Il filtro DFT non verra' generato.")
        return original_midi, {'ok': False}
    comps = sorted({int(k) for k in components if 1 <= int(k) <= 6}) or [5]
    F = pc_dft(hist)
    G = np.zeros(12, dtype=complex)
    G[0] = F[0]
    for k in comps:
        G[k] = F[k]
        if k != 6:
            G[12 - k] = F[12 - k]
    profile = np.roll(np.fft.ifft(G).real, int(shift) % 12)   # sfasamento uguale per tutte le componenti = trasposizione
    c = max(2, min(11, int(cardinality)))
    order = sorted(range(12), key=lambda n: (-round(float(profile[n]), 9), n))
    pcs = sorted(order[:c])
    mult = {"M5": 5, "M7": 7}.get(multiplier, 1)
    if mult != 1:
        pcs = sorted({(mult * n) % 12 for n in pcs})
    pcs_set = set(pcs)

    quant, moved = [], 0
    for n in subject:
        p = _snap_to_pcs(n['pitch'], pcs_set)
        moved += int(p != n['pitch'])
        quant.append({'pitch': p, 'start': n['start'], 'end': n['end'], 'velocity': n['velocity']})
    new_midi = _math_new_midi(original_midi)
    ch = _math_voice_channels(original_midi, 1)[0]
    tag = "M5" if mult == 5 else ("M7" if mult == 7 else "")
    _math_add_track(new_midi, f"DFT k={'+'.join(map(str, comps))} -> {len(pcs)} pc{(' ' + tag) if tag else ''}",
                    quant, ch, _generated_program(0))
    ind = [1.0 if n in pcs_set else 0.0 for n in range(12)]
    return new_midi, {'ok': True, 'components': comps, 'pcs': pcs, 'names': [_NOTE_NAMES[pc] for pc in pcs],
                      'mags_piece': pc_dft_magnitudes(hist), 'mags_set': pc_dft_magnitudes(ind),
                      'moved': moved, 'notes': len(subject), 'multiplier': tag or "Nessuno",
                      'shift': int(shift) % 12, 'cardinality': len(pcs)}

# --- Funzioni di Decomposizione ---

@_preserves_meta
def midi_note_remapper(original_midi, target_scale_name, target_key_name, pitch_shift_range, velocity_randomization, preserve_drums=None, seed=None):
    """
    Rimodella le note MIDI in base a una scala, tonalità e randomizzazione di pitch/velocity.
    Il pitch rimappato al note_on viene ricordato (stack per nota/canale) e riusato dal
    note_off corrispondente: nessuna nota resta appesa nella DAW anche con pitch_shift_range > 0.
    Un note_on con velocity 0 (= note_off) non viene mai trasformato in una nota vera.
    """
    rng = _rng(seed, "note_remapper")
    preserve = _drums_preserved(preserve_drums)
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    target_scale_intervals = get_scale_notes(target_scale_name)
    key_offset = get_key_offset(target_key_name)

    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(track, 'name') and track.name:
            new_track.name = track.name
        open_notes = defaultdict(list)  # (nota, canale) -> stack dei pitch rimappati
        for msg in track:
            is_on = msg.type == 'note_on' and msg.velocity > 0
            is_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
            if not (is_on or is_off) or (preserve and msg.channel == DRUM_CHANNEL):
                new_track.append(_clone(msg))
                continue

            key = (msg.note, msg.channel)
            if is_off and open_notes.get(key):
                new_track.append(_clone(msg, note=open_notes[key].pop()))
                continue

            shifted_note = msg.note
            if pitch_shift_range > 0:
                shifted_note += rng.randint(-pitch_shift_range, pitch_shift_range)
            shifted_note = max(0, min(127, shifted_note))

            note_in_octave = (shifted_note - key_offset) % 12
            closest_scale_interval = min(target_scale_intervals, key=lambda x: abs(note_in_octave - x))
            octave = (shifted_note - key_offset) // 12
            new_note_pitch = max(0, min(127, octave * 12 + closest_scale_interval + key_offset))

            new_velocity = msg.velocity
            if is_on:
                open_notes[key].append(new_note_pitch)
                if velocity_randomization > 0:
                    factor = 1 + rng.uniform(-velocity_randomization / 100, velocity_randomization / 100)
                    new_velocity = max(1, min(127, int(round(float(new_velocity) * factor))))

            new_track.append(_clone(msg, note=new_note_pitch, velocity=new_velocity))
        new_midi.tracks.append(new_track)
    return new_midi


@_preserves_meta
def midi_phrase_reconstructor(original_midi, phrase_length_beats, reassembly_style, seed=None):
    """Riorganizza le frasi MIDI."""
    rng = _rng(seed, "phrase_reconstructor")
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_phrase = original_midi.ticks_per_beat * phrase_length_beats

    if ticks_per_phrase == 0:
        warn("La lunghezza della frase è zero. Nessuna riorganizzazione applicata.")
        return original_midi

    for original_track in original_midi.tracks:
        phrases = []
        current_phrase_events = []
        _track_name = original_track.name if hasattr(original_track, 'name') else ''
        _header = _extract_instrument_header(original_track)
        current_phrase_start_tick = 0

        events_with_abs_time = []
        time_since_last_event = 0
        _seen_note = False
        for msg in original_track:
            time_since_last_event += msg.time
            if msg.type in ('note_on', 'note_off'):
                _seen_note = True
            if msg.type == 'program_change' or (msg.type == 'control_change' and msg.control in (0, 32)):
                continue  # gia' catturati in _header, verranno fissati all'inizio
            if not _seen_note and _is_instrument_header_msg(msg):
                continue  # volume/pan/effetti iniziali: stanno nell'header, non si rimescolano con le frasi
            if msg.is_meta and msg.type in _GLOBAL_META_TYPES:
                continue  # tempo/metrica/tonalita' non si rimescolano con le frasi: li ripristina _preserves_meta
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
            _empty_track = mido.MidiTrack()
            if _track_name:
                _empty_track.name = _track_name
            for _h in _header:
                _empty_track.append(_h)
            new_midi.tracks.append(_empty_track)
            continue

        reorganized_phrases = []
        if reassembly_style == "Casuale":
            reorganized_phrases = list(phrases)
            rng.shuffle(reorganized_phrases)
        elif reassembly_style == "Inversione":
            reorganized_phrases = list(reversed(phrases))
        elif reassembly_style == "Ciclico A-B-A":
            if len(phrases) >= 3:
                a_phrase, b_phrase, c_phrase = phrases[0], phrases[1], (phrases[2] if len(phrases) > 2 else phrases[1])
                num_repetitions = max(1, len(phrases) // 3)
                for _ in range(num_repetitions):
                    reorganized_phrases.extend([a_phrase, b_phrase, a_phrase, c_phrase])
            else:
                warn(f"Troppo poche frasi ({len(phrases)}) per lo stile 'Ciclico A-B-A'. Verrà usata la riorganizzazione casuale.")
                reorganized_phrases = list(phrases)
                rng.shuffle(reorganized_phrases)
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
        for _h in _header:
            new_track.append(_h)
        flat_events_for_reconstruction = []
        absolute_time_in_reorganized_seq = 0

        for phrase_block in reorganized_phrases:
            # Contatore delle note aperte in questa frase (per nota/canale).
            # Una nota che inizia in una frase e finisce in quella successiva viene troncata
            # a fine frase, e il suo note_off "orfano" (nell'altra frase) viene scartato:
            # dopo il rimescolamento non resta nessuna nota appesa ne' nessun note_off senza note_on.
            open_notes = defaultdict(int)  # (pitch, channel) -> note_on aperti
            phrase_abs = absolute_time_in_reorganized_seq

            for msg_in_phrase in phrase_block:
                phrase_abs += msg_in_phrase.time
                if msg_in_phrase.type == 'note_on' and msg_in_phrase.velocity > 0:
                    open_notes[(msg_in_phrase.note, msg_in_phrase.channel)] += 1
                elif msg_in_phrase.type == 'note_off' or (msg_in_phrase.type == 'note_on' and msg_in_phrase.velocity == 0):
                    key = (msg_in_phrase.note, msg_in_phrase.channel)
                    if open_notes[key] <= 0:
                        continue  # note_off orfano: il suo note_on sta in un'altra frase
                    open_notes[key] -= 1
                flat_events_for_reconstruction.append({'msg': _clone(msg_in_phrase), 'abs_time': phrase_abs})

            # Chiudi note rimaste aperte alla fine della frase
            phrase_end = phrase_abs
            for (pitch, ch), n_open in open_notes.items():
                for _ in range(n_open):
                    flat_events_for_reconstruction.append({
                        'msg': mido.Message('note_off', note=pitch, velocity=0, channel=ch, time=0),
                        'abs_time': phrase_end
                    })

            absolute_time_in_reorganized_seq = phrase_end

        # Ordina: note_off prima di note_on allo stesso tick
        append_events_to_track(new_track, flat_events_for_reconstruction)

        new_midi.tracks.append(new_track)
    return new_midi

@_preserves_meta
def midi_time_scrambler(original_midi, stretch_factor, quantization_strength, swing_amount):
    """
    Modifica il timing e la durata delle note MIDI.
    """
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    ticks_per_subdivision = original_midi.ticks_per_beat / 4
    if ticks_per_subdivision == 0:
        warn("Ticks per beat è zero o troppo basso. Restituito il MIDI originale.")
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
            events_with_abs_time.append({'msg': _clone(msg), 'abs_time_mod': current_abs_time_stretched})

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
            new_msg = _clone(msg, time=delta_time)
            new_track.append(new_msg)
            last_abs_time_mod = abs_time_mod

        new_midi.tracks.append(new_track)
    return new_midi

@_preserves_meta
@_keeps_tempo_map
def midi_density_transformer(original_midi, add_note_probability, remove_note_probability, polyphony_mode, preserve_drums=None, seed=None):
    """
    Aggiunge o rimuove note per alterare la densita' MIDI.
    Fix: tracce senza note vengono passate intatte.
    Fix: note aggiunte hanno durata esplicita uguale alla nota originale.
    Fix: note_off sempre dopo note_on — abs_time note_off = start + durata originale.
    """
    rng = _rng(seed, "density")
    preserve = _drums_preserved(preserve_drums)
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        _dens_name = original_track.name if hasattr(original_track, 'name') else ''
        _dens_header = _extract_instrument_header(original_track)
        notes = extract_notes(original_track, original_midi.ticks_per_beat)

        # Se la traccia non ha note (metadati, controller, ecc.) — passa intatta
        if not notes:
            new_midi.tracks.append(original_track)
            continue

        # (il sorteggio avviene sempre, cosi' la sequenza casuale delle altre tracce non cambia;
        #  i colpi di batteria con "Preserva la batteria" attivo non vengono mai rimossi)
        modified_notes = [note for note in notes
                          if rng.randint(0, 100) >= remove_note_probability
                          or (preserve and note['channel'] == DRUM_CHANNEL)]

        # Durata minima garantita: almeno 1 tick
        def safe_duration(note):
            return max(1, note['end'] - note['start'])

        final_events = []
        track_end_time = max(n['end'] for n in notes)
        drone_channel = _get_track_default_channel(original_track)  # il drone suona con lo strumento della traccia

        if polyphony_mode == "Droni" and add_note_probability > 0 and drone_channel != DRUM_CHANNEL and rng.randint(0, 100) < add_note_probability:
            drone_pitch = 36
            drone_velocity = 64
            final_events.append({'msg': mido.Message('note_on', note=drone_pitch, velocity=drone_velocity, channel=drone_channel, time=0), 'abs_time': 0})
            final_events.append({'msg': mido.Message('note_off', note=drone_pitch, velocity=0, channel=drone_channel, time=0), 'abs_time': track_end_time + original_midi.ticks_per_beat * 4})

        for note_data in modified_notes:
            dur = safe_duration(note_data)
            note_start = note_data['start']
            note_end   = note_start + dur  # durata esplicita, non dipende da note_off originale

            final_events.append({'msg': mido.Message('note_on',  note=note_data['pitch'], velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_start})
            final_events.append({'msg': mido.Message('note_off', note=note_data['pitch'], velocity=0,                    channel=note_data['channel'], time=0), 'abs_time': note_end})

            if (not (preserve and note_data['channel'] == DRUM_CHANNEL)) and rng.randint(0, 100) < add_note_probability:
                if polyphony_mode == "Riempi Accordo (Triadi)":
                    intervals = [4, 7]
                elif polyphony_mode == "Aggiungi Contro-Melodia":
                    intervals = [rng.choice([-5, -3, -2, 2, 3, 5])]
                else:
                    intervals = []

                for interval in intervals:
                    new_pitch = note_data['pitch'] + interval
                    if 0 <= new_pitch <= 127:
                        # Nota aggiunta: stessa durata della nota originale, note_off esplicito
                        final_events.append({'msg': mido.Message('note_on',  note=new_pitch, velocity=note_data['velocity'], channel=note_data['channel'], time=0), 'abs_time': note_start})
                        final_events.append({'msg': mido.Message('note_off', note=new_pitch, velocity=0,                    channel=note_data['channel'], time=0), 'abs_time': note_end})

        # Ordina per abs_time, note_off prima di note_on allo stesso tick (evita sovrapposizioni)
        new_track = mido.MidiTrack()
        if _dens_name:
            new_track.name = _dens_name
        for _h in _dens_header:
            new_track.append(_h)
        append_events_to_track(new_track, final_events)

        new_midi.tracks.append(new_track)
    return new_midi

@_preserves_meta
def midi_random_pitch_transformer(original_midi, random_pitch_strength, preserve_drums=None, seed=None):
    """
    Randomizes the pitch of notes based on a given strength (probability).
    Usa (pitch, channel) come chiave e un contatore per gestire note duplicate
    sullo stesso pitch/canale — nessuna nota resta aperta nel DAW.
    """
    rng = _rng(seed, "random_pitch")
    preserve = _drums_preserved(preserve_drums)
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)

    for original_track in original_midi.tracks:
        new_track = mido.MidiTrack()
        if hasattr(original_track, 'name') and original_track.name:
            new_track.name = original_track.name

        # pitch_map: (pitch_orig, channel) -> lista di pitch nuovi (stack LIFO)
        # gestisce piu' note_on sullo stesso pitch prima del note_off
        pitch_map = defaultdict(list)

        for msg in original_track:
            if preserve and msg.type in ('note_on', 'note_off') and msg.channel == DRUM_CHANNEL:
                new_track.append(msg)
                continue

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.note, msg.channel)
                if rng.randint(0, 100) < random_pitch_strength:
                    new_pitch = rng.randint(0, 127)
                else:
                    new_pitch = msg.note
                pitch_map[key].append(new_pitch)
                new_track.append(_clone(msg, note=new_pitch))

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if pitch_map[key]:
                    # LIFO: chiude l'ultima nota aperta su questo pitch/canale
                    mapped_pitch = pitch_map[key].pop()
                else:
                    mapped_pitch = msg.note
                new_track.append(_clone(msg, note=mapped_pitch))

            else:
                new_track.append(msg)

        # Chiudi eventuali note rimaste aperte (note_on senza note_off)
        for (orig_pitch, ch), pitches in pitch_map.items():
            for p in pitches:
                new_track.append(mido.Message('note_off', note=p, velocity=0, channel=ch, time=0))

        new_midi.tracks.append(new_track)
    return new_midi


_RB_DRUM_MAP = {"kick": 36, "snare": 38, "hihat_closed": 42}
_RB_INSTRUMENT_NAMES = {"kick": "Cassa", "snare": "Rullante", "hihat_closed": "Hi-hat"}


def _rb_hit(start_tick, duration_ticks, velocity):
    return {'start_tick': start_tick, 'duration_ticks': duration_ticks, 'velocity': velocity}


def _rb_parse_time_signature(time_signature):
    try:
        beats, note_value = map(int, time_signature.split('/'))
        if beats <= 0 or note_value <= 0:
            raise ValueError
        return beats, note_value
    except (ValueError, IndexError):
        warn(f"Metrica non valida: '{time_signature}'. Verrà usata la metrica 4/4.")
        return 4, 4


def _rb_pattern_fixed(kick, snare, hihat, beats_per_measure, tpb):
    """Pattern Fisso (Pop/Rock): cassa su 1 e 3, rullante su 2 e 4, hi-hat in ottavi."""
    pat = {"kick": [], "snare": [], "hihat_closed": []}
    if kick:
        pat["kick"].append(_rb_hit(0, tpb // 8, 100))
        if beats_per_measure >= 3:
            pat["kick"].append(_rb_hit(tpb * 2, tpb // 8, 100))
    if snare:
        if beats_per_measure >= 2:
            pat["snare"].append(_rb_hit(tpb, tpb // 8, 100))
        if beats_per_measure >= 4:
            pat["snare"].append(_rb_hit(tpb * 3, tpb // 8, 100))
    if hihat:
        for i in range(beats_per_measure * 2):
            pat["hihat_closed"].append(_rb_hit(i * tpb // 2, tpb // 8, 80))
    return pat


def _rb_pattern_random(rng, kick, snare, hihat, beats_per_measure, tpb):
    """Pattern Casuale: ogni sedicesimo ha una probabilita' per strumento (ordine delle estrazioni RNG invariato)."""
    pat = {"kick": [], "snare": [], "hihat_closed": []}
    kick_prob, snare_prob, hihat_prob = 0.2, 0.1, 0.4
    sub = tpb // 4
    for i in range(beats_per_measure * 4):
        start_tick = i * sub
        duration = sub // 2
        if kick and rng.random() < kick_prob:
            pat["kick"].append(_rb_hit(start_tick, duration, rng.randint(80, 110)))
        if snare and rng.random() < snare_prob:
            pat["snare"].append(_rb_hit(start_tick, duration, rng.randint(80, 110)))
        if hihat and rng.random() < hihat_prob:
            pat["hihat_closed"].append(_rb_hit(start_tick, duration, rng.randint(60, 90)))
    return pat


def _rb_pattern_adaptive(rng, original_midi, kick, snare, hihat, beats_per_measure, ticks_per_measure, tpb):
    """Pattern Adattivo: cassa/rullante sui tick (snappati al sedicesimo) dove il brano suona di piu'."""
    pat = {"kick": [], "snare": [], "hihat_closed": []}
    note_on_counts = defaultdict(int)
    for track in original_midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
                tick_in_measure = abs_time % ticks_per_measure
                subdivision_ticks = tpb // 4
                note_on_counts[round(tick_in_measure / subdivision_ticks) * subdivision_ticks] += 1

    if not note_on_counts:
        warn("Nessuna nota trovata per un pattern adattivo. Verrà usato un pattern fisso.")
        if kick:  pat["kick"].append(_rb_hit(0, tpb // 8, 100))
        if snare: pat["snare"].append(_rb_hit(tpb, tpb // 8, 100))
        if hihat: pat["hihat_closed"].append(_rb_hit(0, tpb // 2, 80))
        return pat

    most_common_ticks = sorted(note_on_counts, key=note_on_counts.get, reverse=True)

    kick_ticks = []
    if kick:
        for tick in most_common_ticks:
            if len(kick_ticks) >= 3:
                break
            if (beats_per_measure == 4 and (tick == 0 or tick == tpb * 2)) or (len(kick_ticks) < 2 and note_on_counts[tick] > 1):
                kick_ticks.append(tick)
        if not kick_ticks:
            kick_ticks.extend([0, tpb * 2] if beats_per_measure >= 4 else [0])
        for tick in kick_ticks:
            pat["kick"].append(_rb_hit(tick, tpb // 8, 100))

    snare_ticks = []
    if snare:
        for tick in most_common_ticks:
            is_kick_tick = any(abs(tick - kt) < tpb / 4 for kt in kick_ticks)
            if not is_kick_tick and len(snare_ticks) < 2:
                snare_ticks.append(tick)
        if not snare_ticks:
            snare_ticks.extend([tpb, tpb * 3] if beats_per_measure >= 4 else [tpb])
        for tick in snare_ticks:
            pat["snare"].append(_rb_hit(tick, tpb // 8, 100))

    if hihat:
        ticks_per_eighth = tpb // 2
        for i in range(int(ticks_per_measure / ticks_per_eighth)):
            pat["hihat_closed"].append(_rb_hit(i * ticks_per_eighth, ticks_per_eighth // 2, rng.randint(60, 90)))
    return pat


def _rb_render_track(drum_name, hits, num_measures, ticks_per_measure, total_ticks):
    """Replica il pattern di una misura per tutto il brano e lo scrive come traccia di batteria (canale 10)."""
    track = mido.MidiTrack()
    track.name = f"Ritmica: {_RB_INSTRUMENT_NAMES[drum_name]}"
    track.append(mido.Message('program_change', program=0, channel=9, time=0))

    note = _RB_DRUM_MAP[drum_name]
    events = []
    for i in range(num_measures):
        measure_start = i * ticks_per_measure
        for hit in hits:
            start = measure_start + hit['start_tick']
            if start < total_ticks:  # solo eventi che rientrano nella durata del brano
                end = start + hit['duration_ticks']
                events.append({'msg': mido.Message('note_on', note=note, velocity=hit['velocity'], channel=9), 'abs_time': start})
                events.append({'msg': mido.Message('note_off', note=note, velocity=0, channel=9), 'abs_time': min(end, total_ticks)})

    events.sort(key=lambda x: (x['abs_time'], 0 if x['msg'].type == 'note_off' else 1))
    last_abs_time = 0
    for ev in events:
        track.append(_clone(ev['msg'], time=max(0, ev['abs_time'] - last_abs_time)))
        last_abs_time = ev['abs_time']

    # Messaggio finale: garantisce la lunghezza corretta della traccia
    track.append(mido.MetaMessage('end_of_track', time=max(0, total_ticks - last_abs_time)))
    return track


@_preserves_meta
def midi_add_rhythmic_base(original_midi, kick, snare, hihat, time_signature, rhythmic_pattern_style, seed=None):
    """
    Aggiunge una o più tracce con una base ritmica che dura esattamente quanto il brano originale.
    """
    rng = _rng(seed, "rhythmic_base")
    new_midi = mido.MidiFile(ticks_per_beat=original_midi.ticks_per_beat)
    for track in original_midi.tracks:
        new_midi.tracks.append(track)

    beats_per_measure, note_value = _rb_parse_time_signature(time_signature)
    tpb = new_midi.ticks_per_beat
    ticks_per_measure = int(tpb * beats_per_measure * 4 / note_value)
    if ticks_per_measure == 0:
        warn("Ticks per misura è zero. Non è possibile aggiungere la base ritmica.")
        return new_midi

    # Durata totale del brano originale in ticks
    total_ticks = 0
    for track in original_midi.tracks:
        total_ticks = max(total_ticks, sum(msg.time for msg in track))
    if total_ticks == 0:
        warn("Il brano originale non contiene eventi validi per calcolare la lunghezza. La base ritmica non verrà aggiunta.")
        return new_midi

    patterns = {"kick": [], "snare": [], "hihat_closed": []}
    if rhythmic_pattern_style == "Pattern Fisso (Pop/Rock)":
        patterns = _rb_pattern_fixed(kick, snare, hihat, beats_per_measure, tpb)
    elif rhythmic_pattern_style == "Pattern Casuale":
        patterns = _rb_pattern_random(rng, kick, snare, hihat, beats_per_measure, tpb)
    elif rhythmic_pattern_style == "Pattern Adattivo":
        patterns = _rb_pattern_adaptive(rng, original_midi, kick, snare, hihat,
                                        beats_per_measure, ticks_per_measure, tpb)

    num_measures = int(np.ceil(total_ticks / ticks_per_measure))
    for drum_name, hits in patterns.items():
        if hits:
            new_midi.tracks.append(_rb_render_track(drum_name, hits, num_measures, ticks_per_measure, total_ticks))
    return new_midi


# Mappatura General MIDI: numero programma → nome famiglia strumentale
def _gm_track_name(program, channel):
    """Nome GM dello strumento (es. 'Violin', 'Acoustic Grand Piano') dato programma e canale; 'Drums' sul canale 10."""
    if channel == 9:
        return "Drums"
    return GM_PROGRAM_NAMES[max(0, min(127, int(program)))]


def _split_type0_to_tracks(midi):
    """
    Converte un MIDI tipo 0 (1 traccia, N canali) in un MIDI tipo 1
    con una traccia per canale attivo (canali senza note vengono ignorati).
    Nomina ogni traccia con il nome GM reale (Bass, Drums, Guitar, ecc.)
    preservando il canale originale.
    """
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
        meta_track.append(_clone(msg, time=delta))
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
            ch_track.append(_clone(msg, time=delta))
            last_t = abs_t
        new_midi.tracks.append(ch_track)

    return new_midi


@_trims_tail
@_preserves_meta
def _recomposer_style_config(style, tpb):
    """Parametri del `style` del Recomposer (durate/pause in ticks, fattore velocity, passo di pitch)."""
    style_configs = {
        "ambient":             {"note_dur_range": (tpb * 2, tpb * 6),  "gap_range": (tpb // 2, tpb * 2),
                                "vel_factor": 0.6,  "pitch_step": 4},
        "drone":               {"note_dur_range": (tpb * 4, tpb * 8),  "gap_range": (tpb, tpb * 3),
                                "vel_factor": 0.5,  "pitch_step": 7},
        "minimal":             {"note_dur_range": (tpb // 2, tpb * 2), "gap_range": (tpb // 4, tpb),
                                "vel_factor": 0.7,  "pitch_step": 2},
        "armonico":            {"note_dur_range": (tpb // 2, tpb),     "gap_range": (tpb // 8, tpb // 4),
                                "vel_factor": 0.85, "pitch_step": 3},
        "elettronico":         {"note_dur_range": (tpb // 4, tpb // 2), "gap_range": (tpb // 8, tpb // 4),
                                "vel_factor": 0.9,  "pitch_step": 0},
        "minimalismo_ritmico": {"note_dur_range": (tpb // 4, tpb // 2), "gap_range": (tpb // 2, tpb * 2),
                                "vel_factor": 0.8,  "pitch_step": 5},
        "sperimentale":        {"note_dur_range": (tpb // 8, tpb * 3), "gap_range": (0, tpb * 2),
                                "vel_factor": 1.0,  "pitch_step": 0},
    }
    return style_configs.get(style, style_configs["minimal"])


def _recomposer_total_ticks(midi, tpb):
    """Durata in ticks = fine della MUSICA (ultima nota), non la fine grezza del file:
    un end_of_track / marker molto oltre l'ultima nota allungherebbe l'output di minuti."""
    total_ticks = midi_last_note_tick(midi)
    if total_ticks == 0:
        for track in midi.tracks:
            total_ticks = max(total_ticks, sum(msg.time for msg in track))
    if total_ticks == 0:
        total_ticks = tpb * 4 * 32  # fallback 32 battute
    return total_ticks


def _recomposer_pitch_picker(rng, cfg, weighted_pool):
    """Ritorna pick_pitch(base): sceglie il prossimo pitch dal pool (vocabolario della traccia)."""
    # Indice per il "pitch piu' vicino": valori unici ordinati + prima posizione nel pool,
    # cosi' la parita' si risolve come min() (vince il primo del pool) ma in O(log n).
    uniq = sorted(set(weighted_pool))
    first_pos = {}
    for i, p in enumerate(weighted_pool):
        first_pos.setdefault(p, i)

    def nearest_in_pool(candidate):
        k = bisect.bisect_left(uniq, candidate)
        if k == 0:
            return uniq[0]
        if k == len(uniq):
            return uniq[-1]
        lo, hi = uniq[k - 1], uniq[k]
        d_lo, d_hi = candidate - lo, hi - candidate
        if d_lo < d_hi:
            return lo
        if d_hi < d_lo:
            return hi
        return lo if first_pos[lo] < first_pos[hi] else hi

    def pick_pitch(base=None):
        if base is None or cfg["pitch_step"] == 0:
            return rng.choice(weighted_pool)
        direction = rng.choice([-1, 1])
        return nearest_in_pool(base + direction * cfg["pitch_step"])

    return pick_pitch


def _recomposer_build_track(rng, cfg, style, tpb, total_ticks, weighted_pool,
                            vel_min, vel_max, channel, track_name, instrument_header=None):
    """Costruisce una nuova traccia dal pool di pitch di una traccia originale."""
    if not weighted_pool:
        return None
    pick_pitch = _recomposer_pitch_picker(rng, cfg, weighted_pool)

    new_track = mido.MidiTrack()
    new_track.name = track_name
    for h in (instrument_header or []):
        new_track.append(h)

    events = []
    current_tick = 0
    last_pitch = rng.choice(weighted_pool)

    while current_tick < total_ticks:
        pitch = max(0, min(127, pick_pitch(last_pitch)))
        vel   = max(1, min(127, int(rng.randint(vel_min, vel_max) * cfg["vel_factor"])))
        dur   = rng.randint(*cfg["note_dur_range"])
        gap   = rng.randint(*cfg["gap_range"])

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
    for kind, tick, p, v2, ch in events:
        delta = max(0, tick - last_t)
        if kind == "on":
            new_track.append(mido.Message("note_on",  note=p, velocity=v2, channel=ch, time=delta))
        else:
            new_track.append(mido.Message("note_off", note=p, velocity=0,  channel=ch, time=delta))
        last_t = tick
    return new_track


def _recomposer_vocabulary(orig_track):
    """(pitch, velocity, canale) di tutte le note_on della traccia."""
    pitches, velocities, channels = [], [], []
    for msg in orig_track:
        if msg.type == 'note_on' and msg.velocity > 0:
            pitches.append(msg.note)
            velocities.append(msg.velocity)
            channels.append(msg.channel)
    return pitches, velocities, channels


def _recomposer_velocity_range(velocities):
    vel_min, vel_max = min(velocities), max(velocities)
    if vel_min > vel_max:
        vel_min, vel_max = vel_max, vel_min
    vel_min = max(1, vel_min)
    vel_max = min(127, vel_max)
    if vel_min == vel_max:
        vel_min = max(1, vel_max - 10)
    return vel_min, vel_max


def _copy_track(track, name=None):
    """Copia integrale di una traccia (opzionalmente rinominata)."""
    out = mido.MidiTrack()
    if name is not None:
        out.name = name
    for msg in track:
        out.append(_clone(msg))
    return out


@_keeps_tempo_map
def midi_recomposer(original_midi, style, preserve_drums=None, seed=None):
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
    rng = _rng(seed, "recomposer")
    preserve = _drums_preserved(preserve_drums)

    # File tipo 0: esplodi canali in tracce separate prima di ricomporre
    if original_midi.type == 0 or (len(original_midi.tracks) == 1 and
            len({m.channel for t in original_midi.tracks for m in t if hasattr(m, 'channel')}) > 1):
        original_midi = _split_type0_to_tracks(original_midi)

    tpb = original_midi.ticks_per_beat
    total_ticks = _recomposer_total_ticks(original_midi, tpb)
    cfg = _recomposer_style_config(style, tpb)

    new_midi = mido.MidiFile(ticks_per_beat=tpb)
    for track_idx, orig_track in enumerate(original_midi.tracks):
        track_name = orig_track.name if hasattr(orig_track, 'name') and orig_track.name else f"Track {track_idx}"
        pitches, velocities, channels = _recomposer_vocabulary(orig_track)

        # Traccia senza note (es. traccia metadati/tempo) → copiala intatta
        if not pitches:
            new_midi.tracks.append(_copy_track(orig_track, name=track_name))
            continue

        dominant_channel = Counter(channels).most_common(1)[0][0]

        # Batteria: traccia copiata intatta
        if preserve and dominant_channel == DRUM_CHANNEL:
            new_midi.tracks.append(_copy_track(orig_track))
            continue

        # Header strumento (program_change/bank select) da preservare
        header = _extract_instrument_header(orig_track)

        # Pool di pitch pesato
        weighted_pool = []
        for pitch, count in Counter(pitches).items():
            weighted_pool.extend([pitch] * max(1, count))

        vel_min, vel_max = _recomposer_velocity_range(velocities)
        new_track = _recomposer_build_track(
            rng, cfg, style, tpb, total_ticks, weighted_pool, vel_min, vel_max,
            dominant_channel, track_name, instrument_header=header)

        if new_track is None:
            # Fallback: traccia vuota con nome originale (e strumento preservato)
            new_track = mido.MidiTrack()
            new_track.name = track_name
            for h in header:
                new_track.append(h)
        new_midi.tracks.append(new_track)

    return new_midi


# --- Funzione Report ---
_METHOD_LABELS_EN = {
    "MIDI Note Remapper": "🎶 Note Remapping (Vertical)",
    "MIDI Phrase Reconstructor": "🔄 Phrase Reorganization (Horizontal)",
    "MIDI Time Scrambler": "⏳ Rhythm/Duration Manipulation (Horizontal)",
    "MIDI Density Transformer": "🎲 Density Control (Harmony/Counterpoint)",
    "MIDI Random Pitch Transformer": "❓ Total Pitch Randomization (Chaos)",
    "MIDI Rhythmic Base": "🥁 Add Rhythmic Base",
    "MIDI Recomposer": "🔁 Recomposition (new piece from the original material)",
    "MIDI Costas Sequencer": "🧮 Scott Rickard — Costas Sequencer",
    "MIDI Stockhausen Punktuelle": "🎯 Karlheinz Stockhausen — Punktuelle Musik",
    "MIDI Boulez Multiplication": "🔷 Pierre Boulez — Chord Multiplication",
    "MIDI Xenakis Stochastic": "☁️ Iannis Xenakis — Stochastic Music (Clouds)",
    "MIDI Cage Chance": "☯️ John Cage — Chance Operations (I Ching)",
    "MIDI Eno Generative": "🌫️ Brian Eno — Generative Music (Asynchronous Loops)",
    "MIDI Bach Canon": "🎻 Johann Sebastian Bach — Strict Canon",
    "MIDI Glass Additive": "➕ Philip Glass — Additive Process",
    "MIDI Messiaen Modes": "🕊️ Olivier Messiaen — Modes of Limited Transposition",
    "MIDI Part Tintinnabuli": "🔔 Arvo Pärt — Tintinnabuli",
    "MIDI Reich Phasing": "🌀 Steve Reich — Phasing",
    "MIDI Canon per Tonos": "🔺 Bach/Hofstadter — Canon per Tonos (endlessly rising canon)",
    "MIDI Shepard Penrose": "🪜 Escher/Shepard — Impossible Scale (Shepard-Penrose)",
    "MIDI Escher Metamorphosis": "🦎 Escher — Metamorphosis (one track into another)",
    "MIDI MIU System": "🧩 Hofstadter — MIU Formal System",
    "MIDI Escher Symmetry": "🔳 Escher — Frieze Symmetries",
    "MIDI Hamiltonian Cycle": "🕸️ Graph Theory — Hamiltonian Cycle",
    "MIDI Norgard Infinity": "♾️ Per Nørgård — Infinity Series (musical fractal)",
    "MIDI Euclidean Rhythm": "🥁 Toussaint — Euclidean Rhythms",
    "MIDI Neo Riemannian": "🔁 Lewin/Tymoczko — Neo-Riemannian PLR Transformations",
    "MIDI Spiral Array": "🌀 Chew — Spiral Array (harmony and tension)",
    "MIDI Maximally Even": "⭕ Clough/Douthett — Maximally Even Sets",
    "MIDI PC DFT": "📈 Lewin/Quinn/Amiot — Fourier Transform of Pitch-Class Sets",
}


def _describe_method(lang, key, p):
    """Righe di report (lang 'it' o 'en') con i parametri usati da un metodo."""
    def L(a, b):
        return a if lang == "it" else b
    yes, no = L("Sì", "Yes"), "No"
    out = []
    try:
        if key == "MIDI Note Remapper":
            out.append(L(f"   * Scala: {p[0]} | Tonalita': {p[1]}", f"   * Scale: {p[0]} | Key: {p[1]}"))
            out.append(L(f"   * Pitch Shift: +/-{p[2]} semitoni | Velocity: {p[3]}%", f"   * Pitch shift: +/-{p[2]} semitones | Velocity: {p[3]}%"))
        elif key == "MIDI Phrase Reconstructor":
            out.append(L(f"   * Lunghezza frase: {p[0]} beat | Stile: {p[1]}", f"   * Phrase length: {p[0]} beats | Style: {p[1]}"))
        elif key == "MIDI Time Scrambler":
            out.append(L(f"   * Stretch: {p[0]}x | Quantizzazione: {p[1]}% | Swing: {p[2]}%", f"   * Stretch: {p[0]}x | Quantization: {p[1]}% | Swing: {p[2]}%"))
        elif key == "MIDI Density Transformer":
            out.append(L(f"   * Aggiungi note: {p[0]}% | Rimuovi note: {p[1]}% | Polifonia: {p[2]}", f"   * Add notes: {p[0]}% | Remove notes: {p[1]}% | Polyphony: {p[2]}"))
        elif key == "MIDI Random Pitch Transformer":
            out.append(L(f"   * Forza randomizzazione: {p[0]}%", f"   * Randomization strength: {p[0]}%"))
        elif key == "MIDI Rhythmic Base":
            names_it = [n for flag, n in zip(p[:3], ("Cassa", "Rullante", "Hi-hat")) if flag]
            names_en = [n for flag, n in zip(p[:3], ("Kick", "Snare", "Hi-hat")) if flag]
            out.append(L(f"   * Elementi: {', '.join(names_it) if names_it else 'Nessuno'}", f"   * Elements: {', '.join(names_en) if names_en else 'None'}"))
            out.append(L(f"   * Metrica: {p[3]} | Pattern: {p[4]}", f"   * Meter: {p[3]} | Pattern: {p[4]}"))
        elif key == "MIDI Recomposer":
            out.append(L(f"   * Stile di ricomposizione: {p[0]}", f"   * Recomposition style: {p[0]}"))
        elif key == "MIDI Costas Sequencer":
            mode = p[0]
            out.append(L(f"   * Modalità: {mode} | Ordine richiesto: {p[1]}", f"   * Mode: {mode} | Requested order: {p[1]}"))
            if mode == "Permutazione Pitch (Cromatica)":
                out.append(L(f"   * Trasposizione: {p[2]} ottave | Costruzione di Welch, n=12, p=13", f"   * Transposition: {p[2]} octaves | Welch construction, n=12, p=13"))
            elif mode == "Griglia Ritmica Costas":
                out.append(L("   * Costruzione di Welch (Scott Rickard / J.P. Costas)", "   * Welch construction (Scott Rickard / J.P. Costas)"))
            else:
                out.append(L(f"   * Pitch base: {p[2]} | Estensione: {p[3]} semitoni | Passo: {p[4]} beat", f"   * Base pitch: {p[2]} | Range: {p[3]} semitones | Step: {p[4]} beats"))
        elif key == "MIDI Stockhausen Punktuelle":
            dur_on, dyn_on, timbre_on, iso_on, row_used = p
            out.append(L(f"   * Fila dodecafonica: {row_used}", f"   * Twelve-tone row: {row_used}"))
            fl_it = [n for f, n in ((dur_on, "Durata"), (dyn_on, "Dinamica"), (timbre_on, "Timbro")) if f]
            fl_en = [n for f, n in ((dur_on, "Duration"), (dyn_on, "Dynamics"), (timbre_on, "Timbre")) if f]
            out.append(L(f"   * Parametri serializzati: Altezza (P), {', '.join(fl_it) if fl_it else 'solo Altezza'}", f"   * Serialized parameters: Pitch (P), {', '.join(fl_en) if fl_en else 'pitch only'}"))
            out.append(L(f"   * Isolamento punti (staccato): {yes if iso_on else no}", f"   * Point isolation (staccato): {yes if iso_on else no}"))
            out.append(L("   * Serialismo integrale (Stockhausen/Boulez, rad. Messiaen 'Mode de valeurs')", "   * Integral serialism (Stockhausen/Boulez, after Messiaen 'Mode de valeurs')"))
        elif key == "MIDI Boulez Multiplication":
            set_size, chord_density, register_spread, set_a, set_b = p
            out.append(L(f"   * Dimensione insiemi A/B: {set_size} | Densità accordo: {'completa' if not chord_density else chord_density} | Ottave: {register_spread}",
                         f"   * Set size A/B: {set_size} | Chord density: {'full' if not chord_density else chord_density} | Octaves: {register_spread}"))
            out.append(L(f"   * Insieme A: {set_a} | Insieme B: {set_b}", f"   * Set A: {set_a} | Set B: {set_b}"))
            out.append(L("   * Moltiplicazione d'accordi (Boulez, 'Le Marteau sans maître' / Structures II)", "   * Chord multiplication (Boulez, 'Le Marteau sans maître' / Structures II)"))
        elif key == "MIDI Xenakis Stochastic":
            sieve_str, mean_ev, pc, ps, dm, vm, vs = p
            out.append(L(f"   * Crivello (sieve): {sieve_str} | Tasso eventi: {mean_ev}/beat (processo di Poisson)", f"   * Sieve: {sieve_str} | Event rate: {mean_ev}/beat (Poisson process)"))
            out.append(L(f"   * Altezza: Gauss(μ={pc}, σ={ps}) | Durata: Exp(μ={dm} beat) | Dinamica: Gauss(μ={vm}, σ={vs})",
                         f"   * Pitch: Gauss(μ={pc}, σ={ps}) | Duration: Exp(μ={dm} beats) | Dynamics: Gauss(μ={vm}, σ={vs})"))
            out.append(L("   * Musica stocastica (Xenakis, Pithoprakta/Achorripsis, 1955-57)", "   * Stochastic music (Xenakis, Pithoprakta/Achorripsis, 1955-57)"))
        elif key == "MIDI Cage Chance":
            silence_p, dur_var = p
            out.append(L(f"   * Probabilità di silenzio per evento: {silence_p:.0%} | Varietà durata: {yes if dur_var else no}", f"   * Silence probability per event: {silence_p:.0%} | Duration variety: {yes if dur_var else no}"))
            out.append(L("   * Operazioni di caso via I Ching, metodo delle tre monete (Cage, 'Music of Changes', 1951)", "   * Chance operations via I Ching, three-coin method (Cage, 'Music of Changes', 1951)"))
        elif key == "MIDI Eno Generative":
            n_loops, min_lb, max_lb, nlr, dm, vb = p
            out.append(L(f"   * Numero loop: {n_loops} | Lunghezza: {min_lb}-{max_lb} beat | Estensione durata: ×{dm}", f"   * Loops: {n_loops} | Length: {min_lb}-{max_lb} beats | Duration stretch: ×{dm}"))
            out.append(L(f"   * Rapporto durata nota/loop: {nlr} | Velocity base: {vb}", f"   * Note/loop duration ratio: {nlr} | Base velocity: {vb}"))
            out.append(L("   * Loop asincroni a lunghezze incommensurabili (Eno, 'Music for Airports'/'Discreet Music')", "   * Asynchronous loops of incommensurable lengths (Eno, 'Music for Airports'/'Discreet Music')"))
        elif key == "MIDI Bach Canon":
            n_v, interval, delay, transf = p
            out.append(L(f"   * Voci: {n_v} | Intervallo tra voci: {interval} semitoni | Ritardo (comes): {delay} beat", f"   * Voices: {n_v} | Interval between voices: {interval} semitones | Delay (comes): {delay} beats"))
            out.append(L(f"   * Trasformazione: {transf}", f"   * Transformation: {transf}"))
            out.append(L("   * Canone rigoroso (Bach, Arte della Fuga / Offerta Musicale)", "   * Strict canon (Bach, Art of Fugue / Musical Offering)"))
        elif key == "MIDI Glass Additive":
            cell_len, direction, repeats = p
            out.append(L(f"   * Lunghezza cellula: {cell_len} note | Direzione: {direction} | Ripetizioni per stadio: {repeats}", f"   * Cell length: {cell_len} notes | Direction: {direction} | Repetitions per stage: {repeats}"))
            out.append(L("   * Processo additivo (Glass, 'Two Pages'/'1+1'/'Music in Twelve Parts')", "   * Additive process (Glass, 'Two Pages'/'1+1'/'Music in Twelve Parts')"))
        elif key == "MIDI Messiaen Modes":
            mode_n, transp, nrr, cell_notes = p
            out.append(L(f"   * Modo a trasposizione limitata: Modo {mode_n} | Trasposizione: +{transp} semitoni", f"   * Mode of limited transposition: Mode {mode_n} | Transposition: +{transp} semitones"))
            out.append(L(f"   * Ritmo non retrogradabile: {yes if nrr else no} (cellula: {cell_notes} valori)", f"   * Non-retrogradable rhythm: {yes if nrr else no} (cell: {cell_notes} values)"))
            out.append(L("   * Modi a trasposizione limitata (Messiaen, 'Technique de mon langage musical', 1944)", "   * Modes of limited transposition (Messiaen, 'Technique de mon langage musical', 1944)"))
        elif key == "MIDI Part Tintinnabuli":
            tonic, triad, position = p
            out.append(L(f"   * Tonica: {tonic} {triad} | Posizione voce T: {position}", f"   * Tonic: {tonic} {triad} | T-voice position: {position}"))
            out.append(L("   * Tintinnabuli (Pärt, dal 1976: 'Spiegel im Spiegel', 'Für Alina')", "   * Tintinnabuli (Pärt, since 1976: 'Spiegel im Spiegel', 'Für Alina')"))
        elif key == "MIDI Reich Phasing":
            cell_len, cycles, shift_units, shift_every = p
            out.append(L(f"   * Lunghezza cellula: {cell_len} note | Cicli: {cycles}", f"   * Cell length: {cell_len} notes | Cycles: {cycles}"))
            out.append(L(f"   * Sfasamento: +{shift_units} unità ogni {shift_every} cicli", f"   * Phase shift: +{shift_units} units every {shift_every} cycles"))
            out.append(L("   * Phasing processuale (Reich, 'Piano Phase'/'Clapping Music')", "   * Process phasing (Reich, 'Piano Phase'/'Clapping Music')"))
        elif key == "MIDI Canon per Tonos":
            n_v, step, laps, delay, v_int, subj, close = p
            out.append(L(f"   * Voci: {n_v} (intervallo {v_int} st, ritardo {delay} beat) | Modulazione: +{step} st a giro | Ottave: {laps}",
                         f"   * Voices: {n_v} (interval {v_int} st, delay {delay} beats) | Modulation: +{step} st per lap | Octaves: {laps}"))
            out.append(L(f"   * Soggetto: {subj} note | Giro finale di arrivo in tonalita': {yes if close else no}",
                         f"   * Subject: {subj} notes | Closing arrival lap in the home key: {yes if close else no}"))
            out.append(L("   * Canon per Tonos (Bach, Offerta Musicale, 1747; Hofstadter, GEB)", "   * Canon per Tonos (Bach, Musical Offering, 1747; Hofstadter, GEB)"))
        elif key == "MIDI Shepard Penrose":
            direction, steps, step_b, layers, scale = p
            out.append(L(f"   * Direzione: {direction} | Passi: {steps} | Durata passo: {step_b} beat | Strati: {layers}",
                         f"   * Direction: {direction} | Steps: {steps} | Step length: {step_b} beats | Layers: {layers}"))
            out.append(L(f"   * Scala: {scale}", f"   * Scale: {scale}"))
            out.append(L("   * Scala di Shepard (1964) / scala impossibile di Penrose, cfr. Escher 'Salendo e scendendo' (1960)",
                         "   * Shepard scale (1964) / Penrose impossible staircase, cf. Escher 'Ascending and Descending' (1960)"))
        elif key == "MIDI Escher Metamorphosis":
            a_name, b_name, n_notes, snap = p
            out.append(L(f"   * Da: {a_name} → A: {b_name} | Note generate: {n_notes}", f"   * From: {a_name} → To: {b_name} | Notes generated: {n_notes}"))
            out.append(L(f"   * Quantizzazione sul pool di pitch A∪B: {yes if snap else no}", f"   * Snap to the A∪B pitch pool: {yes if snap else no}"))
            out.append(L("   * Interpolazione di altezza, attacco, durata e velocity (Escher, serie 'Metamorphosis')", "   * Interpolation of pitch, onset, duration and velocity (Escher, 'Metamorphosis' series)"))
        elif key == "MIDI MIU System":
            steps, max_len, cell_n, cell_b, final = p
            out.append(L(f"   * Passi di derivazione: {steps} | Lunghezza massima: {max_len} | Cellula: {cell_n} note, {cell_b} beat",
                         f"   * Derivation steps: {steps} | Max length: {max_len} | Cell: {cell_n} notes, {cell_b} beats"))
            out.append(L(f"   * Stringa finale: {final}", f"   * Final string: {final}"))
            out.append(L("   * Sistema MIU (Hofstadter, GEB, 1979): I = cellula, U = cellula invertita, M = pedale; MU non e' derivabile",
                         "   * MIU system (Hofstadter, GEB, 1979): I = cell, U = inverted cell, M = pedal; MU is not derivable"))
        elif key == "MIDI Escher Symmetry":
            pattern, motif_n, cells = p
            out.append(L(f"   * Fregio: {pattern} | Motivo: {motif_n} note | Celle: {cells}", f"   * Frieze: {pattern} | Motif: {motif_n} notes | Cells: {cells}"))
            out.append(L("   * Isometrie tempo-altezza: identita', inversione, retrogrado, rotazione di 180° (Escher; du Sautoy, Blueprints)",
                         "   * Time-pitch isometries: identity, inversion, retrograde, 180° rotation (Escher; du Sautoy, Blueprints)"))
        elif key == "MIDI Hamiltonian Cycle":
            max_int, max_nodes, repeats, nodes, status, covered, expansions = p
            out.append(L(f"   * Salto massimo: {max_int} st | Nodi: {nodes} (max {max_nodes}) | Ripetizioni: {repeats}",
                         f"   * Max leap: {max_int} st | Nodes: {nodes} (max {max_nodes}) | Repetitions: {repeats}"))
            res_it = {'full': "ciclo hamiltoniano completo", 'partial': f"ciclo sulla componente principale ({covered}/{nodes} nodi)",
                      'path': f"nessun ciclo: usato il percorso piu' lungo ({covered}/{nodes} nodi)"}.get(status, status)
            res_en = {'full': "complete Hamiltonian cycle", 'partial': f"cycle on the main component ({covered}/{nodes} nodes)",
                      'path': f"no cycle: longest path used ({covered}/{nodes} nodes)"}.get(status, status)
            out.append(L(f"   * Esito: {res_it} ({expansions} espansioni)", f"   * Result: {res_en} ({expansions} expansions)"))
            out.append(L("   * Ciclo hamiltoniano (W. R. Hamilton, Icosian game, 1857; du Sautoy, Blueprints)", "   * Hamiltonian cycle (W. R. Hamilton, Icosian game, 1857; du Sautoy, Blueprints)"))
        elif key == "MIDI Norgard Infinity":
            n_notes, step_b, layers, scale = p
            out.append(L(f"   * Note per lo strato veloce: {n_notes} | Passo: {step_b} beat | Strati: {layers} | Scala: {scale}",
                         f"   * Notes in the fast layer: {n_notes} | Step: {step_b} beats | Layers: {layers} | Scale: {scale}"))
            out.append(L("   * Serie infinita di Nørgård: a(2n) = -a(n), a(2n+1) = a(n)+1 (autosimile)", "   * Nørgård infinity series: a(2n) = -a(n), a(2n+1) = a(n)+1 (self-similar)"))
        elif key == "MIDI Euclidean Rhythm":
            steps, pulses, rot, step_b, layers, cycles, subj, pats = p
            out.append(L(f"   * Passi: {steps} | Attacchi: {pulses} | Rotazione: {rot} | Passo: {step_b} beat | Strati: {layers} | Cicli: {cycles} | Soggetto: {subj} note",
                         f"   * Steps: {steps} | Pulses: {pulses} | Rotation: {rot} | Step: {step_b} beats | Layers: {layers} | Cycles: {cycles} | Subject: {subj} notes"))
            out.append(L(f"   * Pattern: {pats}", f"   * Patterns: {pats}"))
            out.append(L("   * Ritmi euclidei (G. Toussaint, 2005): algoritmo di Bjorklund, attacchi distribuiti nel modo piu' uniforme possibile",
                         "   * Euclidean rhythms (G. Toussaint, 2005): Bjorklund algorithm, onsets spread as evenly as possible"))
        elif key == "MIDI Neo Riemannian":
            pattern, steps, chord_b, start, tie, period, vl_total, chords = p
            out.append(L(f"   * Schema: {pattern} | Accordi: {steps} | Durata: {chord_b} beat | Partenza: {start} | Note comuni legate: {yes if tie else no}",
                         f"   * Pattern: {pattern} | Chords: {steps} | Length: {chord_b} beats | Start: {start} | Common tones tied: {yes if tie else no}"))
            out.append(L(f"   * Sequenza: {chords}", f"   * Sequence: {chords}"))
            period_it = f"si richiude dopo {period} accordi" if period else "non si richiude entro la sequenza"
            period_en = f"closes after {period} chords" if period else "does not close within the sequence"
            out.append(L(f"   * Ciclo: {period_it} | Voice leading totale: {vl_total} semitoni",
                         f"   * Cycle: {period_en} | Total voice leading: {vl_total} semitones"))
            out.append(L("   * P, L, R (D. Lewin, 1987) con voice leading di distanza minima (D. Tymoczko, Science, 2006)",
                         "   * P, L, R (D. Lewin, 1987) with minimal-distance voice leading (D. Tymoczko, Science, 2006)"))
        elif key == "MIDI Spiral Array":
            win_b, bass, key_name, n_win, mean_t, prog = p
            out.append(L(f"   * Finestra: {win_b} beat ({n_win} finestre) | Basso: {yes if bass else no} | Tonalita': {key_name}",
                         f"   * Window: {win_b} beats ({n_win} windows) | Bass: {yes if bass else no} | Key: {key_name}"))
            out.append(L(f"   * Progressione: {prog}", f"   * Progression: {prog}"))
            out.append(L(f"   * Tensione media (distanza CE-tonalita'): {mean_t:.3f} | velocity proporzionale alla tensione",
                         f"   * Mean tension (CE-to-key distance): {mean_t:.3f} | velocity proportional to tension"))
            out.append(L("   * Spiral Array (E. Chew, 2000): altezze su un'elica lungo la linea delle quinte, centro d'effetto pesato sulle durate",
                         "   * Spiral Array (E. Chew, 2000): pitches on a helix along the line of fifths, duration-weighted center of effect"))
        elif key == "MIDI Maximally Even":
            c, mode, tonic, sub, chord_b, subj, names, ivs, euclid, moved, n_notes = p
            out.append(L(f"   * Cardinalita': {c} su 12 | Modo: {mode} | Tonica: {tonic} | Sotto-insieme: {('ME(%d,%d)' % (sub, c)) if sub else 'nessuno'} | Accordi: {chord_b} beat | Soggetto: {subj} note",
                         f"   * Cardinality: {c} of 12 | Mode: {mode} | Tonic: {tonic} | Subset: {('ME(%d,%d)' % (sub, c)) if sub else 'none'} | Chords: {chord_b} beats | Subject: {subj} notes"))
            out.append(L(f"   * Insieme: {names} | Intervalli: {ivs} | Note spostate: {moved}/{n_notes}",
                         f"   * Set: {names} | Intervals: {ivs} | Notes moved: {moved}/{n_notes}"))
            if euclid:
                out.append(L(f"   * Verificato: ME({c},12) e' una rotazione del ritmo euclideo E({c},12)",
                             f"   * Verified: ME({c},12) is a rotation of the Euclidean rhythm E({c},12)"))
            out.append(L("   * Insiemi massimamente uniformi (J. Clough, J. Douthett, 1991): ME(c,d) = { floor(k*d/c) }, k = 0..c-1",
                         "   * Maximally even sets (J. Clough, J. Douthett, 1991): ME(c,d) = { floor(k*d/c) }, k = 0..c-1"))
        elif key == "MIDI PC DFT":
            comps, c, shift, mult, names, mags_p, mags_s, moved, n_notes = p
            out.append(L(f"   * Componenti: {comps} | Cardinalita': {c} | Sfasamento: {shift} semitoni | Moltiplicazione: {mult}",
                         f"   * Components: {comps} | Cardinality: {c} | Phase shift: {shift} semitones | Multiplication: {mult}"))
            out.append(L(f"   * Insieme risultante: {names} | Note spostate: {moved}/{n_notes}",
                         f"   * Resulting set: {names} | Notes moved: {moved}/{n_notes}"))
            out.append(L(f"   * |F1..F6| del brano: {mags_p}", f"   * |F1..F6| of the piece: {mags_p}"))
            out.append(L(f"   * |F1..F6| dell'insieme: {mags_s}", f"   * |F1..F6| of the set: {mags_s}"))
            out.append(L("   * DFT dei pc-set (D. Lewin; J. Quinn, 2006; E. Amiot, 2016): F_k = somma di exp(-2*pi*i*k*n/12); k=5 diatonico, k=3 triadico, k=4 ottatonico, k=6 toni interi",
                         "   * Pitch-class set DFT (D. Lewin; J. Quinn, 2006; E. Amiot, 2016): F_k = sum of exp(-2*pi*i*k*n/12); k=5 diatonic, k=3 triadic, k=4 octatonic, k=6 whole-tone"))
    except (IndexError, ValueError, TypeError):
        out.append(L("   * (parametri non disponibili)", "   * (parameters unavailable)"))
    return out


def build_report(original_file, original_midi, output_midi, selected_methods, parameters, midi_methods, stile=None, val=None, daw_info=None, drum_info=None):
    """Report bilingue IT/EN in formato '::' (protocollo Loop507). `parameters` puo' essere un dict
    {metodo: parametri} oppure una lista allineata a selected_methods (serve per le catene con metodi ripetuti)."""
    if val is None:
        val = validate_midi_output(original_midi, output_midi)
    tpb = original_midi.ticks_per_beat
    dur = lambda d: f"{d:.2f}" if d is not None else "n/a"
    gm_label = _generated_program_label()

    def methods_block(lang):
        lines = []
        for i, key in enumerate(selected_methods):
            p = parameters.get(key, []) if isinstance(parameters, dict) else parameters[i]
            label = midi_methods.get(key, key)
            if lang == "en":
                label = _METHOD_LABELS_EN.get(key, label)
            lines.append(f"{i+1}. {label}")
            lines.extend(_describe_method(lang, key, p))
        return "\n".join(lines)

    status_it = "OK" if val['ok'] else "ATTENZIONE: note non bilanciate o tempo/metrica persi"
    status_en = "OK" if val['ok'] else "WARNING: unbalanced notes or tempo/meter lost"
    yn_it = "sì" if val['meta_ok'] else "NO"
    yn_en = "yes" if val['meta_ok'] else "NO"

    r = "[MIDI_DECOMPOSER] // VOL_01 // MIDI // STRUCTURAL_DECOMPOSITION\n"
    r += ":: MOTORE / ENGINE: midi_decomposer [v6.0]\n"
    r += f":: FILE: {original_file}\n"
    if stile:
        r += f":: STILE / STYLE: {stile}\n"
    r += f":: TRACCE / TRACKS: {val['tracks_in']} | DURATA / DURATION: {dur(val['duration_in'])} sec | TICKS/BEAT: {tpb}\n"
    r += f":: SEED: {_current_run_seed()} | BATTERIA / DRUMS: {'preservata / preserved' if _drums_preserved() else 'non preservata / not preserved'}\n"
    if gm_label:
        r += f":: STRUMENTO TRACCE GENERATE / GENERATED-TRACK INSTRUMENT: {gm_label}\n"
    if drum_info is not None:
        r += (f":: VARIAZIONI BATTERIA / DRUM VARIATIONS: intensita' {drum_info['amount']:.0%} | ghost note: {drum_info['ghost']}, "
              f"fill: {drum_info['fills']}, crash: {drum_info['crash']}, colpi tolti: {drum_info['dropped']} "
              "(altezze del kit invariate / kit pitches unchanged)\n")
    if daw_info is not None:
        r += (":: DAW: tempo+metrica in prima traccia, un canale per traccia "
              f"(tracce divise: {daw_info['tracks_split']}, aggiunte: {daw_info['tracks_added']}) / "
              "tempo+meter in first track, one channel per track\n")
    r += "\n====[ IT ]====\n"
    r += "\"Il file e' entrato come partitura. E' uscito come esperimento.\"\n\n"
    r += "> METODI APPLICATI (in ordine):\n" + methods_block("it") + "\n\n"
    r += "> VALIDAZIONE:\n"
    r += f"* Note: {val['notes_in']} -> {val['notes_out']} | Durata: {dur(val['duration_in'])} -> {dur(val['duration_out'])} sec\n"
    r += f"* Note non bilanciate: {val['unbalanced']} | Tempo/metrica preservati: {yn_it}\n"
    r += f"* Esito: {status_it}\n\n"
    r += "> TECHNICAL LOG SHEET:\n"
    r += f"* Tracce originali: {val['tracks_in']} -> Tracce output: {val['tracks_out']}\n"
    r += f"* Metodi applicati: {len(selected_methods)}\n\n"
    r += "> Regia e Algoritmo: Loop507\n"
    r += "\n====[ EN ]====\n"
    r += "\"The file came in as a score. It came out as an experiment.\"\n\n"
    r += "> APPLIED METHODS (in order):\n" + methods_block("en") + "\n\n"
    r += "> VALIDATION:\n"
    r += f"* Notes: {val['notes_in']} -> {val['notes_out']} | Duration: {dur(val['duration_in'])} -> {dur(val['duration_out'])} sec\n"
    r += f"* Unbalanced notes: {val['unbalanced']} | Tempo/meter preserved: {yn_en}\n"
    r += f"* Result: {status_en}\n\n"
    r += "> TECHNICAL LOG SHEET:\n"
    r += f"* Original tracks: {val['tracks_in']} -> Output tracks: {val['tracks_out']}\n"
    r += f"* Methods applied: {len(selected_methods)}\n\n"
    r += "> Direction & Algorithm: Loop507\n\n"
    r += "#loop507 #mididecomposer #generativemusic #midiprocessing\n"
    r += "#structuraldecomposition #algorithmicmusic #experimentalmusic"
    return r


# =====================================================================
# [2] INTERFACCIA STREAMLIT
# =====================================================================

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

# Collega il motore a questa sessione (impostazioni da session_state, avvisi su st.warning)
configure(settings=st.session_state, warn=st.warning)

# --- Session State ---
if 'midi_ready'   not in st.session_state: st.session_state.midi_ready   = False
if 'midi_bytes'   not in st.session_state: st.session_state.midi_bytes   = None
if 'midi_report'  not in st.session_state: st.session_state.midi_report  = ""
if 'midi_filename' not in st.session_state: st.session_state.midi_filename = ""


# --- Esito: salvataggio persistente in session_state ---
def finalize_result(original_file, original_midi, output_midi, selected_methods, parameters, midi_methods, suffix, stile=None, chain=False):
    """Salva il risultato in session_state (bytes, nome file, report) per il download persistente.
    Unico punto di uscita per tutti i pulsanti dell'app. Con chain=True il passo viene accodato alla
    cronologia dei passi precedenti (stesso file di partenza): il report elenca l'intera catena."""
    methods = list(selected_methods)
    plist = [parameters.get(k, ()) for k in methods] if isinstance(parameters, dict) else list(parameters)
    history = []
    if chain and st.session_state.get('chain_source') == original_file:
        history = list(st.session_state.get('chain_history', []))
    for k, p in zip(methods, plist):
        history.append({'method': k, 'params': p, 'suffix': suffix, 'stile': stile})
    suffixes = list(dict.fromkeys(h['suffix'] for h in history))
    stiles = list(dict.fromkeys(h['stile'] for h in history if h['stile']))

    drum_info = None
    if st.session_state.get('drum_var', False):
        _amount = st.session_state.get('drum_var_amount', 40) / 100.0
        output_midi, drum_info = drum_variations(output_midi, _amount)
        drum_info['amount'] = _amount
    daw_info = None
    if st.session_state.get('daw_optimize', True):
        output_midi, daw_info = daw_optimize(output_midi)

    buf = io.BytesIO()
    output_midi.save(file=buf)
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    st.session_state.midi_bytes    = buf.getvalue()
    st.session_state.midi_filename = f"{base_name}_{'_'.join(suffixes)}.mid"
    val = validate_midi_output(original_midi, output_midi)
    st.session_state.midi_report   = build_report(
        original_file, original_midi, output_midi,
        [h['method'] for h in history], [h['params'] for h in history], midi_methods,
        stile=" → ".join(stiles) if stiles else None, val=val, daw_info=daw_info, drum_info=drum_info
    )
    st.session_state.midi_ready    = True
    st.session_state.chain_history = history
    st.session_state.chain_source  = original_file
    if not val['ok']:
        st.warning("⚠️ Il risultato contiene note non bilanciate o ha perso tempo/metrica: vedi la sezione VALIDAZIONE del report.")


# --- Player MIDI in-browser (web component html-midi-player, no dipendenze server) ---
def _player_html(midi_bytes, label, key_suffix=""):
    """HTML del player. Se la libreria (CDN) non carica entro 5 secondi mostra un avviso invece di un riquadro vuoto."""
    b64_midi = base64.b64encode(midi_bytes).decode("utf-8")
    data_uri = f"data:audio/midi;base64,{b64_midi}"
    return f"""
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
    <div style="background:#111;border-radius:8px;padding:12px;font-family:sans-serif;">
        <p style="color:#ddd;margin:0 0 8px 0;font-size:14px;">🎧 {label}</p>
        <div id="midi-fallback-{key_suffix}" style="display:none;color:#f5b041;font-size:13px;margin-bottom:8px;">
            ⚠️ Anteprima non disponibile (libreria del player non raggiungibile: rete bloccata o CDN offline).
            Il file e' comunque corretto: scaricalo e aprilo in una DAW.
        </div>
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
    <script>
        setTimeout(function () {{
            if (!window.customElements || !customElements.get('midi-player')) {{
                var f = document.getElementById('midi-fallback-{key_suffix}');
                if (f) f.style.display = 'block';
            }}
        }}, 5000);
    </script>
    """


def render_midi_player(midi_bytes, label, key_suffix=""):
    """
    Incorpora un lettore/visualizzatore MIDI direttamente nel browser dell'utente,
    usando la libreria 'html-midi-player' (Tone.js + soundfont via CDN).
    Nessuna sintesi lato server: funziona anche su Streamlit Cloud.
    """
    html = _player_html(midi_bytes, label, key_suffix)
    if hasattr(st, "iframe"):  # Streamlit recenti: components.v1.html e' deprecato in favore di st.iframe
        st.iframe(html, height=260)
    else:
        components.html(html, height=260, scrolling=False)


@st.cache_resource(show_spinner=False, max_entries=4)
def _load_midi(raw):
    """Lettura + normalizzazione + anteprima + durata del MIDI caricato, calcolate UNA volta per file
    (Streamlit riesegue lo script a ogni click: senza cache si rileggeva tutto ogni volta).
    Il motore non modifica mai il MIDI in ingresso, quindi l'oggetto puo' essere condiviso."""
    midi = mido.MidiFile(file=io.BytesIO(raw))
    midi, was_split = normalize_midi(midi)
    buf = io.BytesIO()
    midi.save(file=buf)
    return midi, was_split, buf.getvalue(), midi_length_seconds(midi)


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
        midi_data, _was_split, _orig_midi_bytes, _orig_seconds = _load_midi(uploaded_midi_file.getvalue())
        if _was_split:
            st.info(f"ℹ️ File MIDI di tipo 0 (una sola traccia con più canali): diviso automaticamente in "
                    f"{len(midi_data.tracks)} tracce (una per canale), così tutti i metodi lavorano per strumento.")
        st.subheader("File MIDI Caricato: Panoramica")
        st.write(f"Nome file: **{uploaded_midi_file.name}**")
        st.write(f"Numero di tracce: **{len(midi_data.tracks)}**")
        st.write(f"Durata (stimata): **{_orig_seconds:.2f} secondi**" if _orig_seconds is not None
                 else "Durata (stimata): **n/d** (file MIDI di tipo 2)")

        with st.expander("🎧 Ascolta il MIDI originale"):
            render_midi_player(_orig_midi_bytes, "MIDI originale", key_suffix="original")

        st.markdown("---")
        st.subheader("⚙️ Modalita' di Decomposizione")
        _gs_col1, _gs_col2 = st.columns([1, 2])
        with _gs_col1:
            _global_seed_input = st.text_input(
                "🎲 Seed globale (opzionale)", value="", key="global_seed",
                help="Stesso file + stessi parametri + stesso seed = stesso risultato. "
                     "Se vuoto ne viene scelto uno casuale, riportato nel report (riga ':: SEED'). "
                     "Il seed specifico di un compositore, se compilato, ha la precedenza.",
            )
        _gs_value = _global_seed_input.strip()
        st.session_state["_run_seed"] = int(_gs_value) if _gs_value.isdigit() else random.SystemRandom().randrange(2 ** 31)
        with _gs_col2:
            st.caption(f"Seed di questa esecuzione: **{st.session_state['_run_seed']}**")
        st.checkbox(
            "🥁 Preserva la batteria (canale MIDI 10)",
            value=True,
            key="preserve_drums",
            help="Se attivo, le note del canale 10 (batteria General MIDI) non vengono trasposte, "
                 "rimappate o ricomposte dai metodi che alterano l'altezza o la struttura. "
                 "Eccezioni: il Time Scrambler stira/quantizza anche la batteria, per restare in sincrono con le altre "
                 "tracce; Riorganizzazione Frasi riordina nel tempo anche i colpi di batteria (stesse altezze). "
                 "I compositori generativi (Eno, Bach, Glass, Reich, Xenakis, Costas...) non creano parti di batteria: "
                 "la lasciano com'e' (usa 'Varia il groove della batteria' per variarla).",
        )

        _drum_var_on = st.checkbox(
            "🥁 Varia il groove della batteria",
            value=False,
            key="drum_var",
            help="Opzionale, per qualsiasi metodo: umanizza dinamica e tempi, toglie qualche colpo, aggiunge ghost note "
                 "del rullante e fill (rullo di tom + crash) a fine frase. Usa solo le altezze gia' presenti nella "
                 "batteria del brano: il kit non cambia. Se non c'e' batteria non fa nulla.",
        )
        st.slider(
            "Intensita' variazioni batteria (%)", min_value=10, max_value=100, value=40, step=5,
            key="drum_var_amount", disabled=not _drum_var_on,
        )

        st.checkbox(
            "🎚️ Ottimizza l'output per la DAW",
            value=True,
            key="daw_optimize",
            help="Tempo e metrica nella prima traccia (come prevede lo standard MIDI) ed espliciti all'inizio; "
                 "una traccia per canale. Ableton Live ignora i canali MIDI e unirebbe in un'unica clip le parti "
                 "di una traccia multi-canale; FL Studio importa un canale alla volta. Note, tempi e durata "
                 "restano identici.",
        )

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
            "MIDI Xenakis Stochastic": "☁️ Iannis Xenakis — Musica Stocastica (Nuvole)",
            "MIDI Cage Chance": "☯️ John Cage — Operazioni di Caso (I Ching)",
            "MIDI Eno Generative": "🌫️ Brian Eno — Musica Generativa (Cicli Asincroni)",
            "MIDI Bach Canon": "🎻 Johann Sebastian Bach — Canone Rigoroso",
            "MIDI Glass Additive": "➕ Philip Glass — Processo Additivo",
            "MIDI Messiaen Modes": "🕊️ Olivier Messiaen — Modi a Trasposizione Limitata",
            "MIDI Part Tintinnabuli": "🔔 Arvo Pärt — Tintinnabuli",
            "MIDI Reich Phasing": "🌀 Steve Reich — Phasing",
            # --- Matematica Sperimentale (vedi modalita' dedicata "🧪 Matematica Sperimentale") ---
            "MIDI Canon per Tonos": "🔺 Bach/Hofstadter — Canon per Tonos (canone eterno ascendente)",
            "MIDI Shepard Penrose": "🪜 Escher/Shepard — Scala Impossibile (Shepard-Penrose)",
            "MIDI Escher Metamorphosis": "🦎 Escher — Metamorfosi (da una traccia all'altra)",
            "MIDI MIU System": "🧩 Hofstadter — Sistema Formale MIU",
            "MIDI Escher Symmetry": "🔳 Escher — Simmetrie del Fregio",
            "MIDI Hamiltonian Cycle": "🕸️ Teoria dei Grafi — Ciclo Hamiltoniano",
            "MIDI Norgard Infinity": "♾️ Per Nørgård — Serie Infinita (frattale musicale)",
            "MIDI Euclidean Rhythm": "🥁 Toussaint — Ritmi Euclidei",
            "MIDI Neo Riemannian": "🔁 Lewin/Tymoczko — Trasformazioni Neo-Riemanniane PLR",
            "MIDI Spiral Array": "🌀 Chew — Spiral Array (armonia e tensione)",
            "MIDI Maximally Even": "⭕ Clough/Douthett — Insiemi Massimamente Uniformi",
            "MIDI PC DFT": "📈 Lewin/Quinn/Amiot — Trasformata di Fourier dei Pc-set",
        }
        MATH_METHODS = {
            "🔺 Canon per Tonos — canone eterno ascendente": "MIDI Canon per Tonos",
            "🪜 Scala impossibile — Shepard-Penrose": "MIDI Shepard Penrose",
            "🦎 Metamorfosi — da una traccia all'altra": "MIDI Escher Metamorphosis",
            "🔳 Simmetrie del fregio (Escher)": "MIDI Escher Symmetry",
            "🕸️ Ciclo hamiltoniano": "MIDI Hamiltonian Cycle",
            "♾️ Serie infinita di Nørgård": "MIDI Norgard Infinity",
            "🧩 Sistema formale MIU (Hofstadter)": "MIDI MIU System",
            "🥁 Ritmi euclidei (Toussaint)": "MIDI Euclidean Rhythm",
            "🔁 Trasformazioni neo-riemanniane PLR (Lewin, Tymoczko)": "MIDI Neo Riemannian",
            "🌀 Spiral Array (Chew) — armonia e tensione": "MIDI Spiral Array",
            "⭕ Insiemi massimamente uniformi (Clough-Douthett)": "MIDI Maximally Even",
            "📈 DFT dei pc-set (Fourier sulle altezze)": "MIDI PC DFT",
            "🧮 Scott Rickard — Costas Sequencer": "MIDI Costas Sequencer",
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
            "☁️ Iannis Xenakis — Musica Stocastica": "MIDI Xenakis Stochastic",
            "☯️ John Cage — Operazioni di Caso (I Ching)": "MIDI Cage Chance",
            "🌫️ Brian Eno — Musica Generativa": "MIDI Eno Generative",
            "🎻 Johann Sebastian Bach — Canone Rigoroso": "MIDI Bach Canon",
            "➕ Philip Glass — Processo Additivo": "MIDI Glass Additive",
            "🕊️ Olivier Messiaen — Modi a Trasposizione Limitata": "MIDI Messiaen Modes",
            "🔔 Arvo Pärt — Tintinnabuli": "MIDI Part Tintinnabuli",
            "🌀 Steve Reich — Phasing": "MIDI Reich Phasing",
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
        def _chain_controls(upload_name, base_midi):
            """Controlli comuni a Compositori e Matematica Sperimentale: concatenazione sul risultato
            precedente e strumento delle tracce generate. Ritorna (midi sorgente, concatenazione attiva)."""
            chain_available = bool(
                st.session_state.midi_ready and st.session_state.midi_bytes
                and st.session_state.get('chain_source') == upload_name
            )
            c1, c2 = st.columns(2)
            with c1:
                chain_checked = st.checkbox(
                    "🔗 Concatena sul risultato precedente", value=False, key="chain_composers",
                    help="Applica la tecnica scelta al risultato dell'ultima elaborazione invece che al file caricato "
                         "(es. Messiaen → Boulez → Reich). Il report elenca tutta la catena. "
                         "Ha effetto dopo aver generato un primo risultato.",
                )
            with c2:
                st.selectbox(
                    "🎹 Strumento delle tracce generate", GM_PROGRAM_OPTIONS, key="gen_program_label",
                    help="Programma General MIDI per le tracce nuove create dai compositori e dai metodi di "
                         "Matematica Sperimentale. 'Predefinito' mantiene la scelta originale di ogni metodo.",
                )
            active = bool(chain_checked and chain_available)
            if chain_checked and not chain_available:
                st.caption("🔗 Nessun risultato precedente per questo file: il primo passo parte dal file caricato.")
            source = base_midi
            if active:
                source = mido.MidiFile(file=io.BytesIO(st.session_state.midi_bytes))
                steps_txt = " → ".join(dict.fromkeys(h["suffix"] for h in st.session_state.get("chain_history", [])))
                st.caption(f"🔗 Input: risultato precedente ({steps_txt}). Il nuovo passo verrà aggiunto alla catena.")
            return source, active

        modalita = st.radio("Modalita':", ["🎨 Stile", "🎼 Compositori", "🧪 Matematica Sperimentale", "🔧 Avanzato"], horizontal=True)

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
                    finalize_result(
                        uploaded_midi_file.name, midi_data, recomposed,
                        ["MIDI Recomposer"], {"MIDI Recomposer": (style_key,)},
                        midi_methods, stile=style_label,
                        suffix="Recomposed"
                    )
                    st.success(
                        f"✅ Ricomposizione completata! "
                        f"{len(midi_data.tracks)} tracce originali → "
                        f"{len(recomposed.tracks)} tracce ricomposte."
                    )
                    _dur_out = midi_length_seconds(recomposed)
                    _dur_music = midi_length_seconds(midi_data, end_tick=midi_last_note_tick(midi_data))
                    if _orig_seconds is not None and _dur_out is not None:
                        st.caption(f"⏱️ Durata: originale {_orig_seconds:.0f} s → ricomposto {_dur_out:.0f} s")
                        if _dur_music and abs(_dur_music / _orig_seconds - 1) > 0.05:
                            st.info(
                                f"Il file originale dura {_orig_seconds:.0f} s ma l'ultima nota termina a "
                                f"{_dur_music:.0f} s (il resto e' silenzio / eventi finali): "
                                f"la ricomposizione segue la musica."
                            )

        elif modalita == "🎼 Compositori":
            st.markdown("#### 🎼 Tecniche Compositive Algoritmiche")
            st.markdown(
                "Ricomposizioni che replicano le tecniche di compositori/matematici storici "
                "che hanno usato schemi combinatori — nessuna rete neurale, solo algoritmo puro."
            )
            compositore_label = st.selectbox("Scegli compositore/tecnica:", list(COMPOSITORI.keys()), key="compositore_select")
            compositore_key = COMPOSITORI[compositore_label]

            source_midi, chain_active = _chain_controls(uploaded_midi_file.name, midi_data)

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
                            source_midi, serialize_duration, serialize_dynamics, serialize_timbre, isolamento_punti
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Stockhausen Punktuelle"],
                            {"MIDI Stockhausen Punktuelle": (serialize_duration, serialize_dynamics, serialize_timbre, isolamento_punti, row_used)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Stockhausen"
                        )
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

                _preview_a, _preview_b = derive_boulez_sets(source_midi, set_size)
                _preview_pivot = _preview_b[0] if _preview_b else 0
                _preview_mult = boulez_multiply_sets(_preview_a, _preview_b, _preview_pivot)
                st.caption(f"Anteprima — Insieme A: {_preview_a} | Insieme B: {_preview_b} | Aggregato risultante: {_preview_mult} ({len(_preview_mult)} classi)")

                if st.button("🔷 Applica Moltiplicazione d'Accordi", type="primary", use_container_width=True, key="btn_boulez"):
                    with st.spinner("Moltiplicando gli insiemi di classi di altezza..."):
                        result_midi, sets_info = midi_boulez_multiplication(source_midi, set_size, chord_density, register_spread)
                        set_a, set_b, multiplied = sets_info
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Boulez Multiplication"],
                            {"MIDI Boulez Multiplication": (set_size, chord_density, register_spread, set_a, set_b)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Boulez"
                        )
                        st.success(f"✅ Moltiplicazione applicata! Aggregato: {multiplied} ({len(multiplied)} classi di altezza)")

            elif compositore_key == "MIDI Xenakis Stochastic":
                st.info(
                    "**Musica stocastica** (*Pithoprakta*, 1955-56 / *Achorripsis*, 1956-57) — i parametri "
                    "sonori sono governati da distribuzioni di probabilità anziché da regole fisse: i tempi "
                    "di attacco seguono un **processo di Poisson** (come la densità delle 'nuvole di suoni' "
                    "di Xenakis), l'altezza segue una **Gaussiana** quantizzata su un **crivello** (sieve, "
                    "teoria dei cribles) definito da classi di resto modulari."
                )
                sieve_input = st.text_input(
                    "Crivello (formula m:r, unione, es. '3:0, 4:1'):",
                    value="3:0, 4:1",
                    key="xenakis_sieve_input",
                    help="Ogni coppia m:r seleziona tutti i numeri congrui a r modulo m. Le coppie si uniscono."
                )
                sieve_pairs = parse_sieve_string(sieve_input)
                _sieve_preview = generate_sieve(sieve_pairs, universe=(0, 24)) if sieve_pairs else []
                st.caption(f"Anteprima crivello (0-24): {_sieve_preview if _sieve_preview else 'nessuna coppia valida — verrà usato il range cromatico completo'}")

                col_x1, col_x2 = st.columns(2)
                with col_x1:
                    mean_events_per_beat = st.slider("Densità eventi (per beat, Poisson):", 0.25, 8.0, 2.0, 0.25, key="xenakis_density")
                    pitch_center = st.slider("Centro altezza (MIDI):", 24, 96, 60, key="xenakis_pitch_center")
                    pitch_spread = st.slider("Dispersione altezza (σ, semitoni):", 1, 36, 12, key="xenakis_pitch_spread")
                with col_x2:
                    duration_mean = st.slider("Durata media evento (beat, esponenziale):", 0.1, 4.0, 0.5, 0.1, key="xenakis_dur_mean")
                    velocity_mean = st.slider("Dinamica media (velocity):", 20, 120, 75, key="xenakis_vel_mean")
                    velocity_spread = st.slider("Dispersione dinamica (σ):", 1, 40, 15, key="xenakis_vel_spread")
                xenakis_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="xenakis_seed")
                xenakis_seed = int(xenakis_seed_input) if xenakis_seed_input.strip().isdigit() else None

                if st.button("☁️ Applica Musica Stocastica", type="primary", use_container_width=True, key="btn_xenakis"):
                    with st.spinner("Generando la nuvola stocastica (Poisson + Gauss + crivello)..."):
                        result_midi, sieve_used = midi_xenakis_stochastic(
                            source_midi, sieve_pairs, mean_events_per_beat, pitch_center, pitch_spread,
                            duration_mean, velocity_mean, velocity_spread, seed=xenakis_seed
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Xenakis Stochastic"],
                            {"MIDI Xenakis Stochastic": (sieve_input, mean_events_per_beat, pitch_center, pitch_spread, duration_mean, velocity_mean, velocity_spread)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Xenakis"
                        )
                        st.success(f"✅ Nuvola stocastica generata! Crivello effettivo: {len(sieve_used)} classi disponibili su 128")

            elif compositore_key == "MIDI Cage Chance":
                st.info(
                    "**Operazioni di caso** (*Music of Changes*, 1951) — ogni nota diventa un evento le cui "
                    "proprietà (altezza, durata, dinamica, presenza/assenza di suono) sono determinate da "
                    "**esagrammi indipendenti**, generati con il metodo classico delle tre monete dell'I "
                    "Ching (64 esiti equiprobabili). Il **silenzio** è materiale musicale legittimo quanto "
                    "il suono — stesso principio alla base di *4'33\"*."
                )
                col_cg1, col_cg2 = st.columns(2)
                with col_cg1:
                    silence_probability = st.slider("Probabilità di silenzio per evento:", 0.0, 0.8, 0.15, 0.05, key="cage_silence_prob")
                with col_cg2:
                    duration_variety = st.checkbox("Varietà indipendente della durata", value=True, key="cage_dur_variety")
                cage_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="cage_seed")
                cage_seed = int(cage_seed_input) if cage_seed_input.strip().isdigit() else None

                if st.button("☯️ Applica Operazioni di Caso", type="primary", use_container_width=True, key="btn_cage"):
                    with st.spinner("Lanciando le monete dell'I Ching (64 esagrammi per parametro)..."):
                        result_midi, hexagram_log = midi_cage_chance_operations(
                            source_midi, silence_probability, duration_variety, seed=cage_seed
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Cage Chance"],
                            {"MIDI Cage Chance": (silence_probability, duration_variety)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Cage"
                        )
                        n_eventi_originali = len(hexagram_log)
                        n_silenzi = sum(1 for h in hexagram_log if (h[3] / 64.0) < silence_probability)
                        st.success(f"✅ Operazioni di caso applicate! {n_eventi_originali - n_silenzi} suoni, {n_silenzi} silenzi su {n_eventi_originali} esagrammi lanciati")

            elif compositore_key == "MIDI Eno Generative":
                st.info(
                    "**Musica generativa a cicli asincroni** (*Music for Airports*, 1978 / *Discreet Music*, "
                    "1975) — ogni loop ripete una singola nota (derivata dal brano) al proprio periodo "
                    "indipendente. Le lunghezze dei loop sono scelte 'incommensurabili' tra loro (via numeri "
                    "primi distinti), cosi' che l'intero sistema impieghi un tempo lunghissimo prima di "
                    "ripetersi esattamente, pur restando gli elementi di base sempre gli stessi. "
                    "*\"Non compongo la musica, compongo i sistemi che la generano\"* (Eno)."
                )
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    num_loops = st.slider("Numero di loop:", 2, 12, 6, key="eno_num_loops")
                    min_loop_beats = st.slider("Lunghezza minima loop (beat):", 2, 32, 8, key="eno_min_loop")
                    max_loop_beats = st.slider("Lunghezza massima loop (beat):", 8, 128, 32, key="eno_max_loop")
                with col_e2:
                    note_length_ratio = st.slider("Durata nota (frazione del loop):", 0.05, 0.9, 0.35, 0.05, key="eno_note_ratio")
                    duration_multiplier = st.slider("Estensione durata brano (×):", 1, 12, 4, key="eno_dur_mult")
                    velocity_base = st.slider("Velocity base:", 15, 90, 55, key="eno_vel_base")
                eno_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="eno_seed")
                eno_seed = int(eno_seed_input) if eno_seed_input.strip().isdigit() else None

                if st.button("🌫️ Applica Musica Generativa", type="primary", use_container_width=True, key="btn_eno"):
                    with st.spinner("Costruendo i cicli asincroni (lunghezze basate su numeri primi)..."):
                        result_midi, loops_info = midi_eno_generative(
                            source_midi, num_loops, min_loop_beats, max_loop_beats,
                            note_length_ratio, duration_multiplier, velocity_base, seed=eno_seed
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Eno Generative"],
                            {"MIDI Eno Generative": (num_loops, min_loop_beats, max_loop_beats, note_length_ratio, duration_multiplier, velocity_base)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Eno"
                        )
                        loop_desc = ", ".join(f"{p}t" for _, p, _ in loops_info[:6])
                        st.success(f"✅ Sistema generativo creato! {len(loops_info)} loop asincroni, cicli: {loop_desc}{'...' if len(loops_info) > 6 else ''}")

            elif compositore_key == "MIDI Bach Canon":
                st.info(
                    "**Canone rigoroso** (*L'Arte della Fuga*, *Offerta Musicale*) — il soggetto (dux) "
                    "viene estratto dalle prime note del brano; ogni voce successiva (comes) lo "
                    "ripropone a distanza di tempo fissa, trasposta a un dato intervallo, ed "
                    "eventualmente **invertita** (canone al rovescio), **retrogradata** (canone "
                    "cancrizzante) o **aumentata ritmicamente**. Costruzione interamente deterministica."
                )
                col_bc1, col_bc2 = st.columns(2)
                with col_bc1:
                    num_voices = st.slider("Numero di voci (dux + comes):", 2, 4, 2, key="bach_num_voices")
                    interval_semitones = st.slider("Intervallo tra voci (semitoni):", -12, 12, 7, key="bach_interval")
                with col_bc2:
                    delay_beats = st.slider("Ritardo del comes (beat):", 0.5, 8.0, 2.0, 0.5, key="bach_delay")
                    transformation = st.selectbox(
                        "Trasformazione del comes:",
                        ["Nessuna (canone rigoroso)", "Inversione (canone al rovescio)",
                         "Retrogrado (canone cancrizzante)", "Aumentazione ritmica (comes raddoppiato)"],
                        key="bach_transformation"
                    )
                augmentation_factor = 2

                if st.button("🎻 Applica Canone", type="primary", use_container_width=True, key="btn_bach"):
                    with st.spinner("Costruendo il canone (dux/comes)..."):
                        result_midi, voices_info = midi_bach_canon(
                            source_midi, num_voices, interval_semitones, delay_beats, transformation, augmentation_factor
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Bach Canon"],
                            {"MIDI Bach Canon": (num_voices, interval_semitones, delay_beats, transformation)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Bach"
                        )
                        st.success(f"✅ Canone applicato! {len(voices_info)} voci generate.")

            elif compositore_key == "MIDI Glass Additive":
                st.info(
                    "**Processo additivo** (*Two Pages*, *1+1*, *Music in Twelve Parts*) — una cellula "
                    "breve viene estratta dal brano e ripetuta a lunghezza crescente: prima 1 nota, poi "
                    "le prime 2, poi le prime 3... fino alla cellula intera (e opzionalmente indietro). "
                    "Un processo aritmetico esplicito applicato alla lunghezza, non alla sostanza."
                )
                col_gl1, col_gl2 = st.columns(2)
                with col_gl1:
                    cell_length_notes = st.slider("Lunghezza cellula (note):", 3, 16, 8, key="glass_cell_len")
                    repeats_per_stage = st.slider("Ripetizioni per stadio:", 1, 8, 2, key="glass_repeats")
                with col_gl2:
                    direction = st.selectbox(
                        "Direzione del processo:",
                        ["Additivo (solo crescita)", "Additivo-sottrattivo (cresce poi decresce)"],
                        key="glass_direction"
                    )

                if st.button("➕ Applica Processo Additivo", type="primary", use_container_width=True, key="btn_glass"):
                    with st.spinner("Costruendo il processo additivo..."):
                        result_midi, stages = midi_glass_additive(
                            source_midi, cell_length_notes, direction, repeats_per_stage
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Glass Additive"],
                            {"MIDI Glass Additive": (cell_length_notes, direction, repeats_per_stage)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Glass"
                        )
                        st.success(f"✅ Processo additivo generato! {len(stages)} stadi.")

            elif compositore_key == "MIDI Messiaen Modes":
                st.info(
                    "**Modi a trasposizione limitata** (*Technique de mon langage musical*, 1944) — "
                    "ogni altezza del brano viene riquantizzata sulla classe più vicina del modo "
                    "scelto (scale a simmetria interna, con meno di 12 trasposizioni distinte). "
                    "Opzionalmente il ritmo viene sostituito da una sequenza **non retrogradabile** "
                    "(palindroma: identica letta avanti o indietro)."
                )
                col_me1, col_me2 = st.columns(2)
                with col_me1:
                    mode_number = st.selectbox("Modo:", [2, 3, 4, 6, 7], index=0, key="messiaen_mode",
                                                format_func=lambda m: f"Modo {m} ({len(MESSIAEN_MODES[m])} note)")
                    transposition = st.slider("Trasposizione (semitoni):", 0, 11, 0, key="messiaen_transp")
                with col_me2:
                    non_retrogradable_rhythm = st.checkbox("Ritmo non retrogradabile (palindromo)", value=True, key="messiaen_nrr")
                    rhythm_cell_notes = st.slider("Lunghezza cellula ritmica:", 3, 15, 7, key="messiaen_rhythm_len") if non_retrogradable_rhythm else 7
                messiaen_seed_input = st.text_input("Seed (opzionale, per riproducibilità):", value="", key="messiaen_seed")
                messiaen_seed = int(messiaen_seed_input) if messiaen_seed_input.strip().isdigit() else None

                if st.button("🕊️ Applica Modi di Messiaen", type="primary", use_container_width=True, key="btn_messiaen"):
                    with st.spinner("Riquantizzando sul modo scelto..."):
                        result_midi, mode_used = midi_messiaen_modes(
                            source_midi, mode_number, transposition, non_retrogradable_rhythm, rhythm_cell_notes, seed=messiaen_seed
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Messiaen Modes"],
                            {"MIDI Messiaen Modes": (mode_number, transposition, non_retrogradable_rhythm, rhythm_cell_notes)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Messiaen"
                        )
                        st.success(f"✅ Modo {mode_number} applicato! Classi: {mode_used}")

            elif compositore_key == "MIDI Part Tintinnabuli":
                st.info(
                    "**Tintinnabuli** (dal 1976: *Spiegel im Spiegel*, *Für Alina*) — la voce **M** "
                    "mantiene il materiale melodico originale; la voce **T** suona sempre una nota "
                    "della triade di tonica, la più vicina sopra o sotto la nota M corrente, secondo "
                    "una regola fissa e deterministica. Nessuna scelta armonica libera."
                )
                col_pa1, col_pa2 = st.columns(2)
                with col_pa1:
                    tonic_key = st.selectbox("Tonica:", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], key="part_tonic")
                    triad_type = st.selectbox("Triade:", ["Minore", "Maggiore"], key="part_triad")
                with col_pa2:
                    t_voice_position = st.selectbox(
                        "Posizione voce T:",
                        ["T-1 (piu' vicina sotto)", "T+1 (piu' vicina sopra)", "T-2 (seconda piu' vicina sotto)", "T+2 (seconda piu' vicina sopra)"],
                        key="part_t_position"
                    )

                _key_est = estimate_key(source_midi)
                if _key_est:
                    _est_tonic, _est_mode = _NOTE_NAMES[_key_est[0]], _key_est[1]
                    _use_est = st.checkbox(
                        f"🔍 Usa la tonalità stimata dal brano ({_est_tonic} {_est_mode.lower()})", value=False, key="part_auto_key",
                        help="Stima Krumhansl-Schmuckler dalle durate delle note (batteria esclusa). Se attivo sostituisce Tonica e Triade scelte sopra.",
                    )
                    if _use_est:
                        tonic_key, triad_type = _est_tonic, _est_mode
                if st.button("🔔 Applica Tintinnabuli", type="primary", use_container_width=True, key="btn_part"):
                    with st.spinner("Calcolando la voce tintinnabuli..."):
                        result_midi, triad_used = midi_part_tintinnabuli(
                            source_midi, tonic_key, triad_type, t_voice_position
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Part Tintinnabuli"],
                            {"MIDI Part Tintinnabuli": (tonic_key, triad_type, t_voice_position)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Part"
                        )
                        st.success(f"✅ Tintinnabuli applicato! Triade: {triad_used}")

            elif compositore_key == "MIDI Reich Phasing":
                st.info(
                    "**Phasing processuale** (*Piano Phase*, 1967 / *Clapping Music*, 1972) — una "
                    "cellula viene estratta dal brano e suonata in loop su due voci identiche: la voce "
                    "A resta fissa, la voce B si sfasa gradualmente di un incremento fisso ogni N cicli. "
                    "Un processo udibile, passo per passo, non un effetto casuale."
                )
                col_re1, col_re2 = st.columns(2)
                with col_re1:
                    cell_length_notes_r = st.slider("Lunghezza cellula (note):", 3, 16, 8, key="reich_cell_len")
                    num_cycles = st.slider("Numero di cicli:", 4, 96, 32, key="reich_cycles")
                with col_re2:
                    phase_shift_units = st.slider("Ampiezza sfasamento (unità cellula):", 1, 4, 1, key="reich_shift_units")
                    shift_every_n_cycles = st.slider("Sfasa ogni N cicli:", 1, 16, 4, key="reich_shift_every")

                if st.button("🌀 Applica Phasing", type="primary", use_container_width=True, key="btn_reich"):
                    with st.spinner("Costruendo lo sfasamento processuale..."):
                        result_midi, final_phase = midi_reich_phasing(
                            source_midi, cell_length_notes_r, num_cycles, phase_shift_units, shift_every_n_cycles
                        )
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Reich Phasing"],
                            {"MIDI Reich Phasing": (cell_length_notes_r, num_cycles, phase_shift_units, shift_every_n_cycles)},
                            midi_methods, stile=compositore_label,
                            chain=chain_active,
                            suffix="Reich"
                        )
                        st.success(f"✅ Phasing applicato! Sfasamento finale: {final_phase} tick.")

        elif modalita == "🧪 Matematica Sperimentale":
            st.markdown("#### 🧪 Matematica Sperimentale")
            st.markdown(
                "Strutture matematiche trasformate in musica, ispirate a *Gödel, Escher, Bach* (Hofstadter), a "
                "*Blueprints* (du Sautoy) e alla teoria matematica della musica (Toussaint, Lewin, Tymoczko, Chew, Clough-Douthett, Amiot). Le tracce originali restano intatte: il risultato si aggiunge come nuove "
                "tracce. Nessuna rete neurale, solo algoritmo puro."
            )
            math_label = st.selectbox("Scegli la struttura:", list(MATH_METHODS.keys()), key="math_select")
            math_key = MATH_METHODS[math_label]
            source_midi, chain_active = _chain_controls(uploaded_midi_file.name, midi_data)

            if math_key == "MIDI Canon per Tonos":
                st.info(
                    "**Canon per Tonos** (*Offerta Musicale*, Bach, 1747) — il canone ricomincia a ogni giro "
                    "un tono più in alto; dopo sei giri è tornato nella tonalità d'origine, un'ottava sopra: "
                    "sale all'infinito senza mai uscire di tonalità. Il soggetto è preso dalle prime note del brano."
                )
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    tonos_voices = st.slider("Voci:", 1, 4, 2, key="tonos_voices")
                    tonos_step = st.select_slider("Modulazione per giro (semitoni):", options=[1, 2, 3, 4, 6], value=2, key="tonos_step",
                                                  help="Solo divisori di 12, così il giro si chiude. 2 = il canone di Bach (6 giri).")
                    tonos_laps = st.slider("Ottave da salire:", 1, 3, 1, key="tonos_laps")
                with col_t2:
                    tonos_delay = st.slider("Ritardo fra le voci (beat):", 0.0, 8.0, 2.0, 0.5, key="tonos_delay")
                    tonos_interval = st.slider("Intervallo fra le voci (semitoni):", -12, 12, 7, key="tonos_interval")
                    tonos_subject = st.slider("Lunghezza soggetto (note):", 4, 24, 12, key="tonos_subject")
                tonos_close = st.checkbox("Giro finale d'arrivo nella tonalità d'origine", value=True, key="tonos_close")
                if st.button("🔺 Applica Canon per Tonos", type="primary", use_container_width=True, key="btn_tonos"):
                    with st.spinner("Costruendo il canone ascendente..."):
                        result_midi, info = midi_canon_per_tonos(
                            source_midi, tonos_voices, tonos_step, tonos_laps, tonos_delay, tonos_interval, tonos_subject, tonos_close)
                        if info.get('ok'):
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Canon per Tonos"],
                                {"MIDI Canon per Tonos": (tonos_voices, info['step'], tonos_laps, tonos_delay, tonos_interval, tonos_subject, tonos_close)},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Tonos")
                            st.success(f"✅ Canon per Tonos creato! {info['cycles']} giri, salita totale {info['total_shift']} semitoni, {info['voices']} voci.")

            elif math_key == "MIDI Shepard Penrose":
                st.info(
                    "**Scala di Shepard-Penrose** (Shepard 1964; *Salendo e scendendo*, Escher 1960) — più strati di "
                    "ottave con un'intensità a campana sul registro: la scala sale (o scende) per sempre senza arrivare "
                    "mai in cima. Funziona meglio con suoni sostenuti (pad, archi, organo). La scala può essere quella del brano."
                )
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    shep_dir = st.selectbox("Direzione:", ["Ascendente", "Discendente"], key="shep_dir")
                    shep_steps = st.slider("Passi:", 12, 240, 48, key="shep_steps")
                    shep_scale = st.selectbox("Scala:", ["Dal brano", "Cromatica"], key="shep_scale")
                with col_s2:
                    shep_beats = st.slider("Durata del passo (beat):", 0.125, 2.0, 0.5, 0.125, key="shep_beats")
                    shep_layers = st.slider("Strati di ottave:", 3, 8, 6, key="shep_layers")
                    shep_vel = st.slider("Velocity massima:", 30, 127, 100, key="shep_vel")
                if st.button("🪜 Applica Shepard-Penrose", type="primary", use_container_width=True, key="btn_shepard"):
                    with st.spinner("Costruendo la scala impossibile..."):
                        result_midi, info = midi_shepard_penrose(source_midi, shep_dir, shep_steps, shep_beats, shep_layers, shep_scale, shep_vel)
                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi, ["MIDI Shepard Penrose"],
                            {"MIDI Shepard Penrose": (shep_dir, shep_steps, shep_beats, info['layers'], info['scale'])},
                            midi_methods, stile=math_label, chain=chain_active, suffix="Shepard")
                        st.success(f"✅ Scala di Shepard-Penrose creata! {info['steps']} passi, {info['layers']} strati, scala: {info['scale']}.")

            elif math_key == "MIDI Escher Metamorphosis":
                st.info(
                    "**Metamorfosi** (Escher, serie *Metamorphosis*) — una traccia si trasforma gradualmente in un'altra: "
                    "altezza, attacco, durata e velocity vengono interpolati nota per nota da A a B, come i pesci che "
                    "diventano uccelli. Servono almeno due tracce con note."
                )
                _src_for_tracks = source_midi
                if source_midi.type == 0 or (len(source_midi.tracks) == 1 and
                        len({m.channel for t in source_midi.tracks for m in t if hasattr(m, 'channel')}) > 1):
                    _src_for_tracks = _split_type0_to_tracks(source_midi)  # tipo 0: i canali diventano tracce
                _mel = math_melodic_tracks(_src_for_tracks)
                if len(_mel) < 2:
                    st.warning("Questo MIDI ha meno di due tracce con note: la Metamorfosi non è applicabile.")
                else:
                    _labels = {i: f"{i}: {nm} ({cnt} note)" for i, nm, cnt in _mel}
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        meta_a = st.selectbox("Traccia di partenza (A):", list(_labels), format_func=lambda x: _labels[x], index=0, key="meta_a")
                    with col_m2:
                        meta_b = st.selectbox("Traccia di arrivo (B):", list(_labels), format_func=lambda x: _labels[x], index=1, key="meta_b")
                    meta_max = st.slider("Note massime generate:", 16, 256, 128, key="meta_max")
                    meta_snap = st.checkbox("Quantizza sui pitch di A e B (resta nel vocabolario del brano)", value=True, key="meta_snap")
                    if meta_a == meta_b:
                        st.caption("A e B coincidono: verranno usate le prime due tracce.")
                    if st.button("🦎 Applica Metamorfosi", type="primary", use_container_width=True, key="btn_meta"):
                        with st.spinner("Trasformando una traccia nell'altra..."):
                            result_midi, info = midi_escher_metamorphosis(_src_for_tracks, meta_a, meta_b, meta_max, meta_snap)
                            if info.get('ok'):
                                finalize_result(
                                    uploaded_midi_file.name, midi_data, result_midi, ["MIDI Escher Metamorphosis"],
                                    {"MIDI Escher Metamorphosis": (info['a_name'], info['b_name'], info['notes'], meta_snap)},
                                    midi_methods, stile=math_label, chain=chain_active, suffix="Metamorfosi")
                                st.success(f"✅ Metamorfosi creata! {info['notes']} note da «{info['a_name']}» a «{info['b_name']}».")

            elif math_key == "MIDI Escher Symmetry":
                st.info(
                    "**Simmetrie del fregio** (Escher; du Sautoy, *Blueprints*) — un motivo del brano viene ripetuto "
                    "applicando le isometrie del piano tempo-altezza: **inversione** (riflessione dell'altezza), "
                    "**retrogrado** (riflessione del tempo) e **rotazione di 180°** (retrogrado-inversione). "
                    "Gli schemi a due voci usano due tracce."
                )
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    sym_pattern = st.selectbox("Schema del fregio:", list(FRIEZE_PATTERNS.keys()), key="sym_pattern")
                with col_e2:
                    sym_motif = st.slider("Lunghezza del motivo (note):", 3, 16, 8, key="sym_motif")
                    sym_cells = st.slider("Celle del fregio:", 2, 48, 16, key="sym_cells")
                if st.button("🔳 Applica Simmetrie", type="primary", use_container_width=True, key="btn_symmetry"):
                    with st.spinner("Costruendo il fregio..."):
                        result_midi, info = midi_escher_symmetry(source_midi, sym_pattern, sym_motif, sym_cells)
                        if info.get('ok'):
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Escher Symmetry"],
                                {"MIDI Escher Symmetry": (sym_pattern, sym_motif, sym_cells)},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Fregio")
                            st.success(f"✅ Fregio creato! {info['cells']} celle, {info['slots']} voce/i.")

            elif math_key == "MIDI Hamiltonian Cycle":
                st.info(
                    "**Ciclo hamiltoniano** (W. R. Hamilton, 1857; du Sautoy, *Blueprints*) — ogni pitch del brano è un "
                    "nodo; due pitch sono collegati se distano al massimo N semitoni. Un ciclo hamiltoniano visita ogni "
                    "nodo una sola volta e richiude il cerchio: una melodia senza ripetizioni che riparte da capo senza salti. "
                    "Durata e velocity di ogni nota sono quelle tipiche del pitch nel brano. Se il ciclo non esiste, "
                    "viene usato il percorso più lungo trovato (alza il salto massimo per ottenerlo)."
                )
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    ham_interval = st.slider("Salto massimo fra note collegate (semitoni):", 1, 12, 7, key="ham_interval")
                    ham_nodes = st.slider("Pitch massimi (nodi):", 4, 24, 16, key="ham_nodes")
                with col_h2:
                    ham_repeats = st.slider("Ripetizioni del ciclo:", 1, 16, 4, key="ham_repeats")
                if st.button("🕸️ Applica Ciclo Hamiltoniano", type="primary", use_container_width=True, key="btn_ham"):
                    with st.spinner("Cercando il ciclo hamiltoniano..."):
                        result_midi, info = midi_hamiltonian_cycle(source_midi, ham_interval, ham_nodes, ham_repeats)
                        if info.get('ok'):
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Hamiltonian Cycle"],
                                {"MIDI Hamiltonian Cycle": (ham_interval, ham_nodes, ham_repeats, info['nodes'], info['status'], info['covered'], info['expansions'])},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Hamilton")
                            if info['status'] == 'full':
                                st.success(f"✅ Ciclo hamiltoniano trovato su {info['nodes']} pitch!")
                            elif info['status'] == 'partial':
                                st.warning(f"Ciclo solo sulla componente principale ({info['covered']}/{info['nodes']} pitch): alza il salto massimo per collegare tutti i pitch.")
                            else:
                                st.warning(f"Nessun ciclo: usato il percorso più lungo ({info['covered']}/{info['nodes']} pitch). Prova ad alzare il salto massimo.")

            elif math_key == "MIDI Norgard Infinity":
                st.info(
                    "**Serie infinita di Nørgård** — a(0)=0, a(2n) = −a(n), a(2n+1) = a(n)+1: 0, 1, −1, 2, 1, 0, −2, 3… "
                    "È autosimile: prendendo un elemento ogni 2, 4, 8… si riottiene la serie (invertita a ogni raddoppio). "
                    "Gli strati lenti sono quindi lo scheletro dello strato veloce. La serie è proiettata sulla scala del brano."
                )
                col_n1, col_n2 = st.columns(2)
                with col_n1:
                    nor_notes = st.slider("Note dello strato veloce:", 16, 512, 64, key="nor_notes")
                    nor_layers = st.slider("Strati (scale temporali):", 1, 4, 3, key="nor_layers")
                with col_n2:
                    nor_beats = st.slider("Durata del passo (beat):", 0.125, 2.0, 0.5, 0.125, key="nor_beats")
                    nor_scale = st.selectbox("Scala:", ["Dal brano", "Cromatica"], key="nor_scale")
                if st.button("♾️ Applica Serie di Nørgård", type="primary", use_container_width=True, key="btn_norgard"):
                    with st.spinner("Generando la serie infinita..."):
                        result_midi, info = midi_norgard_infinity(source_midi, nor_notes, nor_beats, nor_layers, nor_scale)
                        if info.get('ok'):
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Norgard Infinity"],
                                {"MIDI Norgard Infinity": (nor_notes, nor_beats, info['layers'], info['scale'])},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Norgard")
                            st.success(f"✅ Serie di Nørgård creata! {info['layers']} strati, scala: {info['scale']}. Inizio serie: {info['series_head'][:8]}")

            elif math_key == "MIDI Euclidean Rhythm":
                st.info(
                    "**Ritmi euclidei** (G. Toussaint, 2005) — l'algoritmo di Euclide, nella forma di Bjorklund, distribuisce "
                    "*k* attacchi su *n* passi nel modo più uniforme possibile. Molti ritmi tradizionali del mondo sono di questo tipo: "
                    "E(3,8) = x..x..x. · E(5,8) = x.xx.xx. · E(7,12) = x.xx.x.xx.x. Le altezze sono quelle del soggetto del brano, "
                    "ciclate sugli attacchi."
                )
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    euc_steps = st.slider("Passi (n):", 4, 32, 16, key="euc_steps")
                    euc_pulses = st.slider("Attacchi (k):", 1, 32, 5, key="euc_pulses",
                                           help="Se k supera n viene ridotto a n.")
                    euc_rot = st.slider("Rotazione (passi):", 0, 31, 0, key="euc_rot")
                    euc_subject = st.slider("Note del soggetto:", 1, 16, 8, key="euc_subject")
                with col_e2:
                    euc_beats = st.slider("Durata del passo (beat):", 0.125, 1.0, 0.25, 0.125, key="euc_beats")
                    euc_layers = st.slider("Strati:", 1, 3, 1, key="euc_layers",
                                           help="Ogni strato successivo ha 2 attacchi in meno e suona un'ottava sotto: poliritmia euclidea.")
                    euc_cycles = st.slider("Cicli:", 1, 32, 4, key="euc_cycles")
                if st.button("🥁 Applica Ritmi Euclidei", type="primary", use_container_width=True, key="btn_euclid"):
                    with st.spinner("Calcolando i ritmi euclidei..."):
                        result_midi, info = midi_euclidean_rhythm(
                            source_midi, euc_steps, euc_pulses, euc_rot, euc_beats, euc_layers, euc_cycles, euc_subject)
                        if info.get('ok'):
                            pats_txt = " · ".join(f"E({k},{info['steps']}) = {ps}" for k, ps in info['patterns'])
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Euclidean Rhythm"],
                                {"MIDI Euclidean Rhythm": (info['steps'], euc_pulses, info['rotation'], euc_beats, info['layers'], euc_cycles, euc_subject, pats_txt)},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Euclideo")
                            st.success(f"✅ Ritmi euclidei creati! {info['layers']} strati.")
                            st.code("\n".join(f"E({k},{info['steps']})  {ps}" for k, ps in info['patterns']), language=None)

            elif math_key == "MIDI Neo Riemannian":
                st.info(
                    "**Trasformazioni neo-riemanniane** (D. Lewin, *Generalized Musical Intervals and Transformations*, 1987) — "
                    "una triade è un oggetto e **P** (parallela: Do↔Do min), **L** (nota sensibile: Do↔Mi min), **R** (relativa: "
                    "Do↔La min) sono le funzioni che la trasformano; ognuna muove una sola voce. P-L alternate danno il ciclo esatonico "
                    "(6 accordi), P-R l'ottatonico (8), L-R tutte le 24 triadi. Il voicing usa il voice leading di distanza minima "
                    "(D. Tymoczko, *Science*, 2006): l'accordo è un punto, il voice leading una distanza."
                )
                _nr_starts = ["Dal brano"] + [n for n in _NOTE_NAMES] + [n + "m" for n in _NOTE_NAMES]
                col_n1, col_n2 = st.columns(2)
                with col_n1:
                    nr_pattern = st.selectbox("Schema:", list(NR_PATTERNS.keys()), key="nr_pattern")
                    nr_steps = st.slider("Accordi:", 4, 48, 12, key="nr_steps")
                with col_n2:
                    nr_start = st.selectbox("Triade di partenza:", _nr_starts, key="nr_start",
                                            help="'Dal brano' usa la tonalità stimata (Krumhansl-Schmuckler). Il suffisso 'm' indica una triade minore.")
                    nr_beats = st.slider("Durata dell'accordo (beat):", 0.5, 8.0, 2.0, 0.5, key="nr_beats")
                nr_tie = st.checkbox("Lega le note comuni fra accordi vicini", value=True, key="nr_tie",
                                     help="Le note che restano ferme (due su tre) non vengono riattaccate: si sente la voce che si muove.")
                if st.button("🔁 Applica Trasformazioni PLR", type="primary", use_container_width=True, key="btn_nr"):
                    with st.spinner("Percorrendo lo spazio delle triadi..."):
                        result_midi, info = midi_neo_riemannian(source_midi, nr_pattern, nr_steps, nr_beats, nr_start, nr_tie)
                        if info.get('ok'):
                            chords_txt = " – ".join(info['chords'][:24]) + (" …" if len(info['chords']) > 24 else "")
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Neo Riemannian"],
                                {"MIDI Neo Riemannian": (nr_pattern, nr_steps, nr_beats, info['start'], nr_tie, info['period'], info['vl_total'], chords_txt)},
                                midi_methods, stile=math_label, chain=chain_active, suffix="PLR")
                            st.success(f"✅ Trasformazioni PLR create! Partenza: {info['start']}, voice leading totale {info['vl_total']} semitoni.")
                            if info['period']:
                                st.caption(f"Il ciclo si richiude dopo {info['period']} accordi.")
                            st.caption("Sequenza: " + chords_txt)

            elif math_key == "MIDI Spiral Array":
                st.info(
                    "**Spiral Array** (E. Chew, 2000) — le altezze stanno su un'elica lungo la linea delle quinte; un accordo è il "
                    "baricentro delle sue note, una tonalità quello dei suoi accordi I, V, IV. Il **centro d'effetto** di ogni finestra "
                    "del brano (baricentro delle note pesato sulle durate) indica l'accordo più vicino; la distanza del centro "
                    "d'effetto dalla tonalità è la **tensione** e governa la velocity. Il risultato è una riduzione armonica del brano "
                    "(accordi + basso) con voice leading a distanza minima."
                )
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    sp_window = st.select_slider("Finestra d'analisi (beat):", options=[1.0, 2.0, 4.0, 8.0], value=4.0, key="sp_window",
                                                 help="Finestre brevi con poche note danno accordi poco affidabili: 4 beat (una battuta in 4/4) è un buon punto di partenza.")
                with col_p2:
                    sp_bass = st.checkbox("Aggiungi basso sulla fondamentale", value=True, key="sp_bass")
                if st.button("🌀 Applica Spiral Array", type="primary", use_container_width=True, key="btn_spiral"):
                    with st.spinner("Analizzando l'armonia sull'elica..."):
                        result_midi, info = midi_spiral_array(source_midi, sp_window, sp_bass)
                        if info.get('ok'):
                            prog_txt = " – ".join(info['progression'][:24]) + (" …" if len(info['progression']) > 24 else "")
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Spiral Array"],
                                {"MIDI Spiral Array": (sp_window, sp_bass, info['key'], info['windows'], info['mean_tension'], prog_txt)},
                                midi_methods, stile=math_label, chain=chain_active, suffix="Spiral")
                            st.success(f"✅ Spiral Array applicato! Tonalità: {info['key']}, tensione media {info['mean_tension']:.3f}.")
                            st.caption("Progressione: " + prog_txt)

            elif math_key == "MIDI Maximally Even":
                st.info(
                    "**Insiemi massimamente uniformi** (J. Clough e J. Douthett, 1991) — i *c* punti più regolari possibili sul cerchio "
                    "delle 12 altezze: ME(c,12) = { floor(12·k / c) }. c=7 è la scala diatonica, c=5 la pentatonica, c=6 i toni interi, "
                    "c=4 la settima diminuita, c=3 la triade aumentata; le rotazioni sono i modi (con c=7 il modo 1 è l'Ionio). "
                    "È lo stesso oggetto dei ritmi euclidei, sul cerchio delle altezze. Il soggetto del brano viene quantizzato "
                    "sull'insieme e armonizzato con un sotto-insieme annidato: ME(3,7) dentro ME(7,12) = le terze diatoniche."
                )
                _me_tonics = ["Dal brano"] + list(_NOTE_NAMES)
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    me_c = st.slider("Elementi dell'insieme (c):", 2, 11, 7, key="me_c")
                    me_mode = st.slider("Modo (rotazione):", 0, 10, 1, key="me_mode",
                                        help="Viene ridotto modulo c. Con c=7: 0 Locrio, 1 Ionio, 2 Dorico, 3 Frigio, 4 Lidio, 5 Misolidio, 6 Eolio.")
                    me_tonic = st.selectbox("Tonica:", _me_tonics, key="me_tonic",
                                            help="'Dal brano' usa la tonica stimata (Krumhansl-Schmuckler).")
                with col_m2:
                    me_sub = st.slider("Sotto-insieme per gli accordi (0 = nessuno):", 0, 10, 3, key="me_sub",
                                       help="Deve essere minore di c: gli accordi sono ME(s,c) costruito su ogni grado.")
                    me_chord_beats = st.select_slider("Durata degli accordi (beat):", options=[1.0, 2.0, 4.0, 8.0], value=4.0, key="me_chord_beats")
                    me_subject = st.slider("Note del soggetto:", 8, 128, 32, key="me_subject")
                if st.button("⭕ Applica Insiemi Massimamente Uniformi", type="primary", use_container_width=True, key="btn_me"):
                    with st.spinner("Calcolando l'insieme massimamente uniforme..."):
                        result_midi, info = midi_maximally_even(source_midi, me_c, me_mode, me_tonic, me_sub, me_chord_beats, me_subject)
                        if info.get('ok'):
                            names_txt = " ".join(info['names'])
                            ivs_txt = "-".join(str(i) for i in info['intervals'])
                            sub_used = len(info['sub_idx'])
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI Maximally Even"],
                                {"MIDI Maximally Even": (info['cardinality'], info['mode'], info['tonic'], sub_used, me_chord_beats, me_subject,
                                                         names_txt, ivs_txt, info['euclid_equiv'], info['moved'], info['notes'])},
                                midi_methods, stile=math_label, chain=chain_active, suffix="ME")
                            st.success(f"✅ Insieme ME({info['cardinality']},12) creato! {names_txt} — intervalli {ivs_txt}.")
                            st.caption(f"Note del soggetto spostate sull'insieme: {info['moved']}/{info['notes']}.")

            elif math_key == "MIDI PC DFT":
                st.info(
                    "**DFT dei pc-set** (D. Lewin; J. Quinn, 2006; E. Amiot, 2016) — un insieme di altezze è un segnale sui 12 punti del "
                    "cerchio e la sua trasformata di Fourier F_k dice quanto somiglia alla divisione regolare dell'ottava in k parti "
                    "(k=5 diatonico, k=3 triadico, k=4 ottatonico, k=6 toni interi). Il metodo calcola la DFT dell'istogramma del brano, "
                    "tiene solo le componenti scelte, prende le c altezze più forti del profilo ricostruito e quantizza il soggetto su quell'insieme. "
                    "La moltiplicazione M5/M7 scambia le componenti (F'_k = F_5k): il diatonico diventa un cluster."
                )
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    dft_comps = st.multiselect("Componenti di Fourier da tenere:", [1, 2, 3, 4, 5, 6], default=[5], key="dft_comps",
                                               format_func=lambda k: DFT_COMPONENT_LABELS[k],
                                               help="Se non ne scegli nessuna viene usata la 5.")
                    dft_card = st.slider("Altezze dell'insieme (c):", 3, 11, 7, key="dft_card")
                with col_d2:
                    dft_shift = st.slider("Sfasamento (semitoni):", 0, 11, 0, key="dft_shift",
                                          help="Sfasare le componenti di 2*pi*k*t/12 equivale a traspore il profilo di t semitoni.")
                    dft_mult = st.selectbox("Moltiplicazione dell'insieme:", ["Nessuno", "M5", "M7"], key="dft_mult")
                    dft_subject = st.slider("Note del soggetto:", 8, 128, 32, key="dft_subject")
                if st.button("📈 Applica Filtro DFT", type="primary", use_container_width=True, key="btn_dft"):
                    with st.spinner("Filtrando nello spazio di Fourier..."):
                        result_midi, info = midi_pc_dft_filter(source_midi, dft_comps, dft_card, dft_shift, dft_mult, dft_subject)
                        if info.get('ok'):
                            names_txt = " ".join(info['names'])
                            mags_p = " ".join(f"{x:.2f}" for x in info['mags_piece'])
                            mags_s = " ".join(f"{x:.2f}" for x in info['mags_set'])
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI PC DFT"],
                                {"MIDI PC DFT": ("+".join(str(k) for k in info['components']), info['cardinality'], info['shift'], info['multiplier'],
                                                 names_txt, mags_p, mags_s, info['moved'], info['notes'])},
                                midi_methods, stile=math_label, chain=chain_active, suffix="DFT")
                            st.success(f"✅ Filtro DFT applicato! Insieme: {names_txt}.")
                            st.caption(f"|F1..F6| del brano: {mags_p}  ·  dell'insieme: {mags_s}  ·  note spostate: {info['moved']}/{info['notes']}")

            elif math_key == "MIDI Costas Sequencer":
                st.caption(
                    "Basato sulla costruzione di Welch (Scott Rickard / J.P. Costas): "
                    "una permutazione algoritmica in cui nessun vettore-differenza tra "
                    "coppie si ripete. Nessuna rete neurale, nessun random puro — pura DSP/combinatoria."
                )
                costas_mode = st.selectbox(
                    "Modalità Costas:",
                    ["Permutazione Pitch (Cromatica)", "Griglia Ritmica Costas", "Generatore Costas (Nuova Melodia)"],
                    key="costas_mode_math"
                )

                if costas_mode == "Permutazione Pitch (Cromatica)":
                    st.info("Usa una matrice di Costas di ordine 12 (p=13) come cifrario di sostituzione fisso per le 12 classi di altezza. Ritmo invariato.")
                    transpose_octave = st.slider("Trasposizione (ottave):", -2, 2, 0, key="costas_transpose_math")
                    costas_params = (costas_mode, 12, transpose_octave, 0, 1.0)

                elif costas_mode == "Griglia Ritmica Costas":
                    costas_order_req = st.slider("Ordine minimo della matrice (n):", 3, 24, 8, key="costas_order_grid_math")
                    _p_preview = _costas_find_prime(costas_order_req)
                    st.caption(f"Ordine effettivo: n = {_p_preview - 1} (primo p = {_p_preview})")
                    costas_params = (costas_mode, costas_order_req, 0, 0, 1.0)

                else:  # Generatore Costas (Nuova Melodia)
                    costas_order_req = st.slider("Ordine minimo della matrice (n):", 3, 24, 12, key="costas_order_gen_math")
                    _p_preview = _costas_find_prime(costas_order_req)
                    st.caption(f"Ordine effettivo: n = {_p_preview - 1} (primo p = {_p_preview})")
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        base_pitch = st.slider("Pitch base (MIDI):", 24, 96, 60, key="costas_base_pitch_math")
                    with col_c2:
                        pitch_range_semitones = st.slider("Estensione (semitoni):", 6, 48, 24, key="costas_pitch_range_math")
                    step_beats = st.slider("Durata di ogni passo (in battute/beat):", 0.125, 2.0, 0.25, 0.125, key="costas_step_beats_math")
                    costas_params = (costas_mode, costas_order_req, base_pitch, pitch_range_semitones, step_beats)

                if st.button("🧮 Applica Costas Sequencer", type="primary", use_container_width=True, key="btn_costas"):
                    with st.spinner("Generando la matrice di Costas (costruzione di Welch)..."):
                        cmode, corder, cp1, cp2, cp3 = costas_params
                        if cmode == "Permutazione Pitch (Cromatica)":
                            result_midi, costas_info = midi_costas_pitch_permutation(source_midi, transpose_octave=cp1)
                        elif cmode == "Griglia Ritmica Costas":
                            result_midi, costas_info = midi_costas_rhythmic_grid(source_midi, corder)
                        else:
                            result_midi, costas_info = midi_costas_generator(source_midi, corder, cp1, cp2, cp3)

                        finalize_result(
                            uploaded_midi_file.name, midi_data, result_midi,
                            ["MIDI Costas Sequencer"], {"MIDI Costas Sequencer": costas_params},
                            midi_methods, stile=math_label,
                            chain=chain_active,
                            suffix="Costas"
                        )
                        n_costas, p_costas, g_costas = costas_info
                        st.success(f"✅ Costas Sequencer applicato! Ordine effettivo n={n_costas} (p={p_costas}, g={g_costas})")

            else:  # Sistema formale MIU
                st.info(
                    "**Sistema formale MIU** (Hofstadter, *Gödel, Escher, Bach*, cap. 1) — alfabeto {M, I, U}, assioma MI e "
                    "quattro regole di riscrittura: xI→xIU · Mx→Mxx · III→U · UU→(nulla). A ogni passo viene applicata una "
                    "regola applicabile scelta a caso (seed). Ogni stringa diventa una sezione: **I** = cellula del brano, "
                    "**U** = cellula invertita, **M** = pedale. Curiosità: MU non è mai derivabile (il numero di I non è mai un multiplo di 3)."
                )
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    miu_steps = st.slider("Passi di derivazione:", 2, 16, 8, key="miu_steps")
                    miu_maxlen = st.slider("Lunghezza massima della stringa:", 6, 32, 16, key="miu_maxlen")
                with col_u2:
                    miu_cell = st.slider("Note della cellula:", 2, 6, 3, key="miu_cell")
                    miu_beats = st.slider("Durata di ogni simbolo (beat):", 0.5, 4.0, 2.0, 0.5, key="miu_beats")
                if st.button("🧩 Applica Sistema MIU", type="primary", use_container_width=True, key="btn_miu"):
                    with st.spinner("Derivando stringhe MIU..."):
                        result_midi, info = midi_miu_system(source_midi, miu_steps, miu_maxlen, miu_cell, miu_beats)
                        if info.get('ok'):
                            finalize_result(
                                uploaded_midi_file.name, midi_data, result_midi, ["MIDI MIU System"],
                                {"MIDI MIU System": (miu_steps, miu_maxlen, miu_cell, miu_beats, info['final'])},
                                midi_methods, stile=math_label, chain=chain_active, suffix="MIU")
                            st.success(f"✅ Sistema MIU applicato! {len(info['history']) - 1} passi, stringa finale: {info['final']}")
                            if len(info['history']) - 1 < miu_steps:
                                st.info(f"La derivazione si è fermata dopo {len(info['history']) - 1} passi su {miu_steps}: "
                                        "nessuna regola era più applicabile entro la lunghezza massima. Alza il limite per proseguire.")
                            st.caption("Derivazione: " + " → ".join(info['history'][:8]) + (" …" if len(info['history']) > 8 else ""))

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
                        finalize_result(
                            uploaded_midi_file.name, midi_data, decomposed_midi_file,
                            selected_methods_keys, parameters, midi_methods, stile=None,
                            suffix="Decomposed"
                        )
                    else:
                        st.error("Impossibile generare il MIDI decomposto. Controlla i messaggi di avviso.")

    except Exception as e:
        st.error(f"❌ Errore durante la lettura o l'elaborazione del file MIDI: {str(e)}")
        st.error("Assicurati che sia un file MIDI valido (.mid o .midi) e riprova.")
        st.exception(e)
else:
    st.info("👆 Carica un file MIDI (.mid o .midi) per iniziare la decomposizione.")
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
        * **🔁 MIDI Recomposer**: Ricompone ogni traccia in un nuovo brano usando solo le sue note come vocabolario (7 stili).

        **🎼 Compositori:** Stockhausen (Punktuelle), Boulez (moltiplicazione d'accordi), Xenakis (nuvole stocastiche), Cage (I Ching), Eno (cicli asincroni), Bach (canone), Glass (processo additivo), Messiaen (modi a trasposizione limitata), Pärt (tintinnabuli), Reich (phasing).

        **🧪 Matematica Sperimentale:** Canon per Tonos (canone eterno ascendente), scala di Shepard-Penrose, Metamorfosi di Escher fra due tracce, simmetrie del fregio, ciclo hamiltoniano, serie infinita di Nørgård e sistema formale MIU di Hofstadter (ispirati a *Gödel, Escher, Bach* e a *Blueprints*); ritmi euclidei (Toussaint), trasformazioni neo-riemanniane PLR con voice leading minimo (Lewin, Tymoczko) Spiral Array (Chew), insiemi massimamente uniformi (Clough-Douthett) e DFT dei pc-set (Lewin, Quinn, Amiot), dalla *mathematical music theory*; Costas Sequencer di Scott Rickard (permutazioni di Costas, costruzione di Welch).

        **🥁 Batteria:** con *Preserva la batteria* (attivo di default) il canale MIDI 10 resta intatto nei metodi che alterano altezza o struttura.

        **⏱️ Tempo e metrica:** tempo, metrica e tonalità del file originale vengono mantenuti nel file generato.

        **🎲 Seed:** con lo stesso *Seed globale* (o quello scritto nel report, riga `:: SEED`) ottieni lo stesso risultato.

        **🔗 Concatenazione:** nei Compositori puoi applicare una tecnica al risultato della precedente; il report elenca tutta la catena.

        **🎹 Strumento GM:** scegli lo strumento delle tracce generate dai compositori.

        **✅ Validazione e report bilingue:** ogni risultato è controllato (note bilanciate, tempo/metrica) e il report è in italiano e inglese.

        **📂 File tipo 0:** i file con una sola traccia e più canali vengono divisi automaticamente in una traccia per canale.
        """)
# RISULTATI PERSISTENTI
if st.session_state.midi_ready and st.session_state.midi_bytes:
    st.markdown("---")
    st.subheader("🎧 Ascolta il risultato prima di scaricare")
    render_midi_player(st.session_state.midi_bytes, "MIDI decomposto/ricomposto", key_suffix="result")

    st.subheader("Scarica il tuo MIDI Decomposto")
    st.caption(
        "🍎 **Logic Pro:** apri il file con *File ▸ Apri* (non *Importa*). L'importazione carica solo le note e i "
        "controller e **ignora tempo, metrica, nomi delle tracce e marker**; con *Apri* Logic crea una traccia di "
        "strumento software per ogni traccia MIDI, scegliendo lo strumento dal program change."
    )
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

    # --- Download per singola traccia (persistente: vive fuori da qualsiasi st.button) ---
    try:
        _result_midi = mido.MidiFile(file=io.BytesIO(st.session_state.midi_bytes))
    except Exception:
        _result_midi = None
    if _result_midi is not None and len(_result_midi.tracks) > 0:
        st.markdown("---")
        st.subheader("Scarica Singole Tracce del MIDI")
        _track_labels = [get_track_display_name(t, i) for i, t in enumerate(_result_midi.tracks)]
        _sel_tracks = st.multiselect(
            "Seleziona una o più tracce da scaricare singolarmente:",
            options=list(range(len(_result_midi.tracks))),
            format_func=lambda x: _track_labels[x],
            key="single_tracks_sel",
            help="Ogni traccia viene scaricata come file MIDI separato, con tempo e metrica del brano."
        )
        _base = os.path.splitext(st.session_state.midi_filename or "midi")[0]
        for _ti in _sel_tracks:
            if _ti >= len(_result_midi.tracks):
                continue
            _buf = io.BytesIO()
            track_to_midi_file(_result_midi, _ti).save(file=_buf)
            _safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in _track_labels[_ti])
            st.download_button(
                label=f"💾 Scarica {_track_labels[_ti]}",
                data=_buf.getvalue(),
                file_name=f"{_base}_{_safe}.mid",
                mime="audio/midi",
                key=f"download_track_{_ti}"
            )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><em>MIDI Decomposer by loop507</em></p>
    <p>Sperimenta la destrutturazione MIDI</p>
    <p style='font-size: 0.8em;'>Powered by Streamlit & Mido</p>
</div>
""", unsafe_allow_html=True)
