#
# Copyright 2026 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Implementation of the Unicode Bidirectional Algorithm (UAX #9) as of Unicode
17.0.0. This module is a replacement for the pre-6.3 UBA implementation from
python-bidi. This is an unoptimized, literal python port of the spec. Rules P1
and L3 are not implemented.
"""

from bisect import bisect_right
from functools import lru_cache
from typing import Literal, Optional

from kraken.lib.bidi._data import BIDI_CLASS_RANGES, BRACKETS, MIRRORED, UCD_VERSION

__all__ = ['get_display', 'get_display_map', 'bidi_class', 'UCD_VERSION']

# maximum explicit embedding depth (BD2)
MAX_DEPTH = 125
# open bracket stack limit (BD16)
_MAX_PAIRING_DEPTH = 63

_ISOLATE_INITIATORS = frozenset(('LRI', 'RLI', 'FSI'))
_ISOLATES = frozenset(('LRI', 'RLI', 'FSI', 'PDI'))
_EXPLICIT_EMBEDDINGS = frozenset(('RLE', 'LRE', 'RLO', 'LRO'))
# neutral or isolate formatting characters (NI)
_NI = frozenset(('B', 'S', 'WS', 'ON', 'LRI', 'RLI', 'FSI', 'PDI'))

# canonical equivalences for bracket matching (N0)
_CANON_BRACKETS = {0x3008: 0x2329, 0x3009: 0x232A}

_CLASS_STARTS = tuple(r[0] for r in BIDI_CLASS_RANGES)


@lru_cache(maxsize=None)
def bidi_class(codepoint: int) -> str:
    """
    Returns the Bidi_Class of a code point.
    """
    idx = bisect_right(_CLASS_STARTS, codepoint) - 1
    if idx >= 0 and codepoint <= BIDI_CLASS_RANGES[idx][1]:
        return BIDI_CLASS_RANGES[idx][2]
    return 'L'


def _direction(level: int) -> str:
    return 'R' if level % 2 else 'L'


def _first_strong_level(types: list[str], start: int, end: int) -> int:
    """
    Rules P2/P3: embedding level of the first strong character in
    types[start:end], skipping characters between an isolate initiator and
    its matching PDI. Defaults to 0.
    """
    depth = 0
    for typ in types[start:end]:
        if typ in _ISOLATE_INITIATORS:
            depth += 1
        elif typ == 'PDI':
            if depth > 0:
                depth -= 1
        elif depth == 0:
            if typ == 'L':
                return 0
            elif typ in ('R', 'AL'):
                return 1
    return 0


def _match_isolates(types: list[str]) -> tuple[dict[int, Optional[int]], set[int]]:
    """
    Rule BD9: matches isolate initiators with their PDIs. Returns a map from
    initiator index to PDI index (None if unmatched) and the set of matched
    PDI indices.
    """
    matching_pdi = {}
    matched_pdis = set()
    stack = []
    for idx, typ in enumerate(types):
        if typ in _ISOLATE_INITIATORS:
            stack.append(idx)
        elif typ == 'PDI' and stack:
            initiator = stack.pop()
            matching_pdi[initiator] = idx
            matched_pdis.add(idx)
    for initiator in stack:
        matching_pdi[initiator] = None
    return matching_pdi, matched_pdis


def _explicit_levels(types: list[str],
                     wtypes: list[str],
                     matching_pdi: dict[int, Optional[int]],
                     para_level: int) -> list[Optional[int]]:
    """
    Rules X1-X9: resolves explicit embedding levels using the directional
    status stack and marks the characters removed by X9 (RLE, LRE, RLO, LRO,
    PDF, BN) with a level of None. Directional overrides are applied to
    ``wtypes`` in place.
    """
    levels: list[Optional[int]] = [para_level] * len(types)
    # (embedding level, directional override, directional isolate status)
    stack = [(para_level, 'N', False)]
    overflow_isolates = overflow_embeddings = valid_isolates = 0

    for idx, typ in enumerate(types):
        if typ in _EXPLICIT_EMBEDDINGS:
            # X2-X5
            levels[idx] = None
            new_level = (stack[-1][0] + 1) | 1 if typ[0] == 'R' else (stack[-1][0] + 2) & ~1
            if new_level <= MAX_DEPTH and not overflow_isolates and not overflow_embeddings:
                stack.append((new_level, {'RLO': 'R', 'LRO': 'L'}.get(typ, 'N'), False))
            elif not overflow_isolates:
                overflow_embeddings += 1
        elif typ in _ISOLATE_INITIATORS:
            # X5a-X5c
            level, override, _ = stack[-1]
            levels[idx] = level
            if override != 'N':
                wtypes[idx] = override
            if typ == 'FSI':
                pdi = matching_pdi[idx]
                rtl = _first_strong_level(types, idx + 1, pdi if pdi is not None else len(types)) == 1
            else:
                rtl = typ == 'RLI'
            new_level = (level + 1) | 1 if rtl else (level + 2) & ~1
            if new_level <= MAX_DEPTH and not overflow_isolates and not overflow_embeddings:
                valid_isolates += 1
                stack.append((new_level, 'N', True))
            else:
                overflow_isolates += 1
        elif typ == 'PDI':
            # X6a
            if overflow_isolates:
                overflow_isolates -= 1
            elif valid_isolates:
                overflow_embeddings = 0
                while not stack[-1][2]:
                    stack.pop()
                stack.pop()
                valid_isolates -= 1
            level, override, _ = stack[-1]
            levels[idx] = level
            if override != 'N':
                wtypes[idx] = override
        elif typ == 'PDF':
            # X7
            levels[idx] = None
            if not overflow_isolates:
                if overflow_embeddings:
                    overflow_embeddings -= 1
                elif not stack[-1][2] and len(stack) > 1:
                    stack.pop()
        elif typ == 'B':
            # X8
            levels[idx] = para_level
        elif typ == 'BN':
            levels[idx] = None
        else:
            # X6
            level, override, _ = stack[-1]
            levels[idx] = level
            if override != 'N':
                wtypes[idx] = override
    return levels


def _isolating_run_sequences(types: list[str],
                             levels: list[Optional[int]],
                             keep: list[int],
                             matching_pdi: dict[int, Optional[int]],
                             matched_pdis: set[int],
                             para_level: int) -> list[tuple[list[int], str, str]]:
    """
    Rules X10/BD13: computes the isolating run sequences of a paragraph and
    their sos/eos types. Returns (character indices, sos, eos) triples.
    """
    if not keep:
        return []
    # level runs in X9-compacted order
    runs = []
    run_of = {}
    for pos in keep:
        if runs and levels[pos] == levels[runs[-1][-1]]:
            runs[-1].append(pos)
        else:
            runs.append([pos])
        run_of[pos] = len(runs) - 1

    keep_index = {pos: k for k, pos in enumerate(keep)}
    sequences = []
    for run in runs:
        if types[run[0]] == 'PDI' and run[0] in matched_pdis:
            # continuation of the sequence containing the matching initiator
            continue
        seq = list(run)
        while types[seq[-1]] in _ISOLATE_INITIATORS and matching_pdi[seq[-1]] is not None:
            seq.extend(runs[run_of[matching_pdi[seq[-1]]]])
        first, last = seq[0], seq[-1]
        k = keep_index[first]
        prev_level = levels[keep[k - 1]] if k > 0 else para_level
        sos = _direction(max(levels[first], prev_level))
        if types[last] in _ISOLATE_INITIATORS and matching_pdi[last] is None:
            next_level = para_level
        else:
            k = keep_index[last]
            next_level = levels[keep[k + 1]] if k + 1 < len(keep) else para_level
        eos = _direction(max(levels[last], next_level))
        sequences.append((seq, sos, eos))
    return sequences


def _resolve_weak(seq: list[int], sos: str, eos: str, wtypes: list[str]) -> None:
    """
    Rules W1-W7 over one isolating run sequence.
    """
    # W1: resolve NSMs
    prev = sos
    for pos in seq:
        if wtypes[pos] == 'NSM':
            wtypes[pos] = 'ON' if prev in _ISOLATES else prev
        prev = wtypes[pos]

    # W2: EN after strong AL -> AN
    strong = sos
    for pos in seq:
        if wtypes[pos] == 'EN' and strong == 'AL':
            wtypes[pos] = 'AN'
        if wtypes[pos] in ('L', 'R', 'AL'):
            strong = wtypes[pos]

    # W3: AL -> R
    for pos in seq:
        if wtypes[pos] == 'AL':
            wtypes[pos] = 'R'

    # W4: single separator between numbers of the same type
    for k in range(1, len(seq) - 1):
        typ = wtypes[seq[k]]
        prev_typ, next_typ = wtypes[seq[k - 1]], wtypes[seq[k + 1]]
        if typ == 'ES' and prev_typ == next_typ == 'EN':
            wtypes[seq[k]] = 'EN'
        elif typ == 'CS' and prev_typ == next_typ and prev_typ in ('EN', 'AN'):
            wtypes[seq[k]] = prev_typ

    # W5: ET sequence adjacent to EN -> EN
    k = 0
    while k < len(seq):
        if wtypes[seq[k]] == 'ET':
            j = k
            while j < len(seq) and wtypes[seq[j]] == 'ET':
                j += 1
            before = wtypes[seq[k - 1]] if k > 0 else sos
            after = wtypes[seq[j]] if j < len(seq) else eos
            if before == 'EN' or after == 'EN':
                for m in range(k, j):
                    wtypes[seq[m]] = 'EN'
            k = j
        else:
            k += 1

    # W6: remaining separators and terminators -> ON
    for pos in seq:
        if wtypes[pos] in ('ET', 'ES', 'CS'):
            wtypes[pos] = 'ON'

    # W7: EN after strong L -> L
    strong = sos
    for pos in seq:
        if wtypes[pos] == 'EN' and strong == 'L':
            wtypes[pos] = 'L'
        if wtypes[pos] in ('L', 'R'):
            strong = wtypes[pos]


def _n0_strong(typ: str) -> Optional[str]:
    # within N0, EN and AN are treated as R
    if typ == 'L':
        return 'L'
    elif typ in ('R', 'EN', 'AN'):
        return 'R'
    return None


def _resolve_brackets(seq: list[int],
                      sos: str,
                      types: list[str],
                      wtypes: list[str],
                      cps: list[int],
                      embedding_dir: str) -> None:
    """
    Rule N0 with BD14-BD16: resolves paired brackets.
    """
    # BD16: identify bracket pairs
    stack = []
    pairs = []
    for k, pos in enumerate(seq):
        if wtypes[pos] != 'ON':
            continue
        bracket = BRACKETS.get(cps[pos])
        if bracket is None:
            continue
        if bracket[1] == 'o':
            if len(stack) == _MAX_PAIRING_DEPTH:
                break
            stack.append((_CANON_BRACKETS.get(cps[pos], cps[pos]), k))
        else:
            opener = _CANON_BRACKETS.get(bracket[0], bracket[0])
            for si in range(len(stack) - 1, -1, -1):
                if stack[si][0] == opener:
                    pairs.append((stack[si][1], k))
                    del stack[si:]
                    break
    pairs.sort()

    opposite = 'L' if embedding_dir == 'R' else 'R'
    for open_k, close_k in pairs:
        # N0 b: strong type matching the embedding direction inside the pair
        found_opposite = False
        new_type = None
        for k in range(open_k + 1, close_k):
            strong = _n0_strong(wtypes[seq[k]])
            if strong == embedding_dir:
                new_type = embedding_dir
                break
            elif strong is not None:
                found_opposite = True
        if new_type is None:
            if found_opposite:
                # N0 c: opposite strong types inside, resolve from context
                context = sos
                for k in range(open_k - 1, -1, -1):
                    strong = _n0_strong(wtypes[seq[k]])
                    if strong is not None:
                        context = strong
                        break
                new_type = opposite if context == opposite else embedding_dir
            else:
                # N0 d: no strong types inside
                continue
        wtypes[seq[open_k]] = new_type
        wtypes[seq[close_k]] = new_type
        # NSMs following a re-typed bracket take its type
        for bracket_k in (open_k, close_k):
            for k in range(bracket_k + 1, len(seq)):
                if types[seq[k]] != 'NSM':
                    break
                wtypes[seq[k]] = new_type


def _resolve_neutrals(seq: list[int], sos: str, eos: str, wtypes: list[str],
                      embedding_dir: str) -> None:
    """
    Rules N1/N2: resolves sequences of neutral and isolate formatting
    characters.
    """
    k = 0
    ln = len(seq)
    while k < ln:
        if wtypes[seq[k]] in _NI:
            j = k
            while j < ln and wtypes[seq[j]] in _NI:
                j += 1
            before = wtypes[seq[k - 1]] if k > 0 else sos
            after = wtypes[seq[j]] if j < ln else eos
            if before in ('EN', 'AN'):
                before = 'R'
            if after in ('EN', 'AN'):
                after = 'R'
            fill = before if before == after else embedding_dir
            for m in range(k, j):
                wtypes[seq[m]] = fill
            k = j
        else:
            k += 1


def resolve_levels(types: list[str],
                   cps: Optional[list[int]],
                   para_level: int) -> tuple[list[Optional[int]], list[int]]:
    """
    Runs rules X1-L2 over a paragraph given as a list of bidirectional
    character types and optionally the corresponding code points (needed
    for bracket pair resolution in rule N0).

    Returns:
        A tuple of the resolved embedding levels (None for characters
        removed by rule X9) and the visual order of the surviving character
        indices from left to right.
    """
    wtypes = list(types)
    matching_pdi, matched_pdis = _match_isolates(types)
    levels = _explicit_levels(types, wtypes, matching_pdi, para_level)
    keep = [idx for idx, level in enumerate(levels) if level is not None]

    for seq, sos, eos in _isolating_run_sequences(types, levels, keep,
                                                  matching_pdi, matched_pdis,
                                                  para_level):
        embedding_dir = _direction(levels[seq[0]])
        _resolve_weak(seq, sos, eos, wtypes)
        if cps is not None:
            _resolve_brackets(seq, sos, types, wtypes, cps, embedding_dir)
        _resolve_neutrals(seq, sos, eos, wtypes, embedding_dir)

    # I1/I2
    for pos in keep:
        if levels[pos] % 2:
            if wtypes[pos] != 'R':
                levels[pos] += 1
        elif wtypes[pos] == 'R':
            levels[pos] += 1
        elif wtypes[pos] in ('EN', 'AN'):
            levels[pos] += 2

    # L1, using the original character types
    reset = True
    for pos in reversed(keep):
        if types[pos] in ('B', 'S'):
            levels[pos] = para_level
            reset = True
        elif types[pos] in ('WS', 'LRI', 'RLI', 'FSI', 'PDI'):
            if reset:
                levels[pos] = para_level
        else:
            reset = False

    # L2
    visual = list(keep)
    if visual:
        highest = max(levels[pos] for pos in visual)
        lowest_odd = min((levels[pos] for pos in visual if levels[pos] % 2),
                         default=None)
        if lowest_odd is not None:
            for level in range(highest, lowest_odd - 1, -1):
                k = 0
                while k < len(visual):
                    if levels[visual[k]] >= level:
                        j = k
                        while j < len(visual) and levels[visual[j]] >= level:
                            j += 1
                        visual[k:j] = visual[k:j][::-1]
                        k = j
                    else:
                        k += 1
    return levels, visual


def get_display_map(text: str,
                    base_dir: Optional[Literal['L', 'R']] = None) -> tuple[str, list[int]]:
    """
    Reorders a string from logical order into display order according to the
    Unicode 17.0.0 bidirectional algorithm. The text is treated as a single
    paragraph.

    Args:
        text: Text in logical order.
        base_dir: Base (paragraph) direction, 'L' or 'R'. When None the
                  direction is resolved from the first strong character
                  (rules P2/P3).

    Returns:
        A tuple of the reordered string and the index of each of its
        characters in ``text``. Mirrored characters (e.g. parentheses in
        right-to-left runs) are replaced by their mirror image; explicit
        directional formatting characters (LRE, RLE, LRO, RLO, PDF, LRI,
        RLI, FSI, PDI) and characters with bidirectional class BN are
        removed from the output.
    """
    if base_dir not in (None, 'L', 'R'):
        raise ValueError(f'Invalid base direction {base_dir!r}')
    cps = [ord(ch) for ch in text]
    types = [bidi_class(cp) for cp in cps]
    if base_dir is None:
        para_level = _first_strong_level(types, 0, len(types))
    else:
        para_level = 0 if base_dir == 'L' else 1
    levels, visual = resolve_levels(types, cps, para_level)
    display = []
    order = []
    for k in visual:
        if types[k] in _ISOLATES:
            continue
        cp = cps[k]
        # L4
        if levels[k] % 2 and cp in MIRRORED:
            cp = MIRRORED[cp]
        display.append(chr(cp))
        order.append(k)
    return ''.join(display), order


def get_display(text: str, base_dir: Optional[Literal['L', 'R']] = None) -> str:
    """
    Reorders a string from logical order into display order according to the
    Unicode 17.0.0 bidirectional algorithm. The text is treated as a single
    paragraph.

    Args:
        text: Text in logical order.
        base_dir: Base (paragraph) direction, 'L' or 'R'. When None the
                  direction is resolved from the first strong character
                  (rules P2/P3).

    Returns:
        The reordered string, with mirrored characters replaced by their
        mirror image and directional formatting characters and BN-class
        characters removed.
    """
    return get_display_map(text, base_dir)[0]
