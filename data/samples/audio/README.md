# Sample Audio — Expedition First Voice

Public-domain audio and original MIDI files used for First Voice inference passes.

## Files

| File | Content | Source | License |
|------|---------|--------|---------|
| `stein_gertrude_if_i_told_him.mp3` | Gertrude Stein reading "If I Told Him: A Completed Portrait of Picasso" | pytorch-roulette project | — |
| `stein_gertrude_if_i_told_him.wav` | Same recording in WAV format | pytorch-roulette project | — |
| `bach_cello_suite1_prelude.mid` | J.S. Bach — Cello Suite No. 1 in G major (BWV 1007) Prelude, generated | Synthesized — public domain composition | Public domain |
| `beethoven_moonlight_sonata.mid` | Beethoven — Piano Sonata Op. 27 No. 2 "Moonlight" Adagio theme, generated | Synthesized — public domain composition | Public domain |

## Notes

- WAV files are preferred over MP3 for ASR/audio-classification models (PCM format, no decoding overhead).
- MIDI files represent the underlying musical score; most audio models expect PCM audio, so MIDI is used as thematic reference rather than direct model input.
- Bach and Beethoven compositions are public domain (both composers died before 1900).

## Usage

`lib/expedition/sampler.py` selects a WAV (preferred) or MP3 from this directory
for `automatic-speech-recognition` and `audio-classification` tasks during the
First Voice inference pass.
