# Sample Images — Expedition First Voice

Public-domain and freely-licensed images used for First Voice inference passes.
Selected for thematic coherence: 1960s modernity, documentary photography, space
exploration, and modernist portraiture.

## Files

| File | Subject | Source | License |
|------|---------|--------|---------|
| `worlds_fair_unisphere_1964.jpg` | Unisphere at the 1964 New York World's Fair, Queens | Wikimedia Commons | Public domain (pre-1978 US) |
| `dorothea_lange_migrant_mother.jpg` | Migrant Mother, Nipomo, California (1936) — Dorothea Lange | Library of Congress / Wikimedia Commons | Public domain (US government work) |
| `nasa_pillars_of_creation.jpg` | Pillars of Creation, Eagle Nebula — Hubble/Webb composite | NASA / ESA / STScI | Public domain (NASA) |
| `stein_gertrude_portrait.jpg` | Gertrude Stein, Paris (c. 1934) | Public domain portrait | Public domain |
| `taxi_street_scene.jpg` | Midcentury street scene with taxi | Local asset | — |
| `test_cat.jpg` | Cat — standard vision model benchmark subject | Local asset | — |

## Usage

`lib/expedition/sampler.py` selects a random image from this directory for
`image-classification`, `object-detection`, `semantic-segmentation`,
`depth-estimation`, `image-to-text`, `visual-question-answering`, and
`image-captioning` tasks during the First Voice inference pass.
