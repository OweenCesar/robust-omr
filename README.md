robust-omr

Simple OMR (optical mark recognition) utilities for reading and grading bubble sheets.

Clone
-----

git clone https://github.com/OweenCesar/robust-omr.git
cd robust-omr

Setup
-----

Create and activate a virtual environment, then install requirements:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run
---

Run the main script:

python src/main.py

Notes
-----
- Raw input images go in `data/raw/`.
- Outputs will be written to `outputs/` (see `outputs/annotated/` and `outputs/warped/`).
