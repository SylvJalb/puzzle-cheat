# Puzzle Cheat

🚨 Currently under development 🚨

Cheat at the puzzle game.
Give it a good quality photo of the puzzle image.
Then, just take a photo of the piece and the app will tell you where it goes.

## Installation

Run a virtual environment and install the requirements.

```bash
pip install -r requirements.txt
```

(If you are using a Ubuntu/Debian based system, you might need to install the following packages
```bash
sudo apt-get install libjpeg-dev zlib1g-dev
```
)

```bash

## Usage

1. Take a photo of the puzzle image.
2. Take a photo of the piece (or multiple pieces already assembled).
3. The app will tell you the zone where the piece can be placed.


## How does it work technically?

1. Remove the background from the puzzle image and the piece.
2. Feature extraction (SURF)
3. Feature Matching (Exhaustive Matching)