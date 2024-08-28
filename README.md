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


```bash

## Usage

1. Take a photo of the puzzle image.
2. Take a photo of the piece (or multiple pieces already assembled).
3. The app will tell you the zone where the piece can be placed.


## How does it work technically?

1. Remove the background from the piece(s) image.
2. Extract features
3. Match the piece features with the features of the puzzle image.