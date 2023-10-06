#!/bin/bash
pkill Xvfb
Xvfb :99 -screen 0 1280x1024x16 &
export DISPLAY=:99
export SDL_AUDIODRIVER=dummy

