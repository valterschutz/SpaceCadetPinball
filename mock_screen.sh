#!/bin/bash
pkill Xvfb
nohup Xvfb :99 -screen 0 1280x1024x16 > xvfb.log 2>&1 &
export DISPLAY=:99
export SDL_AUDIODRIVER=dummy

