#!/usr/bin/env bash
# build_mac_app.sh
# ==================
# Packages handwriting_gui.py into a standalone macOS .app bundle using
# PyInstaller. Run this from inside your project directory (the one
# containing handwriting_gui.py, handwriting_bot.py, handwriting_ebb.py,
# text_extractor.py, and the fonts/ folder), with your venv activated.
#
# Usage:
#     chmod +x build_mac_app.sh
#     ./build_mac_app.sh
#
# Result: dist/Handwriting Robot.app -- double-clickable, movable to
# /Applications like any other Mac app.
#
# NOTES
# -----
# - Uses --onedir (not --onefile). Onefile Tk/Tcl apps extract themselves
#   to a temp folder on every launch, which is slower and occasionally
#   flaky with Tk's asset paths. Onedir launches faster and is more
#   reliable for GUI apps like this.
#
# - --collect-all customtkinter is REQUIRED. CustomTkinter ships its own
#   theme JSON files, fonts, and images as package data. Without this
#   flag, PyInstaller only bundles the .py code and the packaged app will
#   either crash on startup or render with broken/missing styling.
#
# - --add-data bundles your fonts/ folder (for the EMS Casual Hand preset
#   etc.) into the app itself. macOS uses ':' as the separator (Windows
#   would use ';' -- not relevant here, but noted in case you ever build
#   cross-platform).
#
# - The RNN engine's handwriting-synthesis/TensorFlow dependency is NOT
#   bundled here. handwriting_gui.py already guards that import with a
#   try/except, so the packaged app will work fine with the Hershey
#   engine -- selecting "RNN (neural handwriting)" will just show the
#   same clear error message it does today, rather than crashing. Bundling
#   TensorFlow via PyInstaller is a much heavier, more fragile process --
#   worth doing as a separate step later if you actually need RNN inside
#   the packaged app, not as part of this basic build.
#
# - First launch of an unsigned/un-notarized app will trigger macOS
#   Gatekeeper's "Apple could not verify..." warning. Right-click the
#   app -> Open (instead of double-clicking) the first time to bypass
#   this. Full notarization (to avoid this warning even on other
#   people's Macs) requires an Apple Developer account and is out of
#   scope for a personal/local build like this.

set -e

APP_NAME="Handwriting Robot"
ENTRY_POINT="handwriting_gui.py"

if [ ! -f "$ENTRY_POINT" ]; then
    echo "ERROR: $ENTRY_POINT not found in the current directory."
    echo "cd into your project folder first, then re-run this script."
    exit 1
fi

echo "[1/3] Ensuring PyInstaller is installed..."
pip install --quiet pyinstaller

echo "[2/3] Building $APP_NAME.app (this can take a minute or two)..."

ADD_DATA_ARGS=()
if [ -d "fonts" ]; then
    ADD_DATA_ARGS+=(--add-data "fonts:fonts")
else
    echo "  (no fonts/ folder found next to $ENTRY_POINT -- skipping; EMS Casual Hand preset won't be available in the packaged app unless you add it back in later)"
fi

pyinstaller \
    --name "$APP_NAME" \
    --windowed \
    --onedir \
    --noconfirm \
    --collect-all customtkinter \
    --hidden-import=serial.tools.list_ports \
    "${ADD_DATA_ARGS[@]}" \
    "$ENTRY_POINT"

echo "[3/3] Done."
echo ""
echo "Your app is at: dist/$APP_NAME.app"
echo "You can drag it into /Applications, or just double-click it from dist/."
echo ""
echo "NOTE: On first launch, macOS will likely warn that it can't verify"
echo "the developer. Right-click the app -> Open (instead of double-"
echo "clicking) to bypass this the first time."