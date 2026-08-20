#!/usr/bin/env python3
"""
handwriting_gui.py
====================
A point-and-click desktop app for the handwriting robot pipeline --
no terminal commands needed. Wraps handwriting_bot.py, handwriting_ebb.py,
handwriting_rnn.py, and text_extractor.py.

Run it with:
    python3 handwriting_gui.py

Requires everything in requirements.txt, plus tkinter (usually built into
Python -- on some Linux distros install it separately with
`sudo apt install python3-tk`; on macOS with Homebrew Python, `brew install
python-tk`; Windows installers include it by default) AND customtkinter:
    pip install customtkinter

For the RNN engine option, handwriting_rnn.py and its handwriting-synthesis
dependency must also be importable from this same environment -- see
handwriting_rnn.py's docstring for setup. If that dependency isn't
installed, the Hershey engine (including EMS Casual Hand and all other
existing font presets) still works exactly as before; only selecting
"RNN (neural handwriting)" in the Engine dropdown requires it.

UI NOTE: this uses a persistent serial connection (self.ser), established
explicitly via the "Connect" button in the top connection bar, instead of
each action opening/closing its own short-lived connection. Every action
(pen test, motor controls, homing, writing) requires self.connected to be
True first -- see _require_connected().

Built on CustomTkinter for real rounded cards/buttons and light/dark-mode-
aware widgets; the sidebar and status dots use explicit fixed colors (not
the global appearance mode) so they stay dark regardless of system theme.
"""

import io
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from handwriting_bot import (HandwritingGenerator, DEFAULTS as BASE_DEFAULTS,
                               PRINT_STYLE_OVERRIDES, PAPER_SIZES, paper_size_to_cfg,
                               check_page_bounds)
from handwriting_ebb import EBB_DEFAULTS, strokes_to_ebb_commands
from text_extractor import extract_text, extract_blocks, clean_text

try:
    from handwriting_rnn import RNNHandwritingGenerator
except ImportError:
    RNNHandwritingGenerator = None  # RNN engine option will still show in the
                                     # dropdown, but selecting it reports a
                                     # clear error instead of crashing the app
                                     # at import time -- see _generate().

try:
    from handwriting_verify import HandwritingVerifier, VerifiedRNNHandwritingGenerator
except ImportError:
    HandwritingVerifier = None
    VerifiedRNNHandwritingGenerator = None  # Verify checkbox still shows, but
                                             # checking it reports a clear error
                                             # instead of crashing -- see _generate().

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Fonts bundled with the app, selectable directly from a dropdown instead of
# needing to browse for a file. Each preset can also pre-fill the recommended
# style settings that were found to work well for that specific font.
FONT_PRESETS = {
    "Built-in cursive/script": {
        "path": None,
    },
    "EMS Casual Hand": {
        "path": os.path.join(APP_DIR, "fonts", "casual_hand.json"),
        "exclusive": True,
        "print_style": True,
        "exclude_letters": "",
    },
    "Custom font file...": {
        "path": "BROWSE",  # sentinel: reveals the file-picker instead of using a fixed path
    },
}

# Limit switch pin assignments, confirmed on hardware (Port B on the EBB
# board's unused IO header): B5 = X-axis minimum, B6 = X-axis maximum,
# B7 = Y-axis reference. Homing drives toward B6/B7 (the "home corner");
# B5 is available if you want to also measure full X travel later.
# Each tuple is (port, pin, triggered_value) -- triggered_value is the PI
# response that means "switch active": the mechanical switches here read
# '1' when pressed, but the hall-effect sensor on B7 is active-LOW
# (confirmed: reads '1' at rest with no magnet present, '0' when the
# magnet is present) -- so its triggered_value is '0', the opposite of
# the mechanical switches.
SWITCH_X_MIN = ("B", "5", "1")
SWITCH_X_MAX = ("B", "6", "1")
SWITCH_Y = ("B", "7", "0")  # hall-effect sensor, active-low

# --------------------------------------------------------------------------
# Visual palette. Light appearance mode overall, with a permanently-dark
# sidebar (explicit fixed colors, not tied to the global appearance mode).
# --------------------------------------------------------------------------
SIDEBAR_BG = "#18181b"
SIDEBAR_ACTIVE_BG = "#2a2a2f"
SIDEBAR_TEXT = "#b7b6ba"
SIDEBAR_TEXT_ACTIVE = "#ffffff"
SIDEBAR_MUTED = "#6f6e73"

CONTENT_BG = "#f2f1ee"
CARD_BG = "#ffffff"
CARD_BORDER = "#e3e1dc"

TEXT_DARK = "#1f1e1c"
TEXT_MUTED = "#8a8378"

ACCENT = "#2f8f52"
ACCENT_HOVER = "#26743f"

STATUS_COLORS = {
    "connected": "#33c46a",
    "disconnected": "#9a978f",
    "connecting": "#e0a83c",
    "error": "#e2584f",
}

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")


# --------------------------------------------------------------------------
# A queue-backed writer so background-thread print() output can be safely
# shown in the Tk log widget (Tkinter itself is not thread-safe).
# --------------------------------------------------------------------------

class QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)

    def flush(self):
        pass


class HandwritingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Robot")
        self.geometry("1180x780")
        self.minsize(980, 640)
        self.configure(fg_color=CONTENT_BG)

        self.log_queue = queue.Queue()
        self.busy = False
        self.pen_test_event = threading.Event()
        self._cached_verifier = None  # lazy-loaded on first use -- see _get_verifier()

        # Persistent serial connection state -- established explicitly via
        # the Connect button in the top bar, reused by every action instead
        # of each one opening/closing its own short-lived connection.
        self.ser = None
        self.connected = False

        self._build_ui()
        self._poll_log_queue()
        self._refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI ---

    def _build_ui(self):
        # Shared page-layout variables, created up front so both the Page
        # Setup page and the Advanced page's fields can bind to the exact
        # same StringVars -- changing one updates the other.
        self.layout_vars = {
            key: tk.StringVar(value=str(EBB_DEFAULTS[key]))
            for key in ("font_size_mm", "page_width_mm", "page_height_mm",
                        "x_offset_mm", "y_offset_mm", "jitter_amp_mm")
        }

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()

        content = ctk.CTkFrame(self, fg_color=CONTENT_BG, corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self._build_connection_bar(content)

        # Page container -- one frame per nav section, swapped via
        # _show_page(). Built once up front so widgets/vars persist.
        self.page_container = ctk.CTkFrame(content, fg_color=CONTENT_BG, corner_radius=0)
        self.page_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(4, 4))
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        self._build_write_page()
        self._build_style_page()
        self._build_page_setup_page()
        self._build_robot_page()
        self._build_advanced_page()

        self._build_footer(content)

        self._show_page("write")

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color=SIDEBAR_BG)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        title = ctk.CTkLabel(sidebar, text="\u270e  Handwriting\n   Robot",
                              text_color=SIDEBAR_TEXT_ACTIVE, font=("TkDefaultFont", 15, "bold"),
                              justify="left", anchor="w")
        title.pack(fill="x", padx=18, pady=(22, 18))

        sep = ctk.CTkFrame(sidebar, fg_color=SIDEBAR_ACTIVE_BG, height=1, corner_radius=0)
        sep.pack(fill="x", padx=18, pady=(0, 10))

        section_label = ctk.CTkLabel(sidebar, text="SECTIONS", text_color=SIDEBAR_MUTED,
                                      font=("TkDefaultFont", 10, "bold"), anchor="w")
        section_label.pack(fill="x", padx=18, pady=(4, 6))

        self.nav_buttons = {}
        nav_items = [
            ("write", "\u270d  Write"),
            ("style", "\u270f  Style & Font"),
            ("page_setup", "\u2b1a  Page Setup"),
            ("robot", "\u2699  Robot Controls"),
            ("advanced", "\u2699  Advanced / Calibration"),
        ]
        for key, label in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w", corner_radius=8,
                fg_color=SIDEBAR_BG, hover_color=SIDEBAR_ACTIVE_BG,
                text_color=SIDEBAR_TEXT, font=("TkDefaultFont", 12),
                command=lambda k=key: self._show_page(k))
            btn.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[key] = btn

        spacer = ctk.CTkFrame(sidebar, fg_color=SIDEBAR_BG, corner_radius=0)
        spacer.pack(fill="both", expand=True)

        hint = ctk.CTkLabel(sidebar, text="Connect to the robot using\nthe bar at the top right.",
                             text_color=SIDEBAR_MUTED, font=("TkDefaultFont", 10),
                             justify="left", anchor="w")
        hint.pack(fill="x", padx=18, pady=(0, 18))

    def _show_page(self, key):
        for name, frame in self.pages.items():
            if name == key:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()
        for name, btn in self.nav_buttons.items():
            if name == key:
                btn.configure(fg_color=SIDEBAR_ACTIVE_BG, text_color=SIDEBAR_TEXT_ACTIVE)
            else:
                btn.configure(fg_color=SIDEBAR_BG, text_color=SIDEBAR_TEXT)

    def _build_connection_bar(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14,
                            border_width=1, border_color=CARD_BORDER)
        bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 10))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(inner, text="Robot connection", text_color=TEXT_DARK,
                     font=("TkDefaultFont", 14, "bold")).pack(side="left")

        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="right")

        # Status dot + label
        self.conn_dot = ctk.CTkLabel(right, text="\u25cf", text_color=STATUS_COLORS["disconnected"],
                                      font=("TkDefaultFont", 14), width=16)
        self.conn_dot.pack(side="left", padx=(0, 4))

        self.conn_status_var = tk.StringVar(value="Disconnected")
        ctk.CTkLabel(right, textvariable=self.conn_status_var, text_color=TEXT_MUTED,
                     width=170, anchor="w").pack(side="left", padx=(0, 14))

        self.connect_btn = ctk.CTkButton(right, text="Connect", fg_color=ACCENT,
                                          hover_color=ACCENT_HOVER, width=100,
                                          command=self._on_connect_toggle)
        self.connect_btn.pack(side="left", padx=(0, 10))

        ctk.CTkButton(right, text="Refresh", width=80, fg_color="#e9e7e2",
                      hover_color="#dcdad4", text_color=TEXT_DARK,
                      command=self._refresh_ports).pack(side="left", padx=(0, 8))

        self.port_var = tk.StringVar(value="No ports found")
        self.port_menu = ctk.CTkOptionMenu(right, variable=self.port_var, values=["No ports found"],
                                            width=220, fg_color="#e9e7e2", button_color="#dcdad4",
                                            button_hover_color="#cfcdc7", text_color=TEXT_DARK,
                                            dropdown_fg_color="#ffffff", dropdown_text_color=TEXT_DARK)
        self.port_menu.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(right, text="Serial port:", text_color=TEXT_MUTED).pack(side="left")

    def _new_page(self, key):
        page = ctk.CTkFrame(self.page_container, fg_color=CONTENT_BG, corner_radius=0)
        self.pages[key] = page
        return page

    def _scrollable(self, parent):
        """A scrollable page body, built on CTkScrollableFrame so long
        pages (e.g. Advanced) don't get clipped on smaller windows."""
        body = ctk.CTkScrollableFrame(parent, fg_color=CONTENT_BG)
        body.pack(fill="both", expand=True)
        return body

    def _card(self, parent, title):
        """A rounded 'card' section with a bold title -- CustomTkinter's
        CTkFrame supports real rounded corners, unlike plain ttk/tk."""
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14,
                             border_width=1, border_color=CARD_BORDER)
        card.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(card, text=title, text_color=TEXT_DARK,
                     font=("TkDefaultFont", 14, "bold")).pack(anchor="w", padx=18, pady=(16, 8))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        return body

    # ---------------- WRITE PAGE ----------------

    def _build_write_page(self):
        page = self._new_page("write")
        body = self._scrollable(page)

        src = self._card(body, "What to write")

        self.source_mode = tk.StringVar(value="type")
        mode_row = ctk.CTkFrame(src, fg_color="transparent")
        mode_row.pack(fill="x", pady=(0, 8))
        ctk.CTkRadioButton(mode_row, text="Type text", variable=self.source_mode,
                            value="type", command=self._update_source_mode).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(mode_row, text="Load a file (.txt / .docx / .pdf)", variable=self.source_mode,
                            value="file", command=self._update_source_mode).pack(side="left")

        text_wrap = ctk.CTkFrame(src, fg_color="#ffffff", corner_radius=8,
                                  border_width=1, border_color=CARD_BORDER)
        text_wrap.pack(fill="both", expand=True, pady=(4, 10))
        self.text_box = tk.Text(text_wrap, height=8, wrap="word", relief="flat", bd=0,
                                 highlightthickness=0, padx=10, pady=8)
        self.text_box.pack(fill="both", expand=True)
        self.text_box.insert("1.0", "Type what you want the robot to write here...")

        file_row = ctk.CTkFrame(src, fg_color="transparent")
        file_row.pack(fill="x", pady=(0, 8))
        self.file_path_var = tk.StringVar(value="No file selected")
        ctk.CTkButton(file_row, text="Choose file...", width=140,
                      command=self._choose_file).pack(side="left")
        ctk.CTkLabel(file_row, textvariable=self.file_path_var, text_color=TEXT_MUTED).pack(side="left", padx=10)

        self.preserve_structure_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(src, text="Preserve document structure (headings, bullets, numbered lists)",
                         variable=self.preserve_structure_var).pack(anchor="w")

        self._update_source_mode()

    # ---------------- STYLE PAGE ----------------

    def _build_style_page(self):
        page = self._new_page("style")
        body = self._scrollable(page)

        engine_card = self._card(body, "Engine")
        row0 = ctk.CTkFrame(engine_card, fg_color="transparent")
        row0.pack(fill="x", pady=4)
        ctk.CTkLabel(row0, text="Engine:", width=90, anchor="w").pack(side="left")
        self.engine_var = tk.StringVar(value="Hershey (font-based)")
        engine_menu = ctk.CTkOptionMenu(
            row0, variable=self.engine_var,
            values=["Hershey (font-based)", "RNN (neural handwriting)"],
            width=240, command=lambda choice: self._on_engine_change())
        engine_menu.pack(side="left", padx=8)

        self.rnn_controls_frame = ctk.CTkFrame(engine_card, fg_color="transparent")
        self.rnn_controls_frame.pack(fill="x", pady=(8, 0))

        rnn_row = ctk.CTkFrame(self.rnn_controls_frame, fg_color="transparent")
        rnn_row.pack(fill="x")
        ctk.CTkLabel(rnn_row, text="Neatness (0=messy, 1=neat):").pack(side="left")
        self.rnn_bias_var = tk.StringVar(value="0.75")
        ctk.CTkEntry(rnn_row, textvariable=self.rnn_bias_var, width=60).pack(side="left", padx=(6, 20))

        ctk.CTkLabel(rnn_row, text="Style (0-12):").pack(side="left")
        self.rnn_style_var = tk.StringVar(value="9")
        ctk.CTkEntry(rnn_row, textvariable=self.rnn_style_var, width=60).pack(side="left", padx=6)

        self.verify_var = tk.BooleanVar(value=False)
        self.verify_chk = ctk.CTkCheckBox(
            self.rnn_controls_frame,
            text="Verify with recognition model before writing (slower, catches garbled letters)",
            variable=self.verify_var)
        self.verify_chk.pack(anchor="w", pady=(8, 0))

        font_card = self._card(body, "Font")
        frow0 = ctk.CTkFrame(font_card, fg_color="transparent")
        frow0.pack(fill="x", pady=4)
        ctk.CTkLabel(frow0, text="Font:", width=90, anchor="w").pack(side="left")
        self.font_preset_var = tk.StringVar(value="Built-in cursive/script")
        self.font_preset_combo = ctk.CTkOptionMenu(
            frow0, variable=self.font_preset_var, values=list(FONT_PRESETS.keys()),
            width=240, command=lambda choice: self._on_font_preset_change())
        self.font_preset_combo.pack(side="left", padx=8)

        custom_row = ctk.CTkFrame(font_card, fg_color="transparent")
        custom_row.pack(fill="x", pady=(8, 4))
        self.custom_font_path_var = tk.StringVar(value="No font file selected")
        self.choose_font_btn = ctk.CTkButton(custom_row, text="Choose font JSON...", width=160,
                                              command=self._choose_font)
        self.choose_font_btn.pack(side="left")
        ctk.CTkLabel(custom_row, textvariable=self.custom_font_path_var,
                     text_color=TEXT_MUTED).pack(side="left", padx=10)

        self.custom_exclusive_var = tk.BooleanVar(value=True)
        self.custom_exclusive_chk = ctk.CTkCheckBox(
            font_card, text="Use only this font (don't mix with built-in shapes)",
            variable=self.custom_exclusive_var)
        self.custom_exclusive_chk.pack(anchor="w", pady=2)

        self.print_style_var = tk.BooleanVar(value=False)
        self.print_style_chk = ctk.CTkCheckBox(
            font_card, text="Print-style font (disable cursive slant, add letter spacing)",
            variable=self.print_style_var)
        self.print_style_chk.pack(anchor="w", pady=2)

        exclude_row = ctk.CTkFrame(font_card, fg_color="transparent")
        exclude_row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(exclude_row, text="Letters to always use built-in shapes for (e.g. EJ):").pack(side="left")
        self.exclude_letters_var = tk.StringVar(value="")
        ctk.CTkEntry(exclude_row, textvariable=self.exclude_letters_var, width=80).pack(side="left", padx=8)

        self._on_font_preset_change()
        self._on_engine_change()

    # ---------------- PAGE SETUP PAGE ----------------

    def _build_page_setup_page(self):
        page = self._new_page("page_setup")
        body = self._scrollable(page)

        card = self._card(body, "Page setup")

        row0 = ctk.CTkFrame(card, fg_color="transparent")
        row0.pack(fill="x", pady=4)
        ctk.CTkLabel(row0, text="Paper size:", width=90, anchor="w").pack(side="left")
        self.paper_size_var = tk.StringVar(value="Letter (8.5x11in)")
        paper_menu = ctk.CTkOptionMenu(
            row0, variable=self.paper_size_var,
            values=list(PAPER_SIZES.keys()) + ["Custom (set below)"],
            width=200, command=lambda choice: self._on_paper_size_change())
        paper_menu.pack(side="left", padx=8)

        ctk.CTkLabel(row0, text="Margin (mm):").pack(side="left", padx=(16, 4))
        self.margin_var = tk.StringVar(value="15")
        margin_entry = ctk.CTkEntry(row0, textvariable=self.margin_var, width=60)
        margin_entry.pack(side="left")
        margin_entry.bind("<FocusOut>", lambda e: self._on_paper_size_change())

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", pady=8)
        ctk.CTkLabel(row1, text="Font size (mm):", width=90, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.layout_vars["font_size_mm"], width=80).pack(side="left", padx=8)

        self.page_info_var = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self.page_info_var, text_color=TEXT_MUTED,
                     anchor="w", justify="left").pack(fill="x", pady=(4, 0))

        self._on_paper_size_change()

    # ---------------- ROBOT CONTROLS PAGE ----------------

    def _build_robot_page(self):
        page = self._new_page("robot")
        body = self._scrollable(page)

        pen_card = self._card(body, "Pen test")
        pen_row = ctk.CTkFrame(pen_card, fg_color="transparent")
        pen_row.pack(fill="x")
        ctk.CTkButton(pen_row, text="Test pen up/down", command=self._on_test_pen).pack(side="left")
        self.pen_next_btn = ctk.CTkButton(pen_row, text="Next step ->", command=self._pen_test_next,
                                           state="disabled")
        self.pen_next_btn.pack(side="left", padx=8)

        motor_card = self._card(body, "Motors & homing")
        motor_row = ctk.CTkFrame(motor_card, fg_color="transparent")
        motor_row.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(motor_row, text="Enable motors", command=self._on_enable_motors).pack(side="left")
        ctk.CTkButton(motor_row, text="Disable motors (free to move by hand)",
                      command=self._on_disable_motors).pack(side="left", padx=8)
        ctk.CTkButton(motor_row, text="Go to Home (0,0)", command=self._on_go_home).pack(side="left", padx=8)

        home_switches_row = ctk.CTkFrame(motor_card, fg_color="transparent")
        home_switches_row.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(home_switches_row, text="Home to limit switches",
                      command=self._on_home_switches).pack(side="left")
        ctk.CTkLabel(home_switches_row,
                     text="(drives to X-max/Y switches, sets that as origin)",
                     text_color=TEXT_MUTED).pack(side="left", padx=10)

        self.motor_status_var = tk.StringVar(value="Motors: unknown")
        ctk.CTkLabel(motor_card, textvariable=self.motor_status_var, text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x")

    # ---------------- ADVANCED PAGE ----------------

    def _build_advanced_page(self):
        page = self._new_page("advanced")
        body = self._scrollable(page)

        cal_card = self._card(body, "Calibration (pre-filled with known-working values -- "
                                      "only change these for different hardware)")

        self.adv_vars = {}

        def add_field(key, label, default, width=90):
            row = ctk.CTkFrame(cal_card, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=label, width=220, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            ctk.CTkEntry(row, textvariable=var, width=width).pack(side="left", padx=8)
            self.adv_vars[key] = var

        add_field("steps_per_mm", "Steps per mm:", EBB_DEFAULTS["steps_per_mm"])
        add_field("pen_up_value", "Pen UP value (SP,?):", EBB_DEFAULTS["pen_up_value"])
        add_field("pen_down_value", "Pen DOWN value (SP,?):", EBB_DEFAULTS["pen_down_value"])
        add_field("pen_move_settle_ms", "Pen settle time (ms):", EBB_DEFAULTS["pen_move_settle_ms"])
        add_field("baud", "Baud rate (applies on next Connect):", EBB_DEFAULTS.get("baud", 115200))

        self.flip_x_var = tk.BooleanVar(value=EBB_DEFAULTS["flip_x"])
        self.flip_y_var = tk.BooleanVar(value=EBB_DEFAULTS["flip_y"])
        self.reverse_line_var = tk.BooleanVar(value=EBB_DEFAULTS["reverse_line_direction"])
        ctk.CTkCheckBox(cal_card, text="Flip X (if writing comes out mirrored left-right)",
                         variable=self.flip_x_var).pack(anchor="w", pady=(8, 2))
        ctk.CTkCheckBox(cal_card, text="Flip Y (if letter shapes come out mirrored top-bottom)",
                         variable=self.flip_y_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(cal_card, text="Reverse line direction (if new lines appear above the previous line)",
                         variable=self.reverse_line_var).pack(anchor="w", pady=2)

        layout_card = self._card(body, "Page layout (also settable from Page Setup)")

        def add_layout_field(key, label, width=80):
            row = ctk.CTkFrame(layout_card, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=label, width=140, anchor="w").pack(side="left")
            var = self.layout_vars[key]  # created in _build_ui, shared with Page Setup
            ctk.CTkEntry(row, textvariable=var, width=width).pack(side="left", padx=8)

        add_layout_field("font_size_mm", "Font size (mm):")
        add_layout_field("page_width_mm", "Page width (mm):")
        add_layout_field("page_height_mm", "Page height (mm):")
        add_layout_field("x_offset_mm", "X offset (mm):")
        add_layout_field("y_offset_mm", "Y offset (mm):")
        add_layout_field("jitter_amp_mm", "Jitter amount (mm):")

        seed_card = self._card(body, "Reproducibility")
        seed_row = ctk.CTkFrame(seed_card, fg_color="transparent")
        seed_row.pack(fill="x")
        ctk.CTkLabel(seed_row, text="Random seed (blank = random each time):").pack(side="left")
        self.seed_var = tk.StringVar(value="")
        ctk.CTkEntry(seed_row, textvariable=self.seed_var, width=100).pack(side="left", padx=8)

    # ---------------- FOOTER (persistent across all pages) ----------------

    def _build_footer(self, parent):
        footer = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=14,
                               border_width=1, border_color=CARD_BORDER)
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 18))

        top_row = ctk.CTkFrame(footer, fg_color="transparent")
        top_row.pack(fill="x", padx=18, pady=(14, 8))

        self.status_dot = ctk.CTkLabel(top_row, text="\u25cf", text_color=STATUS_COLORS["disconnected"],
                                        font=("TkDefaultFont", 14), width=16)
        self.status_dot.pack(side="left", padx=(0, 4))

        self.status_label_var = tk.StringVar(value="Idle")
        ctk.CTkLabel(top_row, textvariable=self.status_label_var,
                     font=("TkDefaultFont", 12, "bold")).pack(side="left")

        self.progress = ctk.CTkProgressBar(top_row, mode="determinate", width=160)
        self.progress.pack(side="left", padx=16)
        self.progress.set(0)

        ctk.CTkButton(top_row, text="Clear log", width=90, fg_color="#e9e7e2",
                      hover_color="#dcdad4", text_color=TEXT_DARK,
                      command=self._clear_log).pack(side="right")
        self.send_btn = ctk.CTkButton(top_row, text="Write with robot", fg_color=ACCENT,
                                       hover_color=ACCENT_HOVER, command=self._on_send)
        self.send_btn.pack(side="right", padx=6)
        self.preview_btn = ctk.CTkButton(top_row, text="Save file only (no robot needed)",
                                          fg_color="#e9e7e2", hover_color="#dcdad4",
                                          text_color=TEXT_DARK, command=self._on_preview)
        self.preview_btn.pack(side="right", padx=6)

        log_wrap = ctk.CTkFrame(footer, fg_color="#1e1e1e", corner_radius=10)
        log_wrap.pack(fill="both", expand=True, padx=18, pady=(2, 16))

        self.log_text = tk.Text(
            log_wrap, height=8, state="disabled", wrap="word",
            bg="#1e1e1e", fg="#e8e6e1", insertbackground="#e8e6e1",
            font=("Menlo", 12) if sys.platform == "darwin" else ("Consolas", 11),
            padx=12, pady=10, relief="flat", borderwidth=0, highlightthickness=0)
        log_scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True, padx=(2, 0), pady=2)
        log_scroll.pack(side="right", fill="y")

        self.log_text.tag_configure("error", foreground="#ff6b6b")
        self.log_text.tag_configure("warning", foreground="#e8b84b")
        self.log_text.tag_configure("success", foreground="#6bcf7f")
        self.log_text.tag_configure("info", foreground="#7fb8e8")
        self.log_text.tag_configure("normal", foreground="#e8e6e1")

    # ---------------------------------------------------------- UI logic ---

    def _update_source_mode(self):
        mode = self.source_mode.get()
        if mode == "type":
            self.text_box.configure(state="normal")
        else:
            self.text_box.configure(state="disabled")

    def _on_engine_change(self):
        """
        Toggles between Hershey mode (Font preset dropdown + related options
        active, RNN neatness/style fields inert) and RNN mode (the reverse).
        EMS Casual Hand and every other Hershey font preset are untouched by
        this -- they simply become inactive while RNN is selected, and
        immediately usable again the moment Engine is switched back.
        """
        is_rnn = self.engine_var.get().startswith("RNN")

        font_state = "disabled" if is_rnn else "normal"
        self.font_preset_combo.configure(state=font_state)
        self.custom_exclusive_chk.configure(state="disabled" if is_rnn else "normal")
        self.print_style_chk.configure(state="disabled" if is_rnn else "normal")
        self.verify_chk.configure(state="normal" if is_rnn else "disabled")

        if not is_rnn:
            # Restore the correct choose-font button enabled/disabled state,
            # which depends on which font preset is currently selected.
            self._on_font_preset_change()
        else:
            self.choose_font_btn.configure(state="disabled")

    def _on_paper_size_change(self):
        choice = self.paper_size_var.get()
        if choice == "Custom (set below)":
            self.page_info_var.set("Using the Advanced page's page width/height/offset values directly.")
            return
        try:
            margin = float(self.margin_var.get())
        except ValueError:
            margin = 15.0
        cfg_bit = paper_size_to_cfg(choice, margin_mm=margin)
        for key, val in cfg_bit.items():
            self.layout_vars[key].set(str(round(val, 1)))
        sheet_w, sheet_h = PAPER_SIZES[choice]
        self.page_info_var.set(
            f"{choice}: {sheet_w:.0f}x{sheet_h:.0f}mm sheet, "
            f"{cfg_bit['page_width_mm']:.0f}x{cfg_bit['page_height_mm']:.0f}mm usable "
            f"with {margin:.0f}mm margins.")

    def _on_font_preset_change(self):
        choice = self.font_preset_var.get()
        preset = FONT_PRESETS.get(choice, {})
        path = preset.get("path")

        if path is None:
            # Built-in cursive/script -- no custom font at all
            self.custom_font_path_var.set("No font file selected")
            self.choose_font_btn.configure(state="disabled")
            self.custom_exclusive_var.set(False)
            self.print_style_var.set(False)
            self.exclude_letters_var.set("")
        elif path == "BROWSE":
            # Let the user pick their own file
            self.choose_font_btn.configure(state="normal")
            if self.custom_font_path_var.get() in ("No font file selected",) or \
               not os.path.exists(self.custom_font_path_var.get()):
                self.custom_font_path_var.set("No font file selected")
            self.custom_exclusive_var.set(True)
        else:
            # A bundled preset font -- fill in its path and recommended settings
            self.custom_font_path_var.set(path)
            self.choose_font_btn.configure(state="disabled")
            self.custom_exclusive_var.set(preset.get("exclusive", True))
            self.print_style_var.set(preset.get("print_style", False))
            self.exclude_letters_var.set(preset.get("exclude_letters", ""))
            if not os.path.exists(path):
                self._log(f"Warning: expected font file not found at {path}\n"
                          f"Make sure the 'fonts' folder is next to handwriting_gui.py.\n")

    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose a document",
            filetypes=[("Documents", "*.txt *.docx *.pdf"), ("All files", "*.*")])
        if path:
            self.file_path_var.set(path)

    def _choose_font(self):
        path = filedialog.askopenfilename(
            title="Choose a font JSON file",
            filetypes=[("Font JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.custom_font_path_var.set(path)

    def _refresh_ports(self):
        ports = []
        if list_ports is not None:
            ports = [p.device for p in list_ports.comports()]
        if ports:
            self.port_menu.configure(values=ports)
            if self.port_var.get() not in ports:
                self.port_var.set(ports[0])
        else:
            self.port_menu.configure(values=["No ports found"])
            self.port_var.set("No ports found")
            self._log("No serial ports found. Plug in the DrawCore board and click Refresh.\n")

    # --------------------------------------------------------- Log utils ---

    def _log(self, msg):
        lower = msg.lower()
        if "error" in lower or "traceback" in lower:
            tag = "error"
        elif "warning" in lower or "does not fit" in lower:
            tag = "warning"
        elif "done" in lower or "saved" in lower or "wrote" in lower or "fits comfortably" in lower:
            tag = "success"
        elif msg.startswith("["):
            tag = "info"
        else:
            tag = "normal"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg, tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__ENABLE_NEXT__":
                    self.pen_next_btn.configure(state="normal")
                elif msg == "__DISABLE_NEXT__":
                    self.pen_next_btn.configure(state="disabled")
                elif msg.startswith("__MOTOR_STATUS__:"):
                    self.motor_status_var.set(msg.split(":", 1)[1])
                elif msg.startswith("__CONN_STATUS__:"):
                    _, state, text = msg.split(":", 2)
                    self._apply_conn_status(state, text)
                elif msg.startswith("__WARN_POPUP__:"):
                    messagebox.showwarning("Text may not fit", msg.split(":", 1)[1])
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.preview_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.connect_btn.configure(state=state)

        if busy:
            self.status_label_var.set("Working...")
            self.status_dot.configure(text_color=STATUS_COLORS["connecting"])
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.status_label_var.set("Idle")
            self.status_dot.configure(text_color=STATUS_COLORS["disconnected"])
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(0)

    # --------------------------------------------------------- Connection ---

    def _apply_conn_status(self, state, text):
        self.conn_status_var.set(text)
        self.conn_dot.configure(text_color=STATUS_COLORS.get(state, STATUS_COLORS["disconnected"]))
        self.connect_btn.configure(text="Disconnect" if state == "connected" else "Connect")

    def _require_connected(self):
        if not self.connected or self.ser is None:
            messagebox.showwarning("Not connected",
                                    "Connect to the robot first using the connection bar at the top.")
            return False
        return True

    def _on_connect_toggle(self):
        if self.busy:
            return
        if self.connected:
            self._disconnect()
            return

        port = self.port_var.get()
        if not port or port == "No ports found":
            messagebox.showwarning("No port selected", "Choose a serial port first.")
            return
        baud = int(self.adv_vars["baud"].get() or 115200)

        def worker():
            self._set_busy(True)
            self.log_queue.put(f"__CONN_STATUS__:connecting:Connecting to {port}...")
            try:
                import serial
                ser = serial.Serial(port, baud, timeout=5)
                time.sleep(2)  # let the board's USB-CDC settle
                ser.reset_input_buffer()
                self.ser = ser
                self.connected = True
                self.log_queue.put(f"__CONN_STATUS__:connected:Connected -- {port}")
                self.log_queue.put(f"Connected to {port} @ {baud} baud.\n")
            except Exception as e:
                self.ser = None
                self.connected = False
                self.log_queue.put("__CONN_STATUS__:error:Connection failed")
                self.log_queue.put(f"ERROR: could not connect -- {e}\n")
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _disconnect(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.connected = False
        self._apply_conn_status("disconnected", "Disconnected")
        self.log_queue.put("Disconnected.\n")

    def _on_close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.destroy()

    # ------------------------------------------------------- Config build ---

    def _build_cfg(self):
        cfg = dict(EBB_DEFAULTS)
        if self.print_style_var.get():
            cfg.update(PRINT_STYLE_OVERRIDES)

        for key, var in self.adv_vars.items():
            raw = var.get().strip()
            if raw == "":
                continue
            try:
                cfg[key] = float(raw) if "." in raw else int(raw)
            except ValueError:
                pass

        for key, var in self.layout_vars.items():
            raw = var.get().strip()
            if raw == "":
                continue
            try:
                cfg[key] = float(raw)
            except ValueError:
                pass

        cfg["flip_x"] = self.flip_x_var.get()
        cfg["flip_y"] = self.flip_y_var.get()
        cfg["reverse_line_direction"] = self.reverse_line_var.get()

        if self.font_preset_var.get() != "Built-in cursive/script":
            path = self.custom_font_path_var.get()
            if path and path != "No font file selected":
                cfg["custom_font_path"] = path
                cfg["custom_font_exclusive"] = self.custom_exclusive_var.get()
                exclude = self.exclude_letters_var.get().strip()
                if exclude:
                    cfg["custom_font_exclude_letters"] = exclude

        # Engine selection + RNN-specific parameters. Prefixed with "_" since
        # these are consumed directly by _generate() rather than being part
        # of the numeric layout/calibration fields the loops above expect.
        cfg["_engine"] = self.engine_var.get()
        try:
            cfg["_rnn_bias"] = float(self.rnn_bias_var.get())
        except ValueError:
            cfg["_rnn_bias"] = 0.75
        try:
            cfg["_rnn_style"] = int(self.rnn_style_var.get())
        except ValueError:
            cfg["_rnn_style"] = 9

        return cfg

    def _get_text_or_blocks(self):
        """Returns (text, blocks) -- exactly one will be non-None."""
        if self.source_mode.get() == "type":
            text = self.text_box.get("1.0", "end").strip()
            if not text:
                raise ValueError("Please type some text first.")
            return text, None

        path = self.file_path_var.get()
        if not path or path == "No file selected":
            raise ValueError("Please choose a file first.")

        if self.preserve_structure_var.get():
            blocks = extract_blocks(path)
            return None, blocks
        else:
            text = clean_text(extract_text(path))
            return text, None

    def _get_seed(self):
        raw = self.seed_var.get().strip()
        if raw == "":
            import random
            return random.randrange(1_000_000)
        try:
            return int(raw)
        except ValueError:
            return None

    # ------------------------------------------------------------ Actions ---

    def _pen_test_next(self):
        self.pen_next_btn.configure(state="disabled")
        self.pen_test_event.set()

    def _wait_for_next_click(self, message):
        """Called from the worker thread. Enables the Next button, shows a
        message, and blocks (this background thread only) until the user
        clicks it -- same idea as the terminal's 'Press Enter to continue',
        but safe to use from a GUI thread."""
        self.pen_test_event.clear()
        self.log_queue.put(message)
        self.log_queue.put("__ENABLE_NEXT__")
        self.pen_test_event.wait()

    def _gui_test_pen(self):
        """
        GUI-safe version of handwriting_ebb.test_pen(), using the already-
        open persistent connection (self.ser) instead of opening its own.
        The original CLI version pauses with input("Press Enter...")
        between toggles, which works in a terminal but hangs forever when
        run from a GUI thread -- this waits for a click on the "Next
        step ->" button instead, so you control the pace and can
        physically check the pen before moving on.
        """
        ser = self.ser
        ser.reset_input_buffer()
        for val in (0, 1, 0, 1):
            self.log_queue.put(f"Sending SP,{val} -- watch/check the pen now.\n")
            ser.write(f"SP,{val}\r".encode())
            time.sleep(0.3)
            resp = ser.readline().decode(errors="replace").strip()
            self.log_queue.put(f"  -> {resp}\n")
            self._wait_for_next_click(
                "  Check the pen, then click 'Next step ->' when ready to continue...\n")

        self.log_queue.put(
            "\nBased on what you observed: whichever value lifted the pen is your "
            "pen up value, and whichever lowered it is your pen down value. Set "
            "these in the Advanced page if they're not already correct (0=up, "
            "1=down is the default assumption).\n"
        )

    def _ebb_send(self, ser, cmd, timeout=5):
        """Send one EBB command, wait for its response (OK or an error), return the raw lines."""
        ser.write((cmd + "\r").encode())
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                lines.append(line)
                if line.upper().startswith("OK") or "err" in line.lower():
                    break
        return lines

    def _ebb_write_no_wait(self, ser, cmd):
        """
        Sends a command WITHOUT reading its response at all. Used for move
        commands (XM/SM) during fast homing, where we don't care about the
        move's own OK acknowledgment -- we only care about the switch state.
        Skipping this read removes a full command round-trip of latency
        from every chunk. The response is left unread in the input buffer;
        see _ebb_read_switch's reset_input_buffer() call, which discards it
        before the next real (switch) read so it never gets confused for
        that read's actual response.
        """
        ser.write((cmd + "\r").encode())

    def _ebb_query_position(self, ser):
        """Query current raw axis step counts via QS. Returns (axis1, axis2) or (None, None)."""
        lines = self._ebb_send(ser, "QS")
        for line in lines:
            parts = line.replace(" ", "").split(",")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    continue
        return None, None

    def _ebb_read_switch(self, ser, port, pin, flush_first=True):
        """
        Send one PI query for a limit switch pin. Returns '0'/'1' or None on
        a bad response.

        flush_first: discards anything already sitting unread in the input
        buffer before sending the query -- specifically the leftover OK
        acknowledgment from a prior _ebb_write_no_wait() move command that
        was never read. Without this, that stale OK could be mistaken for
        this query's response (see _ebb_write_no_wait's docstring). Always
        True during fast homing; safe to leave True elsewhere too since
        there's normally nothing pending to flush.
        """
        if flush_first:
            ser.reset_input_buffer()
        lines = self._ebb_send(ser, f"PI,{port},{pin}", timeout=2)
        for line in lines:
            if line.startswith("PI,"):
                return line.split(",")[1]
        return None

    def _ebb_switch_debounced(self, ser, port, pin, confirm_count=3, poll_interval_s=0.005,
                               triggered_value="1"):
        """Requires confirm_count consecutive reads matching triggered_value before
        treating the switch as actually triggered -- rejects mechanical bounce/noise
        on a single-read basis. triggered_value lets active-low sensors (like the
        hall-effect switch, which reads '0' when triggered) share this same helper
        with the active-high mechanical switches (which read '1' when triggered)."""
        for _ in range(confirm_count):
            if self._ebb_read_switch(ser, port, pin) != triggered_value:
                return False
            time.sleep(poll_interval_s)
        return True

    def _run_motor_action(self, action_name, fn):
        """Shared wrapper: run fn(ser) on the persistent connection, report success/failure."""
        if self.busy:
            return
        if not self._require_connected():
            return

        def worker():
            self._set_busy(True)
            try:
                self.log_queue.put(f"{action_name}...\n")
                fn(self.ser)
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}\n")
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_enable_motors(self):
        def fn(ser):
            self._ebb_send(ser, "EM,1,1")
            self.log_queue.put("Motors enabled -- they'll now hold position and resist being moved by hand.\n")
            self.log_queue.put("__MOTOR_STATUS__:Motors: enabled")
        self._run_motor_action("Enabling motors", fn)

    def _on_disable_motors(self):
        def fn(ser):
            self._ebb_send(ser, "EM,0,0")
            self.log_queue.put("Motors disabled -- the carriage can now be moved freely by hand.\n")
            self.log_queue.put("__MOTOR_STATUS__:Motors: disabled (free)")
        self._run_motor_action("Disabling motors", fn)

    def _on_go_home(self):
        def fn(ser):
            self._ebb_send(ser, "EM,1,1")  # motors must be enabled to move
            self.log_queue.put("__MOTOR_STATUS__:Motors: enabled")
            a, b = self._ebb_query_position(ser)
            if a is None:
                self.log_queue.put("Could not read current position (QS query failed).\n")
                return
            if a == 0 and b == 0:
                self.log_queue.put("Already at home (0,0) -- no move needed.\n")
                return
            dx, dy = -a, -b
            duration_ms = max(400, min(8000, int(max(abs(dx), abs(dy)) * 2)))
            self.log_queue.put(f"Current position: axis1={a}, axis2={b}. Returning to home...\n")
            self._ebb_send(ser, f"SM,{duration_ms},{dx},{dy}")
            time.sleep(duration_ms / 1000.0 + 0.3)
            a2, b2 = self._ebb_query_position(ser)
            self.log_queue.put(f"Done. Position is now axis1={a2}, axis2={b2}.\n")
        self._run_motor_action("Going to home position", fn)

    def _on_home_switches(self):
        """
        Physical homing using the three confirmed limit switches:
            B5 = X minimum, B6 = X maximum, B7 = Y reference (hall-effect).
        Drives toward B6 (X-max) and B7 (Y) -- the "home corner" -- using
        XM moves (same CoreXY-aware command handwriting_ebb.py uses for
        real drawing, so this matches production motion exactly). Once
        both switches confirm a debounced trigger, zeroes the position
        with CS so (0,0) corresponds to that physical corner.

        Firmware note: EBB v2.6.5 has no automatic hardware limit-stop
        (that's a v3.0 feature), so this polls PI in software and issues
        ES the instant a press is detected.

        SPEED: uses a two-phase seek, the same technique real CNC/3D-printer
        firmware uses for homing -- a fast, PIPELINED bulk move (several XM
        chunks queued back-to-back with zero waiting in between, so the
        board's own move queue chains them into continuous motion) to find
        the switch roughly, then a slow, fire-and-forget precise re-approach
        for an accurate debounced stop.
        """
        HOME_DX_SIGN = 1          # toward X-MAX (B6). Confirmed correct.
        HOME_DY_SIGN = 1          # toward Y (B7). Confirmed correct (flipped from -1).

        # Fast seek phase: queues several small XM moves back-to-back with
        # ZERO waiting in between -- the EBB's own onboard move queue
        # chains and executes them itself, so the carriage moves
        # continuously through a whole batch with no serial round-trip
        # gaps stitched into the motion. The switch is only checked once
        # per BATCH, not once per chunk.
        #
        # X (mechanical switch) can safely use a big batch: once the
        # carriage hits a hard physical stop it STAYS triggered, so even a
        # late check still finds it triggered.
        #
        # Y (hall-effect sensor) CANNOT use a big batch: it only reads
        # triggered while the magnet is within its narrow detection zone
        # (a few mm). At the X-axis speed/batch size, one batch covers
        # ~36mm in ~400ms -- the carriage can sail straight through that
        # narrow zone and out the other side within a single blind batch,
        # so the check after the batch finds it already back to idle and
        # the trigger moment is never actually seen. Y therefore uses much
        # smaller steps AND checks every single chunk (batch size 1) so it
        # can't blow past a narrow zone unnoticed.
        FAST_STEP = 360           # ~4.5mm per chunk at steps_per_mm=80 (X)
        FAST_CHUNK_MS = 50        # -> exactly 90mm/s per chunk (X)
        BATCH_SIZE = 8            # chunks queued per batch -> ~36mm, ~400ms of continuous motion (X)
        MAX_FAST_BATCHES = 100    # safety cutoff -- ~100*36mm = 3600mm of travel (X)

        FAST_STEP_Y = FAST_STEP       # same ~4.5mm chunk size as X now -- the actual fix for
                                       # the missed-trigger issue was BATCH_SIZE_Y=1 (checking
                                       # after every chunk, no blind travel across a whole batch),
                                       # not the chunk size itself. Shrinking the step too was
                                       # unnecessary and is why Y looked slower than X at the start.
        FAST_CHUNK_MS_Y = FAST_CHUNK_MS  # -> same 90mm/s as X
        BATCH_SIZE_Y = 1              # check after EVERY chunk -- no blind travel across multiple chunks
        MAX_FAST_BATCHES_Y = 800      # ~800*4.5mm = 3600mm of travel, matching X's range

        # Back-off distance after the fast phase first sees the switch,
        # before the slow precise re-approach. Must comfortably exceed one
        # FULL BATCH's worth of possible overshoot for that axis.
        BACKOFF_STEPS = 3200      # ~40mm (X, matches the bigger X batch)
        BACKOFF_STEPS_Y = 500     # ~6.25mm -- only needs to exceed ONE chunk's overshoot now
                                   # (~4.5mm), since batch_size=1 means the worst-case blind
                                   # travel per check is just one chunk, not a whole batch.

        # Slow precise approach: fire-and-forget, no waiting for XM's OK,
        # with a debounced switch check between steps.
        SLOW_STEP = 30            # ~0.375mm per chunk
        SLOW_CHUNK_MS = 25        # -> ~15mm/s (far slower than the 90mm/s fast phase)
        MAX_SLOW_CHUNKS = 300     # ~300*0.375mm = 112.5mm -- covers the backoff distance
        CONFIRM_COUNT = 2

        def home_one_axis(ser, label, dx_sign, dy_sign, switch_port, switch_pin,
                           switch_triggered_value="1",
                           fast_step=FAST_STEP, fast_chunk_ms=FAST_CHUNK_MS,
                           batch_size=BATCH_SIZE, max_fast_batches=MAX_FAST_BATCHES,
                           backoff_steps=BACKOFF_STEPS):
            self.log_queue.put(f"Homing {label} (fast seek, pipelined)...\n")
            dx_fast, dy_fast = dx_sign * fast_step, dy_sign * fast_step

            found = False
            for _ in range(max_fast_batches):
                # Check BEFORE moving, so an already-triggered switch is
                # caught with zero movement (e.g. if homing is re-run while
                # already sitting at the switch).
                if self._ebb_read_switch(ser, switch_port, switch_pin) == switch_triggered_value:
                    found = True
                    break
                # Queue the whole batch with back-to-back writes -- no read,
                # no sleep between them. The board's own move queue is what
                # chains these into continuous motion; we only block once,
                # after the whole batch, for the combined duration.
                for _ in range(batch_size):
                    self._ebb_write_no_wait(ser, f"XM,{fast_chunk_ms},{dx_fast},{dy_fast}")
                time.sleep(batch_size * fast_chunk_ms / 1000.0)
            self._ebb_send(ser, "ES")  # stop the fast move immediately either way

            if not found:
                self.log_queue.put(f"WARNING: {label} homing timed out during fast seek -- "
                                   f"switch never triggered. Check wiring, or flip the "
                                   f"direction sign if the carriage moved the wrong way.\n")
                return False

            self.log_queue.put(f"{label} switch found -- backing off for precise re-approach...\n")
            self._ebb_send(ser, "EM,1,1")  # re-enable in case ES halted the motors
            dx_back, dy_back = -dx_sign * backoff_steps, -dy_sign * backoff_steps
            # Duration scaled to backoff_steps' actual distance at this
            # axis's own fast-phase speed (fast_step/fast_chunk_ms ratio) --
            # a fixed short duration here would ask for an impossible speed.
            backoff_duration_ms = max(1, round(backoff_steps / fast_step * fast_chunk_ms))
            self._ebb_send(ser, f"XM,{backoff_duration_ms},{dx_back},{dy_back}", timeout=2)
            time.sleep(backoff_duration_ms / 1000.0 + 0.05)

            dx_slow, dy_slow = dx_sign * SLOW_STEP, dy_sign * SLOW_STEP
            for _ in range(MAX_SLOW_CHUNKS):
                if self._ebb_switch_debounced(ser, switch_port, switch_pin, CONFIRM_COUNT,
                                               triggered_value=switch_triggered_value):
                    self._ebb_send(ser, "ES")
                    self.log_queue.put(f"{label} homed precisely.\n")
                    return True
                self._ebb_write_no_wait(ser, f"XM,{SLOW_CHUNK_MS},{dx_slow},{dy_slow}")
                time.sleep(SLOW_CHUNK_MS / 1000.0)
            self._ebb_send(ser, "ES")
            self.log_queue.put(f"WARNING: {label} slow re-approach timed out -- try increasing "
                               f"BACKOFF_STEPS if this keeps happening.\n")
            return False

        def fn(ser):
            self.log_queue.put("Configuring limit switch pins as inputs...\n")
            for port, pin, _triggered in (SWITCH_X_MIN, SWITCH_X_MAX, SWITCH_Y):
                self._ebb_send(ser, f"PD,{port},{pin},1")

            self._ebb_send(ser, "EM,1,1")
            self.log_queue.put("__MOTOR_STATUS__:Motors: enabled")

            x_port, x_pin, x_triggered = SWITCH_X_MAX
            if not home_one_axis(ser, "X (toward max)", HOME_DX_SIGN, 0, x_port, x_pin,
                                  switch_triggered_value=x_triggered):
                return

            self._ebb_send(ser, "EM,1,1")  # re-enable in case ES left motors halted

            y_port, y_pin, y_triggered = SWITCH_Y
            if not home_one_axis(ser, "Y", 0, HOME_DY_SIGN, y_port, y_pin,
                                  switch_triggered_value=y_triggered,
                                  fast_step=FAST_STEP_Y, fast_chunk_ms=FAST_CHUNK_MS_Y,
                                  batch_size=BATCH_SIZE_Y, max_fast_batches=MAX_FAST_BATCHES_Y,
                                  backoff_steps=BACKOFF_STEPS_Y):
                return

            self._ebb_send(ser, "EM,1,1")
            self._ebb_send(ser, "CS")
            self.log_queue.put("__MOTOR_STATUS__:Motors: homed, origin zeroed")
            self.log_queue.put("Homing complete -- origin (0,0) set at the X-max/Y switch corner.\n")

        self._run_motor_action("Homing to limit switches", fn)

    def _on_test_pen(self):
        if self.busy:
            return
        if not self._require_connected():
            return

        def worker():
            self._set_busy(True)
            try:
                self.log_queue.put("Testing pen -- watch the pen physically...\n")
                self._gui_test_pen()
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}\n")
            finally:
                self.log_queue.put("__DISABLE_NEXT__")
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _get_verifier(self):
        """
        Lazily constructs and caches the HandwritingVerifier -- its model
        weights only need to load once per app session, not once per
        generation. First call is slow (model init); subsequent calls
        reuse the same instance instantly.
        """
        if self._cached_verifier is None:
            self._cached_verifier = HandwritingVerifier()
        return self._cached_verifier

    def _generate(self):
        text, blocks = self._get_text_or_blocks()
        cfg = self._build_cfg()
        seed = self._get_seed()

        self.log_queue.put(f"Using random seed {seed}\n")

        if cfg.get("_engine", "").startswith("RNN"):
            if RNNHandwritingGenerator is None:
                raise RuntimeError(
                    "RNN engine selected, but handwriting_rnn.py (or its "
                    "handwriting-synthesis dependency) isn't importable in this "
                    "environment. Switch Engine back to Hershey, or see "
                    "handwriting_rnn.py's docstring for setup instructions."
                )
            gen = RNNHandwritingGenerator(
                cfg, rng_seed=seed,
                bias=cfg.get("_rnn_bias", 0.75),
                style=cfg.get("_rnn_style", 9))

            if self.verify_var.get():
                if VerifiedRNNHandwritingGenerator is None or HandwritingVerifier is None:
                    raise RuntimeError(
                        "Verify option is checked, but handwriting_verify.py (or its "
                        "easyocr dependency) isn't importable in this environment. "
                        "Uncheck Verify, or run: pip install easyocr"
                    )
                self.log_queue.put("Loading handwriting-recognition model for verification"
                                    " (first use only, may take a moment)...\n")
                verifier = self._get_verifier()
                gen = VerifiedRNNHandwritingGenerator(gen, verifier, similarity_threshold=0.55)

            if blocks is not None:
                strokes = gen.generate_blocks(blocks)
            else:
                strokes = gen.generate(text)

            flagged = getattr(gen, "flagged_lines", None)
            if flagged:
                self.log_queue.put(
                    f"WARNING: {len(flagged)} line(s) did not pass handwriting verification "
                    f"even after retries:\n")
                detail_lines = []
                for f in flagged:
                    line_msg = (f"  expected {f['text']!r}, model read back "
                                f"{f['recognized']!r} (similarity {f['similarity']:.2f})\n")
                    self.log_queue.put(line_msg)
                    detail_lines.append(line_msg.strip())
                self.log_queue.put("__WARN_POPUP__:Some lines did not pass handwriting "
                                    "verification:\n\n" + "\n".join(detail_lines) +
                                    "\n\nYou can still write with the robot, but these "
                                    "specific lines may come out garbled or misread.")
        else:
            gen = HandwritingGenerator(cfg, rng_seed=seed)
            if blocks is not None:
                strokes = gen.generate_blocks(blocks)
            else:
                strokes = gen.generate(text)

        bounds = check_page_bounds(strokes, cfg)
        self.log_queue.put(bounds["message"] + "\n")
        if not bounds["fits"]:
            self.log_queue.put("WARNING: the robot may try to write outside your paper or physical "
                                "travel area. Consider shortening the text, reducing font size, or "
                                "choosing a larger paper size.\n")
            self.log_queue.put("__WARN_POPUP__:" + bounds["message"] +
                                "\n\nConsider shortening the text, reducing the font size, "
                                "or choosing a larger paper size in Page Setup.")

        commands = strokes_to_ebb_commands(strokes, cfg, rng_seed=seed)
        self.log_queue.put(f"Generated {len(strokes)} strokes, {len(commands)} robot commands.\n")
        return commands

    def _on_preview(self):
        if self.busy:
            return

        def worker():
            self._set_busy(True)
            old_stdout = sys.stdout
            sys.stdout = QueueWriter(self.log_queue)
            try:
                commands = self._generate()
                path = filedialog.asksaveasfilename(
                    defaultextension=".ebb",
                    filetypes=[("Robot command file", "*.ebb")],
                    title="Save robot command file")
                if path:
                    with open(path, "w") as f:
                        f.write("\n".join(commands) + "\n")
                    self.log_queue.put(f"Saved to {path}\n")
                else:
                    self.log_queue.put("Save cancelled.\n")
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}\n")
            finally:
                sys.stdout = old_stdout
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _stream_ebb_commands(self, commands, pen_settle_s=0.2):
        """
        Streams commands over the persistent connection (self.ser), mirroring
        handwriting_ebb.stream_ebb()'s protocol exactly but reusing the
        already-open connection instead of opening a new one -- keeps the
        Connect/Disconnect indicator accurate through the whole write.
        """
        ser = self.ser
        ser.reset_input_buffer()
        for i, cmd in enumerate(commands):
            ser.write((cmd + "\r").encode())
            deadline = time.time() + 5
            resp_lines = []
            while time.time() < deadline:
                line = ser.readline().decode(errors="replace").strip()
                if line:
                    resp_lines.append(line)
                    if line.upper().startswith("OK") or "err" in line.lower():
                        break
            resp = " | ".join(resp_lines) if resp_lines else "<no response>"
            self.log_queue.put(f"[{i + 1}/{len(commands)}] {cmd}  ->  {resp}\n")
            if any("err" in r.lower() for r in resp_lines):
                self.log_queue.put(f"!! Board reported an error on: {cmd}\n")

            if cmd.startswith("SP,"):
                time.sleep(pen_settle_s)
            elif cmd.startswith("XM,") or cmd.startswith("SM,"):
                try:
                    duration_ms = int(cmd.split(",")[1])
                    time.sleep(duration_ms / 1000.0)
                except (IndexError, ValueError):
                    pass

    def _on_send(self):
        if self.busy:
            return
        if not self._require_connected():
            return
        if not messagebox.askyesno("Confirm", "The robot will start writing now. Make sure the pen and paper "
                                               "are correctly positioned. Continue?"):
            return

        def worker():
            self._set_busy(True)
            old_stdout = sys.stdout
            sys.stdout = QueueWriter(self.log_queue)
            try:
                commands = self._generate()
                cfg = self._build_cfg()
                pen_settle_s = cfg.get("pen_move_settle_ms", 200) / 1000.0
                self.log_queue.put(f"Sending to {self.port_var.get()}...\n")
                self._stream_ebb_commands(commands, pen_settle_s=pen_settle_s)
                self.log_queue.put("Done!\n")
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}\n")
            finally:
                sys.stdout = old_stdout
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = HandwritingApp()
    app.mainloop()