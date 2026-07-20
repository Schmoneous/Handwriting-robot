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
python-tk`; Windows installers include it by default).

For the RNN engine option, handwriting_rnn.py and its handwriting-synthesis
dependency must also be importable from this same environment -- see
handwriting_rnn.py's docstring for setup. If that dependency isn't
installed, the Hershey engine (including EMS Casual Hand and all other
existing font presets) still works exactly as before; only selecting
"RNN (neural handwriting)" in the Engine dropdown requires it.
"""

import io
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from handwriting_bot import (HandwritingGenerator, DEFAULTS as BASE_DEFAULTS,
                               PRINT_STYLE_OVERRIDES, PAPER_SIZES, paper_size_to_cfg,
                               check_page_bounds)
from handwriting_ebb import EBB_DEFAULTS, strokes_to_ebb_commands, stream_ebb, test_pen as ebb_test_pen
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


class HandwritingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Robot")
        self.geometry("760x760")
        self.minsize(680, 620)

        self.log_queue = queue.Queue()
        self.busy = False
        self.pen_test_event = threading.Event()
        self._cached_verifier = None  # lazy-loaded on first use -- see _get_verifier()

        self._build_ui()
        self._poll_log_queue()
        self._refresh_ports()

    # ---------------------------------------------------------------- UI ---

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Shared page-layout variables, created up front so both the main
        # tab's "Page setup" section and the Advanced tab's fields can bind
        # to the exact same StringVars -- changing one updates the other.
        self.layout_vars = {
            key: tk.StringVar(value=str(EBB_DEFAULTS[key]))
            for key in ("font_size_mm", "page_width_mm", "page_height_mm",
                        "x_offset_mm", "y_offset_mm", "jitter_amp_mm")
        }

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        main_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Write")
        notebook.add(advanced_tab, text="Advanced / Calibration")

        # ---------------- MAIN TAB ----------------

        # --- Text source ---
        src_frame = ttk.LabelFrame(main_tab, text="What to write")
        src_frame.pack(fill="x", **pad)

        self.source_mode = tk.StringVar(value="type")
        ttk.Radiobutton(src_frame, text="Type text", variable=self.source_mode,
                         value="type", command=self._update_source_mode).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(src_frame, text="Load a file (.txt / .docx / .pdf)", variable=self.source_mode,
                         value="file", command=self._update_source_mode).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        self.text_box = tk.Text(src_frame, height=6, wrap="word")
        self.text_box.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=8, pady=4)
        self.text_box.insert("1.0", "Type what you want the robot to write here...")
        src_frame.grid_columnconfigure(2, weight=1)

        file_row = ttk.Frame(src_frame)
        file_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        self.file_path_var = tk.StringVar(value="No file selected")
        ttk.Button(file_row, text="Choose file...", command=self._choose_file).pack(side="left")
        ttk.Label(file_row, textvariable=self.file_path_var).pack(side="left", padx=8)

        self.preserve_structure_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(src_frame, text="Preserve document structure (headings, bullets, numbered lists)",
                         variable=self.preserve_structure_var).grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        self._update_source_mode()

        # --- Style ---
        style_frame = ttk.LabelFrame(main_tab, text="Handwriting style")
        style_frame.pack(fill="x", **pad)

        # --- Engine selector ---
        ttk.Label(style_frame, text="Engine:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.engine_var = tk.StringVar(value="Hershey (font-based)")
        engine_combo = ttk.Combobox(
            style_frame, textvariable=self.engine_var,
            values=["Hershey (font-based)", "RNN (neural handwriting)"],
            state="readonly", width=28)
        engine_combo.grid(row=0, column=1, sticky="w", padx=4, pady=6)
        engine_combo.bind("<<ComboboxSelected>>", lambda e: self._on_engine_change())

        # --- RNN-only controls, disabled unless "RNN (neural handwriting)" is selected ---
        self.rnn_controls_frame = ttk.Frame(style_frame)
        self.rnn_controls_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=28, pady=2)

        ttk.Label(self.rnn_controls_frame, text="Neatness (0=messy, 1=neat):").grid(row=0, column=0, sticky="w")
        self.rnn_bias_var = tk.StringVar(value="0.75")
        ttk.Entry(self.rnn_controls_frame, textvariable=self.rnn_bias_var, width=6).grid(row=0, column=1, padx=(4, 16))

        ttk.Label(self.rnn_controls_frame, text="Style (0-12):").grid(row=0, column=2, sticky="w")
        self.rnn_style_var = tk.StringVar(value="9")
        ttk.Entry(self.rnn_controls_frame, textvariable=self.rnn_style_var, width=6).grid(row=0, column=3, padx=4)

        self.verify_var = tk.BooleanVar(value=False)
        self.verify_chk = ttk.Checkbutton(
            self.rnn_controls_frame,
            text="Verify with recognition model before writing (slower, catches garbled letters)",
            variable=self.verify_var)
        self.verify_chk.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # --- Font preset (Hershey mode only -- EMS Casual Hand lives here, unchanged) ---
        ttk.Label(style_frame, text="Font:").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.font_preset_var = tk.StringVar(value="Built-in cursive/script")
        self.font_preset_combo = ttk.Combobox(
            style_frame, textvariable=self.font_preset_var,
            values=list(FONT_PRESETS.keys()), state="readonly", width=28)
        self.font_preset_combo.grid(row=2, column=1, sticky="w", padx=4, pady=6)
        self.font_preset_combo.bind("<<ComboboxSelected>>", lambda e: self._on_font_preset_change())

        custom_row = ttk.Frame(style_frame)
        custom_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=28, pady=2)
        self.custom_font_path_var = tk.StringVar(value="No font file selected")
        self.choose_font_btn = ttk.Button(custom_row, text="Choose font JSON...", command=self._choose_font)
        self.choose_font_btn.pack(side="left")
        ttk.Label(custom_row, textvariable=self.custom_font_path_var).pack(side="left", padx=8)

        self.custom_exclusive_var = tk.BooleanVar(value=True)
        self.custom_exclusive_chk = ttk.Checkbutton(
            style_frame, text="Use only this font (don't mix with built-in shapes)",
            variable=self.custom_exclusive_var)
        self.custom_exclusive_chk.grid(row=4, column=0, columnspan=3, sticky="w", padx=28)

        self.print_style_var = tk.BooleanVar(value=False)
        self.print_style_chk = ttk.Checkbutton(
            style_frame, text="Print-style font (disable cursive slant, add letter spacing)",
            variable=self.print_style_var)
        self.print_style_chk.grid(row=5, column=0, columnspan=3, sticky="w", padx=28)

        exclude_row = ttk.Frame(style_frame)
        exclude_row.grid(row=6, column=0, columnspan=3, sticky="w", padx=28, pady=(2, 8))
        ttk.Label(exclude_row, text="Letters to always use built-in shapes for (e.g. EJ):").pack(side="left")
        self.exclude_letters_var = tk.StringVar(value="")
        ttk.Entry(exclude_row, textvariable=self.exclude_letters_var, width=12).pack(side="left", padx=6)

        self._on_font_preset_change()
        self._on_engine_change()

        # --- Page setup ---
        page_frame = ttk.LabelFrame(main_tab, text="Page setup")
        page_frame.pack(fill="x", **pad)

        ttk.Label(page_frame, text="Paper size:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.paper_size_var = tk.StringVar(value="Letter (8.5x11in)")
        paper_combo = ttk.Combobox(
            page_frame, textvariable=self.paper_size_var,
            values=list(PAPER_SIZES.keys()) + ["Custom (set below)"],
            state="readonly", width=20)
        paper_combo.grid(row=0, column=1, sticky="w", padx=4)
        paper_combo.bind("<<ComboboxSelected>>", lambda e: self._on_paper_size_change())

        ttk.Label(page_frame, text="Margin (mm):").grid(row=0, column=2, sticky="w", padx=(16, 4))
        self.margin_var = tk.StringVar(value="15")
        margin_entry = ttk.Entry(page_frame, textvariable=self.margin_var, width=6)
        margin_entry.grid(row=0, column=3, sticky="w")
        margin_entry.bind("<FocusOut>", lambda e: self._on_paper_size_change())

        ttk.Label(page_frame, text="Font size (mm):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(page_frame, textvariable=self.layout_vars["font_size_mm"], width=8).grid(
            row=1, column=1, sticky="w", padx=4)

        self.page_info_var = tk.StringVar(value="")
        ttk.Label(page_frame, textvariable=self.page_info_var, foreground="#8a8378").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        self._on_paper_size_change()

        # --- Connection ---
        conn_frame = ttk.LabelFrame(main_tab, text="Robot connection")
        conn_frame.pack(fill="x", **pad)

        ttk.Label(conn_frame, text="Serial port:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=30, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(conn_frame, text="Refresh", command=self._refresh_ports).grid(row=0, column=2, padx=6)

        pen_test_row = ttk.Frame(conn_frame)
        pen_test_row.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        ttk.Button(pen_test_row, text="Test pen up/down", command=self._on_test_pen).pack(side="left")
        self.pen_next_btn = ttk.Button(pen_test_row, text="Next step ->", command=self._pen_test_next, state="disabled")
        self.pen_next_btn.pack(side="left", padx=8)

        motor_row = ttk.Frame(conn_frame)
        motor_row.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        ttk.Button(motor_row, text="Enable motors", command=self._on_enable_motors).pack(side="left")
        ttk.Button(motor_row, text="Disable motors (free to move by hand)",
                   command=self._on_disable_motors).pack(side="left", padx=8)
        ttk.Button(motor_row, text="Go to Home (0,0)", command=self._on_go_home).pack(side="left", padx=8)

        self.motor_status_var = tk.StringVar(value="Motors: unknown")
        ttk.Label(conn_frame, textvariable=self.motor_status_var, foreground="#8a8378").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        # --- Actions ---
        action_frame = ttk.Frame(main_tab)
        action_frame.pack(fill="x", **pad)
        self.preview_btn = ttk.Button(action_frame, text="Save file only (no robot needed)", command=self._on_preview)
        self.preview_btn.pack(side="left", padx=4)
        self.send_btn = ttk.Button(action_frame, text="Write with robot", command=self._on_send)
        self.send_btn.pack(side="left", padx=4)

        # --- Log ---
        log_frame = ttk.LabelFrame(main_tab, text="Status")
        log_frame.pack(fill="both", expand=True, **pad)

        status_row = ttk.Frame(log_frame)
        status_row.pack(fill="x", padx=6, pady=(6, 2))

        self.status_dot = tk.Canvas(status_row, width=14, height=14, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(2, 6))
        self.status_dot_id = self.status_dot.create_oval(2, 2, 12, 12, fill="#8a8378", outline="")

        self.status_label_var = tk.StringVar(value="Idle")
        ttk.Label(status_row, textvariable=self.status_label_var,
                  font=("TkDefaultFont", 11, "bold")).pack(side="left")

        self.progress = ttk.Progressbar(status_row, mode="indeterminate", length=150)
        self.progress.pack(side="left", padx=16)

        ttk.Button(status_row, text="Clear log", command=self._clear_log).pack(side="right")

        log_body = tk.Frame(log_frame, bg="#1e1e1e")
        log_body.pack(fill="both", expand=True, padx=6, pady=(2, 6))

        self.log_text = tk.Text(
            log_body, height=14, state="disabled", wrap="word",
            bg="#1e1e1e", fg="#e8e6e1", insertbackground="#e8e6e1",
            font=("Menlo", 12) if sys.platform == "darwin" else ("Consolas", 11),
            padx=10, pady=8, relief="flat", borderwidth=0)
        log_scroll = ttk.Scrollbar(log_body, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.log_text.tag_configure("error", foreground="#ff6b6b")
        self.log_text.tag_configure("warning", foreground="#e8b84b")
        self.log_text.tag_configure("success", foreground="#6bcf7f")
        self.log_text.tag_configure("info", foreground="#7fb8e8")
        self.log_text.tag_configure("normal", foreground="#e8e6e1")

        # ---------------- ADVANCED TAB ----------------

        cal_frame = ttk.LabelFrame(advanced_tab, text="Calibration (pre-filled with known-working values -- "
                                                        "only change these for different hardware)")
        cal_frame.pack(fill="x", padx=10, pady=10)

        self.adv_vars = {}

        def add_field(row, key, label, default, width=10):
            ttk.Label(cal_frame, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            var = tk.StringVar(value=str(default))
            ttk.Entry(cal_frame, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=4, pady=4)
            self.adv_vars[key] = var

        add_field(0, "steps_per_mm", "Steps per mm:", EBB_DEFAULTS["steps_per_mm"])
        add_field(1, "pen_up_value", "Pen UP value (SP,?):", EBB_DEFAULTS["pen_up_value"])
        add_field(2, "pen_down_value", "Pen DOWN value (SP,?):", EBB_DEFAULTS["pen_down_value"])
        add_field(3, "pen_move_settle_ms", "Pen settle time (ms):", EBB_DEFAULTS["pen_move_settle_ms"])
        add_field(4, "baud", "Baud rate:", EBB_DEFAULTS.get("baud", 115200))

        self.flip_x_var = tk.BooleanVar(value=EBB_DEFAULTS["flip_x"])
        self.flip_y_var = tk.BooleanVar(value=EBB_DEFAULTS["flip_y"])
        self.reverse_line_var = tk.BooleanVar(value=EBB_DEFAULTS["reverse_line_direction"])
        ttk.Checkbutton(cal_frame, text="Flip X (if writing comes out mirrored left-right)",
                         variable=self.flip_x_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(cal_frame, text="Flip Y (if letter shapes come out mirrored top-bottom)",
                         variable=self.flip_y_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(cal_frame, text="Reverse line direction (if new lines appear above the previous line)",
                         variable=self.reverse_line_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        layout_frame = ttk.LabelFrame(advanced_tab, text="Page layout "
                                       "(also settable from the Write tab's Page Setup section)")
        layout_frame.pack(fill="x", padx=10, pady=10)

        def add_layout_field(row, col, key, label, default, width=8):
            ttk.Label(layout_frame, text=label).grid(row=row, column=col * 2, sticky="w", padx=8, pady=4)
            var = self.layout_vars[key]  # created in _build_ui, shared with the Write tab
            ttk.Entry(layout_frame, textvariable=var, width=width).grid(row=row, column=col * 2 + 1, sticky="w", padx=4, pady=4)

        add_layout_field(0, 0, "font_size_mm", "Font size (mm):", EBB_DEFAULTS["font_size_mm"])
        add_layout_field(0, 1, "page_width_mm", "Page width (mm):", EBB_DEFAULTS["page_width_mm"])
        add_layout_field(1, 0, "page_height_mm", "Page height (mm):", EBB_DEFAULTS["page_height_mm"])
        add_layout_field(1, 1, "x_offset_mm", "X offset (mm):", EBB_DEFAULTS["x_offset_mm"])
        add_layout_field(2, 0, "y_offset_mm", "Y offset (mm):", EBB_DEFAULTS["y_offset_mm"])
        add_layout_field(2, 1, "jitter_amp_mm", "Jitter amount (mm):", EBB_DEFAULTS["jitter_amp_mm"])

        seed_frame = ttk.LabelFrame(advanced_tab, text="Reproducibility")
        seed_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(seed_frame, text="Random seed (blank = random each time):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=12).grid(row=0, column=1, sticky="w", padx=4)

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

        font_state = "disabled" if is_rnn else "readonly"
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
            self.page_info_var.set("Using the Advanced tab's page width/height/offset values directly.")
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
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        if not ports:
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

        if busy:
            self.status_label_var.set("Working...")
            self.status_dot.itemconfig(self.status_dot_id, fill="#e8b84b")
            self.progress.start(12)
        else:
            self.status_label_var.set("Idle")
            self.status_dot.itemconfig(self.status_dot_id, fill="#8a8378")
            self.progress.stop()

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

    def _gui_test_pen(self, port, baud):
        """
        GUI-safe version of handwriting_ebb.test_pen(). The original CLI
        version pauses with input("Press Enter...") between toggles, which
        works in a terminal but hangs forever when run from a GUI thread
        (there's no terminal to type Enter into). This waits for a click on
        the "Next step ->" button instead, so you control the pace and can
        physically check the pen before moving on.
        """
        import serial

        with serial.Serial(port, baud, timeout=5) as ser:
            time.sleep(2)
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
            "these in the Advanced tab if they're not already correct (0=up, "
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

    def _run_motor_action(self, action_name, fn):
        """Shared wrapper: open the selected serial port, run fn(ser), report success/failure."""
        if self.busy:
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No port selected", "Choose a serial port first.")
            return
        baud = int(self.adv_vars["baud"].get() or 115200)

        def worker():
            self._set_busy(True)
            try:
                import serial
                self.log_queue.put(f"{action_name}...\n")
                with serial.Serial(port, baud, timeout=5) as ser:
                    time.sleep(2)
                    ser.reset_input_buffer()
                    fn(ser)
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

    def _on_test_pen(self):
        if self.busy:
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No port selected", "Choose a serial port first.")
            return
        baud = int(self.adv_vars["baud"].get() or 115200)

        def worker():
            self._set_busy(True)
            try:
                self.log_queue.put(f"Testing pen on {port} -- watch the pen physically...\n")
                self._gui_test_pen(port, baud)
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
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_send(self):
        if self.busy:
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No port selected", "Choose a serial port first.")
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
                baud = int(self.adv_vars["baud"].get() or 115200)
                pen_settle_s = cfg.get("pen_move_settle_ms", 200) / 1000.0
                self.log_queue.put(f"Sending to {port}...\n")
                stream_ebb(commands, port, baud, pen_settle_s=pen_settle_s)
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