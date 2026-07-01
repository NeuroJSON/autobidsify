"""
main.py  (Full pipeline)
========================

AutoBIDSify Full Desktop - Tkinter edition.

Runs the COMPLETE pipeline (ingest -> validate) locally. The user brings their
own AI (OpenAI / Ollama local / Ollama remote / DashScope); no AI is bundled,
and API keys are never written to disk. Replicates the HTML mock-up layout:
data paths, modality, AI engine tabs, dataset details, a 6-stage progress
strip, a live log, and a result line.

The pipeline runs on a background thread; log lines reach the UI via a queue
polled on the UI thread (thread-safe Tkinter pattern).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog

import worker


DARK = {
    "bg": "#0e1116", "panel": "#161b22", "panel2": "#1b212b", "line": "#262d39",
    "txt": "#e6edf3", "txt_dim": "#8b949e", "txt_faint": "#5c6571",
    "accent": "#4ea1ff", "green": "#3fb950", "amber": "#d29922", "red": "#f85149",
    "term_bg": "#0a0d12", "term_txt": "#7ee787", "run_txt": "#06101e",
}
LIGHT = {
    "bg": "#f4f6fa", "panel": "#ffffff", "panel2": "#f0f3f8", "line": "#dfe4ec",
    "txt": "#1b2230", "txt_dim": "#5a6473", "txt_faint": "#97a0ae",
    "accent": "#2563eb", "green": "#1a8a3a", "amber": "#9a6700", "red": "#cf222e",
    "term_bg": "#10151c", "term_txt": "#7ee787", "run_txt": "#ffffff",
}

MODALITIES = ["mri", "nirs", "eeg", "mixed"]
STAGES = ["ingest", "evidence", "trio", "plan", "execute", "validate"]

# engine key -> (label, needs_key, needs_url, model choices)
ENGINES = {
    "openai": ("OpenAI", True, False,
               ["gpt-4o", "gpt-4o-mini", "gpt-5.1"]),
    "ollama-local": ("Ollama - local", False, True,
                     ["qwen3-coder-next:latest", "qwen3-coder-careful:latest",
                      "qwen2.5-coder:7b"]),
    "ollama-remote": ("Ollama - remote", False, True,
                      ["qwen3-coder-next:latest", "qwen3-coder-careful:latest",
                       "qwen2.5-coder:7b"]),
    "dashscope": ("DashScope", True, False,
                  ["qwen-max", "qwen-plus", "qwen-turbo"]),
}

_WIN = sys.platform.startswith("win")
MONO = ("Consolas", 10) if _WIN else ("DejaVu Sans Mono", 10)
SANS = ("Segoe UI", 10) if _WIN else ("DejaVu Sans", 10)


class App:
    _current_state = "idle"

    def __init__(self, root):
        self.root = root
        self.theme = LIGHT
        self.theme_name = "light"
        self.input_dir: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.modality = tk.StringVar(value="mri")
        self.engine = "openai"
        self.running = False
        self.log_queue = queue.Queue()

        self._headers = []
        self._mod_btns = {}
        self._ai_tabs = {}
        self._stage_labels = {}
        self._ids_btns = {}

        root.title("AutoBIDSify - Full")
        root.geometry("1180x1200")
        root.minsize(1000, 900)
        root.resizable(True, True)

        self._build_ui()
        self._apply_theme()
        self._select_engine("openai")
        self._refresh_run_enabled()
        self.root.after(80, self._drain_log_queue)

    # ---------------- UI ----------------
    def _build_ui(self):
        # Outer scrollable container so the whole UI can scroll when the window
        # is shorter than the content (e.g. on Linux/HiDPI).
        outer = tk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        self._outer = outer
        self.canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._vscroll = tk.Scrollbar(outer, orient="vertical",
                                     command=self.canvas.yview)
        self._vscroll.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self._vscroll.set)

        self.main = tk.Frame(self.canvas)
        self._main_window = self.canvas.create_window((0, 0), window=self.main,
                                                      anchor="nw")

        def _on_main_config(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.main.bind("<Configure>", _on_main_config)

        def _on_canvas_config(event):
            # keep inner frame the same width as the canvas
            self.canvas.itemconfigure(self._main_window, width=event.width)
        self.canvas.bind("<Configure>", _on_canvas_config)

        # Mouse-wheel scrolling (Windows/macOS use <MouseWheel>, Linux uses
        # Button-4/5).
        def _on_wheel(event):
            if event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")
            else:
                self.canvas.yview_scroll(int(-1 * (event.delta / 40)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_wheel)
        self.canvas.bind_all("<Button-4>", _on_wheel)
        self.canvas.bind_all("<Button-5>", _on_wheel)

        # title bar
        tb = tk.Frame(self.main); tb.pack(fill="x", padx=20, pady=(16, 8))
        self.lbl_name = tk.Label(tb, text="AutoBIDSify",
                                 font=("Segoe UI", 15, "bold")); self.lbl_name.pack(side="left")
        self.lbl_tag = tk.Label(tb, text="  full pipeline", font=(SANS[0], 9))
        self.lbl_tag.pack(side="left")
        self.btn_theme = tk.Button(tb, text="\u263e", width=3, relief="flat",
                                   cursor="hand2", command=self._toggle_theme)
        self.btn_theme.pack(side="right")
        self.lbl_sub = tk.Label(tb, text="any data -> BIDS \u00b7 runs locally \u00b7 bring your own AI",
                                font=(MONO[0], 8)); self.lbl_sub.pack(side="right", padx=10)
        self._titlebar = tb

        # i. Data paths
        self._section_header("i.", "Data Paths", "")
        self.input_field, self.btn_input = self._path_row(
            "Select input dataset folder...", self._pick_input)
        self.output_field, self.btn_output = self._path_row(
            "Select output folder...", self._pick_output)

        # ii. Modality
        self._section_header("ii.", "Modality", "")
        modrow = tk.Frame(self.main); modrow.pack(fill="x", padx=20, pady=(0, 2))
        self._modrow = modrow
        for m in MODALITIES:
            b = tk.Button(modrow, text=m.upper(), relief="flat", cursor="hand2",
                          padx=12, pady=6, command=lambda mm=m: self._set_modality(mm))
            b.pack(side="left", padx=(0, 6))
            self._mod_btns[m] = b

        # iii. AI engine
        self._section_header("iii.", "AI Engine", "used in trio & plan stages")
        tabrow = tk.Frame(self.main); tabrow.pack(fill="x", padx=20, pady=(0, 2))
        self._tabrow = tabrow
        for key, (label, *_rest) in ENGINES.items():
            b = tk.Button(tabrow, text=label, relief="flat", cursor="hand2",
                          padx=12, pady=6, command=lambda k=key: self._select_engine(k))
            b.pack(side="left", padx=(0, 6))
            self._ai_tabs[key] = b

        self.ai_panel = tk.Frame(self.main)
        self.ai_panel.pack(fill="x", padx=20, pady=(0, 2))
        self.ai_field_label = tk.Label(self.ai_panel, text="", font=(SANS[0], 9),
                                       anchor="w"); self.ai_field_label.pack(fill="x")
        self.ai_entry_var = tk.StringVar()
        self.ai_entry = tk.Entry(self.ai_panel, textvariable=self.ai_entry_var,
                                 font=(MONO[0], 10), show="")
        self.ai_entry.pack(fill="x", pady=(2, 8), ipady=4)
        self.ai_hint = tk.Label(self.ai_panel, text="", font=(MONO[0], 8), anchor="w")
        self.ai_hint.pack(fill="x")
        mrow = tk.Frame(self.ai_panel); mrow.pack(fill="x", pady=(8, 0))
        self._mrow = mrow
        tk.Label(mrow, text="Model:", font=(SANS[0], 9)).pack(side="left")
        self.model_var = tk.StringVar()
        self.model_menu = tk.OptionMenu(mrow, self.model_var, "")
        self.model_menu.pack(side="left", padx=(8, 0))
        tk.Label(mrow, text="or custom:", font=(SANS[0], 9)).pack(side="left", padx=(14, 0))
        self.custom_model_var = tk.StringVar()
        self.custom_model_entry = tk.Entry(mrow, textvariable=self.custom_model_var,
                                           font=(MONO[0], 10), width=24)
        self.custom_model_entry.pack(side="left", padx=(8, 0), ipady=4)

        # iv. Dataset details
        self._section_header("iv.", "Dataset Details", "")
        det = tk.Frame(self.main); det.pack(fill="x", padx=20, pady=(0, 2))
        self._det = det

        r1 = tk.Frame(det); r1.pack(fill="x", pady=3)
        self.lbl_nsub = tk.Label(r1, text="Number of subjects:", font=(SANS[0], 9), width=22, anchor="w")
        self.lbl_nsub.pack(side="left")
        self.nsub_var = tk.StringVar()
        self.nsub_entry = tk.Entry(r1, textvariable=self.nsub_var, font=(MONO[0], 10), width=12)
        self.nsub_entry.pack(side="left", ipady=4)
        self.lbl_nsub_hint = tk.Label(r1, text="  number of subjects (leave empty to auto-detect)",
                                      font=(MONO[0], 8)); self.lbl_nsub_hint.pack(side="left")

        r2 = tk.Frame(det); r2.pack(fill="x", pady=3)
        self.lbl_ids = tk.Label(r2, text="Subject ID strategy:", font=(SANS[0], 9), width=22, anchor="w")
        self.lbl_ids.pack(side="left")
        self.idstrat = tk.StringVar(value="auto")
        for s in ["auto", "numeric", "semantic"]:
            b = tk.Button(r2, text=s, relief="flat", cursor="hand2", padx=12, pady=6,
                          command=lambda ss=s: self._set_idstrat(ss))
            b.pack(side="left", padx=(0, 6))
            self._ids_btns[s] = b
        self.lbl_ids_hint = tk.Label(r2, text="  numeric for most; semantic if IDs not unique; auto = default",
                                     font=(MONO[0], 8)); self.lbl_ids_hint.pack(side="left")

        r3 = tk.Frame(det); r3.pack(fill="x", pady=3)
        desc_left = tk.Frame(r3); desc_left.pack(side="left", anchor="n")
        self.lbl_desc = tk.Label(desc_left, text="Description:", font=(SANS[0], 9),
                                 width=22, anchor="nw")
        self.lbl_desc.pack(side="top", anchor="nw")
        self.lbl_desc_hint = tk.Label(desc_left,
                                      text="describe the dataset\nas fully as possible\n"
                                           "(modality, tasks,\n#subjects, site, etc.)",
                                      font=(MONO[0], 8), anchor="nw", justify="left")
        self.lbl_desc_hint.pack(side="top", anchor="nw", pady=(4, 0))
        self.desc_text = tk.Text(r3, height=6, font=(SANS[0], 10), wrap="word",
                                 relief="flat", borderwidth=1)
        self.desc_text.pack(side="left", fill="x", expand=True)
        self._desc_left = desc_left

        # actions (Run + Clear + pipeline strip on one row)
        act = tk.Frame(self.main); act.pack(fill="x", padx=20, pady=(4, 2))
        self._act = act
        self.btn_run = tk.Button(act, text="\u25b6 Run Full Pipeline",
                                 font=(SANS[0], 10), relief="flat",
                                 cursor="hand2", padx=12, pady=4, command=self._on_run)
        self.btn_run.pack(side="left")
        self.btn_clearlog = tk.Button(act, text="Clear Log", font=(SANS[0], 10),
                                      relief="flat", cursor="hand2",
                                      padx=12, pady=4, command=self._clear_log)
        self.btn_clearlog.pack(side="left", padx=8)

        strip = tk.Frame(act); strip.pack(side="left", padx=(16, 0))
        self._strip = strip
        for i, s in enumerate(STAGES):
            lbl = tk.Label(strip, text=f"\u25cb {s}", font=(MONO[0], 9))
            lbl.pack(side="left", padx=(0, 3))
            self._stage_labels[s] = lbl
            if i < len(STAGES) - 1:
                tk.Label(strip, text="\u2192", font=(MONO[0], 9)).pack(side="left", padx=3)

        # log
        lh = tk.Frame(self.main); lh.pack(fill="x", padx=20, pady=(2, 0))
        self._lh = lh
        self.lbl_logtitle = tk.Label(lh, text="LIVE LOG", font=(MONO[0], 8))
        self.lbl_logtitle.pack(side="left")
        self.lbl_state = tk.Label(lh, text="\u25cf IDLE", font=(MONO[0], 8))
        self.lbl_state.pack(side="right")
        lf = tk.Frame(self.main); lf.pack(fill="x", padx=20, pady=(0, 8))
        self._lf = lf
        self.log = tk.Text(lf, font=MONO, wrap="word", relief="flat", borderwidth=0, height=18)
        sc = tk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sc.set, state="disabled")
        sc.pack(side="right", fill="y"); self.log.pack(side="left", fill="both", expand=True)
        for t, col in [("info", "#7ee787"), ("tag", "#56a8ff"), ("warn", "#e3b341"),
                       ("err", "#f85149"), ("ok", "#3fb950")]:
            self.log.tag_config(t, foreground=col)

    def _section_header(self, no, title, hint):
        h = tk.Frame(self.main); h.pack(fill="x", padx=20, pady=(8, 4))
        a = tk.Label(h, text=no, font=("Georgia", 13, "italic")); a.pack(side="left")
        b = tk.Label(h, text="  " + title, font=(SANS[0], 11, "bold")); b.pack(side="left")
        c = tk.Label(h, text=hint, font=(MONO[0], 8)); c.pack(side="right")
        self._headers.append((h, a, b, c))

    def _path_row(self, placeholder, cmd):
        row = tk.Frame(self.main); row.pack(fill="x", padx=20, pady=(0, 2))
        field = tk.Label(row, text=placeholder, font=(MONO[0], 9), anchor="w",
                         relief="flat", borderwidth=1, padx=10, pady=4)
        field.pack(side="left", fill="x", expand=True, ipady=3)
        btn = tk.Button(row, text="Select Folder", font=(SANS[0], 10), relief="flat",
                        cursor="hand2", padx=12, pady=4, command=cmd)
        btn.pack(side="right", padx=(8, 0))
        return field, btn

    # ---------------- theme ----------------
    def _apply_theme(self):
        c = self.theme
        self.root.configure(bg=c["bg"])
        self._outer.configure(bg=c["panel"])
        self.canvas.configure(bg=c["panel"])
        self.main.configure(bg=c["panel"])
        for w in [self._titlebar, self.lbl_name, self.lbl_tag, self.lbl_sub]:
            w.configure(bg=c["panel"])
        self.lbl_name.configure(fg=c["txt"]); self.lbl_tag.configure(fg=c["accent"])
        self.lbl_sub.configure(fg=c["txt_faint"])
        self.btn_theme.configure(bg=c["panel2"], fg=c["txt_dim"], activebackground=c["panel2"],
                                 text="\u2600" if self.theme_name == "dark" else "\u263e")
        for (h, a, b, d) in self._headers:
            h.configure(bg=c["panel"]); a.configure(bg=c["panel"], fg=c["accent"])
            b.configure(bg=c["panel"], fg=c["txt"]); d.configure(bg=c["panel"], fg=c["txt_faint"])
        for fld, btn in [(self.input_field, self.btn_input),
                         (self.output_field, self.btn_output)]:
            fld.master.configure(bg=c["panel"])
            fld.configure(bg=c["panel2"], fg=c["txt"], highlightbackground=c["line"], highlightthickness=1)
            self._style_btn(btn)
        for frame in [self._modrow, self._tabrow, self.ai_panel, self._mrow, self._det,
                      self._act, self._strip, self._lh, self._lf]:
            frame.configure(bg=c["panel"])
        self._refresh_modality_btns()
        self._refresh_ai_tabs()
        self.ai_field_label.configure(bg=c["panel"], fg=c["txt_dim"])
        self.ai_entry.configure(bg=c["panel2"], fg=c["txt"], insertbackground=c["txt"],
                                highlightbackground=c["line"], highlightthickness=1, relief="flat")
        self.ai_hint.configure(bg=c["panel"], fg=c["txt_faint"])
        self.model_menu.configure(bg=c["panel2"], fg=c["txt"], activebackground=c["line"],
                                  highlightthickness=1, highlightbackground=c["line"], relief="flat")
        self.custom_model_entry.configure(bg=c["panel2"], fg=c["txt"], insertbackground=c["txt"],
                                          highlightbackground=c["line"], highlightthickness=1, relief="flat")
        for child in self._mrow.winfo_children():
            if isinstance(child, tk.Label):
                child.configure(bg=c["panel"], fg=c["txt_dim"])
        for w in [self.lbl_nsub, self.lbl_ids, self.lbl_desc]:
            w.configure(bg=c["panel"], fg=c["txt"])
        for w in [self.lbl_nsub_hint, self.lbl_ids_hint, self.lbl_desc_hint]:
            w.configure(bg=c["panel"], fg=c["txt_faint"])
        self.nsub_entry.configure(bg=c["panel2"], fg=c["txt"], insertbackground=c["txt"],
                                  highlightbackground=c["line"], highlightthickness=1, relief="flat")
        for fr in [self.lbl_nsub.master, self.lbl_ids.master, self.lbl_desc.master,
                   self._desc_left, self.desc_text.master]:
            fr.configure(bg=c["panel"])
        self.desc_text.configure(bg=c["panel2"], fg=c["txt"], insertbackground=c["txt"],
                                 highlightbackground=c["line"], highlightthickness=1)
        self._refresh_idstrat_btns()
        # run / clear buttons: colour follows enabled state
        self.btn_run.master.configure(bg=c["panel"])
        if hasattr(self, "input_dir"):
            self._refresh_run_enabled()
        self._style_btn(self.btn_clearlog)
        self._refresh_stage_labels()
        self.lbl_logtitle.configure(bg=c["panel"], fg=c["txt_dim"])
        self.lbl_state.configure(bg=c["panel"], fg=c["txt_faint"])
        self.log.configure(bg=c["term_bg"], fg=c["term_txt"])

    def _style_btn(self, b):
        c = self.theme
        b.configure(bg=c["panel2"], fg=c["txt"], activebackground=c["line"],
                    highlightbackground=c["line"], highlightthickness=1, borderwidth=0)

    def _toggle_theme(self):
        self.theme, self.theme_name = (LIGHT, "light") if self.theme_name == "dark" else (DARK, "dark")
        self._apply_theme()
        self._set_state(self._current_state)

    # ---------------- modality / engine / idstrat ----------------
    def _set_modality(self, m):
        self.modality.set(m); self._refresh_modality_btns()

    def _refresh_modality_btns(self):
        c = self.theme
        for m, b in self._mod_btns.items():
            if m == self.modality.get():
                b.configure(bg=c["accent"], fg=c["run_txt"], activebackground=c["accent"],
                            highlightthickness=1, highlightbackground=c["line"], borderwidth=0)
            else:
                self._style_btn(b)

    def _set_idstrat(self, s):
        self.idstrat.set(s); self._refresh_idstrat_btns()

    def _refresh_idstrat_btns(self):
        c = self.theme
        for s, b in self._ids_btns.items():
            if s == self.idstrat.get():
                b.configure(bg=c["accent"], fg=c["run_txt"], activebackground=c["accent"],
                            highlightthickness=1, highlightbackground=c["line"], borderwidth=0)
            else:
                self._style_btn(b)

    def _select_engine(self, key):
        self.engine = key
        label, needs_key, needs_url, models = ENGINES[key]
        if needs_key:
            self.ai_field_label.configure(text="API key (required) - never saved, re-enter each run")
            self.ai_entry.configure(show="*")
            self.ai_entry_var.set("")
        elif key == "ollama-local":
            self.ai_field_label.configure(text="Base URL (local Ollama, editable)")
            self.ai_entry.configure(show="")
            self.ai_entry_var.set("http://localhost:11434")
        else:
            self.ai_field_label.configure(text="Base URL (required)")
            self.ai_entry.configure(show="")
            self.ai_entry_var.set("")
        hints = {
            "openai": "OPENAI_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY - Alibaba Cloud cloud alternative",
            "ollama-remote": "OLLAMA_BASE_URL  e.g. http://your-server.com:11434",
            "ollama-local": "run `ollama serve` first",
        }
        self.ai_hint.configure(text=hints.get(key, ""))
        self.ai_entry.configure(state="normal")
        menu = self.model_menu["menu"]; menu.delete(0, "end")
        for m in models:
            menu.add_command(label=m, command=lambda v=m: self.model_var.set(v))
        self.model_var.set(models[0])
        self._refresh_ai_tabs()

    def _refresh_ai_tabs(self):
        c = self.theme
        for key, b in self._ai_tabs.items():
            if key == self.engine:
                b.configure(bg=c["accent"], fg=c["run_txt"], activebackground=c["accent"],
                            highlightthickness=1, highlightbackground=c["line"], borderwidth=0)
            else:
                self._style_btn(b)

    # ---------------- pickers ----------------
    def _pick_input(self):
        d = filedialog.askdirectory(title="Select input dataset folder")
        if d:
            self.input_dir = d; self.input_field.configure(text=d, fg=self.theme["txt"])
            self._refresh_run_enabled()

    def _pick_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir = d; self.output_field.configure(text=d, fg=self.theme["txt"])
            self._refresh_run_enabled()

    # ---------------- run gating ----------------
    def _refresh_run_enabled(self):
        ready = bool(self.input_dir and self.output_dir and not self.running)
        c = self.theme
        if ready:
            run_blue = "#3b6fb0" if self.theme_name == "dark" else "#3b82f6"
            self.btn_run.configure(state="normal", bg=run_blue, fg="#ffffff",
                                   activebackground=run_blue, cursor="hand2")
        else:
            self.btn_run.configure(state="disabled", bg=c["panel2"],
                                   fg=c["txt_faint"], activebackground=c["panel2"],
                                   cursor="arrow")

    # ---------------- run ----------------
    def _on_run(self):
        if self.running:
            return
        if not self.input_dir or not self.output_dir:
            self._append_log("[WARN] Please select both input and output folders."); return
        label, needs_key, needs_url, _models = ENGINES[self.engine]
        val = self.ai_entry_var.get().strip()
        if needs_key and not val:
            self._append_log(f"[WARN] {label} requires an API key."); return
        if needs_url and not val:
            self._append_log(f"[WARN] {label} requires a Base URL."); return

        nsub = self.nsub_var.get().strip()
        nsubjects = int(nsub) if nsub.isdigit() else None
        describe = self.desc_text.get("1.0", "end").strip()

        custom_model = self.custom_model_var.get().strip()
        chosen_model = custom_model if custom_model else self.model_var.get()
        is_url_engine = needs_url or self.engine == "ollama-local"
        cfg = dict(
            input_dir=Path(self.input_dir), output_dir=Path(self.output_dir),
            engine=self.engine, model=chosen_model,
            api_key=val if needs_key else "", base_url=val if is_url_engine else "",
            modality=self.modality.get(), nsubjects=nsubjects,
            id_strategy=self.idstrat.get(), describe=describe,
        )
        self._clear_log()
        self._reset_stages()
        self.running = True
        self._set_state("running")
        self._refresh_run_enabled()
        threading.Thread(target=self._worker_thread, args=(cfg,), daemon=True).start()

    def _worker_thread(self, cfg):
        try:
            code = worker.run(log=lambda l: self.log_queue.put(("log", l)), **cfg)
            self.log_queue.put(("state", "done" if code == 0 else "error"))
        except Exception as e:  # noqa: BLE001
            self.log_queue.put(("log", f"[FATAL] {e}"))
            self.log_queue.put(("state", "error"))

    # ---------------- log queue ----------------
    def _drain_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload); self._update_stage_from_log(payload)
                elif kind == "state":
                    self.running = False
                    self._set_state(payload)
                    self._refresh_run_enabled()
        except queue.Empty:
            pass
        self.root.after(80, self._drain_log_queue)

    def _append_log(self, line):
        tag = "info"
        if "[FATAL]" in line or "[ERROR]" in line:
            tag = "err"
        elif "[WARN" in line:
            tag = "warn"
        elif "\u2713" in line or line.strip().lower().endswith(("complete", "done.")):
            tag = "ok"
        elif "[INFO]" in line or "[WORKER]" in line:
            tag = "tag"
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag); self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal"); self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---------------- stages ----------------
    def _reset_stages(self):
        for s, lbl in self._stage_labels.items():
            lbl.configure(text=f"\u25cb {s}", fg=self.theme["txt_faint"])

    def _refresh_stage_labels(self):
        for s, lbl in self._stage_labels.items():
            txt = lbl.cget("text")
            lbl.configure(bg=self.theme["panel"])
            if txt.startswith("\u2713"):
                lbl.configure(fg=self.theme["green"])
            elif txt.startswith("\u25cf"):
                lbl.configure(fg=self.theme["accent"])
            else:
                lbl.configure(fg=self.theme["txt_faint"])
        for child in self._strip.winfo_children():
            if isinstance(child, tk.Label) and child.cget("text") == "\u2192":
                child.configure(bg=self.theme["panel"], fg=self.theme["txt_faint"])

    def _update_stage_from_log(self, line):
        low = line.lower()
        for s in STAGES:
            if s in low and ("stage" in low or "===" in low or "/7]" in low
                             or "generating" in low or "running" in low):
                hit = False
                for ss in STAGES:
                    lbl = self._stage_labels[ss]
                    if ss == s:
                        lbl.configure(text=f"\u25cf {ss}", fg=self.theme["accent"]); hit = True
                    elif not hit:
                        lbl.configure(text=f"\u2713 {ss}", fg=self.theme["green"])
                break
        if "pipeline complete" in low or line.strip() == "[WORKER] Done.":
            for ss in STAGES:
                self._stage_labels[ss].configure(text=f"\u2713 {ss}", fg=self.theme["green"])

    def _set_state(self, s):
        self._current_state = s
        labels = {"idle": "\u25cf IDLE", "running": "\u25cf RUNNING",
                  "done": "\u25cf DONE", "error": "\u25cf ERROR"}
        colors = {"idle": self.theme["txt_faint"], "running": self.theme["green"],
                  "done": self.theme["green"], "error": self.theme["red"]}
        self.lbl_state.configure(text=labels.get(s, "\u25cf IDLE"), fg=colors.get(s, self.theme["txt_faint"]))


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = App(root)
    # Some Linux window managers ignore the initial geometry() call, and remote
    # desktops (x2go) have limited usable height. After layout, cap the window
    # to the available screen height so it never runs off-screen.
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w = min(900, sw - 40)
    h = min(760, sh - 80)          # leave room for panels/taskbars
    root.geometry(f"{w}x{h}")
    root.mainloop()


if __name__ == "__main__":
    main()
