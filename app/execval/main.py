"""
main.py
=======

AutoBIDSify ExecVal Desktop — Tkinter edition.

A pure-Tkinter GUI (no pywebview / pythonnet / Qt), chosen for the smallest
possible package and rock-solid PyInstaller packaging. It replicates the
HTML mock-up's layout: a title bar, three input sections (plan bundle, input
dataset, output location), a validate option, an Execute button, and a dark
live-log pane. Light/Dark themes are switchable.

The conversion runs on a background thread; log lines are passed to the UI
through a queue and flushed by a periodic poller (the thread-safe Tkinter
pattern). It imports autobidsify directly from the same repository.
"""

from __future__ import annotations

import os
import json
import queue
import sys
import threading
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog

import bundle
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

REQUIRED = ["ingest_info.json", "BIDSPlan.yaml", "dataset_description.json",
            "README.md", "participants.tsv"]
OPTIONAL = ["mat_mapping.json", "headers_normalized.json", "voxel_final_plan.json"]

_WIN = sys.platform.startswith("win")
MONO = ("Consolas", 10) if _WIN else ("DejaVu Sans Mono", 10)
SANS = ("Segoe UI", 10) if _WIN else ("DejaVu Sans", 10)


class App:
    _current_state = "idle"

    def __init__(self, root):
        self.root = root
        self.theme = LIGHT
        self.theme_name = "light"
        self.bundle_paths: List[str] = []
        self.bundle_result = None
        self.input_dir: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.advanced_open = False
        self.running = False
        self.log_queue = queue.Queue()
        self._headers = []
        self._path_rows = []

        root.title("AutoBIDSify")
        root.geometry("1180x1080")
        root.minsize(1000, 900)

        self._build_ui()
        self._apply_theme()
        self._render_bundle_box()
        self._render_chips()
        self._refresh_run_enabled()
        self.root.after(80, self._drain_log_queue)

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
            self.canvas.itemconfigure(self._main_window, width=event.width)
        self.canvas.bind("<Configure>", _on_canvas_config)

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

        self.titlebar = tk.Frame(self.main)
        self.titlebar.pack(fill="x", padx=20, pady=(16, 8))
        self.lbl_name = tk.Label(self.titlebar, text="AutoBIDSify",
                                 font=("Segoe UI", 15, "bold"))
        self.lbl_name.pack(side="left")
        self.lbl_tag = tk.Label(self.titlebar, text="  execute & validate",
                                font=(SANS[0], 9))
        self.lbl_tag.pack(side="left")
        self.btn_theme = tk.Button(self.titlebar, text="☾", width=3,
                                   command=self._toggle_theme, relief="flat",
                                   cursor="hand2")
        self.btn_theme.pack(side="right")
        self.lbl_sub = tk.Label(self.titlebar,
                                text="offline · no API key · data stays local",
                                font=(MONO[0], 8))
        self.lbl_sub.pack(side="right", padx=10)

        self._section_header("1", "Plan Bundle", "from web pipeline")

        # --- Simple mode: one folder field + Select ---
        simple_row = tk.Frame(self.main)
        simple_row.pack(fill="x", padx=20, pady=(0, 4))
        self._simple_row = simple_row
        self.simple_field = tk.Label(
            simple_row, text="Select downloaded folder from our web...",
            font=(MONO[0], 9), anchor="w", relief="flat", borderwidth=1,
            padx=10, pady=6)
        self.simple_field.pack(side="left", fill="x", expand=True, ipady=3)
        self.btn_simple_select = tk.Button(
            simple_row, text="Select", font=(SANS[0], 10), relief="flat",
            cursor="hand2", padx=12, pady=4, command=self._pick_bundle_folder)
        self.btn_simple_select.pack(side="right", padx=(8, 0))

        # --- File check chips (required row + supplementary row) ---
        self.chips = tk.Text(self.main, height=2, font=(MONO[0], 9),
                             wrap="word", relief="flat", borderwidth=0,
                             highlightthickness=0)
        self.chips.pack(fill="x", padx=20, pady=(2, 4))
        self.chips.configure(state="disabled", cursor="arrow")

        # --- Advanced toggle (below the file check) ---
        adv_head = tk.Frame(self.main)
        adv_head.pack(fill="x", padx=20, pady=(0, 2))
        self._adv_head = adv_head
        self.btn_advanced = tk.Button(
            adv_head, text="\u25b8 Advanced \u2014 add files manually",
            font=(SANS[0], 9, "underline"), relief="flat",
            cursor="hand2", borderwidth=0, command=self._toggle_advanced)
        self.btn_advanced.pack(side="left")

        # --- Advanced mode (hidden by default): file list + Add/Clear ---
        bundle_row = tk.Frame(self.main)
        self._adv_body = bundle_row
        self.bundle_box = tk.Text(bundle_row, height=3, font=(MONO[0], 9),
                                  wrap="none", relief="flat", borderwidth=1,
                                  spacing1=6, padx=10, pady=8)
        self.bundle_box.pack(side="left", fill="x", expand=True)
        self.bundle_box.configure(state="disabled")
        btns = tk.Frame(bundle_row)
        btns.pack(side="right", padx=(8, 0))
        self.btn_addfiles = tk.Button(btns, text="Add Files", width=12,
                                      command=self._add_files, relief="flat",
                                      cursor="hand2")
        self.btn_addfiles.pack(fill="x", pady=1)
        self.btn_addfolder = tk.Button(btns, text="Add Folder", width=12,
                                       command=self._add_folder, relief="flat",
                                       cursor="hand2")
        self.btn_addfolder.pack(fill="x", pady=1)
        self.btn_clearbundle = tk.Button(btns, text="Clear", width=12,
                                         command=self._clear_bundle, relief="flat",
                                         cursor="hand2")
        self.btn_clearbundle.pack(fill="x", pady=1)
        # _adv_body stays unpacked until Advanced is opened

        self._section_header("2", "Input Dataset Path", "")
        self.input_field, self.btn_input = self._path_row(
            "Select your extracted dataset folder...", self._pick_input)

        self._section_header("3", "Output Path", "")
        self.output_field, self.btn_output = self._path_row(
            "Select an output folder...", self._pick_output)

        opt = tk.Frame(self.main)
        opt.pack(fill="x", padx=20, pady=(6, 10))
        self.validate_var = tk.BooleanVar(value=True)
        self.chk_validate = tk.Checkbutton(opt, text="Run validate after execute",
                                           variable=self.validate_var, font=SANS)
        self.chk_validate.pack(side="left")

        act = tk.Frame(self.main)
        act.pack(fill="x", padx=20, pady=(0, 12))
        self.btn_run = tk.Button(act, text="\u25b6  Execute",
                                 font=(SANS[0], 11, "bold"),
                                 command=self._on_execute, relief="flat",
                                 cursor="hand2", padx=20, pady=6)
        self.btn_run.pack(side="left")
        self.btn_clearlog = tk.Button(act, text="Clear Log", command=self._clear_log,
                                      relief="flat", cursor="hand2", padx=12, pady=6)
        self.btn_clearlog.pack(side="left", padx=8)

        loghead = tk.Frame(self.main)
        loghead.pack(fill="x", padx=20)
        self.lbl_logtitle = tk.Label(loghead, text="LIVE LOG", font=(MONO[0], 8))
        self.lbl_logtitle.pack(side="left")
        self.lbl_state = tk.Label(loghead, text="\u25cf IDLE", font=(MONO[0], 8))
        self.lbl_state.pack(side="right")

        logframe = tk.Frame(self.main)
        logframe.pack(fill="x", padx=20, pady=(0, 8))
        self.log = tk.Text(logframe, font=MONO, wrap="word", relief="flat",
                           borderwidth=0, height=14)
        scroll = tk.Scrollbar(logframe, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_config("info", foreground="#7ee787")
        self.log.tag_config("tag", foreground="#56a8ff")
        self.log.tag_config("warn", foreground="#e3b341")
        self.log.tag_config("err", foreground="#f85149")
        self.log.tag_config("ok", foreground="#3fb950")

        foot = tk.Frame(self.main)
        foot.pack(fill="x", padx=20, pady=(0, 14))
        self.btn_open = tk.Button(foot, text="Open Output Folder",
                                  command=self._open_output, relief="flat",
                                  cursor="hand2", state="disabled")
        self.btn_open.pack(side="right")

    def _section_header(self, no, title, hint):
        h = tk.Frame(self.main)
        h.pack(fill="x", padx=20, pady=(8, 4))
        lbl_no = tk.Label(h, text=f"{no} \u00b7 {title}", font=(SANS[0], 11, "bold"))
        lbl_no.pack(side="left")
        lbl_t = tk.Label(h, text="", font=(SANS[0], 11, "bold"))
        lbl_t.pack(side="left")
        lbl_h = tk.Label(h, text=hint, font=(MONO[0], 8))
        lbl_h.pack(side="right")
        self._headers.append((h, lbl_no, lbl_t, lbl_h))

    def _path_row(self, placeholder, cmd):
        row = tk.Frame(self.main)
        row.pack(fill="x", padx=20, pady=(0, 4))
        field = tk.Label(row, text=placeholder, font=(MONO[0], 9), anchor="w",
                         relief="flat", borderwidth=1, padx=10, pady=8)
        field.pack(side="left", fill="x", expand=True)
        btn = tk.Button(row, text="Select Folder", command=cmd, relief="flat",
                        cursor="hand2", padx=12)
        btn.pack(side="right", padx=(8, 0))
        self._path_rows.append((row, field, btn))
        return field, btn

    def _apply_theme(self):
        c = self.theme
        self.root.configure(bg=c["bg"])
        self._outer.configure(bg=c["panel"])
        self.canvas.configure(bg=c["panel"])
        self.main.configure(bg=c["panel"])
        for w in [self.titlebar, self.lbl_name, self.lbl_tag, self.lbl_sub]:
            w.configure(bg=c["panel"])
        self.lbl_name.configure(fg=c["txt"])
        self.lbl_tag.configure(fg=c["accent"])
        self.lbl_sub.configure(fg=c["txt_faint"])
        self.btn_theme.configure(bg=c["panel2"], fg=c["txt_dim"],
                                 activebackground=c["panel2"],
                                 text="\u2600" if self.theme_name == "dark" else "\u263e")
        for (h, lbl_no, lbl_t, lbl_h) in self._headers:
            h.configure(bg=c["panel"])
            lbl_no.configure(bg=c["panel"], fg=c["txt"])
            lbl_t.configure(bg=c["panel"], fg=c["txt"])
            lbl_h.configure(bg=c["panel"], fg=c["txt_faint"])
        for (row, field, btn) in self._path_rows:
            row.configure(bg=c["panel"])
            field.configure(bg=c["panel2"], fg=c["txt"],
                            highlightbackground=c["line"], highlightthickness=1)
            self._style_btn(btn)
        self._simple_row.configure(bg=c["panel"])
        self.simple_field.configure(bg=c["panel2"], fg=c["txt"],
                                    highlightbackground=c["line"], highlightthickness=1)
        self._style_btn(self.btn_simple_select)
        self._adv_head.configure(bg=c["panel"])
        self.btn_advanced.configure(bg=c["panel"], fg=c["accent"],
                                    activebackground=c["panel"],
                                    activeforeground=c["accent"])
        self.bundle_box.master.configure(bg=c["panel"])
        self.bundle_box.configure(bg=c["panel2"], fg=c["txt_dim"],
                                  highlightbackground=c["line"], highlightthickness=1,
                                  insertbackground=c["txt"])
        for b in [self.btn_addfiles, self.btn_addfolder, self.btn_clearbundle]:
            self._style_btn(b)
        self.chips.configure(bg=c["panel"], fg=c["txt_dim"],
                             highlightthickness=0)
        self.chips.tag_config("ok", foreground=c["green"])
        self.chips.tag_config("miss", foreground=c["red"])
        self.chips.tag_config("neutral", foreground=c["txt_faint"])
        self.chk_validate.master.configure(bg=c["panel"])
        self.chk_validate.configure(bg=c["panel"], fg=c["txt_dim"],
                                    activebackground=c["panel"],
                                    selectcolor=c["panel2"])
        self.btn_run.master.configure(bg=c["panel"])
        # button colour (blue when ready / grey when disabled) is set by
        # _refresh_run_enabled; call it so a theme toggle keeps the right look
        if hasattr(self, "bundle_result"):
            self._refresh_run_enabled()
        self._style_btn(self.btn_clearlog)
        self.lbl_logtitle.master.configure(bg=c["panel"])
        self.lbl_logtitle.configure(bg=c["panel"], fg=c["txt_dim"])
        self.lbl_state.configure(bg=c["panel"], fg=c["txt_faint"])
        self.log.master.configure(bg=c["panel"])
        self.log.configure(bg=c["term_bg"], fg=c["term_txt"])
        self.btn_open.master.configure(bg=c["panel"])
        self.btn_open.configure(bg=c["panel"], fg=c["accent"],
                                activebackground=c["panel"], borderwidth=0)

    def _style_btn(self, b):
        c = self.theme
        b.configure(bg=c["panel2"], fg=c["txt"], activebackground=c["line"],
                    highlightbackground=c["line"], highlightthickness=1,
                    borderwidth=0)

    def _toggle_theme(self):
        if self.theme_name == "dark":
            self.theme, self.theme_name = LIGHT, "light"
        else:
            self.theme, self.theme_name = DARK, "dark"
        self._apply_theme()
        self._set_state(self._current_state)

    def _pick_bundle_folder(self):
        folder = filedialog.askdirectory(
            title="Select downloaded folder from our web")
        if not folder:
            return
        self.simple_field.configure(text=folder, fg=self.theme["txt"])
        # Treat the chosen folder as the whole bundle (replace any prior).
        self.bundle_paths = [folder]
        self.bundle_result = bundle.resolve_bundle([Path(folder)])
        self._render_bundle_box()
        self._render_chips()
        self._autofill_paths_from_ingest()
        self._refresh_run_enabled()

    def _toggle_advanced(self):
        self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.btn_advanced.configure(text="\u25be Advanced \u2014 add files manually")
            self._adv_body.pack(fill="x", padx=20, pady=(0, 4),
                                after=self._adv_head)
        else:
            self.btn_advanced.configure(text="\u25b8 Advanced \u2014 add files manually")
            self._adv_body.pack_forget()

    def _autofill_paths_from_ingest(self):
        """If ingest_info.json is in the bundle, read input_path/output_dir and
        prefill the Input/Output fields. If a path does not exist locally, show
        it in red so the user knows to pick it manually."""
        found = getattr(self.bundle_result, "found", {}) or {}
        info_path = found.get("ingest_info.json")
        if not info_path:
            return
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            self._append_log(f"[WARN] Could not read ingest_info.json: {e}")
            return

        in_path = data.get("input_path") or data.get("actual_data_path")
        out_path = data.get("output_dir")

        if in_path:
            exists = Path(in_path).exists()
            self.input_dir = in_path if exists else None
            self.input_field.configure(
                text=in_path,
                fg=self.theme["txt"] if exists else self.theme["red"])
            if not exists:
                self._append_log(
                    "[WARN] input_path from ingest_info.json does not exist "
                    "on this machine - please pick it with Select Folder:")
                self._append_log(f"        {in_path}")
        if out_path:
            exists = Path(out_path).exists()
            self.output_dir = out_path if exists else None
            self.output_field.configure(
                text=out_path,
                fg=self.theme["txt"] if exists else self.theme["red"])
            if not exists:
                self._append_log(
                    "[WARN] output_dir from ingest_info.json does not exist "
                    "on this machine - please pick it with Select Folder:")
                self._append_log(f"        {out_path}")

    def _add_files(self):
        files = filedialog.askopenfilenames(title="Add plan files")
        if files:
            self._add_paths(list(files))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Add a folder")
        if folder:
            self._add_paths([folder])

    def _clear_bundle(self):
        self.bundle_paths = []
        self.bundle_result = None
        self.simple_field.configure(text="Select downloaded folder from our web...",
                                    fg=self.theme["txt_faint"])
        self._render_bundle_box()
        self._render_chips()
        self._refresh_run_enabled()

    def _add_paths(self, paths):
        for p in paths:
            if p not in self.bundle_paths:
                self.bundle_paths.append(p)
        self._render_bundle_box()
        self.bundle_result = bundle.resolve_bundle(
            [Path(p) for p in self.bundle_paths])
        self._render_chips()
        self._refresh_run_enabled()

    def _render_bundle_box(self):
        self.bundle_box.configure(state="normal")
        self.bundle_box.delete("1.0", "end")
        if not self.bundle_paths:
            self.bundle_box.insert("end", "No items added yet\n"
                                   "(add files incl. .zip and/or folders)")
        else:
            self.bundle_box.insert("end", f"{len(self.bundle_paths)} item(s):\n")
            for p in self.bundle_paths:
                parts = Path(p).parts
                short = "/".join(parts[-2:]) if len(parts) > 1 else p
                self.bundle_box.insert("end", f"  - {short}\n")
        self.bundle_box.configure(state="disabled")

    def _render_chips(self):
        found = getattr(self.bundle_result, "found", {}) or {}
        provided = self.bundle_result is not None
        c = self.theme

        self.chips.configure(state="normal")
        self.chips.delete("1.0", "end")

        # Row 1: required files
        for i, name in enumerate(REQUIRED):
            if i:
                self.chips.insert("end", "    ")
            if name in found:
                self.chips.insert("end", f"\u2713 {name}", "ok")
            elif not provided:
                self.chips.insert("end", f"\u25cb {name}", "neutral")
            else:
                # missing required -> red, stays red until user provides it
                self.chips.insert("end", f"\u2717 {name}", "miss")
        self.chips.insert("end", "\n")

        # Row 2: supplementary files (not every dataset needs these)
        for i, name in enumerate(OPTIONAL):
            if i:
                self.chips.insert("end", "    ")
            if name in found:
                self.chips.insert("end", f"\u2713 {name}", "ok")
            else:
                self.chips.insert("end", f"\u25cb {name}", "neutral")

        self.chips.configure(state="disabled")

    def _pick_input(self):
        d = filedialog.askdirectory(title="Select input dataset folder")
        if d:
            self.input_dir = d
            self.input_field.configure(text=d, fg=self.theme["txt"])
            self._refresh_run_enabled()

    def _pick_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir = d
            self.output_field.configure(text=d, fg=self.theme["txt"])
            self._refresh_run_enabled()

    def _refresh_run_enabled(self):
        ready = (self.bundle_result is not None
                 and getattr(self.bundle_result, "is_complete", False)
                 and self.input_dir and self.output_dir and not self.running)
        c = self.theme
        if ready:
            run_blue = "#3b6fb0" if self.theme_name == "dark" else "#3b82f6"
            self.btn_run.configure(state="normal", bg=run_blue, fg="#ffffff",
                                   activebackground=run_blue, cursor="hand2")
        else:
            self.btn_run.configure(state="disabled", bg=c["panel2"],
                                   fg=c["txt_faint"], activebackground=c["panel2"],
                                   cursor="arrow")

    def _on_execute(self):
        if self.running:
            return
        self._clear_log()
        self.btn_open.configure(state="disabled")
        self.running = True
        self._set_state("running")
        self._refresh_run_enabled()
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def _worker_thread(self):
        resolver = None
        try:
            resolver = bundle.BundleResolver()
            resolved = resolver.resolve([Path(p) for p in self.bundle_paths])
            found = {k: str(v) for k, v in resolved.found.items()}
            code = worker.run(
                found=found,
                input_root=Path(self.input_dir),
                output_dir=Path(self.output_dir),
                do_validate=self.validate_var.get(),
                log=lambda line: self.log_queue.put(("log", line)),
            )
            self.log_queue.put(("state", "done" if code == 0 else "error"))
        except Exception as e:
            self.log_queue.put(("log", f"[FATAL] {e}"))
            self.log_queue.put(("state", "error"))
        finally:
            if resolver is not None:
                resolver.cleanup()

    def _drain_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "state":
                    self.running = False
                    self._set_state(payload)
                    if payload == "done":
                        self.btn_open.configure(state="normal")
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
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_state(self, s):
        self._current_state = s
        labels = {"idle": "\u25cf IDLE", "running": "\u25cf RUNNING",
                  "done": "\u25cf DONE", "error": "\u25cf ERROR"}
        colors = {"idle": self.theme["txt_faint"], "running": self.theme["green"],
                  "done": self.theme["green"], "error": self.theme["red"]}
        self.lbl_state.configure(text=labels.get(s, "\u25cf IDLE"),
                                 fg=colors.get(s, self.theme["txt_faint"]))

    def _open_output(self):
        if not self.output_dir:
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(self.output_dir)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", self.output_dir])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", self.output_dir])
        except Exception:
            pass


def main():
    # Crisp rendering on Windows high-DPI displays (avoids blurry scaling).
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = App(root)
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w = min(900, sw - 40)
    h = min(760, sh - 80)
    root.geometry(f"{w}x{h}")
    root.mainloop()


if __name__ == "__main__":
    main()