import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import csv
import threading
import cv2
import numpy as np

LABELS_FILE = "labels.json"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# ── InsightFace SCRFD detector ────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        self.app = None
        self.ready = False
        self.error = None

    def load(self, on_done):
        """Load model in background thread so UI doesn't freeze."""
        def _load():
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"]
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                self.app = app
                self.ready = True
            except Exception as e:
                self.error = str(e)
            finally:
                on_done()
        threading.Thread(target=_load, daemon=True).start()

    def predict(self, img_path):
        """Returns (label, n_faces): label 0/1/2, n_faces int."""
        if not self.ready:
            return None, 0
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None, 0
            faces = self.app.get(img)
            n = len(faces)
            if n == 0:
                return 0, 0
            elif n == 1:
                return 1, 1
            else:
                return 2, n
        except Exception:
            return None, 0


# ── Main app ──────────────────────────────────────────────────────────────────
class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler — Detekcija lica")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(900, 640)

        self.images = []
        self.current_index = 0
        self.labels = {}       # gt labels:   filename -> 0/1/2
        self.predictions = {}  # model preds: filename -> 0/1/2
        self.photo = None
        self.folder_path = ""

        self.detector = FaceDetector()

        self._build_ui()
        self._bind_keys()
        self._load_model()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg="#1a1a1a", pady=10)
        top.pack(fill=tk.X, padx=20)

        tk.Button(
            top, text="📂  Otvori mapu",
            command=self.open_folder,
            bg="#2a2a2a", fg="#cccccc",
            relief="flat", font=("Helvetica", 13),
            padx=16, pady=6, cursor="hand2",
            activebackground="#3a3a3a", activeforeground="#ffffff"
        ).pack(side=tk.LEFT)

        self.export_btn = tk.Button(
            top, text="💾  Spremi CSV",
            command=self.export_csv,
            bg="#2a2a2a", fg="#cccccc",
            relief="flat", font=("Helvetica", 13),
            padx=16, pady=6, cursor="hand2",
            activebackground="#3a3a3a", activeforeground="#ffffff",
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT, padx=8)

        self.metrics_btn = tk.Button(
            top, text="📊  Metrike",
            command=self.show_metrics,
            bg="#2a2a2a", fg="#cccccc",
            relief="flat", font=("Helvetica", 13),
            padx=16, pady=6, cursor="hand2",
            activebackground="#3a3a3a", activeforeground="#ffffff",
            state=tk.DISABLED
        )
        self.metrics_btn.pack(side=tk.LEFT, padx=0)

        self.progress_label = tk.Label(
            top, text="",
            bg="#1a1a1a", fg="#888888",
            font=("Helvetica", 12)
        )
        self.progress_label.pack(side=tk.RIGHT)

        self.stats_label = tk.Label(
            top, text="",
            bg="#1a1a1a", fg="#666666",
            font=("Helvetica", 11)
        )
        self.stats_label.pack(side=tk.RIGHT, padx=16)

        # Model status bar
        self.model_bar = tk.Frame(self.root, bg="#2a1a00")
        self.model_bar.pack(fill=tk.X, padx=20, pady=(0, 4))
        self.model_label = tk.Label(
            self.model_bar, text="⏳  Učitavam SCRFD model...",
            bg="#2a1a00", fg="#cc8800",
            font=("Helvetica", 11), padx=8, pady=4
        )
        self.model_label.pack(side=tk.LEFT)

        # Filename + prediction row
        info_row = tk.Frame(self.root, bg="#1a1a1a")
        info_row.pack(fill=tk.X, padx=20)

        self.filename_label = tk.Label(
            info_row, text="",
            bg="#1a1a1a", fg="#888888",
            font=("Helvetica", 11)
        )
        self.filename_label.pack(side=tk.LEFT)

        self.pred_label = tk.Label(
            info_row, text="",
            bg="#1a1a1a", fg="#888888",
            font=("Helvetica", 11, "bold")
        )
        self.pred_label.pack(side=tk.RIGHT)

        # Image canvas
        self.canvas_frame = tk.Frame(self.root, bg="#111111")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(6, 0))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_image())

        # Label indicator bar
        self.label_bar = tk.Frame(self.root, bg="#1a1a1a", height=6)
        self.label_bar.pack(fill=tk.X, padx=20, pady=(4, 0))
        self.label_indicator = tk.Frame(self.label_bar, bg="#1a1a1a", height=6)
        self.label_indicator.pack(fill=tk.X)

        # Bottom buttons
        bottom = tk.Frame(self.root, bg="#1a1a1a", pady=16)
        bottom.pack(fill=tk.X, padx=20)

        btn_cfg = dict(
            relief="flat", font=("Helvetica", 14, "bold"),
            padx=0, pady=0, cursor="hand2", width=12, height=2
        )

        self.btn_face = tk.Button(
            bottom, text="✓  IMA LICE",
            command=lambda: self.label_and_next(1),
            bg="#1a4a1a", fg="#66cc66",
            activebackground="#1f5e1f", activeforeground="#88ee88",
            **btn_cfg
        )
        self.btn_face.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.btn_multi = tk.Button(
            bottom, text="⊕  VIŠE LICA",
            command=lambda: self.label_and_next(2),
            bg="#1a3a4a", fg="#66aacc",
            activebackground="#1f4a5e", activeforeground="#88ccee",
            **btn_cfg
        )
        self.btn_multi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        nav_frame = tk.Frame(bottom, bg="#1a1a1a")
        nav_frame.pack(side=tk.LEFT, padx=4)

        tk.Button(
            nav_frame, text="←", command=self.prev_image,
            bg="#2a2a2a", fg="#aaaaaa", relief="flat",
            font=("Helvetica", 18), padx=10, pady=8,
            cursor="hand2", activebackground="#3a3a3a"
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            nav_frame, text="→", command=self.next_image,
            bg="#2a2a2a", fg="#aaaaaa", relief="flat",
            font=("Helvetica", 18), padx=10, pady=8,
            cursor="hand2", activebackground="#3a3a3a"
        ).pack(side=tk.LEFT, padx=2)

        self.btn_noface = tk.Button(
            bottom, text="✗  NEMA LICA",
            command=lambda: self.label_and_next(0),
            bg="#4a1a1a", fg="#cc6666",
            activebackground="#5e1f1f", activeforeground="#ee8888",
            **btn_cfg
        )
        self.btn_noface.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # Progress bar
        prog_frame = tk.Frame(self.root, bg="#2a2a2a", height=4)
        prog_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.progress_bar = tk.Frame(prog_frame, bg="#4a90d9", height=4)
        self.progress_bar.place(x=0, y=0, relheight=1, width=0)
        self._prog_frame = prog_frame

        tk.Label(
            self.root,
            text="  F = ima lice   |   V = više lica   |   N = nema lica   |   ← → = navigacija   |   Z = poništi oznaku",
            bg="#1a1a1a", fg="#444444", font=("Helvetica", 10)
        ).pack(side=tk.BOTTOM, pady=4)

    def _bind_keys(self):
        self.root.bind("<KeyPress-f>", lambda e: self.label_and_next(1))
        self.root.bind("<KeyPress-F>", lambda e: self.label_and_next(1))
        self.root.bind("<KeyPress-v>", lambda e: self.label_and_next(2))
        self.root.bind("<KeyPress-V>", lambda e: self.label_and_next(2))
        self.root.bind("<KeyPress-n>", lambda e: self.label_and_next(0))
        self.root.bind("<KeyPress-N>", lambda e: self.label_and_next(0))
        self.root.bind("<Left>",       lambda e: self.prev_image())
        self.root.bind("<Right>",      lambda e: self.next_image())
        self.root.bind("<KeyPress-z>", lambda e: self.clear_label())
        self.root.bind("<KeyPress-Z>", lambda e: self.clear_label())

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        def on_done():
            self.root.after(0, self._on_model_ready)
        self.detector.load(on_done)

    def _on_model_ready(self):
        if self.detector.ready:
            self.model_bar.config(bg="#0a2a0a")
            self.model_label.config(
                text="✓  SCRFD model spreman",
                bg="#0a2a0a", fg="#44cc44"
            )
        else:
            self.model_bar.config(bg="#3a0a0a")
            self.model_label.config(
                text=f"✗  Model nije učitan: {self.detector.error}",
                bg="#3a0a0a", fg="#cc4444"
            )

    def _run_prediction(self, img_name):
        """Run detector on current image (called in background)."""
        if not self.detector.ready:
            return
        img_path = os.path.join(self.folder_path, img_name)
        label, n_faces = self.detector.predict(img_path)
        if label is not None:
            self.predictions[img_name] = label
            self._autosave()
            self.root.after(0, self._update_pred_label)

    def _update_pred_label(self):
        if not self.images:
            return
        img_name = self.images[self.current_index]
        pred = self.predictions.get(img_name)
        if pred is None:
            self.pred_label.config(text="model: ⏳", fg="#888888")
        elif pred == 0:
            self.pred_label.config(text="model: ✗ nema lica", fg="#cc5555")
        elif pred == 1:
            self.pred_label.config(text="model: ✓ ima lice", fg="#55cc55")
        elif pred == 2:
            self.pred_label.config(text="model: ⊕ više lica", fg="#55aacc")
        self._maybe_enable_metrics()

    def _maybe_enable_metrics(self):
        paired = sum(1 for k in self.labels if k in self.predictions)
        if paired > 0:
            self.metrics_btn.config(state=tk.NORMAL)

    # ── Folder ────────────────────────────────────────────────────────────────
    def open_folder(self):
        folder = filedialog.askdirectory(title="Odaberi mapu sa slikama")
        if not folder:
            return
        self.folder_path = folder
        found = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                stem = os.path.splitext(f)[0].lower()
                ext  = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXT and stem == "camera":
                    rel = os.path.relpath(os.path.join(dirpath, f), folder)
                    found.append(rel)
        self.images = sorted(found)
        if not self.images:
            messagebox.showwarning("Nema slika", "U odabranoj mapi nema podržanih slika.")
            return

        labels_path = os.path.join(folder, LABELS_FILE)
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                data = json.load(f)
            self.labels      = data.get("gt", data)   # backwards compat
            self.predictions = data.get("pred", {})
            existing = len(self.labels)
            if existing:
                messagebox.showinfo("Nastavi označavanje", f"Učitano {existing} postojećih oznaka.")
                for i, img in enumerate(self.images):
                    if img not in self.labels:
                        self.current_index = i
                        break
                else:
                    self.current_index = 0
            else:
                self.current_index = 0
        else:
            self.labels = {}
            self.predictions = {}
            self.current_index = 0

        self.export_btn.config(state=tk.NORMAL)
        self._show_image()

    # ── Display ───────────────────────────────────────────────────────────────
    def _show_image(self):
        if not self.images:
            return
        img_name = self.images[self.current_index]
        img_path = os.path.join(self.folder_path, img_name)

        try:
            img = Image.open(img_path)
        except Exception:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Ne mogu otvoriti sliku",
                fill="#666666", font=("Helvetica", 14)
            )
            return

        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        img.thumbnail((cw, ch), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.photo)

        label = self.labels.get(img_name)
        self.filename_label.config(text=img_name)

        # Button highlight
        if label == 1:
            self.label_indicator.config(bg="#1a5c1a")
            self.btn_face.config(relief="sunken", bg="#1f6e1f")
            self.btn_multi.config(relief="flat",   bg="#1a3a4a")
            self.btn_noface.config(relief="flat",  bg="#4a1a1a")
        elif label == 2:
            self.label_indicator.config(bg="#1a3a5c")
            self.btn_multi.config(relief="sunken", bg="#1f4a6e")
            self.btn_face.config(relief="flat",    bg="#1a4a1a")
            self.btn_noface.config(relief="flat",  bg="#4a1a1a")
        elif label == 0:
            self.label_indicator.config(bg="#5c1a1a")
            self.btn_noface.config(relief="sunken", bg="#6e1f1f")
            self.btn_face.config(relief="flat",     bg="#1a4a1a")
            self.btn_multi.config(relief="flat",    bg="#1a3a4a")
        else:
            self.label_indicator.config(bg="#2a2a2a")
            self.btn_face.config(relief="flat",   bg="#1a4a1a")
            self.btn_multi.config(relief="flat",  bg="#1a3a4a")
            self.btn_noface.config(relief="flat", bg="#4a1a1a")

        total   = len(self.images)
        labeled = len(self.labels)
        self.progress_label.config(text=f"{self.current_index + 1} / {total}")
        fc = sum(1 for v in self.labels.values() if v == 1)
        mc = sum(1 for v in self.labels.values() if v == 2)
        nc = sum(1 for v in self.labels.values() if v == 0)
        self.stats_label.config(text=f"✓ {fc}   ⊕ {mc}   ✗ {nc}   ? {total - labeled}")

        self.root.update_idletasks()
        fw    = self._prog_frame.winfo_width()
        bar_w = int(fw * labeled / total) if total else 0
        self.progress_bar.place(x=0, y=0, relheight=1, width=bar_w)
        self.root.title(f"Image Labeler — {img_name} ({self.current_index + 1}/{total})")

        # Show cached prediction or trigger new one
        self._update_pred_label()
        if img_name not in self.predictions and self.detector.ready:
            self.pred_label.config(text="model: ⏳", fg="#888888")
            threading.Thread(
                target=self._run_prediction, args=(img_name,), daemon=True
            ).start()

    # ── Labeling ──────────────────────────────────────────────────────────────
    def label_and_next(self, value):
        if not self.images:
            return
        img_name = self.images[self.current_index]
        self.labels[img_name] = value
        self._autosave()

        # Trigger prediction if not yet done
        if img_name not in self.predictions and self.detector.ready:
            threading.Thread(
                target=self._run_prediction, args=(img_name,), daemon=True
            ).start()

        next_idx = self._find_next_unlabeled()
        if next_idx is not None:
            self.current_index = next_idx
        else:
            self.current_index = min(self.current_index + 1, len(self.images) - 1)
        self._show_image()

    def _find_next_unlabeled(self):
        for i in range(self.current_index + 1, len(self.images)):
            if self.images[i] not in self.labels:
                return i
        return None

    def prev_image(self):
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self._show_image()

    def next_image(self):
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self._show_image()

    def clear_label(self):
        if not self.images:
            return
        img_name = self.images[self.current_index]
        if img_name in self.labels:
            del self.labels[img_name]
            self._autosave()
            self._show_image()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _autosave(self):
        if self.folder_path:
            path = os.path.join(self.folder_path, LABELS_FILE)
            with open(path, "w") as f:
                json.dump({"gt": self.labels, "pred": self.predictions}, f, indent=2)

    # ── Metrics popup ─────────────────────────────────────────────────────────
    def show_metrics(self):
        paired = {k: (self.labels[k], self.predictions[k])
                  for k in self.labels if k in self.predictions}
        if not paired:
            messagebox.showinfo("Metrike", "Nema parova GT + Predikcija.")
            return

        # Binary: ima_lice (1 or 2) vs nema_lice (0)
        # treat label 2 (više lica) as positive (lice postoji)
        def to_binary(v):
            return 1 if v in (1, 2) else 0

        TP = TN = FP = FN = 0
        for gt, pred in paired.values():
            g, p = to_binary(gt), to_binary(pred)
            if   g == 1 and p == 1: TP += 1
            elif g == 0 and p == 0: TN += 1
            elif g == 0 and p == 1: FP += 1
            elif g == 1 and p == 0: FN += 1

        n = TP + TN + FP + FN
        accuracy  = (TP + TN) / n if n else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall    = TP / (TP + FN) if (TP + FN) else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0

        win = tk.Toplevel(self.root)
        win.title("Metrike — Binary classifier")
        win.configure(bg="#1a1a1a")
        win.resizable(False, False)

        def row(parent, label, value, color="#cccccc"):
            f = tk.Frame(parent, bg="#1a1a1a")
            f.pack(fill=tk.X, pady=3)
            tk.Label(f, text=label, bg="#1a1a1a", fg="#888888",
                     font=("Helvetica", 12), width=18, anchor="w").pack(side=tk.LEFT)
            tk.Label(f, text=value, bg="#1a1a1a", fg=color,
                     font=("Helvetica", 12, "bold"), anchor="w").pack(side=tk.LEFT)

        pad = tk.Frame(win, bg="#1a1a1a", padx=28, pady=20)
        pad.pack()

        tk.Label(pad, text="Rezultati evaluacije", bg="#1a1a1a", fg="#ffffff",
                 font=("Helvetica", 15, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Label(pad, text=f"Ukupno označenih s predikcijom: {n}",
                 bg="#1a1a1a", fg="#666666", font=("Helvetica", 10)).pack(anchor="w", pady=(0, 12))

        # Confusion matrix
        cm_frame = tk.Frame(pad, bg="#1a1a1a")
        cm_frame.pack(fill=tk.X, pady=(0, 14))

        def cm_cell(parent, text, val, bg, fg, r, c):
            f = tk.Frame(parent, bg=bg, width=90, height=60)
            f.grid(row=r, column=c, padx=3, pady=3)
            f.grid_propagate(False)
            tk.Label(f, text=text, bg=bg, fg=fg,
                     font=("Helvetica", 9)).place(relx=0.5, rely=0.3, anchor="center")
            tk.Label(f, text=str(val), bg=bg, fg=fg,
                     font=("Helvetica", 18, "bold")).place(relx=0.5, rely=0.68, anchor="center")

        tk.Label(cm_frame, text="", bg="#1a1a1a", width=10).grid(row=0, column=0)
        tk.Label(cm_frame, text="Pred: Lice", bg="#1a1a1a", fg="#666666",
                 font=("Helvetica", 10)).grid(row=0, column=1)
        tk.Label(cm_frame, text="Pred: Nema", bg="#1a1a1a", fg="#666666",
                 font=("Helvetica", 10)).grid(row=0, column=2)
        tk.Label(cm_frame, text="GT: Lice", bg="#1a1a1a", fg="#666666",
                 font=("Helvetica", 10), anchor="e").grid(row=1, column=0, padx=(0,4))
        tk.Label(cm_frame, text="GT: Nema", bg="#1a1a1a", fg="#666666",
                 font=("Helvetica", 10), anchor="e").grid(row=2, column=0, padx=(0,4))

        cm_cell(cm_frame, "TP", TP, "#0a2a0a", "#44cc44", 1, 1)
        cm_cell(cm_frame, "FN", FN, "#2a0a0a", "#cc4444", 1, 2)
        cm_cell(cm_frame, "FP", FP, "#2a1a00", "#cc8800", 2, 1)
        cm_cell(cm_frame, "TN", TN, "#0a1a2a", "#4488cc", 2, 2)

        tk.Frame(pad, bg="#333333", height=1).pack(fill=tk.X, pady=12)

        row(pad, "Accuracy",    f"{accuracy*100:.1f}%",    "#ffffff")
        row(pad, "Precision",   f"{precision*100:.1f}%",   "#66cc66")
        row(pad, "Recall",      f"{recall*100:.1f}%",      "#66aacc")
        row(pad, "F1 Score",    f"{f1:.4f}",               "#ccaa44")
        row(pad, "Specificity", f"{specificity*100:.1f}%", "#aa66cc")

        tk.Frame(pad, bg="#333333", height=1).pack(fill=tk.X, pady=12)
        tk.Label(pad, text="* lice (1) + više lica (2) = pozitivna klasa",
                 bg="#1a1a1a", fg="#555555", font=("Helvetica", 9)).pack(anchor="w")

        tk.Button(pad, text="Zatvori", command=win.destroy,
                  bg="#2a2a2a", fg="#cccccc", relief="flat",
                  font=("Helvetica", 12), padx=16, pady=6,
                  cursor="hand2").pack(pady=(14, 0))

    # ── Export ────────────────────────────────────────────────────────────────
    def export_csv(self):
        if not self.labels:
            messagebox.showinfo("Nema podataka", "Nema označenih slika za export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV datoteka", "*.csv")],
            initialfile="labels.csv",
            title="Spremi CSV"
        )
        if not path:
            return

        label_name = {0: "nema_lica", 1: "ima_lice", 2: "vise_lica"}
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "gt_label", "gt_name", "pred_label", "pred_name"])
            for fname in self.images:
                if fname in self.labels:
                    gt   = self.labels[fname]
                    pred = self.predictions.get(fname, "")
                    pname = label_name.get(pred, "") if pred != "" else ""
                    writer.writerow([fname, gt, label_name[gt], pred, pname])

        total = len(self.labels)
        fc = sum(1 for v in self.labels.values() if v == 1)
        mc = sum(1 for v in self.labels.values() if v == 2)
        nc = sum(1 for v in self.labels.values() if v == 0)
        messagebox.showinfo(
            "Spremljeno",
            f"Exportirano {total} oznaka u:\n{path}\n\n"
            f"Ima lice: {fc}\nViše lica: {mc}\nNema lica: {nc}\n"
            f"S predikcijom: {len(self.predictions)}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("960x720")
    app = ImageLabeler(root)
    root.mainloop()
