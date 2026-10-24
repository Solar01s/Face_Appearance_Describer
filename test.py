import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

import model
import utils

describes, sizes = utils.get_default_tasks()

ap = model.ApperanceDescriber(list(sizes))
ap = utils.load_model("trained_ap.pt", ap)
ap.eval()

BG = "#07070d"
CARD = "#10101c"
CARD_2 = "#161628"
CYAN = "#00f0ff"
CYAN_HOVER = "#0a3a44"
MAGENTA = "#ff2bd6"
MAGENTA_HOVER = "#c81fa8"
VIOLET = "#8a2be2"
TEXT = "#eaeaff"
MUTED = "#8888aa"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Apperance Describer based on ResNet18")
        self.geometry("960x760")
        self.minsize(900, 720)
        self.configure(fg_color=BG)

        self.file_path = ""
        self.preview_img = None

        self._build_header()
        self._build_body()
        self._build_footer()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(18, 6))

        logo = tk.Canvas(header, width=320, height=110, bg=BG, highlightthickness=0)
        logo.pack()
        cx, cy = 160, 55
        font = ("Arial", 68, "bold")

        glow_layers = [(6, "#1c0636"), (5, "#2f0a5c"), (4, "#4a0f8a"),
                       (3, "#7a1fc0"), (2, "#b030ff")]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        for r, color in glow_layers:
            for dx, dy in dirs:
                logo.create_text(cx + dx * r, cy + dy * r, text="UV", font=font, fill=color)

        logo.create_text(cx - 2, cy, text="UV", font=font, fill=CYAN)
        logo.create_text(cx + 2, cy, text="UV", font=font, fill=MAGENTA)
        # белый основной слой
        logo.create_text(cx, cy, text="UV", font=font, fill="#ffffff")

        title = ctk.CTkLabel(
            header,
            text="Apperance Describer",
            font=ctk.CTkFont("Arial", 26, "bold"),
            text_color=TEXT,
        )
        title.pack(pady=(0, 6))

        line = ctk.CTkFrame(header, height=3, width=340, corner_radius=2, fg_color=MAGENTA)
        line.pack()

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=30, pady=14)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.photo_card = ctk.CTkFrame(
            body, width=300, fg_color=CARD, corner_radius=22,
            border_width=2, border_color=CYAN,
        )
        self.photo_card.grid(row=0, column=0, sticky="ns", padx=(0, 18))
        self.photo_card.grid_propagate(False)
        self.photo_card.pack_propagate(False)

        self.photo_label = ctk.CTkLabel(
            self.photo_card,
            text="👤\n\nЗагрузите картинку лица\nдля оценки",
            font=ctk.CTkFont("Arial", 15, "bold"),
            text_color=MUTED,
            justify="center",
        )
        self.photo_label.pack(expand=True, padx=15, pady=15)

        self.result_card = ctk.CTkFrame(
            body, fg_color=CARD, corner_radius=22,
            border_width=2, border_color=MAGENTA,
        )
        self.result_card.grid(row=0, column=1, sticky="nsew")

        self.status_label = ctk.CTkLabel(
            self.result_card,
            text="Ожидание фотографии…",
            font=ctk.CTkFont("Arial", 20, "bold"),
            text_color=MUTED,
        )
        self.status_label.pack(pady=(22, 10))

        self.progress = ctk.CTkProgressBar(
            self.result_card, mode="indeterminate", width=320, height=6,
            progress_color=CYAN, fg_color=CARD_2,
        )

        self.rows_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.rows_frame.pack(fill="both", expand=True, padx=22, pady=(6, 20))

        self.keys = [
            ("⚧", "Пол"),
            ("💇", "Цвет волос"),
            ("👁", "Цвет глаз"),
            ("🎭", "Испытывает эмоцию"),
            ("🎂", "Возрастная группа"),
        ]
        self.value_labels = []
        for icon, name in self.keys:
            row = ctk.CTkFrame(self.rows_frame, fg_color=CARD_2, corner_radius=14)
            row.pack(fill="x", pady=6)

            ctk.CTkLabel(
                row, text=f"{icon} {name}", font=ctk.CTkFont("Arial", 14),
                text_color=MUTED, anchor="w",
            ).pack(side="left", padx=16, pady=14)

            val = ctk.CTkLabel(
                row, text="—", font=ctk.CTkFont("Arial", 16, "bold"),
                text_color=CYAN, anchor="e",
            )
            val.pack(side="right", padx=16)
            self.value_labels.append(val)

    # ---------- кнопки ----------
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=30, pady=(0, 24))
        footer.grid_columnconfigure((0, 1), weight=1, uniform="btn")

        self.load_btn = ctk.CTkButton(
            footer, text="👤 Загрузить фотографию", command=self.load_file,
            height=56, corner_radius=28,
            font=ctk.CTkFont("Arial", 16, "bold"),
            fg_color="transparent", hover_color=CYAN_HOVER,
            border_width=2, border_color=CYAN, text_color=CYAN,
        )
        self.load_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.process_btn = ctk.CTkButton(
            footer, text="✨ Описать", command=self.process_picture,
            height=56, corner_radius=28,
            font=ctk.CTkFont("Arial", 16, "bold"),
            fg_color=MAGENTA, hover_color=MAGENTA_HOVER, text_color="#ffffff",
            border_width=2, border_color="#ff7be8",
            state="disabled", text_color_disabled="#6b4a66",
        )
        self.process_btn.grid(row=0, column=1, sticky="ew", padx=(10, 0))

    def load_file(self):
        file = filedialog.askopenfilename(
            title="Выбор фотографии лица",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp *.jfif"),
                ("Все файлы", "*.*"),
            ],
        )
        if not file:
            return

        try:
            pil_image = Image.open(file).convert("RGB")
        except Exception as e:
            self.photo_label.configure(
                image=None, text=f"Ошибка загрузки:\n{e}", text_color="#ff5c5c"
            )
            print(e)
            return

        self.file_path = file

        pil_image.thumbnail((260, 380))
        self.preview_img = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=pil_image.size
        )
        self.photo_label.configure(image=self.preview_img, text="")

        self.process_btn.configure(state="normal")
        self.status_label.configure(text="Фото загружено. Нажмите «Описать»", text_color=TEXT)
        for v in self.value_labels:
            v.configure(text="—")

    def process_picture(self):
        if not self.file_path:
            return

        self.process_btn.configure(state="disabled")
        self.load_btn.configure(state="disabled")
        self.status_label.configure(text="Анализирую…", text_color=CYAN)
        self.progress.pack(after=self.status_label, pady=(0, 6))
        self.progress.start()

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            indexes = utils.choices(ap.from_file(self.file_path))
            values = [
                describes["task_1"][indexes[0]],
                describes["task_2"][indexes[1]],
                describes["task_3"][indexes[2]],
                describes["task_4"][indexes[3]],
                describes["task_5"][indexes[4]],
            ]
            self.after(0, self._show_result, values, None)
        except Exception as e:
            self.after(0, self._show_result, None, e)

    def _show_result(self, values, error):
        self.progress.stop()
        self.progress.pack_forget()
        self.load_btn.configure(state="normal")
        self.process_btn.configure(state="normal")

        if error is not None:
            self.status_label.configure(text=f"Ошибка: {error}", text_color="#ff5c5c")
            print(error)
            return

        self.status_label.configure(text="Описание завершено ✅", text_color=TEXT)
        for label, value in zip(self.value_labels, values):
            label.configure(text=str(value))


if __name__ == "__main__":
    app = App()
    app.mainloop()
