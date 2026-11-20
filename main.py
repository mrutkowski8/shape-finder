import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from shape_model import SHAPES, load_model, predict_percentages


LIVE_GUESS_INTERVAL_MS = 50


class DrawingCanvas(tk.Canvas):
    BORDER_WIDTH = 1
    _ERASER_RADIUS = 10

    def __init__(self, window: tk.Tk, size: int, tool: tk.StringVar) -> None:
        super().__init__(
            window,
            width=size,
            height=size,
            background="white",
            borderwidth=0,
            highlightthickness=self.BORDER_WIDTH,
            highlightbackground="#b8b8b8",
            cursor="crosshair",
        )
        self._tool = tool
        self._previous_point: tuple[int, int] | None = None
        self.drawing = Image.new("L", (size, size), color=255)
        self._brush = ImageDraw.Draw(self.drawing)
        self._preview = ImageTk.PhotoImage(self.drawing, master=self)
        self.create_image(self.BORDER_WIDTH, self.BORDER_WIDTH, anchor="nw", image=self._preview)
        self._eraser_outline = self.create_oval(
            0, 0, 0, 0, fill="", outline="#757575", width=2, state="hidden"
        )

        self.bind("<Enter>", self._track_pointer)
        self.bind("<Motion>", self._track_pointer)
        self.bind("<Leave>", self._hide_eraser_outline)
        self.bind("<Button-1>", self._start_stroke)
        self.bind("<B1-Motion>", self._extend_stroke)
        self.bind("<ButtonRelease-1>", self._finish_stroke)

    def update_cursor(self) -> None:
        self.configure(cursor="none" if self._tool.get() == "eraser" else "crosshair")
        x, y = self.winfo_pointerxy()
        self._move_eraser_outline(x - self.winfo_rootx(), y - self.winfo_rooty())

    def _track_pointer(self, event: tk.Event) -> None:
        self._move_eraser_outline(event.x, event.y)

    def _hide_eraser_outline(self, _event: tk.Event | None = None) -> None:
        self.itemconfigure(self._eraser_outline, state="hidden")

    def _move_eraser_outline(self, x: int, y: int) -> None:
        if self._tool.get() != "eraser" or not (
            0 <= x < self.winfo_width() and 0 <= y < self.winfo_height()
        ):
            self._hide_eraser_outline()
            return

        radius = self._ERASER_RADIUS
        self.coords(self._eraser_outline, x - radius, y - radius, x + radius, y + radius)
        self.itemconfigure(self._eraser_outline, state="normal")

    def clear(self) -> None:
        self._previous_point = None
        self.drawing.paste(255, (0, 0, self.drawing.width, self.drawing.height))
        self._preview.paste(self.drawing)

    def _position(self, event: tk.Event) -> tuple[int, int]:
        return event.x - self.BORDER_WIDTH, event.y - self.BORDER_WIDTH

    def _start_stroke(self, event: tk.Event) -> None:
        self._previous_point = self._position(event)
        self._extend_stroke(event)

    def _extend_stroke(self, event: tk.Event) -> None:
        self._track_pointer(event)
        if self._previous_point is None:
            return

        x, y = self._position(event)
        fill, radius = (255, self._ERASER_RADIUS) if self._tool.get() == "eraser" else (0, 2)
        self._brush.line(
            (self._previous_point, (x, y)),
            fill=fill,
            width=2 * radius + 1,
        )
        self._brush.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        self._previous_point = (x, y)
        self._preview.paste(self.drawing)

    def _finish_stroke(self, event: tk.Event) -> None:
        self._extend_stroke(event)
        self._previous_point = None


def create_guess_panel(window: tk.Tk) -> tuple[tk.Frame, dict[str, tk.Label]]:
    panel = tk.Frame(
        window,
        background="#f7f7f7",
        highlightthickness=1,
        highlightbackground="#c5c5c5",
        padx=12,
        pady=10,
    )
    panel.columnconfigure(0, weight=1, minsize=88)
    panel.columnconfigure(1, minsize=30)
    label_style = {"background": panel["background"], "padx": 0, "pady": 0}
    tk.Label(
        panel,
        text="current guess:",
        font=("Segoe UI", 11, "bold"),
        foreground="#242b33",
        **label_style,
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
    percentages = {}
    for row, shape in enumerate(SHAPES, start=1):
        tk.Label(panel, text=shape, foreground="#242b33", **label_style).grid(
            row=row, column=0, sticky="w", pady=2
        )
        value = tk.Label(panel, text="0%", foreground="#525b66", **label_style)
        value.grid(row=row, column=1, sticky="e", pady=2)
        percentages[shape] = value
    return panel, percentages


def main() -> None:
    window = tk.Tk()
    window.withdraw()
    window.title("shape-finder")
    window.configure(background="#e8e8e8")

    project_dir = Path(__file__).resolve().parent
    icon = tk.PhotoImage(master=window, file=str(project_dir/ "icons" / "icon.png"))
    window.iconphoto(True, icon)

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    width = screen_width // 2
    height = screen_height // 2
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")
    window.resizable(False, False)

    title = tk.Label(
        window,
        text="find your shape!",
        background=window["background"],
        foreground="#242b33",
        font=("Segoe UI", 20, "bold"),
        padx=0,
        pady=0,
    )
    guess_panel, guess_values = create_guess_panel(window)
    window.update_idletasks()

    toolbar_width = 40
    gap = 14
    margin = 12
    left_space = guess_panel.winfo_reqwidth() + gap + margin
    right_space = toolbar_width + gap + margin
    top_space = title.winfo_reqheight() + gap + margin
    canvas_border = 2 * DrawingCanvas.BORDER_WIDTH
    canvas_size = max(
        1,
        min(
            min(width, height) * 3 // 4,
            width - left_space - right_space - canvas_border,
            height - 2 * top_space - canvas_border,
        ),
    )
    canvas_offset = max(0, left_space - (width - canvas_size - canvas_border) // 2)
    tool = tk.StringVar(window, value="pencil")
    canvas = DrawingCanvas(window, canvas_size, tool)
    canvas.place(relx=0.5, x=canvas_offset, rely=0.5, anchor="center")
    title.place(in_=canvas, relx=0.5, y=-gap, anchor="s", bordermode="outside")
    guess_panel.place(in_=canvas, rely=0.5, x=-gap, anchor="e", bordermode="outside")

    try:
        model = load_model()
    except FileNotFoundError:
        model = None
        print("Run python train.py to enable shape guesses.", file=sys.stderr)

    pending_guess: str | None = None

    def update_guess() -> None:
        nonlocal pending_guess
        pending_guess = None
        if model is None:
            return
        for shape, percentage in predict_percentages(model, canvas.drawing).items():
            guess_values[shape].configure(text=f"{percentage}%")

    def schedule_guess(_event: tk.Event) -> None:
        nonlocal pending_guess
        if model is not None and pending_guess is None:
            pending_guess = canvas.after(LIVE_GUESS_INTERVAL_MS, update_guess)

    def cancel_pending_guess(_event: tk.Event | None = None) -> None:
        nonlocal pending_guess
        if pending_guess is not None:
            canvas.after_cancel(pending_guess)
            pending_guess = None

    def refresh_guess(_event: tk.Event | None = None) -> None:
        cancel_pending_guess()
        update_guess()

    def clear_canvas() -> None:
        canvas.clear()
        refresh_guess()

    canvas.bind("<Button-1>", schedule_guess, add="+")
    canvas.bind("<B1-Motion>", schedule_guess, add="+")
    canvas.bind("<ButtonRelease-1>", refresh_guess, add="+")
    canvas.bind("<Destroy>", cancel_pending_guess, add="+")

    toolbar = tk.Frame(window, background=window["background"])
    toolbar.place(
        in_=canvas,
        relx=1,
        rely=0.5,
        x=gap,
        anchor="w",
        width=toolbar_width,
        bordermode="outside",
    )
    icons = {
        name: tk.PhotoImage(master=window, file=str(project_dir / "icons" / f"{name}.png"))
        for name in ("pencil", "eraser", "trash")
    }
    button_style = {
        "height": toolbar_width - 2,
        "padx": 0,
        "pady": 0,
        "background": "#f7f7f7",
        "activebackground": "#e3edfb",
        "relief": "flat",
        "borderwidth": 0,
        "highlightthickness": 1,
        "highlightbackground": "#c5c5c5",
        "highlightcolor": "#4874b2",
        "cursor": "hand2",
    }
    for name, label in (("pencil", "Pencil"), ("eraser", "Eraser")):
        tk.Radiobutton(
            toolbar,
            image=icons[name],
            text=label,
            variable=tool,
            value=name,
            command=canvas.update_cursor,
            indicatoron=False,
            offrelief="flat",
            selectcolor="#cfe2fb",
            **button_style,
        ).pack(fill="x", pady=(0, 8))
    tk.Button(
        toolbar, image=icons["trash"], text="Clear canvas", command=clear_canvas, **button_style
    ).pack(fill="x")

    window.deiconify()
    window.mainloop()


if __name__ == "__main__":
    main()
