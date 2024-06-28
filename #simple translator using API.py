#simple translator using API
import tkinter as tk
from tkinter import scrolledtext
from tkinter import PhotoImage
#importing translator model
from googletrans import Translator, LANGUAGES

def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

def start_translation():
    text = text_entry.get("1.0", tk.END).strip()
    target_lang = lang_var.get()
    try:
        translation = translate_text(text, target_lang)
        result_text.set(translation)
    except Exception as e:
        result_text.set(f"Translation error: {e}")

#main window using tk
rt = tk.Tk()
rt.title("Text Translator")
rt.configure(bg="plum")
icon = PhotoImage(file=r"C:\Users\Paru\OneDrive\Pictures\kitty3.png")
rt.iconphoto(False, icon)
header_label = tk.Label(rt, text="Hello, welcome to ASH_TRANSLATES", bg="orange")
header_label.grid(pady=20)
rt.columnconfigure(0, weight=1)
rt.rowconfigure(0, weight=1)

#for framing content of the UI
content_frame = tk.Frame(rt, padx=10, pady=10, bg="thistle")
content_frame.grid(row=1, column=0, sticky="nsew")
description_label = tk.Label(content_frame, text="This application is based on GOOGLE TRANSLATE API and i have craeted an interactive translator using this ML pipeline, hoping for more improvement using datatsets.", bg="lightblue", wraplength=400, justify=tk.LEFT)
description_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

#adding image for illustration
small_image = PhotoImage(file=r"C:\Users\Paru\Downloads\WhatsApp Image 2024-06-23 at 7.35.46 PM (1).png").subsample(9, 9)
image_label = tk.Label(content_frame, image=small_image, bg="thistle")
image_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")

text_label = tk.Label(content_frame, text="Enter text to translate:", bg="lightpink")
text_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
text_entry = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, width=50, height=10, bg="lavender", fg="black")
text_entry.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
lang_var = tk.StringVar(value="ml")  # Default to Malayalam

#dropdown for language list
lang_label = tk.Label(content_frame, text="Select target language:", bg="lightpink")
lang_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
lang_menu = tk.OptionMenu(content_frame, lang_var, *LANGUAGES.values())
lang_menu.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
lang_menu.configure(bg="pink", fg="black")
translate_button = tk.Button(content_frame, text="Start Translation", command=start_translation, bg="yellow", fg="black")
translate_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
result_text = tk.StringVar()
result_label = tk.Label(content_frame, textvariable=result_text, wraplength=400, justify=tk.LEFT, font=("Helvetica", 20), bg="orange")
result_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

rt.mainloop()
