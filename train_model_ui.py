import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading

import joblib

from train_model import load_dataset, predict_emotion, train_text_classifier


DEFAULT_DATA_PATH = "train.txt"
DEFAULT_MODEL_PATH = "text_classifier.joblib"
DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# Color scheme
BG_COLOR = "#f0f0f0"
HEADER_COLOR = "#2c3e50"
BUTTON_COLOR = "#3498db"
SUCCESS_COLOR = "#27ae60"
ERROR_COLOR = "#e74c3c"
TEXT_COLOR = "#2c3e50"


class TrainUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Classifier - Train & Test")
        self.geometry("900x700")
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)
        
        self.model = None
        self.vectorizer = None
        self.training = False

        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        # Header
        header = tk.Frame(self, bg=HEADER_COLOR, height=60)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="🤖 Emotion Classifier", 
                        font=("Arial", 20, "bold"), bg=HEADER_COLOR, fg="white")
        title.pack(pady=10)

        # Main container
        main_frame = tk.Frame(self, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Training Section
        train_frame = tk.LabelFrame(main_frame, text="📚 Training Configuration", 
                                   font=("Arial", 11, "bold"), bg=BG_COLOR, 
                                   fg=TEXT_COLOR, padx=10, pady=10)
        train_frame.pack(fill=tk.X, pady=(0, 15))

        # Data file
        tk.Label(train_frame, text="Training Data File:", font=("Arial", 10), 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, sticky="w", pady=5)
        self.data_entry = tk.Entry(train_frame, width=60, font=("Arial", 9))
        self.data_entry.grid(row=0, column=1, padx=5, pady=5)
        self.data_entry.insert(0, DEFAULT_DATA_PATH)
        
        tk.Button(train_frame, text="Browse", command=self.browse_data,
                 bg=BUTTON_COLOR, fg="white", font=("Arial", 9),
                 padx=10, pady=5).grid(row=0, column=2, padx=5)

        # Model file
        tk.Label(train_frame, text="Model Output File:", font=("Arial", 10),
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, sticky="w", pady=5)
        self.model_entry = tk.Entry(train_frame, width=60, font=("Arial", 9))
        self.model_entry.grid(row=1, column=1, padx=5, pady=5)
        self.model_entry.insert(0, DEFAULT_MODEL_PATH)

        # Vectorizer file
        tk.Label(train_frame, text="Vectorizer Output File:", font=("Arial", 10),
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=2, column=0, sticky="w", pady=5)
        self.vectorizer_entry = tk.Entry(train_frame, width=60, font=("Arial", 9))
        self.vectorizer_entry.grid(row=2, column=1, padx=5, pady=5)
        self.vectorizer_entry.insert(0, DEFAULT_VECTORIZER_PATH)

        # Buttons for training
        button_frame = tk.Frame(train_frame, bg=BG_COLOR)
        button_frame.grid(row=3, column=0, columnspan=3, pady=15)
        
        self.train_button = tk.Button(button_frame, text="▶ Train Model", 
                                     command=self.train_model_thread,
                                     bg=SUCCESS_COLOR, fg="white", 
                                     font=("Arial", 11, "bold"),
                                     padx=20, pady=10)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(button_frame, text="Ready", 
                                    font=("Arial", 9), bg=BG_COLOR, 
                                    fg=TEXT_COLOR)
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Progress bar
        self.progress = ttk.Progressbar(train_frame, length=400, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")

        # Testing Section
        test_frame = tk.LabelFrame(main_frame, text="🧪 Test Emotion Prediction",
                                  font=("Arial", 11, "bold"), bg=BG_COLOR,
                                  fg=TEXT_COLOR, padx=10, pady=10)
        test_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(test_frame, text="Enter text to classify:", font=("Arial", 10),
                bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w", pady=(0, 5))
        
        self.test_text_entry = tk.Entry(test_frame, width=100, font=("Arial", 9))
        self.test_text_entry.pack(fill=tk.X, pady=5)
        self.test_text_entry.insert(0, "i am feeling sad and hopeless")

        test_button_frame = tk.Frame(test_frame, bg=BG_COLOR)
        test_button_frame.pack(anchor="w", pady=10)
        
        self.test_button = tk.Button(test_button_frame, text="🔍 Predict Emotion",
                                    command=self.test_text,
                                    bg=BUTTON_COLOR, fg="white",
                                    font=("Arial", 10, "bold"),
                                    padx=20, pady=8)
        self.test_button.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(test_button_frame, text="",
                                    font=("Arial", 10, "bold"),
                                    bg=BG_COLOR, fg=SUCCESS_COLOR)
        self.result_label.pack(side=tk.LEFT, padx=20)

        # Output Section
        output_frame = tk.LabelFrame(main_frame, text="📋 Output & Logs",
                                    font=("Arial", 11, "bold"), bg=BG_COLOR,
                                    fg=TEXT_COLOR, padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_box = scrolledtext.ScrolledText(output_frame, width=100, height=15,
                                                   font=("Courier", 9),
                                                   bg="white", fg=TEXT_COLOR)
        self.output_box.pack(fill=tk.BOTH, expand=True)
        self.output_box.configure(state="disabled")

        # Configure text tags for styling
        self.output_box.tag_config("success", foreground=SUCCESS_COLOR, font=("Courier", 9, "bold"))
        self.output_box.tag_config("error", foreground=ERROR_COLOR, font=("Courier", 9, "bold"))
        self.output_box.tag_config("info", foreground=BUTTON_COLOR, font=("Courier", 9, "bold"))

    def browse_data(self):
        path = filedialog.askopenfilename(
            title="Select training data file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, path)

    def append_output(self, text: str, tag: str = ""):
        self.output_box.configure(state="normal")
        if tag:
            self.output_box.insert(tk.END, text + "\n", tag)
        else:
            self.output_box.insert(tk.END, text + "\n")
        self.output_box.see(tk.END)
        self.output_box.configure(state="disabled")

    def train_model_thread(self):
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def train_model(self):
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.training = True
        self.train_button.configure(state="disabled")
        self.progress.start()
        self.status_label.configure(text="⏳ Training...", fg="#f39c12")

        data_path = self.data_entry.get().strip() or DEFAULT_DATA_PATH
        model_path = self.model_entry.get().strip() or DEFAULT_MODEL_PATH
        vectorizer_path = self.vectorizer_entry.get().strip() or DEFAULT_VECTORIZER_PATH

        try:
            if not os.path.exists(data_path):
                self.append_output(f"❌ Error: Training data not found: {data_path}", "error")
                self.status_label.configure(text="Error", fg=ERROR_COLOR)
                return

            self.append_output(f"📂 Loading data from: {data_path}", "info")
            df = load_dataset(data_path)
            self.append_output(f"✓ Loaded {len(df)} records", "success")

            self.append_output(f"🔄 Training model...", "info")
            model, vectorizer = train_text_classifier(data_path, model_path, vectorizer_path)
            
            self.model = model
            self.vectorizer = vectorizer
            
            self.append_output(f"✓ Model saved: {model_path}", "success")
            self.append_output(f"✓ Vectorizer saved: {vectorizer_path}", "success")
            self.append_output("✓ Training completed successfully!", "success")
            self.status_label.configure(text="✓ Ready", fg=SUCCESS_COLOR)
            
            messagebox.showinfo("Success", "Model trained successfully!")

        except Exception as exc:
            self.append_output(f"❌ Training failed: {str(exc)}", "error")
            self.status_label.configure(text="Error", fg=ERROR_COLOR)
            messagebox.showerror("Training error", str(exc))

        finally:
            self.training = False
            self.train_button.configure(state="normal")
            self.progress.stop()

    def test_text(self):
        model_path = self.model_entry.get().strip() or DEFAULT_MODEL_PATH
        vectorizer_path = self.vectorizer_entry.get().strip() or DEFAULT_VECTORIZER_PATH
        text = self.test_text_entry.get().strip()

        if not text:
            messagebox.showwarning("Input needed", "Enter text to classify.")
            return

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            self.append_output("❌ Error: Model or vectorizer not found. Train first!", "error")
            messagebox.showerror("Error", "Model or vectorizer file not found. Train first!")
            return

        try:
            if not self.model or not self.vectorizer:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
            
            prediction = predict_emotion(text, self.model, self.vectorizer)
            
            self.append_output(f"📝 Input: {text}", "info")
            self.append_output(f"🎯 Prediction: {prediction}", "success")
            self.result_label.configure(text=f"Result: {prediction}", fg=SUCCESS_COLOR)

        except Exception as exc:
            self.append_output(f"❌ Prediction failed: {str(exc)}", "error")
            self.result_label.configure(text="Prediction failed", fg=ERROR_COLOR)
            messagebox.showerror("Prediction error", str(exc))


if __name__ == "__main__":
    app = TrainUI()
    app.mainloop()
