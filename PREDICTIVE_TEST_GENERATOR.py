import re
import json
import os
import random
import numpy as np
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from tkinter import font as tkfont
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading


class AdvancedTextGenerator:
    def __init__(self, n=3, user_id="default"):
        self.n = n
        self.user_id = user_id
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.custom_words = set()
        self.start_tokens = ['<START>'] * (self.n - 1)

        # Neural network components
        self.nn_model = None
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.max_sequence_len = 10

        # Smoothing parameters
        self.k = 1  # Laplace smoothing factor
        self.backoff_weights = [0.4, 0.3, 0.3]  # For Katz backoff

        # User adaptation tracking
        self.user_adaptation = defaultdict(int)
        self.adaptation_threshold = 10

        self.load_user_model()

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = text.split()
        return tokens

    def train_ngram(self, text):
        tokens = self.preprocess_text(text)
        self.vocab.update(tokens)
        tokens = self.start_tokens + tokens

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_word = tokens[i + self.n - 1]
            self.ngrams[context][next_word] += 1

    def train_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.train_ngram(text)
            self.train_neural_network(text)
            return True
        except Exception as e:
            print(f"Error training from file: {e}")
            return False

    def train_neural_network(self, text):
        try:
            # Tokenize the text
            self.tokenizer.fit_on_texts([text])
            sequences = self.tokenizer.texts_to_sequences([text])[0]

            # Prepare sequences for RNN training
            sequences = [sequences[i:i + self.max_sequence_len + 1]
                         for i in range(len(sequences) - self.max_sequence_len)]

            if not sequences:
                return

            X = np.array([seq[:-1] for seq in sequences])
            y = np.array([seq[-1] for seq in sequences])

            # Build simple RNN model
            vocab_size = len(self.tokenizer.word_index) + 1

            if self.nn_model is None:
                self.nn_model = Sequential([
                    Embedding(vocab_size, 32, input_length=self.max_sequence_len),
                    SimpleRNN(64),
                    Dense(vocab_size, activation='softmax')
                ])
                self.nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

            self.nn_model.fit(X, y, epochs=10, verbose=0)
        except Exception as e:
            print(f"Error training neural network: {e}")

    def add_custom_word(self, word):
        cleaned_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
        if cleaned_word:
            self.custom_words.add(cleaned_word)
            self.vocab.add(cleaned_word)
            self.user_adaptation[cleaned_word] += 1
            self.check_adaptation()

    def check_adaptation(self):
        if len([k for k, v in self.user_adaptation.items() if v > self.adaptation_threshold]) > 5:
            self.adjust_model_to_user()

    def adjust_model_to_user(self):
        for word, count in self.user_adaptation.items():
            if word in self.vocab and count > self.adaptation_threshold:
                for context in self.ngrams:
                    if word in self.ngrams[context]:
                        self.ngrams[context][word] += count // 2

    def laplace_smoothing(self, context, word):
        return (self.ngrams[context].get(word, 0) + self.k) / (
                sum(self.ngrams[context].values()) + self.k * len(self.vocab))

    def katz_backoff(self, context_tokens, word):
        for i in range(len(context_tokens), 0, -1):
            current_context = tuple(context_tokens[-i:])
            if current_context in self.ngrams and word in self.ngrams[current_context]:
                return self.ngrams[current_context][word] * self.backoff_weights[i - 1]
        return 0

    def get_next_word(self, context, num_suggestions=3):
        suggestions = []

        # Try neural network prediction
        try:
            if self.nn_model:
                nn_preds = self.neural_predict(context, num_suggestions)
                suggestions.extend(nn_preds)
        except Exception as e:
            print(f"Neural prediction error: {e}")

        # Fall back to smoothed n-gram model
        context_tokens = self.preprocess_text(context)

        if len(context_tokens) < self.n - 1:
            context_tokens = self.start_tokens[:(self.n - 1 - len(context_tokens))] + context_tokens
        else:
            context_tokens = context_tokens[-(self.n - 1):]

        context_tuple = tuple(context_tokens)

        # Get candidates with smoothing
        candidates = []
        for word in self.vocab:
            prob = (0.7 * self.laplace_smoothing(context_tuple, word) +
                    0.3 * self.katz_backoff(context_tokens, word))

            if word in self.custom_words:
                prob *= 1.5
            candidates.append((word, prob))

        candidates.sort(key=lambda x: -x[1])
        suggestions.extend([word for word, prob in candidates[:num_suggestions]])

        # Remove duplicates and return
        seen = set()
        unique_suggestions = []
        for word in suggestions:
            if word not in seen:
                seen.add(word)
                unique_suggestions.append(word)
            if len(unique_suggestions) >= num_suggestions:
                break

        return unique_suggestions[:num_suggestions]

    def neural_predict(self, context, num_suggestions=3):
        if not self.nn_model or not hasattr(self.tokenizer, 'word_index'):
            return []

        try:
            sequence = self.tokenizer.texts_to_sequences([context])
            if not sequence or not sequence[0]:
                return []

            sequence = pad_sequences(sequence, maxlen=self.max_sequence_len)
            preds = self.nn_model.predict(sequence, verbose=0)[0]
            top_indices = np.argsort(preds)[-num_suggestions:][::-1]

            suggestions = []
            for idx in top_indices:
                word = self.tokenizer.index_word.get(idx, None)
                if word and word != '<OOV>':
                    suggestions.append(word)

            return suggestions
        except Exception as e:
            print(f"Error in neural prediction: {e}")
            return []

    def save_user_model(self):
        try:
            model_data = {
                'n': self.n,
                'ngrams': {str(k): dict(v) for k, v in self.ngrams.items()},
                'vocab': list(self.vocab),
                'custom_words': list(self.custom_words),
                'user_adaptation': dict(self.user_adaptation),
                'tokenizer_config': self.tokenizer.to_json() if hasattr(self.tokenizer, 'to_json') else None
            }

            os.makedirs('user_models', exist_ok=True)
            with open(f'user_models/{self.user_id}_model.json', 'w') as f:
                json.dump(model_data, f)

            if self.nn_model:
                self.nn_model.save(f'user_models/{self.user_id}_nn_model.keras')
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_user_model(self):
        try:
            model_file = f'user_models/{self.user_id}_model.json'
            if os.path.exists(model_file):
                with open(model_file, 'r') as f:
                    model_data = json.load(f)

                self.n = model_data['n']
                self.ngrams = defaultdict(Counter)
                for k, v in model_data['ngrams'].items():
                    context = tuple(eval(k))
                    self.ngrams[context] = Counter(v)

                self.vocab = set(model_data['vocab'])
                self.custom_words = set(model_data['custom_words'])
                self.user_adaptation = defaultdict(int, model_data['user_adaptation'])
                self.start_tokens = ['<START>'] * (self.n - 1)

                if model_data.get('tokenizer_config'):
                    self.tokenizer = Tokenizer()
                    self.tokenizer.from_json(model_data['tokenizer_config'])

                nn_file = f'user_models/{self.user_id}_nn_model.keras'
                if os.path.exists(nn_file):
                    self.nn_model = load_model(nn_file)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class PredictiveTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Predictive Text Generator")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        # Set application icon (if available)
        try:
            self.root.iconbitmap("icon.ico")  # You can add an icon file
        except:
            pass

        # Create generator instance
        self.generator = AdvancedTextGenerator(user_id="user1")

        # Configure theme colors
        self.bg_color = "#f5f5f5"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#6e9cd2"
        self.accent_color = "#ff6b6b"

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure styles
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 9))
        self.style.configure('Suggest.TButton', font=('Segoe UI', 10, 'bold'),
                             foreground='white', background=self.secondary_color)
        self.style.configure('Primary.TButton', font=('Segoe UI', 9, 'bold'),
                             foreground='white', background=self.primary_color)
        self.style.configure('Accent.TButton', font=('Segoe UI', 9),
                             foreground='white', background=self.accent_color)
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'),
                             foreground=self.primary_color)
        self.style.configure('Status.TLabel', font=('Segoe UI', 9),
                             foreground='#666666')

        # GUI Components
        self.create_widgets()

        # Load some initial training data
        self.load_initial_data()

        # Bind keyboard shortcuts
        self.setup_keyboard_shortcuts()

    def create_widgets(self):
        # Configure root background
        self.root.configure(bg=self.bg_color)

        # Create main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="Text Input")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.stats_tab, text="Statistics")

        # Build each tab
        self.build_main_tab()
        self.build_training_tab()
        self.build_stats_tab()

        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.status_frame, textvariable=self.status_var,
                                    style='Status.TLabel', relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)

        # Progress bar for training operations
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')

    def build_main_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.main_tab, text="Text Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Text area with scrollbar
        text_frame = ttk.Frame(input_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD,
                                                   font=('Segoe UI', 11),
                                                   height=15, padx=10, pady=10)
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.bind('<KeyRelease>', self.on_key_release)
        self.text_area.focus_set()

        # Suggestions frame
        suggestions_frame = ttk.LabelFrame(self.main_tab, text="Word Suggestions", padding=10)
        suggestions_frame.pack(fill=tk.X, pady=(0, 10))

        self.suggestion_buttons = []
        btn_frame = ttk.Frame(suggestions_frame)
        btn_frame.pack(fill=tk.X)

        for i in range(5):  # Increased from 3 to 5 suggestions
            btn = ttk.Button(btn_frame, text="", style='Suggest.TButton',
                             command=lambda idx=i: self.use_suggestion(idx))
            btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.suggestion_buttons.append(btn)

        # Quick actions frame
        actions_frame = ttk.Frame(self.main_tab)
        actions_frame.pack(fill=tk.X)

        ttk.Button(actions_frame, text="Add Custom Word", style='Primary.TButton',
                   command=self.add_custom_word).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Clear Text", style='Accent.TButton',
                   command=self.clear_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Save Text", style='Primary.TButton',
                   command=self.save_text).pack(side=tk.LEFT, padx=5)

    def build_training_tab(self):
        # Training options frame
        training_frame = ttk.LabelFrame(self.training_tab, text="Training Options", padding=15)
        training_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # File training section
        file_frame = ttk.Frame(training_frame)
        file_frame.pack(fill=tk.X, pady=10)

        ttk.Label(file_frame, text="Train from text file:", style='Header.TLabel').pack(anchor=tk.W)

        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_btn_frame, text="Select File", style='Primary.TButton',
                   command=self.train_from_file).pack(side=tk.LEFT, padx=5)

        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        ttk.Label(file_btn_frame, textvariable=self.file_path_var, style='Status.TLabel').pack(side=tk.LEFT, padx=10)

        # Direct text input training
        text_train_frame = ttk.Frame(training_frame)
        text_train_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(text_train_frame, text="Train from text input:", style='Header.TLabel').pack(anchor=tk.W)

        self.train_text_area = scrolledtext.ScrolledText(text_train_frame, wrap=tk.WORD,
                                                         height=8, font=('Segoe UI', 10))
        self.train_text_area.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Button(text_train_frame, text="Train from Text", style='Primary.TButton',
                   command=self.train_from_text).pack(anchor=tk.E, pady=5)

        # Model management
        model_frame = ttk.Frame(training_frame)
        model_frame.pack(fill=tk.X, pady=10)

        ttk.Button(model_frame, text="Save Model", style='Primary.TButton',
                   command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Reset Model", style='Accent.TButton',
                   command=self.reset_model).pack(side=tk.LEFT, padx=5)

    def build_stats_tab(self):
        # Stats frame
        stats_frame = ttk.LabelFrame(self.stats_tab, text="Model Statistics", padding=15)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Stats display
        stats_text_frame = ttk.Frame(stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = scrolledtext.ScrolledText(stats_text_frame, wrap=tk.WORD,
                                                    height=15, font=('Consolas', 10),
                                                    state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Refresh button
        ttk.Button(stats_frame, text="Refresh Statistics", style='Primary.TButton',
                   command=self.update_stats).pack(anchor=tk.E, pady=5)

    def setup_keyboard_shortcuts(self):
        # Bind Ctrl+Number to select suggestions
        for i in range(len(self.suggestion_buttons)):
            self.root.bind(f'<Control-Key-{i + 1}>', lambda e, idx=i: self.use_suggestion(idx))

        # Bind Ctrl+S to save
        self.root.bind('<Control-s>', lambda e: self.save_model())

        # Bind Ctrl+O to open file
        self.root.bind('<Control-o>', lambda e: self.train_from_file())

        # Bind Ctrl+L to clear text
        self.root.bind('<Control-l>', lambda e: self.clear_text())

    def load_initial_data(self):
        # Load default training data if available
        default_files = ['sample_text.txt']
        trained = False

        for file in default_files:
            if os.path.exists(file):
                if self.generator.train_from_file(file):
                    self.status_var.set(f"Trained on {file}")
                    trained = True
                    self.root.update()

        # If no files found, use minimal built-in data
        if not trained:
            default_text = """
                This is an enhanced predictive text generator with neural networks.
                It learns from your typing and adapts to your writing style.
                Try typing something to see word suggestions appear.
                The application includes smoothing techniques for better predictions.
                You can add custom words and train from your own text files.
            """
            self.generator.train_ngram(default_text)
            self.generator.train_neural_network(default_text)
            self.status_var.set("Using default training data")

        # Update stats
        self.update_stats()

    def on_key_release(self, event):
        # Get current text and cursor position
        current_text = self.text_area.get("1.0", tk.END).strip()
        if current_text:
            # Get the last few words for context
            words = current_text.split()
            context = " ".join(words[-5:]) if len(words) > 5 else current_text

            suggestions = self.generator.get_next_word(context, len(self.suggestion_buttons))
            for i, btn in enumerate(self.suggestion_buttons):
                if i < len(suggestions):
                    btn.config(text=suggestions[i], state=tk.NORMAL)
                else:
                    btn.config(text="", state=tk.DISABLED)
        else:
            for btn in self.suggestion_buttons:
                btn.config(text="", state=tk.DISABLED)

    def use_suggestion(self, index):
        suggestion = self.suggestion_buttons[index].cget("text")
        if suggestion:
            # Insert the suggestion at the current cursor position
            self.text_area.insert(tk.INSERT, suggestion + " ")

            # Track usage for adaptation
            self.generator.user_adaptation[suggestion] += 1
            self.generator.check_adaptation()

            # Update suggestions
            self.on_key_release(None)

    def add_custom_word(self):
        # Get the current word at cursor using a simpler approach
        current_pos = self.text_area.index(tk.INSERT)

        # Get the text from the beginning to cursor
        text_before = self.text_area.get("1.0", current_pos)

        # Split into words and get the last word
        words = text_before.split()
        if words:
            word = words[-1]

            # Clean the word (remove any punctuation)
            word = re.sub(r'[^a-zA-Z0-9]', '', word)

            if word:
                self.generator.add_custom_word(word)
                self.status_var.set(f"Added '{word}' to custom words")
                if self.generator.save_user_model():
                    self.status_var.set(f"Added '{word}' and saved model")

                # Update stats
                self.update_stats()
                return

        # If no word was found, show a message
        messagebox.showinfo("Info", "Please place your cursor after a word to add it as a custom word.")

    def train_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select training file",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )

        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            self.status_var.set(f"Training from {os.path.basename(file_path)}...")
            self.progress.pack(fill=tk.X, pady=5)
            self.progress.start()

            # Run training in a separate thread to avoid UI freezing
            def training_thread():
                try:
                    if self.generator.train_from_file(file_path):
                        self.root.after(0, lambda: self.status_var.set(f"Trained on {os.path.basename(file_path)}"))
                        if self.generator.save_user_model():
                            self.root.after(0, lambda: self.status_var.set(f"Trained and saved model"))
                    self.root.after(0, self.progress.stop)
                    self.root.after(0, self.progress.pack_forget)
                    self.root.after(0, self.update_stats)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to train from file: {str(e)}"))
                    self.root.after(0, self.progress.stop)
                    self.root.after(0, self.progress.pack_forget)

            threading.Thread(target=training_thread, daemon=True).start()

    def train_from_text(self):
        text = self.train_text_area.get("1.0", tk.END).strip()
        if text:
            self.status_var.set("Training from text...")
            self.progress.pack(fill=tk.X, pady=5)
            self.progress.start()

            # Run training in a separate thread
            def training_thread():
                try:
                    self.generator.train_ngram(text)
                    self.generator.train_neural_network(text)
                    self.root.after(0, lambda: self.status_var.set("Trained from text input"))
                    if self.generator.save_user_model():
                        self.root.after(0, lambda: self.status_var.set("Trained and saved model"))
                    self.root.after(0, self.progress.stop)
                    self.root.after(0, self.progress.pack_forget)
                    self.root.after(0, self.update_stats)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to train from text: {str(e)}"))
                    self.root.after(0, self.progress.stop)
                    self.root.after(0, self.progress.pack_forget)

            threading.Thread(target=training_thread, daemon=True).start()
        else:
            messagebox.showwarning("Warning", "Please enter some text to train on")

    def save_model(self):
        if self.generator.save_user_model():
            self.status_var.set("Model saved successfully")
            self.update_stats()
        else:
            self.status_var.set("Error saving model")

    def reset_model(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to reset the model? All training data will be lost."):
            self.generator = AdvancedTextGenerator(user_id="user1")
            self.status_var.set("Model reset")
            self.load_initial_data()

    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        for btn in self.suggestion_buttons:
            btn.config(text="", state=tk.DISABLED)

    def save_text(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Text",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_area.get("1.0", tk.END))
                self.status_var.set(f"Text saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text: {str(e)}")

    def update_stats(self):
        stats = f"""Vocabulary Size: {len(self.generator.vocab)}
Custom Words: {len(self.generator.custom_words)}
N-gram Contexts: {len(self.generator.ngrams)}
User Adaptation Words: {len([k for k, v in self.generator.user_adaptation.items() if v > 0])}

Top 10 Custom Words:
"""

        # Get top custom words by usage
        sorted_adaptation = sorted(self.generator.user_adaptation.items(),
                                   key=lambda x: x[1], reverse=True)[:10]
        for word, count in sorted_adaptation:
            if count > 0:
                stats += f"  {word}: {count} uses\n"

        # Enable text widget, update, then disable again
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats)
        self.stats_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = PredictiveTextApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")
        raise