import tkinter as tk # Import the tkinter library
from tkinter import ttk, messagebox, filedialog # Import the ttk, messagebox, and filedialog classes
from PIL import Image, ImageTk # Import the Image and ImageTk classes  # pyright: ignore[reportMissingImports]
import cv2 # Import the OpenCV library  # pyright: ignore[reportMissingImports]
import numpy as np # Import the NumPy library  # pyright: ignore[reportMissingImports]
import tensorflow as tf # Import the TensorFlow library  # pyright: ignore[reportMissingImports, reportUnusedImport, reportUnusedImport]
from tensorflow import keras # Import the Keras library  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split # Import the train_test_split function  # pyright: ignore[reportMissingImports]
from sklearn.utils import class_weight # Import the class_weight class  # pyright: ignore[reportMissingImports, reportUnusedImport, reportUnusedImport, reportUnusedImport, reportUnusedImport]
import os # Import the os library  # pyright: ignore[reportMissingImports]
import pickle # Import the pickle library  # pyright: ignore[reportMissingImports]
import threading # Import the threading library  # pyright: ignore[reportMissingImports]

# Constants
IMG_SIZE = 64 # The size of the image
PADDING_RATIO = 0.05 # The padding ratio
LETTER_LABELS = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 
                 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 
                 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25} # The labels for the characters
CORRECTIONS_FILE = 'user_corrections.pkl' # Global variable for the corrections file

class CursiveGraderGUI:
    def __init__(self, root): # Function to initialize the GUI
        self.root = root # Set the root window
        self.root.title("Cursive Handwriting Grader - Unified System") # Set the title of the window
        self.root.geometry("1400x900") # Set the geometry of the window
        self.root.minsize(1200, 800) # Set the minimum size of the window
        self.root.configure(bg="#f0f0f0") # Configure the background color of the window

        # Shared state
        self.model = None # Initialize the model to None
        self.user_corrections = {'images': [], 'labels': []} # Initialize the user corrections dictionary
        self.load_corrections() # Load the corrections from the file

        # Training tab state
        self.training_data = None # Initialize the training data to None
        self.training_labels = None # Initialize the training labels to None
        self.training_model = None # Initialize the training model to None
        self.training_base_path = None # Initialize the training base path to None

        # Online learning tab state
        self.online_drawing = False # Global variable for drawing
        self.online_img_draw = np.ones((400, 400, 3), dtype=np.uint8) * 255 # Create a canvas
        self.online_last_point = None # Global variable for the last point
        self.online_correction_count = 0 # Initialize the correction count to 0
        self.online_last_prediction = None # Initialize the last prediction to None

        # Grading tab state
        self.grading_image = None # Initialize the grading image to None
        self.grading_characters = [] # Initialize the grading characters list
        self.grading_predictions = [] # Initialize the grading predictions list
        self.grading_char_index = 0 # Initialize the grading character index to 0
        self.grading_current_photo = None # Initialize the grading current photo to None

        self.setup_ui() # Set up the UI

    def setup_ui(self): # Function to set up the UI
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root) # Create a notebook widget
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # Pack the notebook widget

        # Create tabs
        self.training_frame = ttk.Frame(self.notebook, padding=20) # Create a frame for the training tab
        self.online_frame = ttk.Frame(self.notebook, padding=20) # Create a frame for the online learning tab
        self.grading_frame = ttk.Frame(self.notebook, padding=20) # Create a frame for the grading tab

        self.notebook.add(self.training_frame, text="Train Model") # Add the training frame to the notebook
        self.notebook.add(self.online_frame, text="Online Learning") # Add the online learning frame to the notebook
        self.notebook.add(self.grading_frame, text="Grade Students") # Add the grading frame to the notebook

        self.setup_training_tab() # Set up the training tab
        self.setup_online_learning_tab() # Set up the online learning tab
        self.setup_grading_tab() # Set up the grading tab

    # ==================== TRAINING TAB ====================
    def setup_training_tab(self):
        main = ttk.Frame(self.training_frame)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="MODEL TRAINING", font=('Arial', 24, 'bold')).pack(pady=(0, 20))

        # Left panel - Controls
        left = ttk.Frame(main, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left.pack_propagate(False)

        ttk.Label(left, text="1. Select Training Data Folder", font=('Arial', 14, 'bold')).pack(anchor='w', pady=(10, 8))
        ttk.Label(left, text="Folder should contain A.jpg, B.jpg, etc.", font=('Arial', 10), foreground='gray').pack(anchor='w', padx=15)
        ttk.Button(left, text="Select Folder", command=self.select_training_folder).pack(fill=tk.X, pady=5)
        self.training_folder_label = ttk.Label(left, text="No folder selected", foreground='red', font=('Arial', 10))
        self.training_folder_label.pack(anchor='w', padx=15, pady=(0, 20))

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="2. Training Parameters", font=('Arial', 14, 'bold')).pack(anchor='w', pady=(10, 8))
        
        params_frame = ttk.Frame(left)
        params_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(params_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="40")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT, padx=5)

        params_frame2 = ttk.Frame(left)
        params_frame2.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(params_frame2, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.StringVar(value="8")
        ttk.Entry(params_frame2, textvariable=self.batch_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Separator(left).pack(fill=tk.X, pady=20)

        self.train_btn = ttk.Button(left, text="TRAIN MODEL", command=self.start_training, style="Big.TButton")
        self.train_btn.pack(fill=tk.X, pady=20, ipady=15)

        self.training_status = ttk.Label(left, text="", font=('Arial', 11), foreground='blue')
        self.training_status.pack(pady=10)

        # Right panel - Progress and info
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Training Progress", font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.training_log = tk.Text(right, height=25, font=('Consolas', 10), wrap=tk.WORD, bg='#fafafa')
        self.training_log.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(right, command=self.training_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.config(yscrollcommand=scrollbar.set)

    def select_training_folder(self): # Function to select the training folder
        folder = filedialog.askdirectory(title="Select folder containing training images (A.jpg, B.jpg, etc.)") # Ask the user to select a folder
        if folder: # Check if a folder was selected
            self.training_base_path = folder # Set the training base path to the selected folder
            self.training_folder_label.config(text=f"Selected: {os.path.basename(folder)}", foreground='green') # Update the label to show the selected folder
            self.log_training(f"Selected folder: {folder}") # Log the selected folder

    def load_characters_from_image(self, image_path, label, padding_ratio=0.05): # Function to load characters from an image
        """Load characters from a scanned cursive sheet"""
        img = cv2.imread(image_path) # Read the image file
        if img is None: # Check if the image is None
            return [], [] # Return empty lists if the image is None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # Convert the image to binary
        kernel = np.ones((3,3), np.uint8) # Create a kernel for the morphological operation
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1) # Apply the morphological operation
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contours in the binary image
        valid_contours = [c for c in contours if 200 < cv2.contourArea(c) < 50000] # Find the valid contours in the binary image

        processed_chars = [] # Initialize the list of processed characters
        labels = [] # Initialize the list of labels

        for cnt in valid_contours: # Loop through the valid contours
            x, y, w, h = cv2.boundingRect(cnt) # Get the bounding rectangle for the contour
            pad = int(padding_ratio * max(w, h)) # Calculate the padding for the character
            x0 = max(0, x - pad) # Calculate the start x coordinate of the character
            y0 = max(0, y - pad) # Calculate the start y coordinate of the character
            x1 = min(img.shape[1], x + w + pad) # Calculate the end x coordinate of the character
            y1 = min(img.shape[0], y + h + pad) # Calculate the end y coordinate of the character

            cropped = img[y0:y1, x0:x1] # Crop the image to the character
            char_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # Convert the cropped image to grayscale
            h_actual, w_actual = char_gray.shape # Get the actual height and width of the character
            size = max(h_actual, w_actual) # Calculate the size of the character
            square = np.ones((size, size), dtype=np.uint8) * 255 # Create a square of the size of the character
            y_offset = (size - h_actual) // 2 # Calculate the offset for the character
            x_offset = (size - w_actual) // 2 # Calculate the offset for the character
            square[y_offset:y_offset + h_actual, x_offset:x_offset + w_actual] = char_gray # Center the character in the square
            resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE)) # Resize the character to the size of the image
            normalized = resized / 255.0 # Normalize the character
            processed_chars.append(normalized) # Add the normalized character to the list of processed characters
            labels.append(label) # Add the label to the list of labels

        return processed_chars, labels # Return the list of processed characters and the list of labels

    def log_training(self, message): # Function to log training messages
        self.training_log.insert(tk.END, message + "\n") # Insert the message into the training log
        self.training_log.see(tk.END) # Scroll to the end of the log
        self.root.update() # Update the root window

    def start_training(self): # Function to start the training
        if not self.training_base_path: # Check if the training base path is not set
            messagebox.showwarning("Missing", "Please select a training data folder first!") # Show a warning message
            return # Return if the training base path is not set

        self.train_btn.config(state=tk.DISABLED) # Disable the train button
        self.training_status.config(text="Training in progress...", foreground='blue') # Update the training status
        self.training_log.delete(1.0, tk.END) # Clear the training log

        # Run training in separate thread
        thread = threading.Thread(target=self.train_model_thread) # Create a thread for training
        thread.daemon = True # Set the thread as a daemon thread
        thread.start() # Start the thread

    def train_model_thread(self): # Function to train the model in a separate thread
        try: # Try to train the model
            self.log_training("="*50) # Print a separator
            self.log_training("LOADING TRAINING DATA") # Print a message to indicate that training data is being loaded
            self.log_training("="*50) # Print a separator

            all_processed_chars, all_labels = [], [] # Initialize the list of processed characters and the list of labels
            for letter, label in LETTER_LABELS.items(): # Loop through the letter labels
                path = os.path.join(self.training_base_path, f"{letter}.jpg") # Join the path to the training base path with the letter
                if os.path.exists(path): # Check if the path exists
                    chars, labels = self.load_characters_from_image(path, label, padding_ratio=PADDING_RATIO) # Load the characters from the image
                    all_processed_chars.extend(chars) # Add the processed characters to the list of processed characters
                    all_labels.extend(labels) # Add the labels to the list of labels
                    self.log_training(f"→ Loaded {len(chars)} samples from {letter}.jpg") # Print the number of samples loaded from the image
                else: # If the path does not exist
                    self.log_training(f"Warning: {letter}.jpg not found, skipping...") # Print a warning message

            if len(all_processed_chars) == 0: # Check if there are no processed characters
                self.log_training("ERROR: No training data found!") # Print an error message
                return # Return if there are no processed characters

            X = np.array(all_processed_chars).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Reshape the processed characters to the size of the image
            y = np.array(all_labels) # Convert the labels to an array

            self.log_training(f"\nTotal samples: {len(y)}") # Print the total number of samples
            self.log_training("\nData balance:")
            for letter, label in LETTER_LABELS.items():
                count = np.sum(y == label) # Count the number of samples for the letter
                if count > 0: # Check if there are any samples for the letter
                    percentage = (count / len(y)) * 100 # Calculate the percentage of samples for the letter
                    self.log_training(f"  {letter}: {count} samples ({percentage:.1f}%)") # Print the number of samples for the letter and the percentage

            # Build model
            self.log_training("\n" + "="*50)
            self.log_training("BUILDING MODEL")
            self.log_training("="*50)

            model = keras.Sequential([ # Create a sequential model
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                keras.layers.BatchNormalization(), # Create a batch normalization layer
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)), # Create a max pooling layer
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(64, (3, 3), activation='relu'), # Create a convolutional layer
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(64, (3, 3), activation='relu'), # Create a convolutional layer
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(128, (3, 3), activation='relu'), # Create a convolutional layer
                keras.layers.BatchNormalization(),
                keras.layers.Flatten(), # Create a flatten layer
                keras.layers.Dense(128, activation='relu'), # Create a dense layer
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5), # Create a dropout layer
                keras.layers.Dense(len(LETTER_LABELS), activation='softmax') # Create a dense layer
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile the model with the Adam optimizer

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Split the data into training and validation sets

            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train) # Compute the class weights
            class_weight_dict = dict(enumerate(class_weights)) # Create a dictionary of the class weights

            epochs = int(self.epochs_var.get()) # Get the number of epochs
            batch_size = int(self.batch_var.get()) # Get the batch size

            self.log_training("\n" + "="*50)
            self.log_training("TRAINING MODEL")
            self.log_training("="*50)
            self.log_training(f"Epochs: {epochs}, Batch Size: {batch_size}")

            # Custom callback to log progress
            class TrainingCallback(keras.callbacks.Callback):
                def __init__(self, log_func, total_epochs):
                    self.log_func = log_func
                    self.total_epochs = total_epochs
                    
                def on_epoch_end(self, epoch, logs=None):
                    epoch_num = epoch + 1
                    self.log_func(f"Epoch {epoch_num}/{self.total_epochs} - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Acc: {logs['val_accuracy']:.4f}")

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                verbose=0,
                callbacks=[TrainingCallback(self.log_training, epochs)]
            )

            # Save model
            model_path = 'cursive_letter_classifier_improved.h5' # Set the model path
            model.save(model_path) # Save the model to the file
            self.model = model # Set the model
            self.training_model = model # Set the training model

            # Evaluate
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1) # Predict the validation data
            accuracy = np.mean(y_pred == y_val) # Calculate the accuracy
            
            self.log_training("\n" + "="*50) # Print a separator
            self.log_training("TRAINING COMPLETE") # Print a message to indicate that training is complete
            self.log_training("="*50) # Print a separator
            self.log_training(f"Validation Accuracy: {accuracy:.2%}") # Print the validation accuracy
            self.log_training(f"Model saved to: {model_path}") # Print the model path

            self.training_status.config(text=f"Training complete! Accuracy: {accuracy:.2%}", foreground='green') # Update the training status
            self.train_btn.config(state=tk.NORMAL) # Enable the train button

        except Exception as e: # Catch any exceptions
            self.log_training(f"\nERROR: {str(e)}") # Print the error message
            self.training_status.config(text="Training failed!", foreground='red') # Update the training status
            self.train_btn.config(state=tk.NORMAL) # Enable the train button
            messagebox.showerror("Training Error", str(e)) # Show an error message

    # ==================== ONLINE LEARNING TAB ====================
    def setup_online_learning_tab(self):
        main = ttk.Frame(self.online_frame)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="ONLINE LEARNING", font=('Arial', 24, 'bold')).pack(pady=(0, 20))

        # Top panel - Model loading
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(top, text="Load Model:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Select .h5 Model", command=self.load_model_for_online).pack(side=tk.LEFT, padx=5)
        self.online_model_label = ttk.Label(top, text="No model loaded", foreground='red', font=('Arial', 10))
        self.online_model_label.pack(side=tk.LEFT, padx=10)

        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # Main content
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        # Left - Drawing canvas
        left = ttk.Frame(content, width=500)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 20))
        left.pack_propagate(False)

        ttk.Label(left, text="Draw a Character", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        self.draw_canvas = tk.Canvas(left, bg='white', width=400, height=400, highlightthickness=2, highlightbackground="#333")
        self.draw_canvas.pack(pady=10)
        self.draw_canvas.bind("<Button-1>", self.on_draw_start)
        self.draw_canvas.bind("<B1-Motion>", self.on_draw_move)
        self.draw_canvas.bind("<ButtonRelease-1>", self.on_draw_end)

        buttons_frame = ttk.Frame(left)
        buttons_frame.pack(pady=10)
        ttk.Button(buttons_frame, text="Clear", command=self.clear_drawing).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Predict", command=self.predict_drawn).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Retrain with Corrections", command=self.retrain_online).pack(side=tk.LEFT, padx=5)

        # Right - Results and corrections
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Prediction Results", font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.online_result_text = tk.Text(right, height=15, font=('Arial', 12), wrap=tk.WORD, bg='#fafafa')
        self.online_result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(right, text="Correction", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(10, 5))
        correction_frame = ttk.Frame(right)
        correction_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(correction_frame, text="Correct Letter:").pack(side=tk.LEFT, padx=5)
        self.correction_entry = ttk.Entry(correction_frame, width=5)
        self.correction_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(correction_frame, text="Add Correction", command=self.add_correction_online).pack(side=tk.LEFT, padx=5)

        self.online_status = ttk.Label(right, text="", font=('Arial', 11), foreground='blue')
        self.online_status.pack(pady=10)

    def load_model_for_online(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.h5")])
        if path:
            try:
                self.model = keras.models.load_model(path)
                self.online_model_label.config(text=f"Loaded: {os.path.basename(path)}", foreground='green')
                self.online_status.config(text="Model loaded successfully!", foreground='green')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def on_draw_start(self, event): # Function to handle the start of drawing
        self.online_drawing = True # Set the drawing flag to True
        self.online_last_point = (event.x, event.y) # Set the last point to the current point
        cv2.circle(self.online_img_draw, (event.x, event.y), 3, (0, 0, 0), -1) # Draw a circle at the current point
        self.draw_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill='black', outline='black') # Draw an oval on the canvas

    def on_draw_move(self, event): # Function to handle the movement of the mouse while drawing
        if self.online_drawing and self.online_last_point: # Check if drawing is active and there is a last point
            cv2.line(self.online_img_draw, self.online_last_point, (event.x, event.y), (0, 0, 0), 3) # Draw a line from the last point to the current point
            self.draw_canvas.create_line(self.online_last_point[0], self.online_last_point[1], 
                                         event.x, event.y, fill='black', width=3, capstyle=tk.ROUND) # Draw a line on the canvas
            self.online_last_point = (event.x, event.y) # Set the last point to the current point

    def on_draw_end(self, event): # Function to handle the end of drawing
        if self.online_drawing and self.online_last_point: # Check if drawing is active and there is a last point
            cv2.line(self.online_img_draw, self.online_last_point, (event.x, event.y), (0, 0, 0), 3) # Draw a line from the last point to the current point
            self.draw_canvas.create_line(self.online_last_point[0], self.online_last_point[1], 
                                         event.x, event.y, fill='black', width=3, capstyle=tk.ROUND) # Draw a line on the canvas
        self.online_drawing = False # Set the drawing flag to False

    def clear_drawing(self):
        self.draw_canvas.delete("all")
        self.online_img_draw = np.ones((400, 400, 3), dtype=np.uint8) * 255
        self.online_result_text.delete(1.0, tk.END)
        self.online_last_prediction = None

    def get_drawing_image(self):
        """Get drawing from internal numpy array"""
        return self.online_img_draw.copy()

    def auto_crop_and_process(self, img, padding_ratio=0.05, img_size=64, invert=False): # Function to auto-crop and process the image
        """Auto-crop and process image"""
        if len(img.shape) == 3: # Check if the image has 3 dimensions
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        else: # If the image does not have 3 dimensions
            gray = img.copy() # Copy the image
        
        if invert: # Check if the image should be inverted
            gray = 255 - gray # Invert the image
        
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # Convert the image to binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contours in the binary image
        if not contours: # Check if there are no contours
            return None # Return None if there are no contours
        
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea)) # Get the bounding rectangle for the largest contour
        pad = int(padding_ratio * max(w, h)) # Calculate the padding
        x_start = max(0, x - pad) # Calculate the start x coordinate
        y_start = max(0, y - pad) # Calculate the start y coordinate
        x_end = min(gray.shape[1], x + w + pad) # Calculate the end x coordinate
        y_end = min(gray.shape[0], y + h + pad) # Calculate the end y coordinate
        
        cropped = gray[y_start:y_end, x_start:x_end] # Crop the image
        h, w = cropped.shape # Get the height and width of the cropped image
        size = max(h, w) # Calculate the size
        square = np.ones((size, size), dtype=np.uint8) * 255 # Create a square
        y_offset = (size - h) // 2 # Calculate the y offset
        x_offset = (size - w) // 2 # Calculate the x offset
        square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped # Place the cropped image in the square
        
        resized = cv2.resize(square, (img_size, img_size)) # Resize the square to the image size
        normalized = resized / 255.0 # Normalize the image
        return normalized # Return the normalized image

    def predict_drawn(self):
        if self.model is None:
            messagebox.showwarning("Missing", "Please load a model first!")
            return

        drawn_img = self.get_drawing_image()
        if drawn_img is None:
            return

        processed = self.auto_crop_and_process(drawn_img, padding_ratio=PADDING_RATIO, img_size=IMG_SIZE, invert=True)
        if processed is None:
            self.online_result_text.delete(1.0, tk.END)
            self.online_result_text.insert(tk.END, "No valid character detected. Please draw a character.")
            return

        img_input = processed.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        preds = self.model.predict(img_input, verbose=0)[0]
        pred_label = np.argmax(preds)
        confidence = preds[pred_label]

        label_to_letter = {v: k for k, v in LETTER_LABELS.items()}
        predicted_letter = label_to_letter[pred_label]
        all_probs = {label_to_letter[i]: preds[i] for i in range(len(preds))}

        grade = 100 if confidence >= 0.9 else 0
        feedback = "EXCELLENT!" if grade == 100 else "SLOPPY HANDWRITING!"

        self.online_result_text.delete(1.0, tk.END)
        result = f"PREDICTED LETTER: {predicted_letter}\n\n"
        result += f"CONFIDENCE: {confidence:.1%}\n\n"
        result += f"GRADE: {grade}/100\n\n"
        result += f"{feedback}\n\n"
        result += "All Probabilities:\n"
        for letter, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            result += f"  {letter}: {prob:.1%}\n"

        self.online_result_text.insert(1.0, result)
        self.online_last_prediction = (processed, predicted_letter, confidence)

    def add_correction_online(self):
        if not hasattr(self, 'online_last_prediction'):
            messagebox.showwarning("Missing", "Please predict a character first!")
            return

        correct_letter = self.correction_entry.get().strip().upper()
        if correct_letter not in LETTER_LABELS:
            messagebox.showerror("Invalid", f"Please enter a valid letter (A-Z)")
            return

        processed, predicted_letter, confidence = self.online_last_prediction
        
        if correct_letter == predicted_letter:
            self.online_status.config(text="Prediction was correct! No correction needed.", foreground='green')
            return

        self.user_corrections['images'].append(processed)
        self.user_corrections['labels'].append(LETTER_LABELS[correct_letter])
        self.save_corrections()
        self.online_correction_count += 1

        self.online_status.config(text=f"Correction added! ({self.online_correction_count} total corrections)", foreground='green')
        self.correction_entry.delete(0, tk.END)

        # Auto-retrain after 5 corrections
        if self.online_correction_count >= 5:
            self.online_status.config(text=f"Auto-retraining with {self.online_correction_count} corrections...", foreground='blue')
            self.retrain_online()

    def retrain_online(self):
        if self.model is None:
            messagebox.showwarning("Missing", "Please load a model first!")
            return

        if len(self.user_corrections['images']) == 0:
            messagebox.showinfo("No Corrections", "No corrections to train on yet!")
            return

        self.online_status.config(text="Retraining model... This may take a moment.", foreground='blue')
        self.root.update()

        try:
            correction_imgs = np.array(self.user_corrections['images']).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            correction_labels = np.array(self.user_corrections['labels'])

            # Freeze early layers
            for layer in self.model.layers[:-4]:
                layer.trainable = False

            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            self.model.fit(correction_imgs, correction_labels, epochs=5, batch_size=8, verbose=0)

            # Unfreeze
            for layer in self.model.layers:
                layer.trainable = True

            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.model.save('cursive_letter_classifier_improved.h5')

            self.online_correction_count = 0
            self.online_status.config(text="Model updated and saved successfully!", foreground='green')
            messagebox.showinfo("Success", "Model retrained and saved!")

        except Exception as e:
            self.online_status.config(text=f"Retraining failed: {str(e)}", foreground='red')
            messagebox.showerror("Error", f"Retraining failed:\n{e}")

    # ==================== GRADING TAB ====================
    def setup_grading_tab(self):
        main = ttk.Frame(self.grading_frame)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="STUDENT GRADING", font=('Arial', 24, 'bold')).pack(pady=(0, 20))

        # Top controls
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(top, text="Load Model:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Select .h5 Model", command=self.load_model_for_grading).pack(side=tk.LEFT, padx=5)
        self.grading_model_label = ttk.Label(top, text="No model loaded", foreground='red', font=('Arial', 10))
        self.grading_model_label.pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Load Image:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(top, text="Select Student Image", command=self.load_image_for_grading).pack(side=tk.LEFT, padx=5)
        self.grading_image_label = ttk.Label(top, text="No image loaded", foreground='red', font=('Arial', 10))
        self.grading_image_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="EXTRACT & GRADE", command=self.extract_and_grade, style="Big.TButton").pack(side=tk.LEFT, padx=20)

        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # Content area
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        # Left - Original image
        left = ttk.Frame(content, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left.pack_propagate(False)

        ttk.Label(left, text="Original Image", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        self.grading_original_canvas = tk.Canvas(left, bg='white', height=300, highlightthickness=2, highlightbackground="#ccc")
        self.grading_original_canvas.pack(fill=tk.BOTH, expand=True)

        # Right - Results
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grading_status = ttk.Label(right, text="Load model and image, then click 'EXTRACT & GRADE'", 
                                        font=('Arial', 12), foreground='gray')
        self.grading_status.pack(pady=20)

    def load_model_for_grading(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.h5")])
        if path:
            try:
                self.model = keras.models.load_model(path)
                self.grading_model_label.config(text=f"Loaded: {os.path.basename(path)}", foreground='green')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def load_image_for_grading(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.grading_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.grading_image is not None:
                self.grading_image_label.config(text=f"Loaded: {os.path.basename(path)}", foreground='green')
                self.display_grading_image()

    def display_grading_image(self):
        if self.grading_image is None:
            return
        h, w = self.grading_image.shape
        canvas_w = self.grading_original_canvas.winfo_width() or 380
        canvas_h = self.grading_original_canvas.winfo_height() or 280
        scale = min(canvas_w / w, canvas_h / h)
        disp_h, disp_w = int(h * scale), int(w * scale)
        disp = cv2.resize(self.grading_image, (disp_w, disp_h))
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.grading_original_canvas.delete("all")
        self.grading_original_canvas.create_image(canvas_w//2, canvas_h//2, image=photo, anchor=tk.CENTER)
        self.grading_original_canvas.image = photo

    def extract_and_grade(self): # Function to extract and grade characters
        if not self.model or self.grading_image is None: # Check if the model or image is not loaded
            messagebox.showwarning("Missing", "Please load both model and image first!") # Show a warning message
            return # Return if the model or image is not loaded

        gray = self.grading_image.copy() # Copy the grading image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # Convert the image to binary
        kernel = np.ones((3,3), np.uint8) # Create a kernel for the morphological operation
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # Apply the morphological operation
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contours in the binary image

        valid = [c for c in contours if 200 < cv2.contourArea(c) < 50000] # Find the valid contours
        if not valid: # Check if there are no valid contours
            messagebox.showinfo("No Letters", "No characters found in the image!") # Show an info message
            return # Return if there are no valid contours

        sorted_contours = sorted(valid, key=lambda c: cv2.boundingRect(c)[0]) # Sort the contours by x coordinate

        self.grading_characters.clear() # Clear the grading characters list
        self.grading_predictions.clear() # Clear the grading predictions list

        label_to_letter = {v: k for k, v in LETTER_LABELS.items()} # Create a dictionary to map labels to letters

        for cnt in sorted_contours: # Loop through the sorted contours
            x, y, w, h = cv2.boundingRect(cnt) # Get the bounding rectangle of the contour
            pad = int(0.05 * max(w, h)) # Calculate the padding
            crop = gray[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            size = max(crop.shape) # Calculate the size of the crop
            sq = np.full((size, size), 255, np.uint8) # Create a square image
            offset_y = (size - crop.shape[0]) // 2 # Calculate the offset of the crop
            offset_x = (size - crop.shape[1]) // 2 # Calculate the offset of the crop
            sq[offset_y:offset_y+crop.shape[0], offset_x:offset_x+crop.shape[1]] = crop
            processed = cv2.resize(sq, (64, 64)).astype(np.float32) / 255.0 # Resize the crop to 64x64

            pred = self.model.predict(processed.reshape(1, 64, 64, 1), verbose=0)[0] # Predict the letter
            idx = np.argmax(pred) # Get the index of the predicted letter
            conf = float(pred[idx]) # Get the confidence of the predicted letter
            letter = label_to_letter.get(idx, '?')
            grade = 100 if conf >= 0.9 else 80 if conf >= 0.7 else 50 # Calculate the grade based on the confidence
            feedback = "EXCELLENT" if grade == 100 else "Good" if grade >= 80 else "Keep practicing!" # Calculate the feedback based on the grade

            self.grading_characters.append({'processed': processed}) # Add the processed character to the grading characters list
            self.grading_predictions.append({ # Add the prediction to the grading predictions list
                'letter': letter, # Add the letter to the prediction
                'confidence': conf, # Add the confidence to the prediction
                'grade': grade, # Add the grade to the prediction
                'feedback': feedback # Add the feedback to the prediction
            })

        self.grading_status.config(text=f"Found {len(self.grading_characters)} letters — Showing results...", foreground='green') # Update the grading status
        self.setup_grading_report_ui() # Setup the grading report UI

    def setup_grading_report_ui(self):
        # Clear the current grading tab content
        for widget in self.grading_frame.winfo_children(): # Loop through the children of the grading frame
            widget.destroy() # Destroy the widget

        # Header
        header = ttk.Frame(self.grading_frame, padding=20) # Create a header frame
        header.pack(fill=tk.X)

        total = len(self.grading_predictions) # Calculate the total number of predictions
        avg = sum(p['grade'] for p in self.grading_predictions) / total if total > 0 else 0 # Calculate the average grade
        color = "#2e8b57" if avg >= 90 else "#ff8c00" if avg >= 70 else "#dc143c" # Calculate the color based on the average grade
        feedback = "EXCELLENT!" if avg >= 90 else "GOOD!" if avg >= 70 else "KEEP PRACTICING!" # Calculate the feedback based on the average grade

        ttk.Label(header, text="FINAL RESULTS", font=('Arial', 28, 'bold')).pack() # Create a label for the final results
        ttk.Label(header, text=f"{avg:.1f}/100", font=('Arial', 80, 'bold'), foreground=color).pack(pady=10)
        ttk.Label(header, text=feedback, font=('Arial', 32, 'bold'), foreground=color).pack() # Create a label for the feedback
        ttk.Label(header, text=f"Total letters: {total} → {' '.join(p['letter'] for p in self.grading_predictions)}",
                 font=('Arial', 16)).pack(pady=15) # Create a label for the total letters and the predicted letters

        # Main content
        content = ttk.Frame(self.grading_frame) # Create a content frame
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=(10, 140))

        # Left - Character preview
        left_panel = ttk.Frame(content) # Create a left panel frame
        left_panel.pack(side=tk.LEFT, padx=(0, 40))

        self.grading_report_canvas = tk.Canvas(left_panel, bg='white', width=400, height=400,
                                              highlightthickness=4, highlightbackground="#222") # Create a canvas for the character preview
        self.grading_report_canvas.pack(pady=20) # Pack the canvas for the character preview

        # Right - Details
        right_panel = ttk.Frame(content) # Create a right panel frame
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grading_report_text = tk.Text(right_panel, font=('Consolas', 20), bg='#fdfdfd', padx=40, pady=40,
                                          spacing1=12, spacing3=12, wrap=tk.WORD) # Create a text widget for the details
        self.grading_report_text.pack(fill=tk.BOTH, expand=True) # Pack the text widget for the details

        # Navigation bar
        nav = tk.Frame(self.grading_frame, bg="#2c3e50", height=100) # Create a navigation bar frame
        nav.place(relx=0.5, rely=1.0, relwidth=1.0, anchor='s') # Place the navigation bar frame

        self.grading_prev_btn = tk.Button(nav, text="Previous", font=('Arial', 18, 'bold'), bg="#34495e", fg="white",
                                         width=12, height=2, relief='flat', command=self.prev_grading_char) # Create a previous button
        self.grading_prev_btn.place(relx=0.25, rely=0.5, anchor='center')

        self.grading_counter = tk.Label(nav, text="", font=('Arial', 28, 'bold'), bg="#2c3e50", fg="white") # Create a counter label
        self.grading_counter.place(relx=0.5, rely=0.5, anchor='center') # Place the counter label

        self.grading_next_btn = tk.Button(nav, text="Next", font=('Arial', 18, 'bold'), bg="#34495e", fg="white",
                                          width=12, height=2, relief='flat', command=self.next_grading_char) # Create a next button
        self.grading_next_btn.place(relx=0.75, rely=0.5, anchor='center')

        back_btn = tk.Button(nav, text="Back", font=('Arial', 16, 'bold'), bg="#c0392b", fg="white",
                            command=self.back_to_grading_main, relief='flat') # Create a back button
        back_btn.place(relx=0.95, rely=0.5, anchor='center')

        self.grading_char_index = 0 # Initialize the grading character index to 0
        self.update_grading_display()

    def update_grading_display(self):
        if not self.grading_predictions: # Check if there are no predictions
            return

        i = self.grading_char_index # Get the current character index
        p = self.grading_predictions[i]

        # Show character
        img = (self.grading_characters[i]['processed'] * 255).astype(np.uint8) # Convert the processed character to an image
        img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_NEAREST)
        photo = ImageTk.PhotoImage(Image.fromarray(img)) # Create a photo image from the image
        self.grading_report_canvas.delete("all")
        self.grading_report_canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
        self.grading_current_photo = photo # Store the current photo

        self.grading_counter.config(text=f"{i+1} / {len(self.grading_predictions)}") # Update the counter

        # Update text
        color = "#2e8b57" if p['grade'] == 100 else "#ff8c00" if p['grade'] >= 70 else "#dc143c"
        self.grading_report_text.delete(1.0, tk.END) # Delete the current text
        text = f"PREDICTED LETTER\n{p['letter']}\n\n"
        text += f"CONFIDENCE\n{p['confidence']:.1%}\n\n"
        text += f"GRADE\n{p['grade']}/100\n\n"
        text += f"{p['feedback']}" # Add the feedback to the text
        self.grading_report_text.insert(1.0, text)

        # Styling
        self.grading_report_text.tag_add("letter", "2.0", "2.end") # Add the letter tag to the text
        self.grading_report_text.tag_config("letter", font=('Arial', 100, 'bold'), foreground="#0078D7", justify='center') # Configure the letter tag
        self.grading_report_text.tag_add("grade", "6.0", "6.end")
        self.grading_report_text.tag_config("grade", font=('Arial', 80, 'bold'), foreground=color, justify='center') # Configure the grade tag
        self.grading_report_text.tag_add("feedback", "8.0", "8.end")
        self.grading_report_text.tag_config("feedback", font=('Arial', 40, 'bold'), foreground=color, justify='center') # Configure the feedback tag

        # Button states
        self.grading_prev_btn.config(state=tk.NORMAL if i > 0 else tk.DISABLED) # Configure the previous button state   
        self.grading_next_btn.config(state=tk.NORMAL if i < len(self.grading_predictions)-1 else tk.DISABLED) # Configure the next button state

    def prev_grading_char(self):
        if self.grading_char_index > 0: # Check if the current character index is greater than 0
            self.grading_char_index -= 1 # Decrement the current character index    
            self.update_grading_display() # Update the grading display

    def next_grading_char(self):
        if self.grading_char_index < len(self.grading_predictions) - 1:
            self.grading_char_index += 1 # Increment the current character index
            self.update_grading_display() # Update the grading display

    def back_to_grading_main(self):
        self.setup_grading_tab() # Setup the grading tab

    # ==================== UTILITY FUNCTIONS ====================
    def load_corrections(self): # Function to load the corrections
        """Load previously saved corrections""" # Load the corrections from the file
        if os.path.exists(CORRECTIONS_FILE): # Check if the corrections file exists
            with open(CORRECTIONS_FILE, 'rb') as f: # Open the corrections file for reading
                self.user_corrections = pickle.load(f) # Load the corrections from the file

    def save_corrections(self): # Function to save the corrections
        """Save corrections to disk""" # Save the corrections to the file
        with open(CORRECTIONS_FILE, 'wb') as f: # Open the corrections file for writing
            pickle.dump(self.user_corrections, f) # Save the corrections to the file

if __name__ == "__main__": # Check if the script is being run directly
    root = tk.Tk() # Create the root window
    style = ttk.Style() # Create a style object
    style.configure("Big.TButton", font=('Arial', 14, 'bold'), padding=10) # Configure the style for big buttons
    app = CursiveGraderGUI(root) # Create the CursiveGraderGUI application
    root.mainloop() # Start the main event loop
