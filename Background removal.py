import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Import ttk module
import subprocess
import os
import glob
import re
import pandas as pd
import numpy as np

class DataProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Background Removal Data Processor")

        # Set the window size
        master.geometry("700x300")

        # Change background color
        master.configure(bg="#f0f0f0")

        # Add a title label
        title_label = tk.Label(master, text="Background Removal Data Processor", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
        title_label.pack(fill="x")

        # Path selection
        path_frame = tk.Frame(master, bg="#f0f0f0")
        path_frame.pack(pady=10)
        path_label = tk.Label(path_frame, text="Directory Path:", font=("Helvetica", 12), bg="#f0f0f0")
        path_label.pack(side="left", padx=5)
        self.path_entry = tk.Entry(path_frame, width=50)
        self.path_entry.pack(side="left", padx=5)
        browse_button = tk.Button(path_frame, text="Browse", command=self.browse_directory)
        browse_button.pack(side="left", padx=5)

        # Step size input
        step_size_frame = tk.Frame(master, bg="#f0f0f0")
        step_size_frame.pack(pady=10)
        step_size_label = tk.Label(step_size_frame, text="Step Size (Hz):", font=("Helvetica", 12), bg="#f0f0f0")
        step_size_label.pack(side="left", padx=5)
        self.step_size_entry = tk.Entry(step_size_frame, width=20)
        self.step_size_entry.pack(side="left", padx=5)
        self.step_size_entry.insert(0, "1000000000")  # Default value

        # Run button
        run_button = tk.Button(master, text="Run", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", command=self.process_data)
        run_button.pack(pady=20)

        # Progress bar
        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill="x", padx=20, pady=10)

    def browse_directory(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory_path)

    def process_data(self):
        directory_path = self.path_entry.get()
        step_size = int(self.step_size_entry.get())

        if not directory_path:
            messagebox.showerror("Error", "Please select a directory path.")
            return

        try:
            self.process_files(directory_path, step_size)
            messagebox.showinfo("Success", f"Extracted data saved to {os.path.join(directory_path, 'background removal')}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_files(self, directory_path, step_size):
        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
        file_paths_sorted = sorted(file_paths, key=lambda x: int(re.search(r"(\d+)", x).group()))
        file_series = pd.Series(file_paths_sorted)

        # Extract frequency values from the first file (assuming they are the same for all files)
        first_file_path = file_series.iloc[0]
        data = np.loadtxt(first_file_path)
        freq_values = data[:, 0]

        # Create an array of evenly spaced frequency values with the specified step size
        min_freq = min(freq_values)
        max_freq = max(freq_values)
        index_freq_values = np.arange(min_freq, max_freq + step_size, step_size)

        # Initialize an empty dictionary to store filtered data
        filtered_freq_data = {}

        path = os.path.join(directory_path, "background removal")
        if not os.path.exists(path):
            os.makedirs(path)

        # Iterate through each frequency value
        for i, freq_value in enumerate(index_freq_values):
            x = []  # Magnetic field values
            y = []  # DS21 values

            for file_path in file_series:
                data = np.loadtxt(file_path)
                noise = np.loadtxt(first_file_path)
                y_value = data[np.where(data[:, 0] == freq_value), 1] - noise[np.where(noise[:,0]==freq_value),1]

                if len(y_value) > 0:
                    # Ensure y_value is a scalar (extract the value from the array)
                    y.append(float(y_value[0][0]))
                    file_name = os.path.basename(file_path)
                    file_name_without_ext = float(os.path.splitext(file_name)[0])
                    x.append(file_name_without_ext)

            # Sort the magnetic field values in ascending order
            sorted_indices = np.argsort(x)
            x_data = np.array(x)[sorted_indices]
            y_data = np.array(y)[sorted_indices]

            filtered_freq_data[freq_value] = {"mag_field": pd.Series(x_data), "s21": pd.Series(y_data)}

            # Create a DataFrame from the filtered data
            df = pd.DataFrame({'mag_field(oe)': x_data, 's21': y_data})

            # Save the DataFrame to a CSV file
            csv_path = os.path.join(path, str(freq_value) + ".csv")
            df.to_csv(csv_path, index=False)

            # Update progress bar
            self.progress.set((i+1) / len(index_freq_values) * 100)
            self.master.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataProcessorGUI(root)
    root.mainloop()
