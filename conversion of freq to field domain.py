#-------------------------------------------------
# Author:      Suraj Joshi
# Created:     21-08-2024
# Copyright:   (c) Suraj Joshi 2024
#--------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Import ttk module
import subprocess
import os
import sys
import glob
import pandas as pd
import re
import numpy as np


class FMRConversionApp:
    def __init__(self, master):
        self.master = master
        master.title("FMR Frequency to Field Domain Conversion")

        # Set window size and background color
        master.geometry("500x450")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="FMR Frequency to Field Domain Conversion",
                                    font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.label = tk.Label(master, text="Convert Frequency Domain Data to Field Domain", font=("Helvetica", 12),
                              bg="#f0f0f0")
        self.label.pack(pady=10)

        self.select_dir_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"),
                                           bg="#4CAF50", fg="white", command=self.select_directory)
        self.select_dir_button.pack(pady=5)

        self.step_size_label = tk.Label(master, text="Step Size for Frequency:", font=("Helvetica", 12), bg="#f0f0f0")
        self.step_size_label.pack(pady=5)

        self.step_size_entry = tk.Entry(master)
        self.step_size_entry.insert(0, "1e9")
        self.step_size_entry.pack(pady=5)

        self.run_button = tk.Button(master, text="Run Conversion", font=("Helvetica", 10, "bold"), bg="#4CAF50",
                                    fg="white", command=self.run_conversion)
        self.run_button.pack(pady=20)

        # Progress bar
        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill="x", padx=20, pady=10)

        self.creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"),
                                      bg="#f0f0f0", fg="#555555")
        self.creator_label.pack(side="bottom", pady=10)

        self.directory = None

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        if self.directory:
            messagebox.showinfo("Selected Directory", f"Directory: {self.directory}")

    def run_conversion(self):
        if not self.directory:
            messagebox.showwarning("Missing Information", "Please select a directory.")
            return

        try:
            step_size = float(self.step_size_entry.get())
            self.convert_freq_to_field(self.directory, self.directory, step_size)
            messagebox.showinfo("Success", "Conversion completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def convert_freq_to_field(self, input_directory, output_directory, step_size):
        field_domain_dir = os.path.join(output_directory, "field domain data")
        os.makedirs(field_domain_dir, exist_ok=True)

        file_paths = glob.glob(os.path.join(input_directory, "*.txt"))
        file_paths_sorted = sorted(file_paths, key=lambda x: int(re.search(r"(\d+)", x).group()))
        file_series = pd.Series(file_paths_sorted)

        first_file_path = file_series.iloc[0]
        data = np.loadtxt(first_file_path)
        freq_values = data[:, 0]

        min_freq = min(freq_values)
        max_freq = max(freq_values)
        index_freq_values = np.arange(min_freq, max_freq + step_size, step_size)

        filtered_freq_data = {}

        df = []

        for i, freq_value in enumerate(index_freq_values):
            x = []
            y = []

            for file_path in file_series:
                data = np.loadtxt(file_path)
                y_value = data[np.where(data[:, 0] == freq_value), 1]
                if len(y_value) > 0:
                    y.append(float(y_value[0][0]))
                    file_name = os.path.basename(file_path)
                    file_name_without_ext = float(os.path.splitext(file_name)[0])
                    x.append(file_name_without_ext)

            sorted_indices = np.argsort(x)
            x_data = np.array(x)[sorted_indices]
            y_data = np.array(y)[sorted_indices]

            filtered_freq_data[freq_value] = {"mag_field": pd.Series(x_data), "s21": pd.Series(y_data)}
            df = pd.DataFrame({'mag_field(oe)': x_data, 's21': y_data})

            csv_path = os.path.join(field_domain_dir, str(freq_value) + ".csv")
            df.to_csv(csv_path, index=False)

            # Update progress bar
            self.progress.set((i+1) / len(index_freq_values) * 100)
            self.master.update_idletasks()

        print(f"Extracted data saved to {field_domain_dir}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FMRConversionApp(root)
    root.mainloop()
