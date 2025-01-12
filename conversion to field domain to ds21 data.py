import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import pandas as pd


class DerivativeCalculationApp:
    def __init__(self, master):
        self.master = master
        master.title("FMR Derivative Calculation")

        # Set window size and background color
        master.geometry("500x400")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="FMR Derivative Calculation", font=("Helvetica", 16, "bold"),
                                    bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.label = tk.Label(master, text="Calculate Derivative of Field Domain Data", font=("Helvetica", 12),
                              bg="#f0f0f0")
        self.label.pack(pady=10)

        self.select_dir_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"),
                                           bg="#4CAF50", fg="white", command=self.select_directory)
        self.select_dir_button.pack(pady=5)

        self.run_button = tk.Button(master, text="Run Calculation", font=("Helvetica", 10, "bold"), bg="#4CAF50",
                                    fg="white", command=self.run_calculation)
        self.run_button.pack(pady=20)

        # Add the creator's name at the bottom
        self.creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"),
                                      bg="#f0f0f0", fg="#555555")
        self.creator_label.pack(side="bottom", pady=10)

        self.directory = None

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        if self.directory:
            messagebox.showinfo("Selected Directory", f"Directory: {self.directory}")

    def run_calculation(self):
        if not self.directory:
            messagebox.showwarning("Missing Information", "Please select a directory.")
            return

        try:
            self.calculate_derivative(self.directory, self.directory)
            messagebox.showinfo("Success", "Derivative calculation completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def calculate_derivative(self, input_directory, output_directory):
        output_directory = os.path.join(output_directory, 'ds21')
        os.makedirs(output_directory, exist_ok=True)

        def calculate_derivative(H, S21, dH=10):
            dS21_dH = (S21[2:] - S21[:-2]) / (2 * dH)
            H_mid = H[1:-1]
            return H_mid, dS21_dH

        for filename in os.listdir(input_directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(input_directory, filename)
                data = pd.read_csv(filepath)

                H = data['mag_field(oe)'].values
                S21 = data['s21'].values

                H_mid, dS21_dH = calculate_derivative(H, S21)

                derivative_data = pd.DataFrame({'Magnetic Field': H_mid, 'dS21/dH': dS21_dH})
                output_filepath = os.path.join(output_directory, f"{filename}")
                derivative_data.to_csv(output_filepath, index=False)

        print("Derivative calculation and saving completed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = DerivativeCalculationApp(root)
    root.mainloop()
