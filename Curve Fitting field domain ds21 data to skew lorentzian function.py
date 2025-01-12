import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import re
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model

class LorentzianFittingApp:
    def __init__(self, master):
        self.master = master
        master.title("Skew Lorentzian Fitting GUI")

        # Set window size and background color
        master.geometry("450x760")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="Skew Lorentzian Fitting Parameters", font=("Helvetica", 16, "bold"),
                                    bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        # Add description label
        self.label = tk.Label(master, text="Set the parameters and directories for skew Lorentzian fitting.", font=("Helvetica", 12),
                              bg="#f0f0f0")
        self.label.pack(pady=10)

        # Initialize variables
        self.input_dir_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.delta_x = tk.DoubleVar()
        self.A = tk.DoubleVar()
        self.LW = tk.DoubleVar()
        self.alpha = tk.DoubleVar()
        self.r2_threshold = tk.DoubleVar()

        self.create_widgets()

        # Add the creator's name at the bottom
        self.creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"),
                                      bg="#f0f0f0", fg="#555555")
        self.creator_label.pack(side="bottom", pady=10)

    def create_widgets(self):
        self.create_label_button_entry("Input Directory:", self.select_input_directory, self.input_dir_path, 50)
        self.create_label_button_entry("Output Directory:", self.select_output_directory, self.output_dir_path, 50)
        self.create_label_entry("Range (delta_x):", self.delta_x, 200)
        self.create_label_entry("Initial Parameter A:", self.A, -1)
        self.create_label_entry("Initial Parameter LW:", self.LW, 40)
        self.create_label_entry("Initial Parameter alpha(asymmetry term):", self.alpha, 0.02)
        self.create_label_entry("R2 Value Threshold:", self.r2_threshold, 0.9)

        self.run_button = tk.Button(self.master, text="Run Fitting", font=("Helvetica", 10, "bold"),
                                    bg="#4CAF50", fg="white", command=self.run_fitting)
        self.run_button.pack(pady=20)

        # Progress bar
        self.progress = ttk.Progressbar(self.master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)
        self.progress["value"] = 0

    def create_label_button_entry(self, label_text, button_command, variable, width):
        label = tk.Label(self.master, text=label_text, font=("Helvetica", 10), bg="#f0f0f0")
        label.pack(pady=5)
        button = tk.Button(self.master, text="Browse", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=button_command)
        button.pack(pady=5)
        entry = tk.Entry(self.master, textvariable=variable, width=width)
        entry.pack(pady=5)

    def create_label_entry(self, label_text, variable, default_value):
        label = tk.Label(self.master, text=label_text, font=("Helvetica", 10), bg="#f0f0f0")
        label.pack(pady=5)
        variable.set(default_value)
        entry = tk.Entry(self.master, textvariable=variable)
        entry.pack(pady=5)

    def select_input_directory(self):
        directory = filedialog.askdirectory()
        self.input_dir_path.set(directory)

    def select_output_directory(self):
        directory = filedialog.askdirectory()
        self.output_dir_path.set(directory)

    def run_fitting(self):
        input_directory = self.input_dir_path.get()
        output_directory = os.path.join(self.output_dir_path.get(), 'Skew Lorentzian Fits')
        os.makedirs(output_directory, exist_ok=True)

        delta_x = self.delta_x.get()
        A = self.A.get()
        LW = self.LW.get()
        alpha = self.alpha.get()
        r2_threshold = self.r2_threshold.get()

        csv_files = [file for file in os.listdir(input_directory) if file.endswith(".csv")]
        csv_files_sorted = sorted(csv_files, key=lambda x: int(re.search(r"(\d+)", x).group()))

        # Update progress bar maximum value
        self.progress["maximum"] = len(csv_files_sorted)

        def derivative_lorentzian(new_x, A, H_res, LW, alpha):
            numerator = -2 * A * (new_x - H_res) * (LW / 2 * (1 + alpha * (new_x - H_res)))
            denominator = np.pi * ((new_x - H_res) ** 2 + (LW / 2 * (1 + alpha * (new_x - H_res))) ** 2) ** 2
            return numerator / denominator

        fitted_params_df = pd.DataFrame(columns=["Frequency (Hz)", "A", "LW", "H_res", "alpha", "R2"])

        for index, csv_file in enumerate(csv_files_sorted):
            fig_name = os.path.splitext(csv_file)[0]
            file_path = os.path.join(input_directory, csv_file)
            df = pd.read_csv(file_path)
            x_data = df['Magnetic Field']
            y_data = df['dS21/dH']

            sorted_indices = np.argsort(x_data)
            x = np.array(x_data)[sorted_indices]
            y = np.array(y_data)[sorted_indices]

            if len(x) == 0 or len(y) == 0:
                print(f"Empty data in file: {csv_file}")
                continue

            min_y_index = np.argmax(y)
            x_min = x[min_y_index] - delta_x
            x_max = x[min_y_index] + delta_x
            new_x = x[(x >= x_min) & (x <= x_max)]
            new_y = y[(x >= x_min) & (x <= x_max)]

            if len(new_x) == 0 or len(new_y) == 0:
                print(f"No data in the specified range for file: {csv_file}")
                continue

            H_res_guess = (new_x.max() + new_x.min()) / 2

            model = Model(derivative_lorentzian)
            params = model.make_params(A=A, LW=LW, H_res=H_res_guess, alpha=alpha)
            params['A'].set(min=-30, max=0)
            params['LW'].set(min=10, max=100)
            params['H_res'].set(min=H_res_guess - 100, max=H_res_guess + 100)
            params['alpha'].set(min=-0.1, max=0.1)

            try:
                result = model.fit(new_y, params, new_x=new_x)
            except Exception as e:
                print(f"Error fitting file {csv_file}: {e}")
                continue

            x_fit = np.linspace(new_x.min(), new_x.max(), 1000)
            y_fit = derivative_lorentzian(x_fit, result.params["A"].value, result.params["H_res"].value, result.params["LW"].value, result.params["alpha"].value)

            r2 = r2_score(new_y, result.best_fit)

            if r2 > r2_threshold:
                fitted_params_df = fitted_params_df._append(
                    {
                        "Frequency (Hz)": fig_name,
                        "A": result.params["A"].value,
                        "LW": result.params["LW"].value,
                        "alpha": result.params["alpha"].value,
                        "H_res": result.params["H_res"].value,
                        "R2": r2,
                    },
                    ignore_index=True,
                )
                plt.scatter(new_x, new_y, label="Limited range")
                plt.plot(x_fit, y_fit, "r-", label="Best Fit")
                plt.xlabel('Magnetic Field')
                plt.ylabel('dS21')
                plt.title(f"FMR Data for Frequency {fig_name} Hz\nH_res: {result.params['H_res'].value:.2f} Oe, LW: {result.params['LW'].value:.2f} Oe, alpha: {result.params['alpha'].value:.6f}")
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(output_directory, f"{fig_name}.png"))
                plt.clf()
            else:
                print(f"Low R2 value for file: {csv_file}, R2: {r2}")

            # Update progress bar value
            self.progress["value"] = index + 1
            self.master.update_idletasks()

        csv_file_path = os.path.join(output_directory, "field domain parameters.csv")
        fitted_params_df.to_csv(csv_file_path, index=False)
        print(f"Fitted parameters and R2 values saved to {csv_file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LorentzianFittingApp(root)
    root.mainloop()
