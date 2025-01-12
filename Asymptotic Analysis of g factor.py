#-------------------------------------------------
# Author:      Suraj Joshi
# Created:     21-08-2024
# Copyright:   (c) Suraj Joshi 2024
#--------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model

class KittelFittingApp:
    def __init__(self, master):
        self.master = master
        master.title("Kittel Equation Fitting")

        # Set window size and background color
        master.geometry("600x750")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="Kittel Equation Fitting", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        # Path for data
        self.data_path_label = tk.Label(master, text="CSV File Path:", font=("Helvetica", 12), bg="#f0f0f0")
        self.data_path_label.pack(pady=5)
        self.data_path_button = tk.Button(master, text="Select CSV File", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.select_data_file)
        self.data_path_button.pack(pady=5)

        # Path for saving plots
        self.plot_path_label = tk.Label(master, text="Directory Path for Saving Plots:", font=("Helvetica", 12), bg="#f0f0f0")
        self.plot_path_label.pack(pady=5)
        self.plot_path_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.select_plot_directory)
        self.plot_path_button.pack(pady=5)

        # Segment size
        self.segment_size_label = tk.Label(master, text="Segment Size (default: 4):", font=("Helvetica", 12), bg="#f0f0f0")
        self.segment_size_label.pack(pady=5)
        self.segment_size_entry = tk.Entry(master)
        self.segment_size_entry.insert(0, "4")
        self.segment_size_entry.pack(pady=5)

        # Initial parameters
        self.m_eff_label = tk.Label(master, text="Initial Parameter M_eff (default: 1):", font=("Helvetica", 12), bg="#f0f0f0")
        self.m_eff_label.pack(pady=5)
        self.m_eff_entry = tk.Entry(master)
        self.m_eff_entry.insert(0, "1")
        self.m_eff_entry.pack(pady=5)

        self.h_k_label = tk.Label(master, text="Initial Parameter H_k (default: 0.0017):", font=("Helvetica", 12), bg="#f0f0f0")
        self.h_k_label.pack(pady=5)
        self.h_k_entry = tk.Entry(master)
        self.h_k_entry.insert(0, "0.0017")
        self.h_k_entry.pack(pady=5)

        self.gamma_label = tk.Label(master, text="Initial Parameter gamma (default: 29):", font=("Helvetica", 12), bg="#f0f0f0")
        self.gamma_label.pack(pady=5)
        self.gamma_entry = tk.Entry(master)
        self.gamma_entry.insert(0, "29")
        self.gamma_entry.pack(pady=5)

        # Run button
        self.run_button = tk.Button(master, text="Run Fitting", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.run_fitting)
        self.run_button.pack(pady=20)

        # Add the creator's name at the bottom
        self.creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.creator_label.pack(side="bottom", pady=10)

        self.data_file_path = None
        self.plot_directory_path = None

    def select_data_file(self):
        self.data_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.data_file_path:
            messagebox.showinfo("Selected CSV File", f"File: {self.data_file_path}")

    def select_plot_directory(self):
        self.plot_directory_path = filedialog.askdirectory()
        if self.plot_directory_path:
            messagebox.showinfo("Selected Directory for Plots", f"Directory: {self.plot_directory_path}")

    def run_fitting(self):
        segment_size = int(self.segment_size_entry.get())
        m_eff = float(self.m_eff_entry.get())
        h_k = float(self.h_k_entry.get())
        gamma = float(self.gamma_entry.get())

        if not self.data_file_path:
            messagebox.showwarning("Missing Information", "Please select a CSV file.")
            return
        if not self.plot_directory_path:
            messagebox.showwarning("Missing Information", "Please select a directory for saving plots.")
            return

        self.perform_fitting(self.data_file_path, self.plot_directory_path, segment_size, m_eff, h_k, gamma)

    def perform_fitting(self, data_file_path, plot_directory_path, segment_size, m_eff, h_k, gamma):
        def f_kittel(x_T, M_eff, H_k, gamma):
            return gamma * (((x_T + H_k) * (x_T + M_eff + H_k)) ** 0.5)

        df = pd.read_csv(data_file_path)

        y_hz = df["Frequency (Hz)"]
        y = y_hz * 1e-9  # frequency in GHz
        x = df["H_res"]  # in Oe
        x_T = 1e-4 * x  # in T

        kittel_model = Model(f_kittel)

        def piecewise_fit(x_T, y, segment_size):
            g_factors = []
            g_errors = []
            upper_frequencies = []
            fits = []
            for i in range(segment_size, len(x_T) + 1, segment_size):
                x_segment = x_T[:i]
                y_segment = y[:i]
                params = kittel_model.make_params(M_eff=m_eff, H_k=h_k, gamma=gamma)
                result = kittel_model.fit(y_segment, params, x_T=x_segment)
                g_factor = 2 * np.pi * (result.params["gamma"].value) / 87.99  # T/GHz
                if result.params["gamma"].stderr is not None:
                    g_error = 2 * np.pi * (result.params["gamma"].stderr) / 87.99  # T/GHz
                else:
                    g_error = 0  # or some default value
                g_factors.append(g_factor)
                g_errors.append(g_error)
                upper_frequencies.append(y_segment.iloc[-1])
                fits.append((x_segment, result.eval(x_T=x_segment)))

            # Handle the remaining points
            if len(x_T) % segment_size != 0:
                x_segment = x_T
                y_segment = y
                params = kittel_model.make_params(M_eff=m_eff, H_k=h_k, gamma=gamma)
                result = kittel_model.fit(y_segment, params, x_T=x_segment)
                g_factor = 2 * np.pi * (result.params["gamma"].value) / 87.99  # T/GHz
                if result.params["gamma"].stderr is not None:
                    g_error = 2 * np.pi * (result.params["gamma"].stderr) / 87.99  # T/GHz
                else:
                    g_error = 0  # or some default value
                g_factors.append(g_factor)
                g_errors.append(g_error)
                upper_frequencies.append(y_segment.iloc[-1])
                fits.append((x_segment, result.eval(x_T=x_segment)))

            return upper_frequencies, g_factors, g_errors, fits

        upper_frequencies, g_factors, g_errors, fits = piecewise_fit(x_T, y, segment_size)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.errorbar(upper_frequencies, g_factors, yerr=g_errors, fmt='o', color='b', label='Error Bar')
        plt.axhline(y=g_factors[-1], color='r', linestyle='--', label=f'Asymptotic g-factor: {g_factors[-1]:.3f} ± {g_errors[-1]:.3f}')
        plt.xlabel('Upper Fitting Frequency (GHz)')
        plt.ylabel('Fitted g-factor')
        plt.title('Upper Fitting Frequency vs Fitted g-factor')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(x_T, y, label='Data')
        for x_segment, fit in fits:
            plt.plot(x_segment, fit, label='Kittel Fit')
        plt.xlabel('Magnetic Field (T)')
        plt.ylabel('Frequency (GHz)')
        plt.title('Kittel Fit')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory_path, "Kittel_fit_asymptotic.png"))
        plt.show()

        messagebox.showinfo("Success", "Fitting completed and plot saved!")

if __name__ == "__main__":
    root = tk.Tk()
    app = KittelFittingApp(root)
    root.mainloop()
