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
        master.geometry("500x500")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="Kittel Equation Fitting", font=("Helvetica", 16, "bold"),
                                    bg="#4CAF50", fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.label = tk.Label(master, text="Fit Data to Kittel Equation", font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack(pady=10)

        self.select_dir_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"),
                                           bg="#4CAF50", fg="white", command=self.select_directory)
        self.select_dir_button.pack(pady=5)

        self.M_eff_label = tk.Label(master, text="Initial Parameter M_eff:", font=("Helvetica", 12), bg="#f0f0f0")
        self.M_eff_label.pack(pady=5)
        self.M_eff_entry = tk.Entry(master)
        self.M_eff_entry.insert(0, "1")
        self.M_eff_entry.pack(pady=5)

        self.H_k_label = tk.Label(master, text="Initial Parameter H_k:", font=("Helvetica", 12), bg="#f0f0f0")
        self.H_k_label.pack(pady=5)
        self.H_k_entry = tk.Entry(master)
        self.H_k_entry.insert(0, "0.01")
        self.H_k_entry.pack(pady=5)

        self.gamma_label = tk.Label(master, text="Initial Parameter gamma:", font=("Helvetica", 12), bg="#f0f0f0")
        self.gamma_label.pack(pady=5)
        self.gamma_entry = tk.Entry(master)
        self.gamma_entry.insert(0, "29")
        self.gamma_entry.pack(pady=5)

        self.run_button = tk.Button(master, text="Run Fitting", font=("Helvetica", 10, "bold"), bg="#4CAF50",
                                    fg="white", command=self.run_fitting)
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

    def run_fitting(self):
        if not self.directory:
            messagebox.showwarning("Missing Information", "Please select a directory.")
            return

        try:
            M_eff = float(self.M_eff_entry.get())
            H_k = float(self.H_k_entry.get())
            gamma = float(self.gamma_entry.get())

            self.fit_kittel(self.directory, self.directory, M_eff, H_k, gamma)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def fit_kittel(self, input_directory, output_directory, M_eff, H_k, gamma):
        def f_kittel(x_T, M_eff, H_k, gamma):
            return gamma * (((x_T + H_k) * (x_T + M_eff + H_k)) ** 0.5)

        file_path = os.path.join(output_directory, 'field domain parameters.csv')
        df = pd.read_csv(file_path)

        y_hz = df["Frequency (Hz)"]
        y = y_hz * 1e-9  # Convert to GHz
        x = df["H_res"]
        x_T = 1e-4 * x  # Convert to Tesla

        kittel_model = Model(f_kittel)

        # Initial parameter guesses
        params = kittel_model.make_params(M_eff=M_eff, H_k=H_k, gamma=gamma)

        try:
            result = kittel_model.fit(y, params, x_T=x_T)
        except Exception as e:
            messagebox.showerror("Error during fitting", f"The model function generated NaN values and the fit aborted! Please check your model function and/or set boundaries on parameters where applicable.")
            return

        if result is None:
            messagebox.showerror("Error", "Fitting process failed.")
            return

        x_fit = np.linspace(min(x_T), max(x_T), 1000)
        y_fit = result.eval(x_T=x_fit)

        print(result.fit_report())

        gfactor = 2 * np.pi * (result.params["gamma"].value) / 87.99

        plt.scatter(x_T, y, label="Data")
        plt.plot(x_fit, y_fit, label="Fitted Curve", color="red")
        plt.ylabel("Frequency (GHz)")
        plt.xlabel("Magnetic Field (T)")
        plt.title("Fitting Data to Kittel Equation")
        plt.legend()
        plt.grid(True)
        fit_parameters = f"M_eff = {result.params['M_eff'].value:.2f} T \ngamma = {result.params['gamma'].value:.2f} GHz/T \nH_k = 0.00 T \ng-factor = {gfactor:.4f}"
        plt.text(0.6, 0.2, fit_parameters, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='gray'))

        plt.savefig(os.path.join(output_directory, "Kittel_fit.png"))
        plt.show()

        # Save fitting results to material parameter.csv
        material_params = pd.DataFrame({
            "Parameter": ["M_eff (T)", "gamma (GHz/T)", "H_k (T)", "g-factor"],
            "Value": [result.params['M_eff'].value, result.params['gamma'].value, H_k, gfactor]
        })
        material_params.to_csv(os.path.join(output_directory, "material parameter.csv"), index=False)
        print(f"Fitted parameters and R2 values saved to {os.path.join(output_directory, 'material parameter.csv')}")

if __name__ == "__main__":
    root = tk.Tk()
    app = KittelFittingApp(root)
    root.mainloop()
