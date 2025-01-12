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
from sklearn.metrics import r2_score


class LinewidthFittingApp:
    def __init__(self, master):
        self.master = master
        master.title("Linewidth Equation Fitting")

        # Set window size and background color
        master.geometry("500x500")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="Linewidth Equation Fitting", font=("Helvetica", 16, "bold"),
                                    bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.label = tk.Label(master, text="Fit Data to Linewidth Equation", font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack(pady=10)

        self.select_dir_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"),
                                           bg="#4CAF50", fg="white", command=self.select_directory)
        self.select_dir_button.pack(pady=5)

        self.material_label = tk.Label(master, text="Material Name:", font=("Helvetica", 12), bg="#f0f0f0")
        self.material_label.pack(pady=5)
        self.material_entry = tk.Entry(master)
        self.material_entry.insert(0, "FeGaB")  # Default material name
        self.material_entry.pack(pady=5)

        self.alpha_label = tk.Label(master, text="Initial Parameter alpha:", font=("Helvetica", 12), bg="#f0f0f0")
        self.alpha_label.pack(pady=5)
        self.alpha_entry = tk.Entry(master)
        self.alpha_entry.insert(0, "0.003")
        self.alpha_entry.pack(pady=5)

        self.DH0_label = tk.Label(master, text="Initial Parameter DH0:", font=("Helvetica", 12), bg="#f0f0f0")
        self.DH0_label.pack(pady=5)
        self.DH0_entry = tk.Entry(master)
        self.DH0_entry.insert(0, "0.0022")
        self.DH0_entry.pack(pady=5)

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
            material_name = self.material_entry.get()
            alpha = float(self.alpha_entry.get())
            DH0 = float(self.DH0_entry.get())
            self.fit_linewidth(self.directory, self.directory, material_name, alpha, DH0)
        except FileNotFoundError:
            messagebox.showerror("Error", "material parameter.csv not found. Please run the Kittel program first.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def fit_linewidth(self, input_directory, output_directory, material_name, alpha, DH0):
        def DH(x, alpha, DH0):
            return ((4 * np.pi * alpha * x) / gamma) + DH0

        # Read gamma from material parameter.csv
        material_file_path = os.path.join(output_directory, 'material parameter.csv')
        if not os.path.isfile(material_file_path):
            raise FileNotFoundError

        material_df = pd.read_csv(material_file_path)
        gamma_row = material_df.loc[material_df['Parameter'] == 'gamma (GHz/T)']
        gamma = 2 * np.pi * gamma_row['Value'].values[0]

        file_path = os.path.join(output_directory, 'field domain parameters.csv')
        df = pd.read_csv(file_path)

        x_hz = df["Frequency (Hz)"]
        x = x_hz * 1e-9  # Convert to GHz
        LW1 = df["LW"]
        LW = LW1 * 1e-4  # Convert to Tesla

        LW_model = Model(DH)

        # Initial parameter guesses
        params = LW_model.make_params(alpha=alpha, DH0=DH0)

        try:
            result = LW_model.fit(LW, params, x=x)
        except Exception as e:
            print("Error during fitting:", e)

        if 'result' in locals():
            LW_fit = np.linspace(min(LW), max(LW), 10000)
            x_fit = np.linspace(min(x), max(x), 10000)
            y_fit = result.eval(x=x_fit)
            r2 = r2_score(LW_fit, y_fit)

            print(result.fit_report())

            plt.scatter(x, LW, label="Data")
            plt.plot(x_fit, y_fit, label="Fitted Curve", color="red")
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Linewidth (DH) (T)")
            plt.title("Fitting Data to Linewidth Equation")
            plt.legend()
            plt.grid(True)
            fit_parameters = f"alpha = {result.params['alpha'].value:.6f} \nDH_0 = {result.params['DH0'].value:.4f}"

            plt.text(0.1, 0.6, fit_parameters, transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', edgecolor='gray'))
            plt.savefig(os.path.join(output_directory, "linewidth_fit.png"))
            plt.show()

            # Update material parameter.csv with fitting results
            material_df = material_df._append(
                {
                    "Parameter": "Material",
                    "Value": material_name
                }, ignore_index=True
            )
            material_df = material_df._append(
                {
                    "Parameter": "alpha",
                    "Value": result.params['alpha'].value
                }, ignore_index=True
            )
            material_df = material_df._append(
                {
                    "Parameter": "DH0 (Oe)",
                    "Value": result.params['DH0'].value
                }, ignore_index=True
            )
            material_df.to_csv(os.path.join(output_directory, 'material parameter.csv'), index=False)
        else:
            print("Fitting process failed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = LinewidthFittingApp(root)
    root.mainloop()
