import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model

class FittingApp:
    def __init__(self, master):
        self.master = master
        master.title("Lorentzian Fitting Application")

        # Set window size and background color
        master.geometry("600x700")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="Lorentzian Fitting Application", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
        self.title_label.pack(fill="x")

        # Path for data
        self.directory_label = tk.Label(master, text="Directory Path for Data:", font=("Helvetica", 12), bg="#f0f0f0")
        self.directory_label.pack(pady=5)
        self.directory_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.select_data_directory)
        self.directory_button.pack(pady=5)

        # Path for results
        self.results_label = tk.Label(master, text="Directory Path for Results:", font=("Helvetica", 12), bg="#f0f0f0")
        self.results_label.pack(pady=5)
        self.results_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.select_results_directory)
        self.results_button.pack(pady=5)

        # Parameters
        self.delta_x_label = tk.Label(master, text="Delta_x (default: 200):", font=("Helvetica", 12), bg="#f0f0f0")
        self.delta_x_label.pack(pady=5)
        self.delta_x_entry = tk.Entry(master)
        self.delta_x_entry.insert(0, "200")
        self.delta_x_entry.pack(pady=5)

        self.A_label = tk.Label(master, text="Initial Parameter A (default: -1):", font=("Helvetica", 12), bg="#f0f0f0")
        self.A_label.pack(pady=5)
        self.A_entry = tk.Entry(master)
        self.A_entry.insert(0, "-1")
        self.A_entry.pack(pady=5)

        self.LW_label = tk.Label(master, text="Initial Parameter LW (default: 40):", font=("Helvetica", 12), bg="#f0f0f0")
        self.LW_label.pack(pady=5)
        self.LW_entry = tk.Entry(master)
        self.LW_entry.insert(0, "40")
        self.LW_entry.pack(pady=5)

        self.R2_label = tk.Label(master, text="R2 Threshold (default: 0.9):", font=("Helvetica", 12), bg="#f0f0f0")
        self.R2_label.pack(pady=5)
        self.R2_entry = tk.Entry(master)
        self.R2_entry.insert(0, "0.9")
        self.R2_entry.pack(pady=5)

        self.run_button = tk.Button(master, text="Run Fitting", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=self.run_fitting)
        self.run_button.pack(pady=20)

        # Add the creator's name at the bottom
        self.creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"), bg="#f0f0f0", fg="#555555")
        self.creator_label.pack(side="bottom", pady=10)

        self.directory_path = None
        self.results_path = None

    def select_data_directory(self):
        self.directory_path = filedialog.askdirectory()
        if self.directory_path:
            messagebox.showinfo("Selected Directory for Data", f"Directory: {self.directory_path}")

    def select_results_directory(self):
        self.results_path = filedialog.askdirectory()
        if self.results_path:
            messagebox.showinfo("Selected Directory for Results", f"Directory: {self.results_path}")

    def run_fitting(self):
        delta_x = int(self.delta_x_entry.get())
        A = float(self.A_entry.get())
        LW = float(self.LW_entry.get())
        R2_threshold = float(self.R2_entry.get())

        if not self.directory_path:
            messagebox.showwarning("Missing Information", "Please select a directory for data.")
            return
        if not self.results_path:
            messagebox.showwarning("Missing Information", "Please select a directory for results.")
            return

        self.perform_fitting(self.directory_path, self.results_path, delta_x, A, LW, R2_threshold)

    def perform_fitting(self, directory_path, results_path, delta_x, A, LW, R2_threshold):
        # Get a list of all CSV files in the directory with sorting
        csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]
        csv_files_sorted = sorted(csv_files, key=lambda x: int(re.search(r"(\d+)", x).group()))

        def derivative_lorentzian(new_x, A, H_res, LW):
            return -(A * LW * (new_x - H_res)) / (np.pi * ((new_x - H_res) ** 2 + (LW / 2) ** 2) ** 2)

        # Loop through each CSV file
        for csv_file in csv_files_sorted:
            fig_name = os.path.splitext(csv_file)[0]
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path)  # Read CSV data into a DataFrame
            x_data = df['Magnetic Field']
            y_data = df['dS21/dH']

            # Sort the magnetic field values in ascending order
            sorted_indices = np.argsort(x_data)
            x = np.array(x_data)[sorted_indices]
            y = np.array(y_data)[sorted_indices]

            # Redefine the y range where there is a dip, over a total length of 1000 Oe field
            min_y_index = np.argmax(y)
            x_min = x[min_y_index] - delta_x
            x_max = x[min_y_index] + delta_x
            new_x = x[(x >= x_min) & (x <= x_max)]
            new_y = y[(x >= x_min) & (x <= x_max)]

            # Estimate H_res as the midpoint of max and min point
            H_res_guess = (new_x.max() + new_x.min()) / 2

            # Fit each dataset to the derivative Lorentzian model
            model = Model(derivative_lorentzian)
            params = model.make_params(A=A, LW=LW, H_res=H_res_guess)

            # Constrain parameters
            params['A'].set(min=-30, max=0)
            params['LW'].set(min=10, max=100)
            params['H_res'].set(min=H_res_guess - 100, max=H_res_guess + 100)

            result = model.fit(new_y, params, new_x=new_x)

            # Calculate R2 value
            r2 = r2_score(new_y, result.best_fit)

            # Plot only if R2 > threshold
            if r2 > R2_threshold:
                # Generate finer x data for smoother curve
                x_fit = np.linspace(new_x.min(), new_x.max(), 1000)
                y_fit = derivative_lorentzian(x_fit, result.params["A"].value, result.params["H_res"].value, result.params["LW"].value)

                # Convert frequency to GHz
                freq_in_ghz = float(fig_name) * 1e-9
                freq_in_ghz = round(freq_in_ghz, 0)

                # Plot experimental data
                plt.scatter(new_x, new_y)
                # Plot fitted result
                plt.plot(x_fit, y_fit, label=f"{freq_in_ghz} GHz")

        plt.xlabel('Magnetic Field (Oe)')
        plt.ylabel('dS21')
        plt.title("FMR Data for Different Frequencies")
        plt.legend(bbox_to_anchor=(1.10, 1), loc='upper right')  # supported values are 'best', 'upper right', etc.
        plt.grid()
        plt.savefig(os.path.join(results_path, "FMR_fitting_results.png"))
        plt.show()

        messagebox.showinfo("Success", "Fitting completed and results saved!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FittingApp(root)
    root.mainloop()
