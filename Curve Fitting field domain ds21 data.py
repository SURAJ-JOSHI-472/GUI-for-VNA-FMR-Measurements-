#-------------------------------------------------
# Author:      Suraj Joshi
# Created:     21-08-2024
# Copyright:   (c) Suraj Joshi 2024
#--------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
        master.title("FMR Lorentzian Fitting")

        # Set window size and background color
        master.geometry("500x600")
        master.configure(bg="#f0f0f0")

        # Add title label with styling
        self.title_label = tk.Label(master, text="FMR Lorentzian Fitting", font=("Helvetica", 16, "bold"), bg="#3F51B5",
                                    fg="white", pady=10)
        self.title_label.pack(fill="x")

        self.label = tk.Label(master, text="Fit Field Domain DS21 Data to Derivative Lorentzian Function",
                              font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack(pady=10)

        self.select_dir_button = tk.Button(master, text="Select Directory", font=("Helvetica", 10, "bold"),
                                           bg="#4CAF50", fg="white", command=self.select_directory)
        self.select_dir_button.pack(pady=5)

        self.delta_x_label = tk.Label(master, text="Desired Range (delta_H):", font=("Helvetica", 12), bg="#f0f0f0")
        self.delta_x_label.pack(pady=5)
        self.delta_x_entry = tk.Entry(master)
        self.delta_x_entry.insert(0, "150")
        self.delta_x_entry.pack(pady=5)

        self.A_label = tk.Label(master, text="Initial Parameter A:", font=("Helvetica", 12), bg="#f0f0f0")
        self.A_label.pack(pady=5)
        self.A_entry = tk.Entry(master)
        self.A_entry.insert(0, "-15")
        self.A_entry.pack(pady=5)

        self.LW_label = tk.Label(master, text="Initial Parameter LW:", font=("Helvetica", 12), bg="#f0f0f0")
        self.LW_label.pack(pady=5)
        self.LW_entry = tk.Entry(master)
        self.LW_entry.insert(0, "40")
        self.LW_entry.pack(pady=5)

        self.H_res_label = tk.Label(master, text="Initial Parameter H_res:", font=("Helvetica", 12), bg="#f0f0f0")
        self.H_res_label.pack(pady=5)
        self.H_res_entry = tk.Entry(master)
        self.H_res_entry.insert(0, "100")
        self.H_res_entry.pack(pady=5)

        self.R2_label = tk.Label(master, text="R2 Value Threshold:", font=("Helvetica", 12), bg="#f0f0f0")
        self.R2_label.pack(pady=5)
        self.R2_entry = tk.Entry(master)
        self.R2_entry.insert(0, "0.9")
        self.R2_entry.pack(pady=5)

        self.run_button = tk.Button(master, text="Run Fitting", font=("Helvetica", 10, "bold"), bg="#4CAF50",
                                    fg="white", command=self.run_fitting)
        self.run_button.pack(pady=20)

        # Add the progress bar
        self.progress = ttk.Progressbar(master, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=5)

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
            delta_x = float(self.delta_x_entry.get())
            A = float(self.A_entry.get())
            LW = float(self.LW_entry.get())
            H_res = float(self.H_res_entry.get())
            R2_threshold = float(self.R2_entry.get())

            self.fit_lorentzian(self.directory, self.directory, delta_x, A, LW, H_res, R2_threshold)
            messagebox.showinfo("Success", "Fitting completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def fit_lorentzian(self, input_directory, output_directory, delta_x, A, LW, H_res, R2_threshold):
        path = os.path.join(output_directory, 'plots')
        os.makedirs(path, exist_ok=True)

        # Get a list of all CSV files in the directory with sorting
        csv_files = [file for file in os.listdir(input_directory) if file.endswith(".csv")]
        csv_files_sorted = sorted(csv_files, key=lambda x: int(re.search(r"(\d+)", x).group()))

        self.progress["maximum"] = len(csv_files_sorted)  # Set progress bar maximum value

        def derivative_lorentzian(new_x, A, H_res, LW):
            return -(A * LW * (new_x - H_res)) / (np.pi * ((new_x - H_res) ** 2 + (LW / 2) ** 2) ** 2)

        # Initialize a DataFrame to store fitted parameters and R2 values
        fitted_params_df = pd.DataFrame(columns=["Frequency (Hz)", "A", "LW", "H_res", "R2"])

        # Loop through each CSV file
        for i, csv_file in enumerate(csv_files_sorted):
            fig_name = os.path.splitext(csv_file)[0]
            file_path = os.path.join(input_directory, csv_file)
            df = pd.read_csv(file_path)  # Read CSV data into a DataFrame
            x_data = df['Magnetic Field']
            y_data = df['dS21/dH']
            # Sort the magnetic field values in ascending order
            sorted_indices = np.argsort(x_data)
            x = np.array(x_data)[sorted_indices]
            y = np.array(y_data)[sorted_indices]

            # Redefine the y range where there is a dip over a total length of 1000 oe field
            min_y_index = np.argmax(y)
            x_min = x[min_y_index] - delta_x
            x_max = x[min_y_index] + delta_x
            new_x = x[(x >= x_min) & (x <= x_max)]
            new_y = y[(x >= x_min) & (x <= x_max)]

            # Fit each dataset to the derivative Lorentzian model
            model = Model(derivative_lorentzian)
            params = model.make_params(A=A, LW=LW, H_res=H_res)
            params['LW'].min = 0  # Constrain LW to be non-negative
            result = model.fit(new_y, params, new_x=new_x)

            # Generate finer x data for smoother curve
            x_fit = np.linspace(new_x.min(), new_x.max(), 1000)
            y_fit = derivative_lorentzian(x_fit, result.params["A"].value, result.params["H_res"].value,
                                          result.params["LW"].value)

            # Calculate R2 value
            r2 = r2_score(new_y, result.best_fit)

            # Append fitted parameters and R2 value to the DataFrame
            if r2 > R2_threshold:
                fitted_params_df = fitted_params_df._append(
                    {
                        "Frequency (Hz)": fig_name,
                        "A": result.params["A"].value,
                        "LW": result.params["LW"].value,
                        "H_res": result.params["H_res"].value,
                        "R2": r2,
                    },
                    ignore_index=True,
                )

                # Plots
                plt.scatter(new_x, new_y, label="Limited range")
                plt.plot(x_fit, y_fit, "r-", label="Best Fit")
                plt.xlabel('Magnetic Field')
                plt.ylabel('dS21')
                plt.title(
                    f"FMR Data for Frequency {fig_name} Hz\nH_res: {result.params['H_res'].value:.2f} Oe, LW: {result.params['LW'].value:.2f} Oe")
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(path, f"{fig_name}.png"))
                plt.clf()

            self.progress["value"] = i + 1  # Update progress bar
            self.master.update_idletasks()  # Force update of the GUI

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(path, "field domain parameters.csv")
        fitted_params_df.to_csv(csv_file_path, index=False)
        print(f"Fitted parameters and R2 values saved to {csv_file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LorentzianFittingApp(root)
    root.mainloop()
