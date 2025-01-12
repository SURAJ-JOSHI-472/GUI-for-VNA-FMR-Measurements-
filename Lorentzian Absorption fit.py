#-------------------------------------------------
# Author:      Suraj Joshi
# Created:     21-08-2024
# Copyright:   (c) Suraj Joshi 2024
#--------------------------------------------------
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

class LorentzianFitGUI:
    def __init__(self, master):
        self.master = master
        master.title("Lorentzian Absorption Fit")

        # Set the window size
        master.geometry("700x600")

        # Change background color
        master.configure(bg="#f0f0f0")

        # Add a title label
        title_label = tk.Label(master, text="Lorentzian Absorption Fit", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
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

        # Initial parameters input
        params_frame = tk.Frame(master, bg="#f0f0f0")
        params_frame.pack(pady=10)
        param_labels = ["A:", "Sigma:", "H_res:"]
        self.param_entries = []
        default_values = ["-20", "20", "200"]
        for i, label_text in enumerate(param_labels):
            param_label = tk.Label(params_frame, text=label_text, font=("Helvetica", 12), bg="#f0f0f0")
            param_label.grid(row=0, column=i * 2, padx=5)
            param_entry = tk.Entry(params_frame, width=10)
            param_entry.grid(row=0, column=i * 2 + 1, padx=5)
            param_entry.insert(0, default_values[i])
            self.param_entries.append(param_entry)

        # Run button
        run_button = tk.Button(master, text="Run", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", command=self.run_fit)
        run_button.pack(pady=20)

        # Progress bar
        self.progress = ttk.Progressbar(master, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=20)
        self.progress["value"] = 0

    def browse_directory(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory_path)

    def run_fit(self):
        directory_path = self.path_entry.get()
        initial_params = [float(entry.get()) for entry in self.param_entries]

        if not directory_path:
            messagebox.showerror("Error", "Please select a directory path.")
            return

        try:
            self.perform_fit(directory_path, initial_params)
            messagebox.showinfo("Success", "Lorentzian fitting completed and data saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def perform_fit(self, directory_path, initial_params):
        path = os.path.join(directory_path, "plots")
        if not os.path.exists(path):
            os.makedirs(path)

        # Get a list of all CSV files in the directory with sorting
        csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]
        csv_files_sorted = sorted(csv_files, key=lambda x: int(re.search(r"(\d+)", x).group()))

        # Update progress bar maximum value
        self.progress["maximum"] = len(csv_files_sorted)

        ## Define S21 function
        def S21(new_x, A, sigma, H_res):
            return (A * sigma) / (np.pi * ((new_x - H_res) ** 2 + sigma ** 2))

        # Initialize a DataFrame to store fitted parameters and R2 values
        fitted_params_df = pd.DataFrame(columns=["Frequency (Hz)", "A", "LW", "H_res", "R2"])

        freq_value = []

        # Loop through each CSV file
        for index, csv_file in enumerate(csv_files_sorted):
            fig_name = os.path.splitext(csv_file)[0]
            freq_value.append(fig_name)
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path)  # Read CSV data into a DataFrame
            x_data = df['mag_field(oe)']
            y_data = df['s21']
            # Sort the magnetic field values in ascending order
            sorted_indices = np.argsort(x_data)
            x = np.array(x_data)[sorted_indices]
            y = np.array(y_data)[sorted_indices]

            # Now redefine the y range where there is a dip over a total length of 1000 oe field
            min_y_index = np.argmin(y)
            delta_x = 120  # Adjust this value as needed
            x_min = x[min_y_index] - delta_x
            x_max = x[min_y_index] + delta_x
            new_x = x[(x >= x_min) & (x <= x_max)]
            new_y = y[(x >= x_min) & (x <= x_max)]

            # Fit each dataset to the Lorentzian model
            model = Model(S21)
            params = model.make_params(A=initial_params[0], sigma=initial_params[1], H_res=initial_params[2])
            params['sigma'].min = 0  # Constrain LW to be non-negative
            params['H_res'].set(min=0, max=2400)
            result = model.fit(new_y, params, new_x=new_x)

            # Calculate R2 value
            y_fit = result.best_fit

            # Calculate R2 value for complex data
            r2 = r2_score(new_y, y_fit)

            # Append fitted parameters and R2 value to the DataFrame
            if r2 > 0.9:
                fitted_params_df = fitted_params_df._append(
                    {
                        "Frequency (Hz)": fig_name,
                        "A": result.params["A"].value,
                        "LW": result.params["sigma"].value * 2,
                        "H_res": result.params["H_res"].value,
                        "R2": r2,
                    },
                    ignore_index=True,
                )

                # Plots
                plt.scatter(x, y, label="Data points")
                plt.scatter(new_x, new_y, label="Limited Range")
                plt.plot(new_x, y_fit, "r-", label="Best Fit")
                plt.xlabel('Magnetic Field')
                plt.ylabel('S21_pure')
                plt.title(f"FMR Data for Frequency {fig_name} Hz")
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(path, f"{fig_name}.png"))
                plt.clf()

            # Update progress bar value
            self.progress["value"] = index + 1
            self.master.update_idletasks()

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(path, "field domain parameters.csv")
        fitted_params_df.to_csv(csv_file_path, index=False)
        print(f"Fitted parameters and R2 values saved to {csv_file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LorentzianFitGUI(root)
    root.mainloop()
