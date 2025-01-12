import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import sys
import webbrowser


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         font=("helvetica", 10, "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, _):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class MasterGUI:
    def __init__(self, master):
        self.master = master
        master.title("FMR Data Processing - Step by Step")

        # Set the window size
        master.geometry("1400x600")

        # Change background color
        master.configure(bg="#f0f0f0")

        # Add a title label
        title_label = tk.Label(master, text="FMR Data Processing - Step by Step", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white", pady=10)
        title_label.pack(fill="x")

        # Create a frame for the main content
        main_frame = tk.Frame(master, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True)

        # Create three frames for horizontal division with equal width
        left_frame = tk.Frame(main_frame, borderwidth=2, relief="solid", padx=10, pady=10, bg="#ffffff")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        middle_frame = tk.Frame(main_frame, borderwidth=2, relief="solid", padx=10, pady=10, bg="#ffffff")
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        right_frame = tk.Frame(main_frame, borderwidth=2, relief="solid", padx=10, pady=10, bg="#ffffff")
        right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Ensure equal resizing for all columns
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)

        # First part: Background Removal and Lorentzian Absorption Fit
        first_part_label = tk.Label(left_frame, text="Background Removal and Lorentzian Absorption Fit", font=("Helvetica", 14, "bold"), bg="#ffffff")
        first_part_label.pack(fill="x", pady=5)

        first_steps = [
            ("Background Removal", "Background removal.py"),
            ("Lorentzian Absorption Fit", "Lorentzian Absorption fit.py")
        ]

        for step, script in first_steps:
            frame = tk.Frame(left_frame, borderwidth=1, relief="solid", padx=10, pady=10, bg="#ffffff")
            frame.pack(padx=10, pady=10, fill="x")
            label = tk.Label(frame, text=step, font=("Helvetica", 12), bg="#ffffff")
            label.pack(side="left")
            button = tk.Button(frame, text="Run Step", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=lambda s=script: self.run_script(s))
            button.pack(side="right")
            ToolTip(button, f"Click to run the {step.lower()} script")

        # Second part: Derivative Divide Method
        second_part_label = tk.Label(middle_frame, text="Derivative Divide Method", font=("Helvetica", 14, "bold"), bg="#ffffff")
        second_part_label.pack(fill="x", pady=5)

        second_steps = [
            ("Conversion of Frequency to Field Domain", "conversion of freq to field domain.py"),
            ("Derivative of Field Domain", "conversion to field domain to ds21 data.py"),
            ("Lorentzian Fitting of dS data", "curve fitting field domain ds21 data.py"),
            ("Skew Lorentzian Fitting of dS data", "curve fitting field domain ds21 data to skew lorentzian function.py"),
            ("Derivative FMR Spectra", "FMR Spectra.py")
        ]

        for step, script in second_steps:
            frame = tk.Frame(middle_frame, borderwidth=1, relief="solid", padx=10, pady=10, bg="#ffffff")
            frame.pack(padx=10, pady=10, fill="x")
            label = tk.Label(frame, text=step, font=("Helvetica", 12), bg="#ffffff")
            label.pack(side="left")
            button = tk.Button(frame, text="Run Step", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=lambda s=script: self.run_script(s))
            button.pack(side="right")
            ToolTip(button, f"Click to run the {step.lower()} script")

        # Third part: Kittel Fit, Linewidth Fitting, Asymptotic Analysis of g factor
        third_part_label = tk.Label(right_frame, text="Magnetic Properties", font=("Helvetica", 14, "bold"), bg="#ffffff")
        third_part_label.pack(fill="x", pady=5)

        third_steps = [
            ("Kittel Fit", "Kittel fit from field domain data.py"),
            ("Linewidth Fitting", "Linewidth Fit.py"),
            ("Asymptotic Analysis of g factor", "Asymptotic Analysis of g factor.py")
        ]

        for step, script in third_steps:
            frame = tk.Frame(right_frame, borderwidth=1, relief="solid", padx=10, pady=10, bg="#ffffff")
            frame.pack(padx=10, pady=10, fill="x")
            label = tk.Label(frame, text=step, font=("Helvetica", 12), bg="#ffffff")
            label.pack(side="left")
            button = tk.Button(frame, text="Run Step", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", command=lambda s=script: self.run_script(s))
            button.pack(side="right")
            ToolTip(button, f"Click to run the {step.lower()} script")

        # Help section with clickable email
        help_label = tk.Label(master, text="For help, please contact: ", font=("Helvetica", 10), bg="#f0f0f0", fg="#555555")
        help_label.pack(side="top", pady=5)
        email_link = tk.Label(master, text="joshisuraj472@gmail.com", font=("Helvetica", 10), bg="#f0f0f0", fg="blue", cursor="hand2")
        email_link.pack(side="top", pady=5)
        email_link.bind("<Button-1>", lambda e: self.open_email("joshisuraj472@gmail.com"))
        ToolTip(email_link, "Click to send an email for assistance")

        # Add the creator's name at the bottom
        creator_label = tk.Label(master, text="Created by Suraj Chandra Joshi", font=("Helvetica", 10, "italic"), bg="#f0f0f0", fg="#555555")
        creator_label.pack(side="bottom", pady=10)

    def run_script(self, script_name):
        python_executable = sys.executable
        script_path = os.path.join(os.getcwd(), script_name)

        try:
            subprocess.Popen([python_executable, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {script_name}:\n{e}")

    def open_email(self, email):
        webbrowser.open(f"mailto:{email}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MasterGUI(root)
    root.mainloop()
