import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from threading import Thread
import time

class LinearRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Visualizer")

        self.points = []
        self.table_frame = ttk.Frame(root)
        self.table_frame.pack(pady=10)

        ttk.Label(self.table_frame, text="X").grid(row=0, column=0)
        ttk.Label(self.table_frame, text="Y").grid(row=0, column=1)

        self.entries = []
        for i in range(5):
            x_entry = ttk.Entry(self.table_frame, width=10)
            y_entry = ttk.Entry(self.table_frame, width=10)
            x_entry.grid(row=i+1, column=0)
            y_entry.grid(row=i+1, column=1)
            self.entries.append((x_entry, y_entry))

        self.plot_button = ttk.Button(root, text="Fit Line", command=self.animate_regression)
        self.plot_button.pack(pady=10)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Line of Best Fit")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # Log box for slope and intercept updates
        self.log_box = tk.Text(root, height=8, width=50)
        self.log_box.pack(pady=10)
        self.log_box.insert(tk.END, "Logs will appear here...\n")

    def get_points(self):
        self.points.clear()
        for x_entry, y_entry in self.entries:
            try:
                x = float(x_entry.get())
                y = float(y_entry.get())
                self.points.append((x, y))
            except ValueError:
                continue

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def animate_regression(self):
        thread = Thread(target=self._animate_regression)
        thread.start()

    def _animate_regression(self):
        self.get_points()
        if len(self.points) < 2:
            self.log("Enter at least two valid points.")
            return

        x_vals = np.array([p[0] for p in self.points])
        y_vals = np.array([p[1] for p in self.points])

        m, b = np.random.randn(), np.random.randn()
        learning_rate = 0.05

        for step in range(200):  # Increase steps
            predictions = m * x_vals + b
            error = predictions - y_vals

            m_grad = 2 * np.dot(error, x_vals) / len(x_vals)
            b_grad = 2 * np.sum(error) / len(x_vals)

            m -= learning_rate * m_grad
            b -= learning_rate * b_grad

            # Plotting
            self.ax.clear()
            self.ax.scatter(x_vals, y_vals, color='blue')
            self.ax.plot(x_vals, m * x_vals + b, color='red')
            self.ax.set_xlim(min(x_vals)-1, max(x_vals)+1)
            self.ax.set_ylim(min(y_vals)-1, max(y_vals)+1)
            self.ax.set_title(f"Step {step+1} | m={m:.2f}, b={b:.2f}")
            self.canvas.draw()

            # Log current m and b
            if step % 10 == 0 or step == 199:
                self.log(f"Step {step+1}: m = {m:.4f}, b = {b:.4f}")

            time.sleep(0.01)

        self.log("Fitting complete!")

# Run GUI locally
root = tk.Tk()
app = LinearRegressionGUI(root)
root.mainloop()

