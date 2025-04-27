import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#%%
class DataVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("数据可视化")

        self.df = None
        self.selected_x_values = []

        self.load_button = tk.Button(master, text="选择文件", command=self.load_csv)
        self.load_button.pack()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.show_plot()

    def show_plot(self):
        self.plot_window = tk.Toplevel(self.master)

        fig, ax = plt.subplots()
        ax.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'b-', label='Data')

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not None:
            self.selected_x_values.append(event.xdata)
            if len(self.selected_x_values) == 2:
                self.plot_zoom()

    def plot_zoom(self):
        self.plot_window_zoom = tk.Toplevel(self.master)

        fig, ax = plt.subplots()
        ax.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'b-', label='Data')
        ax.set_xlim(min(self.selected_x_values), max(self.selected_x_values))

        self.canvas_zoom = FigureCanvasTkAgg(fig, master=self.plot_window_zoom)
        self.canvas_zoom.draw()
        self.canvas_zoom.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.Button(self.plot_window_zoom, text="返回选中X轴值", command=self.return_selected_x_values).pack()

    def return_selected_x_values(self):
        selected_x_values = tuple(self.selected_x_values)
        print("选中的X轴值:", selected_x_values)
        self.master.destroy()

def main():
    root = tk.Tk()
    app = DataVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
#%%
