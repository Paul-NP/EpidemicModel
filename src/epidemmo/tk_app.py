import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from epidemmo import ModelBuilder
from epidemmo.model import EpidemicModel
from epidemmo import Standard


class EpidemicModelApp(tk.Tk):
    def __init__(self, model: EpidemicModel) -> None:
        super().__init__()
        self.model = model
        self.auto_update = tk.BooleanVar(value=True)
        self.create_widgets()
        # self.state('zoomed')  # Развернуть окно на весь экран

    def create_widgets(self) -> None:
        # Create frame for parameters
        parameters_frame = tk.Frame(self)
        parameters_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create sliders for each factor
        self.sliders: dict[str, tk.Scale] = {}
        for i, factor in enumerate(self.model.factors):
            if factor['name'][:3] in ['ind', 'end']:
                continue
            label = tk.Label(parameters_frame, text=factor['name'])
            label.grid(row=i, column=0)
            slider = tk.Scale(parameters_frame, from_=0, to=1,
                              resolution=0.01, orient=tk.HORIZONTAL,
                              command=lambda value, f=factor: self.update_factor(f['name'], value))
            print(factor)
            slider.set(factor['value'])

            slider.grid(row=i, column=1)
            self.sliders[factor['name']] = slider

        # Create checkbox for auto update
        auto_update_checkbox = tk.Checkbutton(parameters_frame, text="Auto update", variable=self.auto_update)
        auto_update_checkbox.grid(row=len(self.model.factors), column=0, columnspan=2)
        # auto_update_checkbox.deselect()

        # Create spinbox for simulation duration
        tk.Label(parameters_frame, text="Simulation duration").grid(row=len(self.model.factors) + 1, column=0)
        # нужна StringVar, чтобы установить значение по умолчанию
        duration_var = tk.StringVar(self, value="100")
        self.simulation_duration = tk.Spinbox(parameters_frame, from_=1, to=1000, width=5, textvariable=duration_var)
        self.simulation_duration.grid(row=len(self.model.factors) + 1, column=1)

        # Create button to update graph
        update_button = tk.Button(parameters_frame, text="Update", command=self.update_graph)
        update_button.grid(row=len(self.model.factors) + 2, column=0, columnspan=2)

        # Create plot area
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10)

        # Update graph initially
        self.update_graph()

    def update_factor(self, factor_name: str, value: str) -> None:
        factor_dict = {factor_name: float(value)}
        print(factor_dict)
        self.model.set_factors(**factor_dict)
        print({factor['name']: float(factor['value']) for factor in self.model.factors})
        if self.auto_update.get():
            self.update_graph()

    def update_graph(self) -> None:
        duration = int(self.simulation_duration.get())
        result_df = self.model.start(duration)
        self.ax.clear()
        result_df.plot(ax=self.ax)
        self.canvas.draw()


def get_two_strains() -> ModelBuilder:
    builder = ModelBuilder()
    # builder.add_stages('SR', 'RS', 'IR', 'RI', 'RR', SS=998, SI=1, IS=1)
    builder.add_stages(SS=998, SI=1, IS=1, SR=0, RS=0, IR=0, RI=0, RR=0)
    builder.add_factor('beta', 0.4, latex_repr=r'\beta').add_factor('eta', 0.2, latex_repr=r'\eta')
    builder.add_factor('gamma', 0.1, latex_repr=r'\gamma').add_factor('delta', 0.1, latex_repr=r'\delta')
    builder.add_flow('SS', 'IS', 'beta', {'IS': 1, 'IR': 1}).add_flow('IS', 'RS', 'gamma')
    builder.add_flow('RS', 'RI', 'eta', {'SI': 1, 'RI': 1}).add_flow('RI', 'RR', 'delta')

    builder.add_flow('SS', 'SI', 'eta', {'SI': 1, 'RI': 1}).add_flow('SI', 'SR', 'delta')
    builder.add_flow('SR', 'IR', 'beta', {'IS': 1, 'IR': 1}).add_flow('IR', 'RR', 'gamma')

    builder.set_model_name('two_strains')

    return builder

if __name__ == "__main__":
    model = get_two_strains().build()
    app = EpidemicModelApp(model)
    app.mainloop()