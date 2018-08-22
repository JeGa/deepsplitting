import datetime
import itertools
import os.path
import yaml
import ipywidgets
import IPython
import enum
from matplotlib import pyplot as plt

import deepsplitting.utils.global_config as global_config

plot_properties = global_config.Params(
    marker_factor=0.04,
    marker_min=4,
    hyperparams_y=-0.1,
    marker=itertools.cycle(('s', 'D', '.', 'o', '^', 'v', '*', '8', 'x')))


def reset_plot_properties():
    plot_properties.marker = itertools.cycle(('s', 'D', '.', 'o', '^', 'v', '*', '8', 'x'))
    plot_properties.hyperparams_y = -0.1


class Section(enum.Enum):
    RUN = 0
    PARAMS = 1
    GLOBALCFG = 2
    TIME = 3
    DATA = 4


def clear_plot(plot=1):
    plt.figure(plot)
    plt.clf()

    reset_plot_properties()


def save_plot(name, plot=1):
    plt.figure(plot)

    plot_filename = os.path.join(
        global_config.cfg.results_folder,
        global_config.cfg.results_subfolders['plots'],
        name + datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S') + '.pdf')

    plt.savefig(plot_filename, bbox_inches='tight')


def add_to_plot(file, loss_key, plot=1):
    yaml_dict = load_yaml(file)

    losses = yaml_dict[Section.DATA.value]['Data'][loss_key]

    every = max(int(plot_properties.marker_factor * len(losses)), plot_properties.marker_min)

    optimizer_key = yaml_dict[Section.RUN.value]
    label = "{} {} {}s".format(optimizer_key, loss_key, yaml_dict[Section.TIME.value]['Time'])
    text = "{}: {}".format(optimizer_key, str(yaml_dict[Section.PARAMS.value]['Parameters']))

    plt.figure(plot)

    plt.plot(losses, label=label,
             linewidth=1.0, marker=next(plot_properties.marker), markevery=every, markerfacecolor='none')

    plt.text(0, plot_properties.hyperparams_y, text, transform=plt.gcf().transFigure)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()

    plot_properties.hyperparams_y -= -0.03

    return optimizer_key


def get_results_data(file):
    yaml_dict = load_yaml(file)

    return yaml_dict[Section.DATA.value]['Data'].keys()


def load_yaml(name):
    folder = os.path.join(global_config.cfg.results_folder, global_config.cfg.results_subfolders['data'])

    with open(os.path.join(folder, name), 'r', newline='') as f:
        yaml_dict = yaml.load(f)

    return yaml_dict


def save_yaml(optimizer_key, name, time, data, params):
    folder = os.path.join(global_config.cfg.results_folder, global_config.cfg.results_subfolders['data'])

    filepath = os.path.join(folder, name)

    with open(filepath, 'w', newline='') as f:
        to_write = []

        if params is not None:
            params = {'Parameters': params.__dict__}
        else:
            params = None

        cfgdict = {'Global_config': global_config.cfg.__dict__}
        timedict = {'Time': time}
        datadict = {'Data': data}

        to_write.append(optimizer_key)
        to_write.append(params)
        to_write.append(cfgdict)
        to_write.append(timedict)
        to_write.append(datadict)

        yaml.dump(to_write, f, default_flow_style=False)

        print("Saved results as {}.".format(filepath))


def time_str(key, timer):
    return "{:.6f}".format(timer.times[key]) if key in timer.times else ''


def save_summary(optimizer, summary, timer):
    for optimizer_key, all_losses in summary.items():
        timerstr = time_str(optimizer_key, timer)

        filename = "{}_{}.yml".format(optimizer_key, datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S'))

        save_yaml(optimizer_key, filename, timerstr, all_losses, optimizer[optimizer_key].hyperparams)


def mkdir_ifnot(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def make_results_folder(cfg):
    mkdir_ifnot(cfg.results_folder)

    for key, f in cfg.results_subfolders.items():
        mkdir_ifnot(os.path.join(cfg.results_folder, f))


class Notebook:
    def __init__(self):
        results_folder = global_config.cfg.results_folder
        results_subfolder = global_config.cfg.results_subfolders

        self.data_folder = os.path.join(results_folder, results_subfolder['data'])
        self.plots_folder = os.path.join(results_folder, results_subfolder['plots'])

        self.results_select = None
        self.data_select = None
        self.results_show = None

        self.show_btn = None
        self.add_to_plot_btn = None
        self.clear_btn = None
        self.save_btn = None

        self.main_widget = None

        self.current_keys = []

    def create_widgets(self, file_dict):
        control_label = ipywidgets.Label(value='Select file and add the result data shown on the right to the plot.')
        results_show_label = ipywidgets.Label(value='Select file to show.')

        self.results_select = ipywidgets.Select(options=file_dict, rows=20)
        self.data_select = ipywidgets.Select(placeholder='Select file.', rows=20,
                                             layout=ipywidgets.Layout(width='30%'))

        self.results_show = ipywidgets.Textarea(layout=ipywidgets.Layout(width='auto', height='100%'))

        self.show_btn = ipywidgets.Button(description='Show file', layout=ipywidgets.Layout(width='auto'))
        self.add_to_plot_btn = ipywidgets.Button(description='Add to plot', layout=ipywidgets.Layout(width='auto'))
        self.clear_btn = ipywidgets.Button(description='Clear', layout=ipywidgets.Layout(width='auto'))
        self.save_btn = ipywidgets.Button(description='Save', layout=ipywidgets.Layout(width='auto'))

        select_box = ipywidgets.HBox([self.results_select, self.data_select])

        control_box = ipywidgets.VBox(
            [control_label, select_box, self.show_btn, self.add_to_plot_btn, self.clear_btn, self.save_btn])

        show_box = ipywidgets.VBox([results_show_label, self.results_show],
                                   layout=ipywidgets.Layout(align_items='stretch', flex='1 1 auto'))

        self.main_widget = ipywidgets.HBox([control_box, show_box], layout=ipywidgets.Layout(border='solid'))

    def create_actions(self):
        def add_to_plot_btn_click(_):
            file = self.results_select.value
            data = self.data_select.value

            if data is not None:
                key = add_to_plot(file, data)

                self.current_keys.append(key + '-' + data)

        def show_btn_click(_):
            selected = self.results_select.value

            with open(os.path.join(self.data_folder, selected), 'r') as file:
                text = file.read()

            self.results_show.value = text

        def clear_btn_click(_):
            clear_plot()
            self.current_keys = []

        def save_btn_click(_):
            name = ''

            for k in self.current_keys:
                name += k + '_'

            save_plot(name)

        self.show_btn.on_click(show_btn_click)
        self.add_to_plot_btn.on_click(add_to_plot_btn_click)
        self.clear_btn.on_click(clear_btn_click)
        self.save_btn.on_click(save_btn_click)

        def f(x):
            results_data = get_results_data(x['new'])

            self.data_select.options = list(results_data)

        self.results_select.observe(f, names='value')

    def get_file_list(self):
        file_list = []

        with os.scandir(self.data_folder) as it:
            for entry in it:
                if entry.is_file():
                    file_list.append(entry.name)

        return file_list

    def run(self):
        file_list = self.get_file_list()

        self.create_widgets(file_list)
        self.create_actions()

        IPython.display.display(self.main_widget)
