import datetime
import itertools
import os.path
import yaml
import ipywidgets
import IPython
import enum
from matplotlib import pyplot as plt

import deepsplitting.utils.global_config as global_config


class Section(enum.Enum):
    RUN = 0
    PARAMS = 1
    GLOBALCFG = 2
    TIME = 3
    DATA = 4


def plot_loss_curve(losses, title=''):
    plt.figure()

    plt.title(title)

    plt.plot(losses, linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()


def plot_yaml(files):
    if len(files) == 0 or files is None:
        raise ValueError("Requires list of result yaml files.")

    marker = itertools.cycle(('s', 'D', '.', 'o', '^', 'v', '*', '8', 'x'))

    marker_factor = 0.04
    marker_min = 4
    hyperparams_y = -0.1

    # Get length of data from the first file.
    N = len(load_yaml(files[0])[Section.DATA.value]['Data']['data_loss'])
    every = max(int(marker_factor * N), marker_min)

    plt.figure(1)

    keys = []

    for file in files:
        yaml_dict = load_yaml(file)

        optimizer_key = yaml_dict[Section.RUN.value]

        for loss_key, losses in yaml_dict[Section.DATA.value]['Data'].items():
            label = "{} {} {}s".format(optimizer_key, loss_key, yaml_dict[Section.TIME.value]['Time'])

            plt.plot(losses, label=label, linewidth=1.0, marker=next(marker), markevery=every, markerfacecolor='none')

        text = "{}: {}".format(optimizer_key, str(yaml_dict[Section.PARAMS.value]['Parameters']))
        plt.text(0, hyperparams_y, text, transform=plt.gcf().transFigure)
        hyperparams_y -= -0.03

        keys.append(optimizer_key)
        every += 1

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    all_optimizer_keys_str = ''
    for k in keys:
        all_optimizer_keys_str += str(k) + '_'

    plot_filename = os.path.join(
        global_config.cfg.results_folder,
        global_config.cfg.results_subfolders['plots'],
        all_optimizer_keys_str + datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S') + '.pdf')

    plt.savefig(plot_filename, bbox_inches='tight')

    # IPython.display.clear_output(wait=True)
    # IPython.display.display(plt.gcf())
    plt.show()


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


def mkdir_ifnot(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


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

    def create_widgets(self, file_dict):

        def plot_btn_click(x, selected):
            plot_yaml(selected.value)

        def show_btn_click(_, selected, results_show):
            selected = selected.value

            if len(selected) != 1:
                print('Select one entry to show.')
                return

            with open(os.path.join(self.data_folder, selected[0]), 'r') as f:
                text = f.read()

            results_show.value = text

        control_label = ipywidgets.Label(value='Select multiple files by holding Strg.')
        results_select = ipywidgets.SelectMultiple(options=file_dict, rows=20,
                                                   layout=ipywidgets.Layout(width='auto'))
        results_show = ipywidgets.Textarea(placeholder='Selected config file', disabled=False,
                                           layout=ipywidgets.Layout(align_items='stretch', flex='1 1 auto'))

        show_btn = ipywidgets.Button(description='Show', layout=ipywidgets.Layout(width='auto'))
        show_btn.on_click(lambda x: show_btn_click(x, results_select, results_show))

        plot_btn = ipywidgets.Button(description='Plot', layout=ipywidgets.Layout(width='auto'))
        plot_btn.on_click(lambda x: plot_btn_click(x, results_select))

        control_box = ipywidgets.VBox([control_label, results_select, show_btn, plot_btn])
        all_box = ipywidgets.HBox([control_box, results_show], layout=ipywidgets.Layout(border='solid'))

        return all_box

    def get_file_list(self):
        file_list = []

        with os.scandir(self.data_folder) as it:
            for entry in it:
                if entry.is_file():
                    file_list.append(entry.name)

        return file_list

    def run(self):
        file_list = self.get_file_list()

        IPython.display.display(self.create_widgets(file_list))
