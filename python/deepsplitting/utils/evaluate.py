import datetime
import itertools
import os.path
import yaml
import ipywidgets
import IPython
import enum
import math
import pathlib
from matplotlib import pyplot as plt

import deepsplitting.utils.global_config as global_config

plot_properties = global_config.Params(
    marker_factor=0.04,
    marker_min=4,
    hyperparams_y=0,
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


def add_to_plot(file, loss_key, plot=1, verbose=True):
    yaml_dict = load_yaml(file)
    losses = yaml_dict[Section.DATA.value]['Data'][loss_key]

    if not hasattr(add_to_plot, 'every'):
        add_to_plot.every = max(int(plot_properties.marker_factor * len(losses)), plot_properties.marker_min)

    if not hasattr(add_to_plot, 'text_y'):
        add_to_plot.text_y = plot_properties.hyperparams_y

    optimizer_key = yaml_dict[Section.RUN.value]
    label = "{} {}".format(optimizer_key, loss_key)
    if verbose:
        label += ' ' + yaml_dict[Section.TIME.value]['Time'] + 's'

    params_text = "Optimizer: {}".format(optimizer_key)
    for key, value in yaml_dict[Section.PARAMS.value]['Parameters'].items():
        params_text += os.linesep + "{}={}".format(key, value)

    global_config_text = "Config: {} {} {}s".format(optimizer_key, loss_key,
                                                    int(float(yaml_dict[Section.TIME.value]['Time'])))

    for key, value in yaml_dict[Section.GLOBALCFG.value]['Global_config'].items():
        global_config_text += os.linesep + "{} = {}".format(key, str(value))

    plt.figure(plot)

    plt.plot(losses, label=label, linewidth=0.4,
             marker=next(plot_properties.marker), markevery=add_to_plot.every, markerfacecolor='none')

    if verbose:
        plt.text(0.05, add_to_plot.text_y, global_config_text,
                 horizontalalignment='left', verticalalignment='top', transform=plt.gcf().transFigure)
        plt.text(0.55, add_to_plot.text_y, params_text,
                 horizontalalignment='left', verticalalignment='top', transform=plt.gcf().transFigure)

        add_to_plot.text_y -= 0.55

    add_to_plot.every += math.ceil(0.1 * add_to_plot.every)
    plt.subplots_adjust(bottom=0.15)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.show()

    return optimizer_key


def clear():
    if hasattr(add_to_plot, 'every'):
        del add_to_plot.every
    if hasattr(add_to_plot, 'text_y'):
        del add_to_plot.text_y


add_to_plot.clear = clear


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

        self.add_to_plot_btn = None
        self.clear_btn = None
        self.save_btn = None

        self.verbose_plot_checkbox = None

        self.main_widget = None

        self.current_keys = []

        reset_plot_properties()
        add_to_plot.clear()

    def create_widgets(self, file_list):
        control_label = ipywidgets.Label(value='Select file and add the result data shown on the right to the plot.')

        def sortfun(x):
            splitted = x.rsplit('_', maxsplit=2)
            return splitted[0], splitted[1], splitted[2][:-4]

        self.results_select = ipywidgets.Select(options=sorted(file_list, key=sortfun), rows=20)
        self.data_select = ipywidgets.Select(placeholder='Select file.', rows=20,
                                             layout=ipywidgets.Layout(width='30%'))

        self.results_show = ipywidgets.Textarea(layout=ipywidgets.Layout(width='auto', height='100%'))

        self.add_to_plot_btn = ipywidgets.Button(description='Add to plot', layout=ipywidgets.Layout(width='auto'))
        self.clear_btn = ipywidgets.Button(description='Clear', layout=ipywidgets.Layout(width='auto'))
        self.save_btn = ipywidgets.Button(description='Save', layout=ipywidgets.Layout(width='auto'))

        self.verbose_plot_checkbox = ipywidgets.Checkbox(value=True, description='Verbose plot.', disabled=False,
                                                         layout=ipywidgets.Layout(width='auto'))

        select_box = ipywidgets.HBox([self.results_select, self.data_select])

        control_box = ipywidgets.VBox([control_label, select_box,
                                       self.add_to_plot_btn, self.clear_btn, self.save_btn,
                                       self.verbose_plot_checkbox])

        show_box = ipywidgets.VBox([self.results_show],
                                   layout=ipywidgets.Layout(align_items='stretch', flex='1 1 auto'))

        self.main_widget = ipywidgets.HBox([control_box, show_box], layout=ipywidgets.Layout(border='solid'))

    def create_actions(self):
        def add_to_plot_btn_click(_):
            file = self.results_select.value
            data = self.data_select.value

            verbose = self.verbose_plot_checkbox.value

            if data is not None:
                key = add_to_plot(file, data, verbose=verbose)

                self.current_keys.append(key + '-' + data)

        def clear_btn_click(_):
            clear_plot()
            self.current_keys = []

            add_to_plot.clear()

        def save_btn_click(_):
            name = ''

            for k in self.current_keys:
                name += k + '_'

            save_plot(name)

        self.add_to_plot_btn.on_click(add_to_plot_btn_click)
        self.clear_btn.on_click(clear_btn_click)
        self.save_btn.on_click(save_btn_click)

        def f(x):
            selected = x['new']

            # Display in data select view.
            results_data = get_results_data(selected)
            self.data_select.options = list(results_data)

            # Display yaml in text box.
            with open(os.path.join(self.data_folder, selected), 'r') as file:
                text = file.read()
            self.results_show.value = text

        self.results_select.observe(f, names='value')

    def get_file_list(self):
        return [p.name for p in pathlib.Path(self.data_folder).glob('*.yml')]

    def run(self):
        file_list = self.get_file_list()

        self.create_widgets(file_list)
        self.create_actions()

        IPython.display.display(self.main_widget)
