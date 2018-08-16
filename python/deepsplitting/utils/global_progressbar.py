import progressbar
import math

bar = None


def init(max_epoch, batch_size, N, custom_text_dict):
    global bar

    bar = BarWrapper(max_epoch, batch_size, N, custom_text_dict)


class BarWrapper:
    def __init__(self, max_epoch, batch_size, N, custom_text_dict):
        self.current_epoch = 0
        self.current_batch = 0

        self.max_batch = int(math.ceil(N / batch_size))
        max_value = max_epoch * self.max_batch

        base_text = '(Epoch: [%(epoch)d/%(max_epoch)d], Batch: [%(batch)d/%(max_batch)d]) '
        base_dict = dict(epoch=1, max_epoch=max_epoch, batch=1, max_batch=self.max_batch)

        custom_text = ''
        for key, value in custom_text_dict.items():
            custom_text += "{} = %({}).6f, ".format(value, key)
            base_dict[key] = -1

        self.format_custom_text = progressbar.FormatCustomText(
            base_text + custom_text, base_dict,
        )

        widgets = [
            progressbar.Percentage(), ' ',
            progressbar.AnimatedMarker(), ' ',
            progressbar.Bar(), ' ',
            progressbar.SimpleProgress(), ' ',
            self.format_custom_text, ' ',
            progressbar.ETA()
        ]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets)
        self.bar.start()

    def next_batch(self, custom_data):
        self.current_batch += 1

        if self.current_batch == self.max_batch + 1:
            self.current_batch = 1
            self.current_epoch += 1

        self.format_custom_text.update_mapping(
            epoch=self.current_epoch,
            batch=self.current_batch)

        self.format_custom_text.update_mapping(**custom_data)

        self.bar += 1

    def finish(self):
        progressbar.streams.flush()
        self.bar.finish()
