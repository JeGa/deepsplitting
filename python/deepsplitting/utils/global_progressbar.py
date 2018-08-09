import progressbar
import math

bar = None


def init(max_epoch, batch_size, N):
    global bar

    bar = BarWrapper(max_epoch, batch_size, N)


class BarWrapper:
    def __init__(self, max_epoch, batch_size, N):
        self.current_epoch = 1
        self.current_batch = 1

        self.max_batch = int(math.ceil(N / batch_size))
        max_value = max_epoch * self.max_batch

        self.format_custom_text = progressbar.FormatCustomText(
            ' (Epoch: %(epoch)d/%(max_epoch)d, Batch: %(batch)d/%(max_batch)d,' +
            ' Data loss = %(dataloss).6f, Lagrangian = %(lagrangian).6f) ',
            dict(epoch=1, max_epoch=max_epoch, batch=1, max_batch=self.max_batch, dataloss=-1, lagrangian=-1),
        )

        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(marker=progressbar.RotatingMarker()),
            progressbar.SimpleProgress(),
            self.format_custom_text,
            progressbar.ETA()
        ]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets)
        self.bar.start()

    def next_batch(self, dataloss=-1, lagrangian=-1):
        self.current_batch += 1

        if self.current_batch == self.max_batch + 1:
            self.current_batch = 1
            self.current_epoch += 1

        self.format_custom_text.update_mapping(
            epoch=self.current_epoch,
            batch=self.current_batch,
            dataloss=dataloss,
            lagrangian=lagrangian)

        self.bar += 1

    def finish(self):
        self.bar.finish()
