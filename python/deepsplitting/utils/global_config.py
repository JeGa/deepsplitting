from deepsplitting.utils.misc import Params


class GlobalParams(Params):
    def __init__(self, **params):
        super(GlobalParams, self).__init__(**params)


cfg = None
