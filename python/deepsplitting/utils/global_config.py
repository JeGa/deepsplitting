class Params:
    def __init__(self, **params):
        self.__dict__.update(params)

    def csv_format(self):
        return ["{}={}".format(key, str(v)) for key, v in self.__dict__.items()]


class GlobalParams(Params):
    def __init__(self, **params):
        super(GlobalParams, self).__init__(**params)


cfg = None
