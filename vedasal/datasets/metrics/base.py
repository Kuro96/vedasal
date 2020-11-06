class BaseMetric:
    def __init__(self):
        pass

    def __repr__(self):
        rpl = ', '.join([f'{k}={v}'
                         for k, v in sorted(self.__dict__.items())
                         if not k.startswith('_')])
        return f'{self.__class__.__name__}({rpl})'
