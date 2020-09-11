from elisa.graphic import data_set_graphics


class Plot(object):
    def __init__(self, instance):
        self.data_set = instance

    def display_observation(self, **plot_kwargs):
        obs_kwargs = dict()

        obs_kwargs.update({
            'x_data': self.data_set.x_data,
            'y_data': self.data_set.y_data,
            'y_err': self.data_set.y_err,
            'x_unit': self.data_set.x_unit,
            'y_unit': self.data_set.y_unit,
            'plot_kwargs': plot_kwargs
        })
        data_set_graphics.display_observations(**obs_kwargs)
