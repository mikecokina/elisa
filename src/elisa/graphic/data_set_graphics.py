import matplotlib.pyplot as plt
from .. import units as u


def display_observations(**kwargs):
    """
    Graphics method for displaying the DataSet content.

    :param kwargs: Dict;
    :**kwargs options**:
        * :x_data: numpy.array;
        * :x_unit: astropy.unit.Unit;
        * :y_data: numpy.array;
        * :y_err: numpy.array;
        * :y_unit: astropy.unit.Unit;
        * :plot_kwargs: Dict; plot arguments fot plotting functions
                              (matplotlib.pyplot.errorbar or matplotlib.pyplot.scatter).

    """
    figure = plt.figure()
    if kwargs['y_err'] is not None:
        plt.errorbar(x=kwargs['x_data'], y=kwargs['y_data'], yerr=kwargs['y_err'], linestyle='none',
                     **kwargs['plot_kwargs'])
    else:
        plt.scatter(x=kwargs['x_data'], y=kwargs['y_data'], **kwargs['plot_kwargs'])

    x_lbl = 'Phase' if kwargs['x_unit'] == u.dimensionless_unscaled else f'Time [{kwargs["x_unit"]}]'
    y_lbl = 'Flux' if kwargs['y_unit'] == u.dimensionless_unscaled else f'Magnitude [{kwargs["y_unit"]}]'
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.subplots_adjust(top=0.98, right=0.98)

    plt.show()
