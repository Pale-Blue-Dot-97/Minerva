from Minerva.utils import visutils
# import numpy as np
# from numpy.testing import assert_array_equal


# def test_de_interlace():
#    x = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
#    x2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
#    assert assert_array_equal(visutils.de_interlace(x, 3), x2) is None


def test_format_names():
    timestamp = "01-01-1970"
    model_name = "tester"
    path = ['test', 'path']
    names = visutils.format_plot_names(model_name, timestamp, path)

    filenames = {'History': f'test/path/{model_name}_{timestamp}_MH.png',
                 'Pred': f'test/path/{model_name}_{timestamp}_TP.png',
                 'CM': f'test/path/{model_name}_{timestamp}_CM.png',
                 'ROC': f'test/path/{model_name}_{timestamp}_ROC.png',
                 'Mask': f'test/path/Masks/{model_name}_{timestamp}_Mask',
                 'PvT': f'test/path/PvTs/{model_name}_{timestamp}_PvT'}

    assert filenames == names
