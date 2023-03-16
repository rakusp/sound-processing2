from PyQt5 import QtWidgets
from gui.functions import *
from scipy.io.wavfile import read
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class AudioPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=9, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel('Time in seconds')
        super(AudioPlot, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.resize(1200, 800)
        self.setWindowTitle('Sound Processing App')
        self.plot = AudioPlot(self)
        self.fps = None
        self.data = None
        self._setup()

    def _setup(self):
        main_layout = QtWidgets.QHBoxLayout()
        plot_container = QtWidgets.QVBoxLayout()
        plot_layout = QtWidgets.QVBoxLayout()
        plot_controls_layout = QtWidgets.QHBoxLayout()

        # plot and controls
        toolbar = NavigationToolbar2QT(self.plot, self)
        load_button = QtWidgets.QPushButton(text='Load File')
        load_button.clicked.connect(self.load_file)
        default_button = QtWidgets.QPushButton(text='Default')
        default_button.clicked.connect(lambda: self._draw_plot(self.fps, scale_data(self.data)))
        ste_button = QtWidgets.QPushButton(text='Short Time Energy')
        ste_button.clicked.connect(lambda: self._draw_by_frames(func=short_time_energy))
        zcr_button = QtWidgets.QPushButton(text='Zero Crossing Rate')
        zcr_button.clicked.connect(lambda: self._draw_by_frames(func=zero_crossing_rate))
        acf_button = QtWidgets.QPushButton(text='Autocorrelation')
        acf_button.clicked.connect(lambda: self._draw_by_frames(func=autocorrelation_function, l=10))
        amd_button = QtWidgets.QPushButton(text='Average Magnitude Difference')
        amd_button.clicked.connect(lambda: self._draw_by_frames(func=average_magnitude_difference, l=10))
        plot_controls_layout.addWidget(load_button)
        plot_controls_layout.addWidget(default_button)
        plot_controls_layout.addWidget(ste_button)
        plot_controls_layout.addWidget(zcr_button)
        plot_controls_layout.addWidget(acf_button)
        plot_controls_layout.addWidget(amd_button)

        # plot info
        info_layout = QtWidgets.QGridLayout()
        ste_label = QtWidgets.QLabel(text='STE:')
        self.ste_field = QtWidgets.QLineEdit()
        self.ste_field.setReadOnly(True)

        zcr_label = QtWidgets.QLabel(text='ZCR:')
        self.zcr_field = QtWidgets.QLineEdit()
        self.zcr_field.setReadOnly(True)

        acf_label = QtWidgets.QLabel(text='ACF:')
        self.acf_field = QtWidgets.QLineEdit()
        self.acf_field.setReadOnly(True)

        amd_label = QtWidgets.QLabel(text='AMD:')
        self.amd_field = QtWidgets.QLineEdit()
        self.amd_field.setReadOnly(True)

        info_layout.addWidget(ste_label, 0, 0)
        info_layout.addWidget(self.ste_field, 0, 1)
        info_layout.addWidget(zcr_label, 1, 0)
        info_layout.addWidget(self.zcr_field, 1, 1)
        info_layout.addWidget(acf_label, 2, 0)
        info_layout.addWidget(self.acf_field, 2, 1)
        info_layout.addWidget(amd_label, 3, 0)
        info_layout.addWidget(self.amd_field, 3, 1)

        # setup layouts
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.plot)
        plot_layout.addLayout(plot_controls_layout)

        plot_container.addLayout(plot_layout)
        main_layout.addLayout(plot_container)
        main_layout.addLayout(info_layout)

        # setup app window
        main = QtWidgets.QWidget()
        main.setLayout(main_layout)
        self.setCentralWidget(main)

    def _setup_plot(self):
        toolbar = NavigationToolbar2QT(self.plot, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.plot)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

    def load_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                            "Audio Files (*.wav)")
        if filename:
            print(f'Loading {filename}...')
            self.fps, self.data = read(filename)
            print(f'File loaded successfully')
            self._draw_plot(self.fps, scale_data(self.data))
            self._set_values()

    def _draw_plot(self, fps, data):
        # fps - frames per second
        self.plot.fig.clear(keep_observers=True)
        self.plot.axes = self.plot.fig.add_subplot(111)
        self.plot.axes.plot(np.array(range(len(data))) / fps, data)
        self.plot.draw()
        self.plot.axes.set_xlabel('Time in seconds')

    def _draw_by_frames(self, func, l=None):
        if (self.fps is None) or (self.data is None):
            return

        frames, frame_length = framing(sig=scale_data(self.data), fs=self.fps, win_len=0.25)

        if l is not None:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames, l=l)
        else:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames)

        self._draw_plot(frame_length, data)

    def _set_values(self):
        data = scale_data(self.data)
        self.ste_field.setText(str(short_time_energy(data)))
        self.zcr_field.setText(str(zero_crossing_rate(data)))
        self.acf_field.setText(str(autocorrelation_function(data, l=10)))
        self.amd_field.setText(str(average_magnitude_difference(data, l=10)))
