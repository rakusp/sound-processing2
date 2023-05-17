import copy

import matplotlib
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIntValidator
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.io.wavfile import read

from gui.functions import *

matplotlib.use('Qt5Agg')


class AudioPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=9, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.fig = fig
        self.axes = fig.subplots(nrows=2)
        self.axes[1].set_xlabel('Time (s)')
        super(AudioPlot, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.resize(1200, 800)
        self.setWindowTitle('Sound Processing App')
        self.plot = AudioPlot(self)
        self.fps = None
        self.data = None
        self.duration = 0
        self.frame_len = 25
        self.frame_hop = 10
        self.lag = 10
        self.freq0 = 0
        self.freq1 = 2000
        self.use_freq = False
        self._setup()

    def _setup(self):
        main_layout = QtWidgets.QHBoxLayout()
        plot_layout = QtWidgets.QVBoxLayout()
        toolbar_layout = QtWidgets.QHBoxLayout()

        # plot and controls
        self.toolbar = NavigationToolbar2QT(self.plot, self)

        load_button = QtWidgets.QPushButton(text='Load File')
        load_button.clicked.connect(self.load_file)
        self.plot_type_menu = QtWidgets.QComboBox()
        # self.plot_type_menu.addItems(['Short Time Energy', 'Zero Crossing Rate', 'Autocorrelation Function',
        #                               'Average Magnitude Difference', 'Fundamental Frequency Detection',
        #                               'Unvoice Phones Detection', 'Rectangular Window', 'Bartlett Window',
        #                               'Hann Window', 'Hamming Window', 'Blackman Window', 'Spectral Centroid',
        #                               'Effective Bandwidth', 'Band Energy Ratio', 'Spectral Flatness Measure',
        #                               'Spectral Crest Factor'])
        self.plot_type_dict = {
            'Short Time Energy': (short_time_energy,),
            'Zero Crossing Rate': (zero_crossing_rate,),
            'Autocorrelation Function': (autocorrelation_function, 'use_lag'),
            'Average Magnitude Difference': (average_magnitude_difference, 'use_lag'),
            'Fundamental Frequency Detection': (fundamental_frequency_detection, 'use_fs'),
            'Unvoice Phones Detection': (unvoice_phones_detection, 'use_fs'),
            'Spectral Centroid': (spectral_centroid, 'use_kwargs'),
            'Effective Bandwidth': (effective_bandwidth, 'use_kwargs'),
            'Band Energy Ratio': (band_energy_ratio, 'use_kwargs'),
            'Spectral Flatness Measure': (spectral_flatness_measure, 'use_kwargs'),
            'Spectral Crest Factor': (spectral_crest_factor, 'use_kwargs'),
            'Rectangular Window': (rectangular_window, 'use_window'),
            'Bartlett Window': (bartlett_window, 'use_window'),
            'Hann Window': (hann_window, 'use_window'),
            'Hamming Window': (hamming_window, 'use_window'),
            'Blackman Window': (blackman_window, 'use_window')
        }

        self.plot_type_menu.addItems(list(self.plot_type_dict.keys()))

        self.plot_type_menu.currentTextChanged.connect(self.change_plot)

        self.frequency_button = QtWidgets.QPushButton(text='Use frequency')
        self.frequency_button.setCheckable(True)
        self.frequency_button.clicked.connect(self._freq_button_clicked)

        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addWidget(self.plot_type_menu)
        toolbar_layout.addWidget(self.frequency_button)
        toolbar_layout.addWidget(load_button)
        plot_layout.addLayout(toolbar_layout)

        plot_sunken_frame = QtWidgets.QFrame()
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.plot)
        plot_sunken_frame.setLayout(box)
        plot_sunken_frame.setLineWidth(3)
        plot_sunken_frame.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        plot_sunken_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        plot_sunken_frame.setStyleSheet('background-color: white;')
        plot_layout.addWidget(plot_sunken_frame)
        tip_label = QtWidgets.QLabel(text='Use left and right mouse buttons to select metrics range')
        tip_label.setFixedHeight(15)
        plot_layout.addWidget(tip_label)
        self.plot.mpl_connect('button_press_event', self.select_range)

        # plot info
        info_layout = QtWidgets.QVBoxLayout()
        metrics_layout = QtWidgets.QGridLayout()

        self.player = QMediaPlayer()
        self.player.stateChanged.connect(self.audio_changed)
        self.player.setNotifyInterval(100)
        self.player.positionChanged.connect(self.display_time)
        self.player.durationChanged.connect(self.duration_changed)

        self.player_label = QtWidgets.QLabel(text='0:00 / 0:00')
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.audio_play)

        range_label = QtWidgets.QLabel(text='Selected range (ms):')
        self.range_field = QtWidgets.QLineEdit()
        self.range_field.setReadOnly(True)

        lster_label = QtWidgets.QLabel(text='LSTER:')
        self.lster_field = QtWidgets.QLineEdit()
        self.lster_field.setReadOnly(True)

        hzcrr_label = QtWidgets.QLabel(text='HZCRR:')
        self.hzcrr_field = QtWidgets.QLineEdit()
        self.hzcrr_field.setReadOnly(True)

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

        metrics_layout.addWidget(self.player_label, 0, 0)
        metrics_layout.addWidget(self.play_button, 0, 1)
        metrics_layout.addWidget(range_label, 1, 0)
        metrics_layout.addWidget(self.range_field, 1, 1)
        metrics_layout.addWidget(lster_label, 2, 0)
        metrics_layout.addWidget(self.lster_field, 2, 1)
        metrics_layout.addWidget(hzcrr_label, 3, 0)
        metrics_layout.addWidget(self.hzcrr_field, 3, 1)
        metrics_layout.addWidget(ste_label, 4, 0)
        metrics_layout.addWidget(self.ste_field, 4, 1)
        metrics_layout.addWidget(zcr_label, 5, 0)
        metrics_layout.addWidget(self.zcr_field, 5, 1)
        metrics_layout.addWidget(acf_label, 6, 0)
        metrics_layout.addWidget(self.acf_field, 6, 1)
        metrics_layout.addWidget(amd_label, 7, 0)
        metrics_layout.addWidget(self.amd_field, 7, 1)

        # plotting params
        params_layout = QtWidgets.QGridLayout()
        win_len_label = QtWidgets.QLabel(text='Frame length (ms):')
        self.win_len_field = QtWidgets.QLineEdit()
        win_len_validator = QIntValidator()
        win_len_validator.setBottom(1)
        self.win_len_field.setValidator(win_len_validator)
        self.win_len_field.setText(str(self.frame_len))
        self.win_len_field.editingFinished.connect(self.params_changed)

        win_hop_label = QtWidgets.QLabel(text='Frame step (ms)')
        self.win_hop_field = QtWidgets.QLineEdit()
        self.win_hop_field.setText(str(self.frame_hop))
        self.win_hop_field.editingFinished.connect(self.params_changed)
        win_hop_validator = QIntValidator()
        win_hop_validator.setBottom(1)
        self.win_hop_field.setValidator(win_hop_validator)

        lag_label = QtWidgets.QLabel(text='lag')
        self.lag_field = QtWidgets.QLineEdit()
        self.lag_field.setText(str(self.lag))
        self.lag_field.editingFinished.connect(self.params_changed)
        lag_validator = QIntValidator()
        lag_validator.setBottom(1)
        self.lag_field.setValidator(lag_validator)

        freq0_label = QtWidgets.QLabel(text='freq_0')
        self.freq0_field = QtWidgets.QLineEdit()
        self.freq0_field.setText(str(self.freq0))
        self.freq0_field.editingFinished.connect(self.params_changed)
        freq0_validator = QIntValidator()
        freq0_validator.setBottom(1)
        self.freq0_field.setValidator(freq0_validator)

        freq1_label = QtWidgets.QLabel(text='freq_1')
        self.freq1_field = QtWidgets.QLineEdit()
        self.freq1_field.setText(str(self.freq1))
        self.freq1_field.editingFinished.connect(self.params_changed)
        freq1_validator = QIntValidator()
        freq1_validator.setBottom(1)
        self.freq1_field.setValidator(freq1_validator)

        params_layout.addWidget(win_len_label, 0, 0)
        params_layout.addWidget(self.win_len_field, 0, 1)
        params_layout.addWidget(win_hop_label, 1, 0)
        params_layout.addWidget(self.win_hop_field, 1, 1)
        params_layout.addWidget(lag_label, 2, 0)
        params_layout.addWidget(self.lag_field, 2, 1)
        params_layout.addWidget(freq0_label, 3, 0)
        params_layout.addWidget(self.freq0_field, 3, 1)
        params_layout.addWidget(freq1_label, 4, 0)
        params_layout.addWidget(self.freq1_field, 4, 1)

        # setup layouts
        metrics_frame = QtWidgets.QFrame()
        metrics_frame.setLayout(metrics_layout)
        metrics_frame.setLineWidth(3)
        metrics_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        metrics_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)

        params_frame = QtWidgets.QFrame()
        params_frame.setLayout(params_layout)
        params_frame.setLineWidth(3)
        params_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        params_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)

        metrics_frame.setMaximumWidth(300)
        params_frame.setMaximumWidth(300)
        info_layout.addWidget(metrics_frame)
        info_layout.addWidget(params_frame)

        plot_frame = QtWidgets.QFrame()
        plot_frame.setLayout(plot_layout)
        plot_frame.setLineWidth(3)
        plot_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        plot_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)

        main_layout.addWidget(plot_frame)
        main_layout.addLayout(info_layout)

        # setup app window
        main = QtWidgets.QWidget()
        main.setLayout(main_layout)
        self.setCentralWidget(main)

    def load_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                            "Audio Files (*.wav)")
        if filename:
            print(f'Loading {filename}...')
            self.fps, self.data = read(filename)
            print(f'File loaded successfully')
            self._draw_plot(self.fps, scale_data(self.data))
            self.change_plot(s=self.plot_type_menu.currentText())

            url = QUrl.fromLocalFile(filename)
            content = QMediaContent(url)
            self.player.setMedia(content)

            self._set_values()
            frames, _ = framing(sig=scale_data(self.data), fs=self.fps,
                                win_len=self.frame_len / 1000, win_hop=self.frame_len / 1000)
            self._mark_silence(axis=0, frames=frames, frame_len=self.frame_len / 1000)
            self.plot.axes[0].legend()
            self.plot.draw()

    def _draw_plot(self, fps, data):
        self.plot.fig.clear(keep_observers=True)
        # self.plot.axes = self.plot.fig.subplots(nrows=2, sharex='col')
        self.plot.axes = self.plot.fig.subplots(nrows=2)
        self.plot.axes[0].plot(np.array(range(len(data))) / fps, data)
        self.plot.axes[1].set_xlabel('Time (s)')
        self.line1 = self.plot.axes[0].axvline(x=0, color='green')
        self.line2 = self.plot.axes[0].axvline(x=(len(data) / fps), color='red')
        self.range_field.setText(str(int(abs(self.line1.get_xdata()[0] - self.line2.get_xdata()[0]) * 1000)))

    def change_plot(self, s):
        func, *args = self.plot_type_dict.get(s)
        if (self.fps is None) or (self.data is None):
            return

        frames, _ = framing(sig=scale_data(self.data), fs=self.fps,
                            win_len=self.frame_len / 1000, win_hop=self.frame_hop / 1000)
        frames2, _ = framing(sig=scale_data(self.data), fs=self.fps,
                             win_len=self.frame_len / 1000, win_hop=self.frame_len / 1000)

        if 'use_window' in args:
            frames = self.data
            # data = np.apply_along_axis(func1d=use_window_function, axis=1, arr=frames, win_fun=func)
            if self.use_freq:
                frames = create_spectrum(frames, self.fps)
            data = use_window_function(frames, func)
        elif 'use_kwargs' in args:
            kwargs = {'lag': self.lag, 'fs': self.fps, 'freq_0': self.freq0, 'freq_1': self.freq1}
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames, **kwargs)
        elif 'use_lag' in args:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames, lag=self.lag)
        elif 'use_fs' in args:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames, fs=self.fps)
        elif func == zero_crossing_rate:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames2)
        else:
            data = np.apply_along_axis(func1d=func, axis=1, arr=frames)

        self.plot.axes[1].clear()
        time_seconds = len(self.data) / self.fps
        self.plot.axes[1].plot(np.linspace(0, time_seconds, len(data)), data)
        self.plot.axes[1].set_xlabel('Time (s)')
        if func == unvoice_phones_detection:
            self.plot.axes[1].hlines(0.45, xmin=0, xmax=time_seconds, colors='orange',
                                     linestyles='dashed', label='the boundary between voiced and unvoiced phones')
        if func == zero_crossing_rate:
            self.plot.axes[1].set_ylim([0, 1])
            self._mark_silence(axis=1, frames=frames2, frame_len=self.frame_len / 1000)
            silence = np.apply_along_axis(detect_silence, 1, frames2, vol_max=10e-3)
            self._mark_audio_type(axis=1, frame_len=self.frame_len / 1000, silence=silence, zcr=data)

        self.plot.axes[1].legend()
        self.plot.draw()

    def _set_values(self):
        x1, x2 = self._get_line_xpos()

        data = scale_data(self.data)[int(x1 * self.fps):int(x2 * self.fps)]
        frames, _ = framing(sig=data, fs=self.fps,
                            win_len=self.frame_len / 1000, win_hop=self.frame_hop / 1000)
        frames2, _ = framing(sig=data, fs=self.fps,
                             win_len=self.frame_len / 1000, win_hop=self.frame_len / 1000)
        if len(data) == 0:
            return

        self.lster_field.setText(str(round(low_short_time_energy_ratio(frames), 3)))
        zcr = np.apply_along_axis(zero_crossing_rate, 1, frames2)
        self.hzcrr_field.setText(str(round(high_zero_crossing_rate_ratio(zcr, zcr.shape[0]), 4)))
        self.ste_field.setText(str(round(short_time_energy(data), 3)))
        self.zcr_field.setText(str(round(zero_crossing_rate(data), 3)))
        self.acf_field.setText(str(round(autocorrelation_function(data, lag=self.lag), 3)))
        self.amd_field.setText(str(round(average_magnitude_difference(data, lag=self.lag), 3)))

    def audio_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def audio_changed(self, s):
        if s == QMediaPlayer.StoppedState:
            self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        elif s == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:  # paused
            self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def display_time(self, s):
        self.player_label.setText(f'{hhmmss(s)} / {hhmmss(self.duration)}')

    def duration_changed(self, s):
        self.duration = s
        self.display_time(0)

    def params_changed(self):
        frame_len = int(self.win_len_field.text())
        frame_hop = int(self.win_hop_field.text())
        lag = int(self.lag_field.text())
        freq0 = int(self.freq0_field.text())
        freq1 = int(self.freq1_field.text())

        if frame_len < frame_hop:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText('Incompatible parameter values')
            msg.setInformativeText('Frame step can\'t be larger than frame length')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.lag = lag
        self.freq0 = freq0
        self.freq1 = freq1

        self.change_plot(s=self.plot_type_menu.currentText())

    def select_range(self, event):
        if (self.data is None) or (event.xdata is None) or (self.toolbar.mode != ''):
            return
        if event.button == 1:  # left
            self.line1.set_xdata(min(self.duration / 1000, max(0, event.xdata)))
        elif event.button == 3:  # right
            self.line2.set_xdata(min(self.duration / 1000, max(0, event.xdata)))
        self.plot.draw()

        x1, x2 = self._get_line_xpos()

        self.range_field.setText(str(int(abs(x1 - x2) * 1000)))
        self._set_values()

    def _mark_silence(self, axis, frames, frame_len):
        silence = np.apply_along_axis(detect_silence, 1, frames, vol_max=10e-3)
        j = 0
        for i in range(len(silence)):
            if silence[i]:
                self._color_region(axis, frame_len * i, frame_len * (i + 1), 'red', '_' * j + 'silence')
                j += 1

    def _mark_audio_type(self, axis, frame_len, silence, zcr):
        music_speech_boundary = 0.15
        # frame_len = self.fps * frame_len
        music_speech_array = copy.deepcopy(silence)
        m = 0
        s = 0
        for i in range(len(music_speech_array)):
            if not silence[i]:
                if zcr[i] > music_speech_boundary:
                    self._color_region(axis, frame_len * i, frame_len * (i + 1), 'orange', '_' * s + 'speech')
                    s += 1
                else:
                    self._color_region(axis, frame_len * i, frame_len * (i + 1), 'green', '_' * m + 'music')
                    m += 1

    def _color_region(self, axis, x1, x2, color, label):
        self.plot.axes[axis].axvspan(x1, x2, facecolor=color, label=label, alpha=0.3)

    def _get_line_xpos(self):
        if isinstance(self.line1.get_xdata(), list):
            x1 = self.line1.get_xdata()[0]
        else:
            x1 = self.line1.get_xdata()
        if isinstance(self.line2.get_xdata(), list):
            x2 = self.line2.get_xdata()[0]
        else:
            x2 = self.line2.get_xdata()

        return min(x1, x2), max(x1, x2)

    def _freq_button_clicked(self):
        self.use_freq = not self.use_freq


def hhmmss(ms):
    s = round(ms / 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))
