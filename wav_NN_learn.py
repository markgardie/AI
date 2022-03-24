import numpy as np
import tensorflow as tf
import random as rn
import os



os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
rn.seed(12345)
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizer_v1 import Adam
# from keras.optimizers import Adam
import time
import pandas as pd
from keras.callbacks import TensorBoard
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from scipy import signal
from scipy.io import wavfile
import random

tf.compat.v1.disable_eager_execution()


def Load_Wav_(WorkDir):
    Input_Files = []
    Source_Samples = []
    for d, dirs, files in os.walk(WorkDir):
        for file in files:
            Input_Files.append(file)
    GFile = []
    for file in Input_Files:
        if file.endswith(".wav"):
            sample_rate, samples = wavfile.read(str(WorkDir) + file)
            if sample_rate != 8000:
                continue
            if max(abs(samples)) < 410:
                continue
            if len(samples) < int(0.1 * sample_rate):
                continue
            GFile.append(file)
            #                 samples = Convert_To_06(samples)
            Source_Samples.append(samples)

    return Source_Samples, GFile


def Add_Zero(specgram, TargetColumnNumber, StartSignalPosition):
    if len(specgram[0]) >= TargetColumnNumber:
        return specgram

    full_array = np.zeros((len(specgram), TargetColumnNumber))
    full_array[:, :len(specgram[0]) - TargetColumnNumber] = specgram
    if StartSignalPosition == 0:
        full_array[:, :len(specgram[0]) - TargetColumnNumber] = specgram
    elif StartSignalPosition == TargetColumnNumber - len(specgram[0]):
        full_array[:, StartSignalPosition:] = specgram
    else:
        full_array[:, StartSignalPosition:StartSignalPosition + len(specgram[0]) - TargetColumnNumber] = specgram

    return full_array


def log_specgram(audio, window_size, sample_rate=8000,
                 eps=1e-10, windoe_fuction='hann'):
    nperseg = int(round(window_size * sample_rate / 1000))
    noverlap = int(round(window_size / 2 * sample_rate / 1000))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window=windoe_fuction,
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.astype(np.float32) + eps)


def Convert_Wav_To_specgram(SamplesList, Input_Files, window_size, windoe_fuction):
    samples = np.zeros(int(0.6 * 8000))
    _, _, specgram = log_specgram(audio=samples, window_size=window_size, windoe_fuction=windoe_fuction)
    TargetColumnNumber = len(specgram[0])
    TargetRow = len(specgram)
    x = []
    y = []
    for i in range(len(SamplesList)):
        _, _, specgram = log_specgram(audio=SamplesList[i], window_size=window_size, windoe_fuction=windoe_fuction)
        specgram = Add_Zero(specgram, TargetColumnNumber, 0)
        x.append(specgram)
        file = Input_Files[i]
        if 'cl_1' in file:
            y.append([0, 0, 1])
        elif 'cl_2' in file:
            y.append([1, 0, 0])
        else:
            y.append([0, 1, 0])

    x = np.array(x)
    x = x.reshape(tuple(list(x.shape) + [1]))
    y = np.array(y)
    return x, y


def RandomozeArrays(SourceArrayX, SourceArrayY):
    TargetArrayX = []
    TargetArrayY = []
    while 0 < len(SourceArrayX):
        Index = random.randint(0, len(SourceArrayX) - 1)
        TargetArrayX.append(SourceArrayX[Index])
        del SourceArrayX[Index]
        TargetArrayY.append(SourceArrayY[Index])
        del SourceArrayY[Index]

    return TargetArrayX, TargetArrayY


def Learn_NN_5L_(TrainDir, ValidDir, RezDir, NN_Name, Epochs=30, window_size=25, windoe_fuction='hann'):
    Source_Samples, Input_Files = Load_Wav_(TrainDir)
    print('end load train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Source_Samples, Input_Files = RandomozeArrays(Source_Samples, Input_Files)
    print('end Randomize train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    X_Train, Y_Train = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                               window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert train data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    Source_Samples, Input_Files = Load_Wav_(ValidDir)
    print('end load valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    X_val1, y_val1 = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                             window_size=window_size, windoe_fuction=windoe_fuction)
    print('end convert valid data', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    input_shape = (X_Train.shape[1], X_Train.shape[2], 1)
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(48, (5, 5), strides=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(40, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(15))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    csv_logger = CSVLogger(RezDir + NN_Name + '_training__log.csv', separator=',', append=False)

    checkpoint = ModelCheckpoint(filepath=RezDir + NN_Name + '_Best.hdf5',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)
    model.fit(X_Train, Y_Train,
              batch_size=64,
              epochs=Epochs, shuffle=True,
              validation_data=(X_val1, y_val1),
              callbacks=[checkpoint, csv_logger])
    model.save(filepath=RezDir + NN_Name + '_Final.hdf5')


def TestNN_(NetName, SourceDir, TargetFile, window_size):
    Input_Files = []
    Source_Samples = []
    for d, dirs, files in os.walk(SourceDir):
        for file in files:
            if file.endswith(".wav"):
                sample_rate, samples = wavfile.read(SourceDir + "/" + file)
                if sample_rate != 8000:
                    continue
                if max(abs(samples)) < 410:
                    continue
                if len(samples) < int(0.1 * sample_rate):
                    continue
                Input_Files.append(file)
                #                 samples = Convert_To_06(samples)
                Source_Samples.append(samples)

    x, y = Convert_Wav_To_specgram(SamplesList=Source_Samples, Input_Files=Input_Files,
                                   window_size=window_size, windoe_fuction='hann')

    new_model = load_model(NetName)
    pred = new_model.predict(x)
    f = open(TargetFile + '_FilesReport.csv', 'w', newline='\n')
    f.write('NetName = %s, Files %s \n' % (NetName, SourceDir))
    f.write('File Name,Marked As,Recognized As,Cl 1,Cl 2,Cl 3, \n')
    CodeList = ['Cl 2', 'Cl 3', 'Cl 1']
    SemplCount = [0, 0, 0]
    StatRez = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(pred)):
        YY = list(y[i])
        Rez = list(pred[i])
        TrueCalss = YY.index(max(YY))
        NNClass = Rez.index(max(Rez))
        SemplCount[TrueCalss] += 1
        StatRez[TrueCalss][NNClass] += 1
        f.write('%s,%s,%s, %f , %f, %f,\n' % (
        Input_Files[i], CodeList[TrueCalss], CodeList[NNClass], pred[i][2], pred[i][0], pred[i][1]))

    f.close()
    f = open(TargetFile + '_Report.csv', 'w', newline='\n')
    f.write('NetName = %s, Files %s \n' % (NetName, SourceDir))
    f.write('Var,Cl 1,Cl 2,Cl 3, \n')

    f.write('Count,%s,%s, %s ,\n' % (SemplCount[2], SemplCount[0], SemplCount[1]))
    f.write('Cl 1 As,%s,%s, %s ,\n' % (StatRez[2][2], StatRez[2][0], StatRez[2][1]))
    f.write('Cl 2 As,%s,%s, %s ,\n' % (StatRez[0][2], StatRez[0][0], StatRez[0][1]))
    f.write('Cl 3 As,%s,%s, %s ,\n' % (StatRez[1][2], StatRez[1][0], StatRez[1][1]))
    trueclass = 100.0 * (StatRez[2][2] + StatRez[0][0] + StatRez[1][1]) / float(sum(SemplCount))

    for i in range(len(SemplCount)):
        for k in range(len(SemplCount)):
            if SemplCount[i] > 0:
                StatRez[i][k] = 100.0 * float(StatRez[i][k]) / SemplCount[i]
    f.write('Cl 1 As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[2][2], StatRez[2][0], StatRez[2][1]))
    f.write('Cl 2  As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[0][2], StatRez[0][0], StatRez[0][1]))
    f.write('Cl 3 As %%,%.3f%%,%.3f%%, %.3f%% ,\n' % (StatRez[1][2], StatRez[1][0], StatRez[1][1]))

    f.write(',\n')

    f.write('Total acc. ,%.3f%% ,\n' % (trueclass))

    f.close()

print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
Learn_NN_5L_(TrainDir=r'E:\Programming\KPI_Projects\AI\train\\',
             ValidDir=r'E:\Programming\KPI_Projects\AI\valid\\',
             RezDir=r'E:\Programming\KPI_Projects\AI\res\\',
             NN_Name='NN_L5', Epochs=10, window_size=25, windoe_fuction='hann')

print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

TestNN_(NetName=r'E:\Programming\KPI_Projects\AI\res\NN_L5_Best.hdf5',
            SourceDir=r'E:\Programming\KPI_Projects\AI\test',
            TargetFile=r'E:\Programming\KPI_Projects\AI\res\NN_L5_rez',
            window_size=25)