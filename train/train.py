from __future__ import print_function

import numpy as np
np.random.seed(42)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import json
import matplotlib.pyplot as plt
import models
import pandas as pd
import sys
import tensorflow.keras as keras
import tensorflow as tf
import yaml

from callbacks import all_callbacks
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from optparse import OptionParser
from qkeras.autoqkeras import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input


def print_model_to_json(keras_model, outfile_name):
    jsonString = keras_model.to_json()
    with open(outfile_name, 'w') as f:
        obj = json.loads(jsonString)
        json.dump(obj, f, sort_keys=True, indent=4, separators=(',', ': '))
        f.write('\n')


def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    # List of features to use
    features = yamlConfig['Inputs']

    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_labels_df = pd.DataFrame(treeArray,
                                      columns=list(set(features+labels)))
    features_labels_df = features_labels_df.drop_duplicates()

    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]

    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()

    # Convert to numpy array
    features_val = features_df.values
    labels_val = labels_df.values

    if 'j_index' in features:
        features_val = features_val[:, :-1]  # drop the j_index feature
    if 'j_index' in labels:
        labels_val = labels_val[:, :-1]  # drop the j_index label

    if yamlConfig['InputType'] == 'Conv1D':
        features_2dval = np.zeros((len(labels_df),
                                   yamlConfig['MaxParticles'],
                                   len(features)-1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[(features_df['j_index'] ==
                                         labels_df['j_index'].iloc[i])]
            index_values = features_df_i.index.values
            features_val_ = features_val[np.array(index_values), :]
            nParticles = len(features_val_)
            # Sort descending by first value (ptrel, usually)
            features_val_ = features_val_[features_val_[:, 0].argsort()[::-1]]
            if nParticles > yamlConfig['MaxParticles']:
                features_val_ = features_val_[0:yamlConfig['MaxParticles'], :]
            else:
                zeros = np.zeros((yamlConfig['MaxParticles']-nParticles,
                                  len(features)-1))
                features_val_ = np.concatenate([features_val_, zeros])
            features_2dval[i, :, :] = features_val_

        features_val = features_2dval

    elif yamlConfig['InputType'] == 'Conv2D':
        features_2dval = np.zeros((len(labels_df),
                                   yamlConfig['BinsX'],
                                   yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[(features_df['j_index'] ==
                                         labels_df['j_index'].iloc[i])]

            xbins = np.linspace(yamlConfig['MinX'],
                                yamlConfig['MaxX'],
                                yamlConfig['BinsX']+1)
            ybins = np.linspace(yamlConfig['MinY'],
                                yamlConfig['MaxY'],
                                yamlConfig['BinsY']+1)

            x = features_df_i[features[0]]
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]

            hist, xedges, yedges = np.histogram2d(x, y, weights=w,
                                                  bins=(xbins, ybins))

            for ix in range(0, yamlConfig['BinsX']):
                for iy in range(0, yamlConfig['BinsY']):
                    features_2dval[i, ix, iy, 0] = hist[ix, iy]
        features_val = features_2dval

    X_train, X_test, y_train, y_test = train_test_split(features_val,
                                                        labels_val,
                                                        test_size=0.2,
                                                        random_state=42)

    # Normalize inputs
    if yamlConfig['NormalizeInputs']:
        if yamlConfig['InputType'] not in ['Conv1D', 'Conv2D']:
            if yamlConfig['KerasLoss'] != 'squared_hinge':
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            elif yamlConfig['KerasLoss'] == 'squared_hinge':
                scaler = MinMaxScaler((-1, 1)).fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
        elif yamlConfig['InputType'] == 'Conv1D':
            shape = X_train.shape
            reshaped = X_train.reshape(shape[0]*shape[1], shape[2])
            scaler = StandardScaler().fit(reshaped)
            for p in range(X_train.shape[1]):
                X_train[:, p, :] = scaler.transform(X_train[:, p, :])
                X_test[:, p, :] = scaler.transform(X_test[:, p, :])

    if 'j_index' in labels:
        labels = labels[:-1]

    return X_train, X_test, y_train, y_test, labels


def parse_config(config_file):
    config = open(config_file, 'r')
    return yaml.load(config)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-i', '--input',
                      action='store', type='string', dest='inputFile',
                      help='Path to input file')
    parser.add_option('-t', '--tree',
                      action='store', type='string', dest='tree',
                      help='Tree name')
    parser.add_option('-o', '--output',
                      action='store', type='string', dest='outputDir',
                      help='Output directory')
    parser.add_option('-c', '--config',
                      action='store', type='string', dest='config',
                      help='Path to configuration file')
    parser.add_option('-q', '--autoq-config',
                      action='store', type='string', dest='autoq_config',
                      help='Path tp AutoQKeras configuration')
    parser.add_option('-r', '--learaning-rate',
                      action='store', type='float', dest='start_lr',
                      default=.002, help='Start learning rate')
    parser.add_option('-b', '--batch-size',
                      action='store', type='int', dest='batch_size',
                      default=1024, help='Batch size')
    parser.add_option('-e', '--epochs',
                      action='store', type='int', dest='epochs',
                      default=60, help='Number of epochs')
    parser.add_option('-v', '--validation-split',
                      action='store', type='float', dest='validation_split',
                      default=.25, help='Validation split')
    (options, args) = parser.parse_args()

    yamlConfig = parse_config(options.config)

    tf.get_logger().setLevel('ERROR')

    # Parameters to scan
    STRESS = np.arange(1, 6)
    RATE = np.arange(2, 6)

    if not os.path.isdir(options.outputDir):
        os.mkdir(options.outputDir)

    X_train, X_test, y_train, y_test, _ = get_features(options, yamlConfig)

    model = getattr(models, yamlConfig['KerasModel'])
    if 'L1RegR' in yamlConfig:
        keras_model = model(Input(shape=X_train.shape[1:]),
                            y_train.shape[1],
                            l1Reg=yamlConfig['L1Reg'],
                            l1RegR=yamlConfig['L1RegR'])
    else:
        keras_model = model(Input(shape=X_train.shape[1:]),
                            y_train.shape[1],
                            l1Reg=yamlConfig['L1Reg'])

    print_model_to_json(keras_model, options.outputDir + '/model.json')

    optimizer = Adam(lr=options.start_lr)
    metrics = [tf.keras.metrics.AUC(), 'accuracy']
    loss = [yamlConfig['KerasLoss']]

    keras_model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics)
    with open(options.autoq_config, 'r') as f:
        run_config = json.load(f)

    foo = []
    for r in RATE:
        foo2 = []
        for stress in STRESS:
            run_config["output_dir"] = "%s/autoq%s-%s" % (options.outputDir,
                                                          r, s)
            run_config["goal"]["params"]["rate"] = r
            run_config["goal"]["params"]["stress"] = s
            autoqk = AutoQKerasScheduler(keras_model, metrics, {}, debug=0,
                                         **run_config)
            autoqk.fit(X_train[0:40000],
                       y_train[0:40000],
                       batch_size=options.batch_size,
                       epochs=options.epochs,
                       validation_split=options.validation_split,
                       shuffle=True)

            qmodel = autoqk.get_best_model()
            callbacks = all_callbacks(stop_patience=1000,
                                      lr_factor=0.5,
                                      lr_patience=10,
                                      lr_epsilon=0.000001,
                                      lr_cooldown=2,
                                      lr_minimum=0.0000001,
                                      outputDir=options.outputDir)
            qmodel.fit(X_train,
                       y_train,
                       batch_size=options.batch_size,
                       epochs=50,
                       validation_split=options.validation_split,
                       shuffle=True,
                       callbacks=callbacks.callbacks)

            save_quantization_dict('%s/optimized%s-%s.dict' %
                                   (options.outputDir, r, s), qmodel)
            qmodel.save('%s/optimized%s-%s.h5' % (options.outputDir, r, s))

            y_pred = qmodel(X_test)
            for i in range(5):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
                aucs.append(auc(fpr, tpr))

            foo2.append((autoqk.get_best_model_stats(), np.mean(aucs)))
        foo.append(foo2)

    # Plot the results
    plt.style.use('../plotting/hls.mplstyle')

    def get_logo():
        return OffsetImage(plt.imread('../plotting/hls4mllogo.jpg',
                                      format='jpg'),
                           zoom=0.08)

    with open('../plotting/palette.json') as json_file:
        palette = json.load(json_file)

    colors = ['indigo', 'orange', 'red', 'green']
    markers = ["o", "v", "*", "D", "P"]

    fig, ax = plt.subplots()
    for r, _rate in enumerate(foo):
        for s, _stress in enumerate(_rate):
            ax.scatter(_stress[0], _stress[1],
                       c=palette[colors[r]]['shade_500'],
                       marker=markers[s],
                       s=12)
    legend_elements = []
    for r in RATE:
        legend_elements.append(Patch(fc=palette[colors[r-2]]['shade_300'],
                                     label='Rate=%s' % r))
    for s in STRESS:
        legend_elements.append(Line2D([0], [0], marker=markers[s-1], color='w',
                               mfc=palette['grey']['shade_700'],
                               label='Stress=%s' % s, markersize=8))

    ax.legend(handles=legend_elements, loc='right', bbox_to_anchor=(1.3, 0.5))
    ax.text(0, 1.02, 'CMS', weight='bold', transform=ax.transAxes,
            color=palette['grey']['shade_900'], fontsize=13)
    ab = AnnotationBbox(get_logo(),
                        xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                        box_alignment=(-0.5, 0.15), frameon=False)
    ax.add_artist(ab)
