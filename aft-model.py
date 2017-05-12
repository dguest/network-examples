#!/usr/bin/env python3

import keras
from keras import layers
from keras.models import Model
import numpy as np

tracks = layers.Input(shape=(None, 4), name='tracks')
rnnip_raw = layers.GRU(5, name='rnnip_lstm')(tracks)
vx_inputs = layers.Input(shape=(1,), name='vertex_info')

ip3d_inputs = layers.Input(shape=(3,), name='ip3d')
dl1_inputs = layers.merge([vx_inputs,ip3d_inputs], mode='concat') #4

dl_inputs = layers.merge([rnnip_raw, vx_inputs], mode='concat') #6
dl1_first_layer = layers.Dense(6, name='DL1_layer1')(dl1_inputs)
dl1_out_layer = layers.Dense(4, name='DL1_shared', activation='softmax')
dl1 = dl1_out_layer(dl1_first_layer)
dl2 = dl1_out_layer(dl_inputs)
rnnip = layers.Dense(4, name='rnnip_out')(rnnip_raw)
model = Model(input=[tracks, vx_inputs, ip3d_inputs], output=[dl1, dl2, rnnip])
model.compile(optimizer='adam', loss='categorical_crossentropy')

with open('ftag-arch.json','w') as archetecture:
    archetecture.write(model.to_json(indent=2, sort_keys=True) )

model.save_weights('ftag-weights.h5')

from keras.utils.visualize_util import model_to_dot
model_to_dot(model).write_pdf('ftag-model.pdf')

trk = np.linspace(-1, 1, 20)[:,None] * np.linspace(-1, 1, 4)[None,:]
vx = np.array([0])[None,:]
ip3d = np.linspace(-1, 1, 3)[None,:]
for output in model.predict([trk[None,:], vx, ip3d]):
    print(output)

