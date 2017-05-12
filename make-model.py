#!/usr/bin/env python3

import keras
from keras import layers
from keras.models import Model
import numpy as np

inputs = layers.Input(shape=(2,))
inputs2 = layers.Input(shape=(2,))
input3 = layers.Input(shape=(None, 3))
lstm = layers.LSTM(10)(input3)
# d = layers.Dense(5, activation='tanh')(inputs)
# d2 = layers.Dense(6, activation='relu')(inputs2)
dall = layers.Dense(6, activation='relu')
d = dall(inputs)
d2 = dall(inputs2)
d3 = layers.Dense(3, activation='tanh')(d2)
cat = layers.merge([d, d3, d2, lstm], mode='concat')
outputs = layers.Dense(2, activation='softmax', name='output_name')(cat)
model = Model(input=[inputs,inputs2, input3], output=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

with open('arch.json','w') as archetecture:
    archetecture.write(model.to_json(indent=2, sort_keys=True) )

a = np.arange(2)[None,:]
b = np.arange(21).reshape(-1, 3)[None,...]
print(model.predict([a, a, b]))

model.save_weights('weights.h5')

# seq = keras.models.Sequential([layers.Dense(6, input_shape=(2,))])
# with open('seq-arch.json','w') as seqarch:
#     seqarch.write(seq.to_json(indent=2, sort_keys=True))
# seq.save_weights('seq-weights.h5')
