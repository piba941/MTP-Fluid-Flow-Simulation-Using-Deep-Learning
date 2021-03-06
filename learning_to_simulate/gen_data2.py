# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d5nDhsxwjUeV6_zqiNG2t4giLaaq3Ata
"""

#!mkdir data
#!wget -O "./data/metadata.json" "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterRamps/metadata.json"
#!wget -O "./data/test.tfrecord" "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterRamps/test.tfrecord"

data_folder = "."


import numpy as np
import tensorflow as tf
import json
import functools

import random
tf.compat.v1.enable_eager_execution()

m = open(data_folder+"/metadata.json",)
meta = json.load(m)
raw_image_dataset = tf.data.TFRecordDataset(data_folder+"/test.tfrecord")


tfrecords_file = './ptrain.tfrecord'

writer = tf.io.TFRecordWriter(tfrecords_file)

_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.

  feature_2 = []

  for v in range(feature2.shape[0]):
    val = feature2[v].tobytes()
    # val = tf.convert_to_tensor(val)
    temp = tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
    feature_2.append(temp)

  # feature_2 = tf.convert_to_tensor(feature_2)
  _CONTEXT_FEATURES = {
    'key': _int64_feature(feature0),
    'particle_type': _bytes_feature(feature1),
  }
  
  _FEATURE_DESCRIPTION = {
    'position':tf.train.FeatureList(feature=feature_2)
  }
  
  example = tf.train.SequenceExample(context=tf.train.Features(feature=_CONTEXT_FEATURES),feature_lists=tf.train.FeatureLists(feature_list=_FEATURE_DESCRIPTION))
  return example.SerializeToString()







def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out


def parse_serialized_simulation_example(example_proto, metadata):
  # parsed_feature = tf.train.SequenceExample.FromString(serialized)
  # print(parsed_feature)
  """Parses a serialized simulation tf.SequenceExample.
  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.
  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.
  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  # print(parsed_features['position'])
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  #print(parsed_features["position"])
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  # print(context["particle_type"].numpy())
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  l = context['particle_type'].numpy()
  # print(l)
  return context, parsed_features


#taking per sampple(each containg n frames)

for c,i in enumerate(raw_image_dataset):
  final_particle_type = []
  final_position = []
  parsed_feature = parse_serialized_simulation_example(i,meta)
  # print(parsed_feature[0]["particle_type"].numpy())


  # no need for a key holder
  particle_type = parsed_feature[0]["particle_type"].numpy()
  key = parsed_feature[0]["key"].numpy()
  position = parsed_feature[1]["position"].numpy()

  # print(key.shape, particle_type.shape, position.shape)

  length = len(particle_type)
  data = np.arange(length)
  # print(data)

  index_list = np.random.choice(data,8*int(length/10), replace= False)
  index_list = np.sort(index_list)
  net_particle_type = []
  for p in index_list:
    net_particle_type.append(particle_type[p])
  
  net_particle_type = np.asarray(net_particle_type)
  # print(net_particle_type.shape)
  # print(index_list)

  net_pos = []
  for j in range(position.shape[0]):
    per_frame = []
    for ind in index_list:
      per_frame.append(position[j,ind,:])
    # per_frame = np.expand_dims(per_frame, axis=0)
    per_frame = np.asarray(per_frame)
    per_frame = per_frame.flatten()
    net_pos.append(per_frame)
  
  net_pos = np.asarray(net_pos)
  # print(net_particle_type.shape,net_pos.shape)
  # print()
  final_particle_type.append(net_particle_type)
  final_position.append(net_pos)
  # print(c)
  example = serialize_example(c,final_particle_type[0].tobytes(),final_position[0])
  writer.write(example)

