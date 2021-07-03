import numpy as np
import tensorflow as tf
import json
import functools
import glob

import random
tf.compat.v1.enable_eager_execution()




# m = open(data_folder+"/metadata.json",)
# meta = json.load(m)
# raw_image_dataset = tf.data.TFRecordDataset(data_folder+"/test.tfrecord")

data_folder = "./d_0.35/"
tfrecords_file = './d_0.35.tfrecord'
writer = tf.io.TFRecordWriter(tfrecords_file)


''' ##################### DO NOT TOUCH ########################'''
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

''' ##################### DO NOT TOUCH ########################'''


for count,sample in enumerate(glob.glob(data_folder+'*')):
  key = count
  arr = []
  #files = glob.glob(sample+'/*')
  #print(files)
  #files.sort()
  #print(files)
  for ind in range(200):
    temp_path = sample+"/"+str(ind)+".txt"
    per_frame =[]
    #print(temp_path)
    f = open(temp_path,'r')
    lines = f.readlines()
    nparticles = len(lines)
    for line in lines:
      line  = line.strip().split()
      val1 = np.float32( np.float32(line[0])/64)
      val2 = np.float32( np.float32(line[1])/64)
      per_frame.append(tuple((val1,val2)))
    per_frame = np.asarray(per_frame)
    per_frame = per_frame.flatten()
    arr.append(per_frame)
  arr = np.asarray(arr)
  #print(type(arr[1,3]),arr[1,3] )

  particle_type = np.full(nparticles,5)
  #print(particle_type, arr.shape)
  example = serialize_example(key, particle_type.tobytes(),arr)
  writer.write(example)

  # print(arr.shape)



