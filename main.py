import tensorflow as tf
import pickle
import re
import string
from keras.models import load_model
from tensorflow.keras import layers


def on_receive_transcript(data):
  l = []
  # Run data chunks through the model
  for chunk in data:
    # run chunk through data model
    l.append(process(chunk))

def process(sentence):
  # Load model
  model = load_model('model.h5')
  # Load custom standardization function
  from_disk = pickle.load(open('vectorize_config.pkl', 'rb'))
  new_v = layers.TextVectorization.from_config(from_disk['config'])
  # new_v.adapt(custom_standardization(sentence))
  new_v.set_weights(from_disk['weights'])
  # Standardize sentence
  sentence = custom_standardization(sentence)
  # Predict sentence
  prediction = model.predict(new_v(tf.expand_dims(sentence, -1)))
  # Print prediction
  print(prediction)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    # stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

if __name__ == '__main__':
  process('This video is sponsored by free')