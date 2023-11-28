import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import compute_class_weight

File = pd.read_excel("11-18-2023File.xlsx")
File = File.sample(frac=1.0, random_state=42)
Category = []
ones = 0
zeros = 0
for i in File["Category"]:
    if i == "Null" or i == "Not" or i == "not" or i == 0:
        Category.append(0)
        zeros += 1
    else:
        Category.append(1)
        ones += 1

Sentence = File["Data"]
Category = np.array(Category)
Tokenizer = tf.keras.preprocessing.text.Tokenizer()
Tokenizer.fit_on_texts(Sentence)
Sentence = Tokenizer.texts_to_sequences(Sentence)
Sentence = tf.keras.utils.pad_sequences(Sentence)
totalVocab = len(Tokenizer.word_index)
inputs = tf.keras.Input((49,))
layer = tf.keras.layers.Embedding(totalVocab + 1, 30)(inputs)
layer = tf.keras.layers.LSTM(25)(layer)
layer = tf.keras.layers.Dense(25, 'relu',
                              kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.02))(layer)
layer = tf.keras.layers.Dropout(0.02)(layer)
output = tf.keras.layers.Dense(1, 'sigmoid')(layer)

# Calculate class weights
class_weights = compute_class_weight("balanced", classes=np.unique(Category), y=Category)

# Create a dictionary with class labels as keys and their respective weights as values
class_weights_dict = dict(zip(np.unique(Category), class_weights))

model = tf.keras.Model(inputs, output)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy()])


# model.load_weights("Save/006")


def fit():
    model.fit(Sentence, Category, batch_size=1, epochs=1000, class_weight=class_weights_dict, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_binary_accuracy", 0, 25, mode='max',
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau("val_binary_accuracy", mode='max', patience=3),
                         tf.keras.callbacks.TerminateOnNaN(),
                         tf.keras.callbacks.TensorBoard('./logs/new')])
    model.save_weights("./Save/better")

# 007


sa = True
if sa:
    fit()
else:
    model.load_weights("Save/better")


def r(x: float, t=0.5):
    if x >= t:
        x = 1
    else:
        x = 0
    return x

Data = []
Category = []
p = pd.read_excel('pr.xlsx')
tex = p["Data"]
tex = Tokenizer.texts_to_sequences(tex)
tex = tf.keras.utils.pad_sequences(tex, 49)
pr = model.predict(tex)
for i in range(tex.shape[0]):
    print(f"{p['Data'][i]} : \t{(pr[i][0], 0.65)}")
    Data.append(p['Data'][i])
    Category.append(r(pr[i][0]))
La = {
    "Data": Data,
    "Category": Category
}
La = pd.DataFrame(La)
La.to_excel("La.xlsx")
print(La)
