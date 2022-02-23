import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras import optimizers

print(tf.__version__)

# Dataset
# For this example let's use the Auto MPG Dataset. Auto MPG dataset consists of different features about automobiles in
# 1970 and 80s. These features include attributes like the number of cylinders, horsepower, weight etc. and we need to
# use this information to predict the fuel-efficiency of automobiles. Thus, it is a regression problem.
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete
# and 5 continuous attributes."

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "Model Year", "Origin"]
df = pd.read_csv(url, names=column_names, sep=" ",
                 comment="\t", na_values="?", skipinitialspace=True)

df.head()

# Create a copy for further processing
dataset = df.copy()
# Get some basic data about dataset
print(len(dataset))

# Preprocessing
dataset.isna().sum()

# There are some na values. We can fill or remove those rows.
dataset['Horsepower'].fillna(dataset['Horsepower'].mean(), inplace=True)
dataset.isna().sum()

# Since the column 'Origin' is not numeric but categorical we can encode it using pd.get_dummies
dataset['Origin'].value_counts()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.head()

# Split the Dataset and create train and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
########################################################
# Separate labels and features
train_features = train_dataset.drop(["MPG"], axis=1)
test_features = test_dataset.drop(["MPG"], axis=1)
train_labels = train_dataset["MPG"]
test_labels = test_dataset["MPG"]


# Let's check some basic data about dataset
train_dataset.describe().transpose()

# One of the best practises is to perform normalization of the input data i.e. scale each feature along with its mean
# and standard deviation.
train_dataset.describe().transpose()[['mean', 'std']]

# But we can also apply normalization using sklearn.
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
########################################################
# Fit on Training Data
feature_scaler.fit(train_features.values)
label_scaler.fit(train_labels.values.reshape(-1, 1))
########################################################
# Transform both training and testing data
train_features = feature_scaler.transform(train_features.values)
test_features = feature_scaler.transform(test_features.values)
train_labels = label_scaler.transform(train_labels.values.reshape(-1, 1))
test_labels = label_scaler.transform(test_labels.values.reshape(-1, 1))


print(train_features.shape)
print(test_features.shape)

print(train_labels.shape)
print(test_labels.shape)

# Now let's create a Deep Neural Network to train a regression model on our data.
model = Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer="RMSProp",
              loss="mean_squared_error")


# Now let's train the model
history = model.fit(epochs=100, x=train_features, y=train_labels,
          validation_data=(test_features, test_labels), verbose=0)

# Let's check the model summary
model.summary()

# Function to plot loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error (Loss)')
    plt.legend()
    plt.grid(True)
########################################################
plot_loss(history)

# Model evaluation on testing dataset
model.evaluate(test_features, test_labels)

# Save model
model.save("trained_model.h5")

# Load and perform predictions
saved_model = models.load_model('trained_model.h5')
results = saved_model.predict(test_features)
########################################################
# We can decode using the scikit-learn object to get the result
decoded_result = label_scaler.inverse_transform(results.reshape(-1,1))
print(decoded_result)

