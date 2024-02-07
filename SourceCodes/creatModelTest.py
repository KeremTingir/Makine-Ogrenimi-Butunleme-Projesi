import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle 

# CSV dosyasını oku
df = pd.read_csv('Datasets/normalizedDataset.csv')

# x2_normalized sütununun tek sayı olup olmadığına göre verileri ayır
train_data = df[df['x2_normalized']*100 % 2 == 1]
test_data = df[df['x2_normalized']*100 % 2 == 0]

# Giriş ve çıkış verilerini ayır
X_train = train_data[['x1_normalized', 'x2_normalized']].values
y_train = train_data['y_normalized'].values

X_test = test_data[['x1_normalized', 'x2_normalized']].values
y_test = test_data['y_normalized'].values

# Yapay Sinir Ağı modelini oluştur
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Model parametrelerini tanımla
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ağırlıkları ve bias'ları başlat
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        # Sigmoid aktivasyon fonksiyonu
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Sigmoid aktivasyon fonksiyonunun türevi
        return x * (1 - x)

    def forward(self, X):
        # İleri hesaplama
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

    def backward(self, X, y):
        # Geriye yayılım
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Ağırlık ve bias güncellemeleri
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        # Modeli eğit
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        # Tahmin
        self.forward(X)
        return self.predicted_output


def find_best_hyperparameters(X_train, y_train, X_test, y_test, learning_rates, epochs_values, hidden_sizes):
    best_loss = float('inf')
    best_hyperparameters = {}

    for learning_rate in learning_rates:
        for epochs in epochs_values:
            for hidden_size in hidden_sizes:
                model = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
                model.train(X_train, y_train.reshape(-1, 1), epochs)

                predictions = model.predict(X_test)
                test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_hyperparameters = {'learning_rate': learning_rate, 'epochs': epochs, 'hidden_size': hidden_size}

    return best_hyperparameters

# Hiperparametre arama için değerler belirle
learning_rates = [0.01, 0.1]
epochs_values = [10000, 20000, 30000, 50000, 75000]
hidden_sizes = [2, 3, 4, 5]

# En iyi hiperparametreleri bul
best_hyperparameters = find_best_hyperparameters(X_train, y_train, X_test, y_test, learning_rates, epochs_values, hidden_sizes)

# En iyi hiperparametrelerle modeli tekrar eğit
best_model = NeuralNetwork(input_size=2, hidden_size=best_hyperparameters['hidden_size'],
                           output_size=1, learning_rate=best_hyperparameters['learning_rate'])
best_model.train(X_train, y_train.reshape(-1, 1), best_hyperparameters['epochs'])

# Test setini kullanarak modeli değerlendir
predictions = best_model.predict(X_test)
test_loss = np.mean((predictions - y_test.reshape(-1, 1))**2)
print(f'En İyi Modelin Test Loss\'u: {test_loss}')
print(f'En İyi Hiperparametreler: {best_hyperparameters}')

# Tahmin için kullanılan fonksiyon ve örneği aynı
x1_input = 1
x2_input = 2

# Girdileri normalize et
x1_normalized_input = x1_input / 100.0
x2_normalized_input = x2_input / 100.0

input_data = np.array([[x1_normalized_input, x2_normalized_input]])
best_model.forward(input_data)  # Modeli ileri doğru hesapla
predicted_result_normalized = best_model.predicted_output

# Tahmini sonucu orijinal ölçeğe dönüştür
predicted_result = predicted_result_normalized * 100.0
print(f'Tahmin Edilen Sonuç: {predicted_result[0, 0]}')

# Modeli kaydet
with open('#####.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
