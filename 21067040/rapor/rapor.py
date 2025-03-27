#1)Veri Setini Yükleme ve Kayıp Değer Ekleme:
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Boston Housing veri setini yükle
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# %10 oranında rastgele kayıp değer ekle
np.random.seed(42)
missing_rate = 0.1
n_samples = df.shape[0]
n_missing = int(n_samples * missing_rate)
missing_indices = np.random.choice(df.index, n_missing, replace=False)
missing_cols = np.random.choice(df.columns, int(df.shape[1] * missing_rate * 2), replace=False) # Birden fazla sütunda kayıp olabilir
for index in missing_indices:
    col = np.random.choice(missing_cols)
    df.loc[index, col] = np.nan

# Hedef değişkeni ve özellikleri ayır
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2)Derin Öğrenme Modelini Tanımlama:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#3)Farklı Kayıp Değer İşleme Yöntemlerini Uygulama ve Modeli Eğitme:
from sklearn.impute import SimpleImputer

results = {}
scalers = {}

# 1. Ortalama ile Doldurma (Mean Imputation)
X_train_mean = X_train.copy()
X_test_mean = X_test.copy()
imputer_mean = SimpleImputer(strategy='mean')
X_train_mean = imputer_mean.fit_transform(X_train_mean)
X_test_mean = imputer_mean.transform(X_test_mean)
scaler_mean = StandardScaler()
X_train_mean_scaled = scaler_mean.fit_transform(X_train_mean)
X_test_mean_scaled = scaler_mean.transform(X_test_mean)
model_mean = create_model()
model_mean.fit(X_train_mean_scaled, y_train, epochs=100, verbose=0)
mse_mean, mae_mean = model_mean.evaluate(X_test_mean_scaled, y_test, verbose=0)
results['Mean Imputation'] = {'MSE': mse_mean, 'MAE': mae_mean}
scalers['Mean Imputation'] = scaler_mean

# 2. İnterpolasyon (Interpolation)
X_train_interp = X_train.interpolate()
X_test_interp = X_test.interpolate()
scaler_interp = StandardScaler()
X_train_interp_scaled = scaler_interp.fit_transform(X_train_interp)
X_test_interp_scaled = scaler_interp.transform(X_test_interp)
model_interp = create_model()
model_interp.fit(X_train_interp_scaled, y_train, epochs=100, verbose=0)
mse_interp, mae_interp = model_interp.evaluate(X_test_interp_scaled, y_test, verbose=0)
results['Interpolation'] = {'MSE': mse_interp, 'MAE': mae_interp}
scalers['Interpolation'] = scaler_interp

# 3. Önceki Değerle Doldurma (Forward Fill)
X_train_ffill = X_train.fillna(method='ffill')
X_test_ffill = X_test.fillna(method='ffill')
scaler_ffill = StandardScaler()
X_train_ffill_scaled = scaler_ffill.fit_transform(X_train_ffill)
X_test_ffill_scaled = scaler_ffill.transform(X_test_ffill_scaled)
model_ffill = create_model()
model_ffill.fit(X_train_ffill_scaled, y_train, epochs=100, verbose=0)
mse_ffill, mae_ffill = model_ffill.evaluate(X_test_ffill_scaled, y_test, verbose=0)
results['Forward Fill'] = {'MSE': mse_ffill, 'MAE': mae_ffill}
scalers['Forward Fill'] = scaler_ffill

# 4. Sonraki Değerle Doldurma (Backward Fill)
X_train_bfill = X_train.fillna(method='bfill')
X_test_bfill = X_test.fillna(method='bfill')
scaler_bfill = StandardScaler()
X_train_bfill_scaled = scaler_bfill.fit_transform(X_train_bfill)
X_test_bfill_scaled = scaler_bfill.transform(X_test_bfill_scaled)
model_bfill = create_model()
model_bfill.fit(X_train_bfill_scaled, y_train, epochs=100, verbose=0)
mse_bfill, mae_bfill = model_bfill.evaluate(X_test_bfill_scaled, y_test, verbose=0)
results['Backward Fill'] = {'MSE': mse_bfill, 'MAE': mae_bfill}
scalers['Backward Fill'] = scaler_bfill

#4)Sonuçları Görüntüleme
import matplotlib.pyplot as plt

print("\nKayıp Değer İşleme Yöntemleri Karşılaştırması:")
print(pd.DataFrame.from_dict(results, orient='index'))

# MSE Değerlerinin Grafiği
mse_values = [results[method]['MSE'] for method in results]
methods = list(results.keys())

plt.figure(figsize=(8, 6))
plt.bar(methods, mse_values, color='skyblue')
plt.ylabel('Ortalama Karesel Hata (MSE)')
plt.title('Farklı Kayıp Değer İşleme Yöntemlerinin MSE Değerleri')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# MAE Değerlerinin Grafiği
mae_values = [results[method]['MAE'] for method in results]

plt.figure(figsize=(8, 6))
plt.bar(methods, mae_values, color='lightcoral')
plt.ylabel('Ortalama Mutlak Hata (MAE)')
plt.title('Farklı Kayıp Değer İşleme Yöntemlerinin MAE Değerleri')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

