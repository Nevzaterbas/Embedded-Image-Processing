import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Veri Setini Yükle (Otomatik İndirir) ---
# Artık manuel dosya yolu belirtmenize gerek yok.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# --- 2. Ön İşleme (Preprocessing) ---
# Piksel değerlerini 0-255 arasından 0-1 arasına sıkıştırıyoruz (Normalizasyon)
train_images = train_images / 255.0
test_images = test_images / 255.0

# --- 3. Model Oluşturma (Multi-Layer Perceptron) ---
# Tek nöron yerine, gizli katmanlı bir ağ kuruyoruz.
model = tf.keras.models.Sequential([
    # 28x28'lik görüntüyü düz bir vektöre (784 piksel) çevirir
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Gizli Katman: 128 nöron, ReLU aktivasyonu (Öğrenme kapasitesini artırır)
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Çıkış Katmanı: 0-9 arası 10 rakam olduğu için 10 nöron.
    # Softmax: Olasılık dağılımı verir (Toplamları 1 olur).
    tf.keras.layers.Dense(10, activation='softmax')
])

# --- 4. Derleme (Compile) ---
model.compile(
    optimizer='adam',
    # Etiketler tamsayı olduğu için (0, 1, 2...) SparseCategorical kullanıyoruz
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# --- 5. Eğitim (Training) ---
print("Eğitim başlıyor...")
history = model.fit(
    train_images, 
    train_labels, 
    epochs=5,           # 5 epoch genelde %98 başarı için yeterlidir
    batch_size=32,
    validation_split=0.1 # Eğitim verisinin %10'unu doğrulama için ayır
)

# --- 6. Değerlendirme (Evaluation) ---
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# --- 7. Confusion Matrix ---
# Tahminleri al
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Matrisi çizdir
cm = confusion_matrix(test_labels, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

plt.figure(figsize=(10,10))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title(f"MNIST Confusion Matrix (Test Acc: {test_acc:.2f})")
plt.show()

# --- 8. Kaydet ---
model.save("mnist_mlp_model.h5")
print("Model kaydedildi: mnist_mlp_model.h5")