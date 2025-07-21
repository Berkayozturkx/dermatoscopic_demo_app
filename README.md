# 🔬 Dermatoskopik Görsel Sınıflandırma Uygulaması Demo

Bu Streamlit uygulaması, dermatoskopik görüntüleri 7 farklı cilt lezyonu kategorisine sınıflandırmak için eğitilmiş DenseNet121 modelini kullanır.

## 📋 Özellikler

- **Görüntü Yükleme**: PNG, JPG, JPEG formatlarında dermatoskopik görüntü yükleme
- **Tahmin**: 7 farklı cilt lezyonu türü için sınıflandırma
- **Güven Skoru**: Her sınıf için güven skorları
- **GradCAM Görselleştirme**: Modelin hangi bölgelere odaklandığını gösteren ısı haritası
- **Klinik Öneriler**: Tahmin sonuçlarına göre tıbbi öneriler
- **Risk Değerlendirmesi**: Malign/benign lezyon ayrımı

## 🏷️ Sınıflandırma Sınıfları

### Malign Sınıflar (Yüksek Risk)
- **MEL (Melanoma)**: En tehlikeli cilt kanseri türü
- **BCC (Basal Cell Carcinoma)**: Yaygın, genellikle yavaş büyüyen cilt kanseri
- **AKIEC (Actinic Keratosis/Bowen's Disease)**: Kanser öncesi cilt durumu

### Benign Sınıflar (Düşük Risk)
- **NV (Melanocytic Nevus)**: Yaygın ben
- **BKL (Benign Keratosis)**: Kanserli olmayan cilt büyümesi
- **DF (Dermatofibroma)**: Benign cilt tümörü
- **VASC (Vascular Lesion)**: Kan damarı anormalliği

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- Model dosyası: `densenet121_66acc.keras`

### Adım 1: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 2: Model Dosyasını Kontrol Edin
Model dosyasının (`densenet121_66acc.keras`) uygulama dizininde olduğundan emin olun.

### Adım 3: Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde çalışacaktır.

## 📊 Model Performansı

- **Doğruluk**: ~%66
- **Sınıf Sayısı**: 7
- **Giriş Boyutu**: 128x128 piksel
- **Mimari**: DenseNet121 (Transfer Learning)

## 🔧 Kullanım

1. **Görüntü Yükleme**: "Upload Dermatoscopic Image" bölümünden bir dermatoskopik görüntü yükleyin
2. **İşleme**: Model görüntüyü işleyecek ve tahmin yapacaktır
3. **Sonuçları İnceleme**: 
   - Tahmin edilen sınıf
   - Güven skoru
   - GradCAM görselleştirmesi
   - Tüm sınıflar için güven skorları
4. **Klinik Öneriler**: Risk seviyesine göre tıbbi öneriler

## 🖼️ Görselleştirmeler

### GradCAM (Gradient-weighted Class Activation Mapping)
- Modelin hangi bölgelere odaklandığını gösterir
- Kırmızı alanlar modelin en çok dikkat ettiği bölgeleri temsil eder
- Klinik karar verme sürecini anlamaya yardımcı olur

### Güven Skorları
- Her sınıf için ayrı güven skorları
- En yüksek güven skoruna sahip sınıf vurgulanır
- Top 3 tahmin listesi

## ⚕️ Klinik Öneriler

### Yüksek Risk Lezyonları (Malign)
- Dermatolog randevusu alın
- Lezyonu fotoğraflarla belgeleyin
- Boyut, renk veya şekil değişikliklerini izleyin
- Kesin tanı için biyopsi düşünün

### Düşük Risk Lezyonları (Benign)
- Düzenli cilt kontrollerine devam edin
- Değişiklikleri izleyin
- Güneş koruması uygulayın
- Yıllık dermatolog ziyareti önerilir

## ⚠️ Tıbbi Sorumluluk Reddi

Bu uygulama sadece eğitim ve gösterim amaçlıdır. Profesyonel tıbbi teşhisin yerini tutmaz. Cilt durumlarının uygun teşhis ve tedavisi için her zaman nitelikli bir sağlık uzmanına danışın.

## 🛠️ Teknik Detaylar

### Ön İşleme Adımları
1. **Saç Kaldırma**: Morfolojik işlemlerle saç/kıl temizleme
2. **Siyah Çerçeve Kaldırma**: Görüntü sınırlarındaki siyah alanları temizleme
3. **CLAHE**: Kontrast sınırlı adaptif histogram eşitleme
4. **Standartlaştırma**: Ortalama 0, standart sapma 1'e normalize etme

### Model Mimarisi
- **Temel Model**: DenseNet121 (ImageNet ağırlıkları)
- **Transfer Learning**: Önceden eğitilmiş ağırlıklar kullanılarak
- **Sınıflandırma Katmanları**: GlobalAveragePooling2D + Dense katmanları
- **Aktivasyon**: Softmax (çok sınıflı sınıflandırma)

## 📈 Performans Metrikleri

### Test Sonuçları
- **Genel Doğruluk**: %64
- **Makro Ortalama F1-Skoru**: 0.47
- **Ağırlıklı Ortalama F1-Skoru**: 0.66

### Sınıf Bazında Performans
- **NV (Nevüs)**: En iyi performans (%82 F1-skoru)
- **MEL (Melanoma)**: %38 F1-skoru (iyileştirme gerekli)
- **BCC (Basal Cell Carcinoma)**: %43 F1-skoru

## 🔍 Hata Analizi

### Yaygın Hatalar
1. **Melanoma-Nevüs Karışıklığı**: En kritik hata türü
2. **BCC-AKIEC/BKL Karışıklığı**: Benzer görünüm nedeniyle
3. **Azınlık Sınıf Performansı**: Düşük temsil nedeniyle

### İyileştirme Önerileri
- Sınıf dengesizliği için veri artırma teknikleri
- Melanoma-nevüs ayrımı için özel özellik mühendisliği
- Daha fazla azınlık sınıf verisi toplama
