import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Dermatoskopik Görüntü Sınıflandırma Demo",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Class labels and their descriptions
CLASS_LABELS = {
    0: "MEL (Melanoma)",
    1: "NV (Melanositik Nevüs)", 
    2: "BCC (Bazal Hücreli Karsinom)",
    3: "AKIEC (Aktinik Keratoz / Bowen Hastalığı)",
    4: "BKL (Benign Keratoz)",
    5: "DF (Dermatofibroma)",
    6: "VASC (Vasküler Lezyon)"
}

CLASS_DESCRIPTIONS = {
    0: "Malign melanom - en tehlikeli cilt kanseri türü",
    1: "Benign melanositik nevüs - yaygın ben",
    2: "Bazal hücreli karsinom - yaygın cilt kanseri, genellikle yavaş büyüyen",
    3: "Aktinik keratoz veya Bowen hastalığı - kanser öncesi cilt durumu",
    4: "Benign keratoz - kanserli olmayan cilt büyümesi",
    5: "Dermatofibroma - benign cilt tümörü",
    6: "Vasküler lezyon - kan damarı anormalliği"
}

@st.cache_resource
def load_trained_model():
    """Eğitilmiş DenseNet121 modelini yükle"""
    try:
        model = load_model('densenet121_66acc.keras')
        return model
    except:
        st.error("Model dosyası 'densenet121_66acc.keras' bulunamadı. Lütfen model dosyasının aynı dizinde olduğundan emin olun.")
        return None

def preprocess_image(image):
    """Model tahmini için görüntüyü ön işle"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Resize to 128x128
    img_resized = cv.resize(img_array, (128, 128))
    
    # Convert BGR to RGB if needed
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    else:
        img_rgb = img_resized
    
    # Apply preprocessing steps from notebook
    # Hair removal
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 17))
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    _, mask = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    mask = cv.dilate(mask, None)
    img_processed = cv.inpaint(img_rgb, mask, 1, cv.INPAINT_TELEA)
    
    # Remove black borders
    gray = cv.cvtColor(img_processed, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(cnt)
        img_cropped = img_processed[y:y+h, x:x+w]
        img_resized = cv.resize(img_cropped, (128, 128))
    else:
        img_resized = cv.resize(img_processed, (128, 128))
    
    # Apply CLAHE
    lab = cv.cvtColor(img_resized, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_clahe = cv.merge((cl,a,b))
    img_clahe = cv.cvtColor(lab_clahe, cv.COLOR_LAB2RGB)
    
    # Standardize
    img_standard = img_clahe.astype(np.float32) / 255.0
    mean = np.mean(img_standard, axis=(0, 1), keepdims=True)
    std = np.std(img_standard, axis=(0, 1), keepdims=True) + 1e-7
    img_final = (img_standard - mean) / std
    
    return img_final

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block16_concat', pred_index=None):
    """Tahmin için GradCAM ısı haritası oluştur"""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_gradcam_visualization(img_array, heatmap, alpha=0.4):
    """GradCAM üst üste bindirme görselleştirmesi oluştur"""
    # Resize heatmap to match image size
    heatmap_resized = cv.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply color map
    heatmap_color = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)
    
    # Convert image to uint8
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    # Create overlay
    overlay = cv.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay

def plot_confidence_bar(predictions, class_labels):
    """Güven skorları için çubuk grafik oluştur"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(predictions)), predictions, color='skyblue', alpha=0.7)
    
    # Highlight the highest confidence
    max_idx = np.argmax(predictions)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(0.8)
    
    ax.set_xlabel('Sınıflar')
    ax.set_ylabel('Güven Skoru')
    ax.set_title('Tahmin Güven Skorları')
    ax.set_xticks(range(len(predictions)))
    ax.set_xticklabels([f"{i}: {label}" for i, label in enumerate(class_labels.values())], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(predictions):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Dermatoskopik Görüntü Sınıflandırma Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Hakkında")
    st.sidebar.markdown("""
    Bu demo uygulaması, ISIC 2018 veri seti üzerinde eğitilmiş DenseNet121 modelini kullanarak
    dermatoskopik görüntüleri 7 farklı cilt lezyonu kategorisine sınıflandırır.
    
    **Model Performansı:**
    - Doğruluk: ~%66
    - Sınıflar: 7 cilt lezyonu türü
    - Giriş boyutu: 128x128 piksel
    """)
    
    st.sidebar.title("🔍 Nasıl Kullanılır")
    st.sidebar.markdown("""
    1. Bir dermatoskopik görüntü yükleyin
    2. Model lezyon türünü tahmin edecek
    3. Güven skorları ve GradCAM görselleştirmesini inceleyin
    4. Klinik önemini anlayın
    """)
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.error("Lütfen 'densenet121_66acc.keras' model dosyasının mevcut olduğundan emin olun.")
        return
    
    # File uploader
    st.markdown("### 📤 Dermatoskopik Görüntü Yükle")
    uploaded_file = st.file_uploader(
        "Bir görüntü dosyası seçin", 
        type=['png', 'jpg', 'jpeg'],
        help="Sınıflandırma için bir dermatoskopik görüntü yükleyin"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Orijinal Görüntü")
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner("Görüntü işleniyor ve tahmin yapılıyor..."):
            # Preprocess image
            img_processed = preprocess_image(image)
            
            # Make prediction
            img_input = np.expand_dims(img_processed, axis=0)
            predictions = model.predict(img_input, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate GradCAM
            try:
                heatmap = make_gradcam_heatmap(img_input, model)
                overlay = create_gradcam_visualization(img_processed, heatmap)
                
                with col2:
                    st.markdown("#### 🔍 GradCAM Görselleştirme")
                    st.image(overlay, caption="GradCAM Üst Üste Bindirme", use_column_width=True)
            except Exception as e:
                st.warning(f"GradCAM görselleştirmesi oluşturulamadı: {str(e)}")
        
        # Display results
        st.markdown("### 🎯 Tahmin Sonuçları")
        
        # Prediction card
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric(
                label="Tahmin Edilen Sınıf",
                value=CLASS_LABELS[predicted_class],
                delta=f"{confidence:.1%} güven"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Güven Skoru",
                value=f"{confidence:.3f}",
                delta="Yüksek" if confidence > 0.8 else "Orta" if confidence > 0.6 else "Düşük"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Determine if malignant
            malignant_classes = [0, 2, 3]  # MEL, BCC, AKIEC
            is_malignant = predicted_class in malignant_classes
            st.metric(
                label="Risk Seviyesi",
                value="Yüksek Risk" if is_malignant else "Düşük Risk",
                delta="⚠️ Malign" if is_malignant else "✅ Benign"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Class description
        st.markdown("#### 📖 Klinik Açıklama")
        st.info(f"**{CLASS_LABELS[predicted_class]}**: {CLASS_DESCRIPTIONS[predicted_class]}")
        
        # Confidence scores visualization
        st.markdown("#### 📊 Tüm Sınıflar İçin Güven Skorları")
        fig = plot_confidence_bar(predictions[0], CLASS_LABELS)
        st.pyplot(fig)
        
        # Detailed analysis
        st.markdown("#### 🔬 Detaylı Analiz")
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        st.markdown("**En İyi 3 Tahmin:**")
        for i, idx in enumerate(top_3_indices):
            confidence_score = predictions[0][idx]
            st.markdown(f"{i+1}. **{CLASS_LABELS[idx]}**: {confidence_score:.3f} ({confidence_score:.1%})")
        
        # Clinical recommendations
        st.markdown("#### ⚕️ Klinik Öneriler")
        
        if predicted_class in [0, 2, 3]:  # Malignant classes
            st.warning("""
            **⚠️ Yüksek Riskli Lezyon Tespit Edildi**
            
            Bu lezyon potansiyel olarak malign olarak sınıflandırılmıştır. 
            **Acil tıbbi konsültasyon şiddetle önerilir.**
            
            - Dermatolog randevusu alın
            - Lezyonu fotoğraflarla belgeleyin
            - Boyut, renk veya şekil değişikliklerini izleyin
            - Kesin tanı için biyopsi düşünün
            """)
        else:
            st.success("""
            **✅ Düşük Riskli Lezyon Tespit Edildi**
            
            Bu lezyon benign olarak sınıflandırılmıştır. Ancak, 
            **düzenli izleme hala önerilir.**
            
            - Düzenli cilt kontrollerine devam edin
            - Değişiklikleri izleyin
            - Güneş koruması uygulayın
            - Yıllık dermatolog ziyareti önerilir
            """)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107;">
        <strong>⚠️ Tıbbi Sorumluluk Reddi:</strong> Bu uygulama sadece eğitim ve gösterim amaçlıdır. 
        Profesyonel tıbbi teşhisin yerini tutmaz. Cilt durumlarının uygun teşhis ve tedavisi için her zaman nitelikli 
        bir sağlık uzmanına danışın.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Show sample images or instructions
        st.markdown("### 📋 Talimatlar")
        st.markdown("""
        1. **Bir görüntü yükleyin** yukarıdaki dosya yükleyiciyi kullanarak
        2. **İşlemeyi bekleyin** - model görüntüyü analiz edecek
        3. **Sonuçları inceleyin** - tahmin, güven ve GradCAM görselleştirmesini görün
        4. **Klinik etkileri anlayın** - tıbbi önerileri okuyun
        
        **Desteklenen formatlar:** PNG, JPG, JPEG
        **Önerilen görüntü türü:** Cilt lezyonlarının dermatoskopik görüntüleri
        """)
        
        # Show class information
        st.markdown("### 🏷️ Sınıflandırma Sınıfları")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Malign Sınıflar (Yüksek Risk):**")
            for class_id in [0, 2, 3]:
                st.markdown(f"- **{CLASS_LABELS[class_id]}**: {CLASS_DESCRIPTIONS[class_id]}")
        
        with col2:
            st.markdown("**Benign Sınıflar (Düşük Risk):**")
            for class_id in [1, 4, 5, 6]:
                st.markdown(f"- **{CLASS_LABELS[class_id]}**: {CLASS_DESCRIPTIONS[class_id]}")

if __name__ == "__main__":
    main() 