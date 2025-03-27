# MKT3434 Makine Öğrenimi GUI - <21067040>

Bu GUI uygulaması, temel makine öğrenimi algoritmalarını eğitmek ve görselleştirmek için geliştirilmiştir. Bu sürümde aşağıdaki yeni özellikler eklenmiştir:

## Yeni Özellikler

* **Dinamik Kayıp Fonksiyonu Seçimi:** Hem Klasik Makine Öğrenimi hem de Derin Öğrenme sekmelerinde, problem tipine (regresyon veya sınıflandırma) göre farklı kayıp fonksiyonları seçilebilir. Bu sayede modelinizi probleme en uygun kayıp fonksiyonu ile eğitebilirsiniz.

* **Support Vector Regression (SVR) ve Support Vector Classification (SVC):** Klasik Makine Öğrenimi sekmesine regresyon ve sınıflandırma için Destek Vektör Makineleri (SVM) algoritmaları eklenmiştir. SVR ile sürekli değerli çıktıları tahmin edebilirken, SVC ile verileri farklı sınıflara ayırabilirsiniz.

* **Veri Ön İşleme (Derin Öğrenme Sekmesi):** Derin Öğrenme sekmesine kayıp değerleri işlemek için bir bölüm eklenmiştir. Bu bölüm sayesinde modelinizi eğitmeden önce verinizdeki eksiklikleri giderebilirsiniz.

* **Gelişmiş Naive Bayes (Klasik ML Sekmesi):** Naive Bayes algoritmasına `var_smoothing` parametresi ve öncelik olasılıkları (uniform veya kullanıcı tanımlı) için seçenekler eklenmiştir. Kullanıcı tanımlı öncelikler sayesinde modelin sınıflara olan başlangıç inancını belirleyebilirsiniz.

## Yeni Özelliklerin Kullanımı

**1. Dinamik Kayıp Fonksiyonu Seçimi:**

* Bir veri seti yükledikten sonra, "Classical ML Algorithms" veya "Deep Learning" sekmelerine gidin.
* Her sekmenin en üstünde bulunan "Loss Function" başlığı altındaki açılır menüden (ComboBox) problem tipinize uygun bir kayıp fonksiyonu seçebilirsiniz.
    * **Regresyon Problemleri İçin (Problem tipi belirlendiğinde aşağıdaki seçenekler görünür):**
        * **MSE (Mean Squared Error):** Ortalama kare hatası.
        * **MAE (Mean Absolute Error):** Ortalama mutlak hata.
        * **Huber:** Hatalara karşı daha az duyarlı olan bir kayıp fonksiyonu.
    * **Sınıflandırma Problemleri İçin (Problem tipi belirlendiğinde aşağıdaki seçenekler görünür):**
        * **Cross-Entropy:** Sınıflandırma problemleri için yaygın olarak kullanılan bir kayıp fonksiyonu.
        * **Hinge:** Özellikle SVM gibi modellerle kullanılan bir kayıp fonksiyonu.
        * **CategoricalCrossentropy (Derin Öğrenme):** Çok sınıflı sınıflandırma için.
        * **BinaryCrossentropy (Derin Öğrenme):** İki sınıflı sınıflandırma için.
* Modelinizi eğitmeden önce bu menüden istediğiniz kayıp fonksiyonunu seçtiğinizden emin olun.

**2. Support Vector Regression (SVR) ve Support Vector Classification (SVC):**

* Bir veri seti yükledikten sonra, "Classical ML Algorithms" sekmesine gidin.
* **Regresyon için (Problem tipi regresyon olduğunda "Regression" başlığı altında):**
    * "Support Vector Regression" başlığı altındaki grubu bulun.
    * **Kernel:** Kullanmak istediğiniz çekirdek tipini seçin (Linear, RBF, Poly).
    * **C:** Düzenlileştirme parametresini ayarlayın (daha küçük değerler daha güçlü düzenlileştirme anlamına gelir).
    * **Epsilon:** Boru genişliği parametresini ayarlayın (modelin ihmal edeceği hata miktarını kontrol eder).
    * Parametreleri ayarladıktan sonra "Train Support Vector Regression" butonuna tıklayarak modeli eğitin.
* **Sınıflandırma için (Problem tipi sınıflandırma olduğunda "Classification" başlığı altında):**
    * "Support Vector Classification" başlığı altındaki grubu bulun.
    * **C:** Düzenlileştirme parametresini ayarlayın.
    * **Kernel:** Kullanmak istediğiniz çekirdek tipini seçin (Linear, RBF, Poly).
    * **Degree (Poly seçildiğinde):** Polinomial çekirdeğin derecesini ayarlayın.
    * **Probability:** Olasılık tahminlerinin hesaplanıp hesaplanmayacağını seçin (daha fazla işlem gücü gerektirebilir).
    * Parametreleri ayarladıktan sonra "Train Support Vector Classification" butonuna tıklayarak modeli eğitin.

**3. Veri Ön İşleme (Derin Öğrenme Sekmesi):**

* Bir veri seti yükledikten sonra, "Deep Learning" sekmesine gidin.
* "Data Preprocessing" başlığı altındaki "Missing Values:" etiketinin yanındaki açılır menüden kayıp değerleri işlemek için bir yöntem seçebilirsiniz.
    * **None:** Kayıp değerleri olduğu gibi bırakır.
    * **Mean Imputation:** Kayıp değerleri sütunun ortalaması ile doldurur.
    * **Interpolation:** Kayıp değerleri doğrusal interpolasyon ile tahmin ederek doldurur.
    * **Forward Fill:** Kayıp değerleri bir önceki geçerli değerle doldurur.
    * **Backward Fill:** Kayıp değerleri bir sonraki geçerli değerle doldurur.
* İstediğiniz yöntemi seçtikten sonra derin öğrenme modelinizi eğitebilirsiniz. Ön işleme, eğitim başlamadan önce otomatik olarak uygulanacaktır.

**4. Gelişmiş Naive Bayes (Klasik ML Sekmesi):**

* Bir sınıflandırma veri seti yükledikten sonra, "Classical ML Algorithms" sekmesine gidin.
* "Classification" başlığı altındaki "Naive Bayes" grubunu bulun.
* **var_smoothing:** Varyans sabitleme için bir değer girin (sayısal kararlılığı artırır).
* **priors:** Sınıf öncelik olasılıklarını nasıl belirlemek istediğinizi seçin:
    * **uniform:** Tüm sınıflara eşit öncelik verir.
    * **user_defined:** Kullanıcı tarafından tanımlanan öncelikleri kullanır. Bu seçeneği seçtiğinizde, altındaki metin kutusu aktif hale gelecektir.
* **user_defined_priors:** "priors" seçeneğini "user_defined" olarak seçtiyseniz, bu metin kutusuna sınıf öncelik olasılıklarını virgülle ayrılmış olarak girin. Örneğin, 3 sınıfınız varsa ve önceliklerinizin 0.3, 0.4 ve 0.3 olmasını istiyorsanız, `0.3, 0.4, 0.3` şeklinde girin. Girdiğiniz sayıların toplamının 1'e eşit olması gerektiğini unutmayın. Sınıf sayısıyla eşleşen sayıda öncelik girdiğinizden emin olun.
* Parametreleri ayarladıktan sonra "Train Naive Bayes" butonuna tıklayarak modeli eğitin.
