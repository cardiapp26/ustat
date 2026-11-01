# uSTAT Restricted Cubic Spline (RCS) Modülü İnceleme, Klinik Değerlendirme ve Geliştirme Planı

**Tarih:** 1 Kasım 2026

**Kapsam:** `/Users/yh/Documents/projects/rcs sources` kaynaklarının sentezi, uSTAT RCS modülü durum analizi, klinik/akademik gereksinimler ve önceliklendirilmiş geliştirme yol haritası.

---

## 1. Yönetici Özeti (Executive Summary)

Restricted Cubic Splines (RCS / Kısıtlanmış Kübik Spline'lar), biyomedikal araştırmalarda ve klinik karar modellerinde sürekli değişkenlerin (yaş, biyobelirteçler, kan basıncı, skorlar) sonuç değişkeni (mortalite, hastalık gelişimi, klinik ölçüm) ile olan **non-lineer (eğrisel)** ilişkilerini modellemek için günümüzdeki altın standarttır.

Bu rapor, `/Users/yh/Documents/projects/rcs sources` dizininde yer alan 5 temel referans kaynağını inceleyerek uSTAT'ın mevcut RCS altyapısını değerlendirmekte; hem **klinisyenlerin hasta başındaki karar süreçleri** hem de **uluslararası SCI tıp dergilerinin hakem (reviewer) beklentileri** doğrultusunda gerekli iyileştirmeleri planlamaktadır.

---

## 2. İncelenen Kaynaklar ve Metodolojik Temeller

### 2.1. Frank E. Harrell, Jr. (2015): *Regression Modeling Strategies (RMS)* (2. Baskı, Springer)
* **Temel Felsefe:** Sürekli değişkenlerin yapay olarak kategorize edilmesi (dichotomization) biyolojik gerçekliği bozar, bilgi kaybına ve istatistiksel gücün düşmesine yol açar (§2.4.1).
* **Knot Yerleşimi ve Sayısı:**
  - $k=3, 4, 5, 6, 7$ düğüm noktaları için önerilen standart marjinal persentiller:
    - 3 knot: `[10, 50, 90]`
    - 4 knot: `[5, 35, 65, 95]` (Önerilen varsayılan)
    - 5 knot: `[5, 27.5, 50, 72.5, 95]`
    - 6 knot: `[5, 23, 41, 59, 77, 95]`
    - 7 knot: `[2.5, 18.33, 34.17, 50, 65.83, 81.67, 97.5]`
  - Küçük örneklemlerde ($n < 100$) en dış iki knot, 5. en küçük ve 5. en büyük gözlemle değiştirilir (§2.4.6).
  - Harrell'a göre 5'ten fazla knot nadiren gerekir; 6 ve 7 knot yalnızca büyük örneklemler için anlamlıdır. Knot sayısı AIC ile veriye dayalı seçilebilir (§2.4.6).
* **Hipotez Testi Ayrımı (Kritik):**
  - **Genel İlişki Testi ($P_{\text{overall}}$):** Lineer ve kübik tüm spline terimlerinin ortak Wald testi ($k-1$ serbestlik derecesi). Değişkenin sonuç üzerinde herhangi bir etkisinin olup olmadığını test eder ($H_0: \beta_1 = \dots = \beta_{k-1} = 0$).
  - **Non-lineerlik Testi ($P_{\text{non-linear}}$):** Yalnızca kübik terimlerin ortak Wald testi ($k-2$ serbestlik derecesi). İlişkinin düz bir doğru olup olmadığını test eder ($H_0: \beta_2 = \dots = \beta_{k-1} = 0$).
* **Serbestlik Derecesi Bütçelemesi (EPV):** Olay başına değişken (Events Per Variable) kuralına göre knot sayısı seçimi ve modellerin AIC üzerinden tarafsız karşılaştırılması.

### 2.2. Hongbin Xu et al. (2025): *rcssci: An R Package for Visualization of Restricted Cubic Spline* (Med Research)
* **Genişletilmiş Model Ailesi:** Sadece Cox, Lojistik ve Lineer modeller değil; sayım ve oran verileri için **Poisson / Quasi-Poisson** modelleme (`rcssci_quasipoisson`).
* **Klinik Görselleştirme Standartları:**
  - Eğri altına entegre edilen marjinal yoğunluk dağılımı (density distribution / histogram).
  - Referans noktasının medyan yerine persentil veya minimum risk/dönüm noktası olarak seçilebilmesi.
  - $HR/OR = 1.0$ referans kesişim çizgileri ve eşik (cut-off) değerleri.
  - U-şekilli, L-şekilli veya N-şekilli eğri morfolojilerinin tespiti.

### 2.3. Peter C. Austin (2025): *Graphical methods to illustrate the nature of the relation between a continuous variable and the outcome when using restricted cubic splines with a Cox proportional hazards model* (Statistical Methods in Medical Research, 34(2), 277–285; çevrimiçi 2024)
* **Göreceli Tehlikeden (HR) Mutlak Riske Geçiş:** Cox modellerinde $HR(x; x_{\text{ref}})$ görecelidir ve seçilen referans değere doğrudan bağımlıdır.
* **Austin'in 4 Grafik Yöntemi:**
  1. **Rölatif Hazard Oranı ($HR(x; x_{\text{ref}})$):** Geleneksel göreceli eğri.
  2. **Kümülatif İnsidans / Mutlak Olay Riski ($F(t_0 | X)$):** Belirlenen bir klinik zaman ufkunda ($t_0 = 1$ yıl, 5 yıl vb.) mutlak olay gerçekleşme olasılığı:
     $$F(t_0 | X) = 1 - S_0(t_0)^{\exp(f(X)\beta)}$$
  3. **Anlık Tehlike ($h(t_0 | X)$)**.
  4. **Kümülatif Tehlike ($H(t_0 | X)$)**.
* **Klinik Sonuç:** $F(t_0 | X)$ mutlak risk eğrisi, klinisyenlerin ve hastaların riski doğrudan yüzde olarak anlamasını sağlayarak klinik kararları doğrudan yönlendirir.

### 2.4. J. Gauthier, Q.V. Wu, T.A. Gooley (2020): *Cubic splines to model relationships between continuous variables and outcomes: a guide for clinicians* (Bone Marrow Transplantation)
* **Klinik Yorumlama Rehberi:** Biyolojik değişkenlerde "normal / yüksek" gibi yapay ikiye bölmelerin yarattığı klinik tehlikeler (örneğin 39 yaş ile 41 yaş arasındaki benzerliği silip, 41 yaş ile 85 yaşı aynı risk grubuna sokmak).
* **Eğri Morfolojileri:** Plato (threshold) etkileri, U ve ters-J şekilli eğrilerin klinik tercümesi.
* **Örneklem ve Olay Sayısı:** 3 ila 5 knot'un çoğu klinik çalışma için optimum olduğu, uç bölgelerdeki geniş güven aralıklarının klinik yorumda dikkate alınması gerektiği.

### 2.5. Maarten L. Buis: *Using and interpreting restricted cubic splines*
* Harrell parametrizasyonu, marjinal etkiler (spline fonksiyonunun türevi / eğim grafiği) ve sürekli × sürekli / sürekli × kategorik etkileşim terimlerinin modellenmesi.

---

## 3. Klinisyenler İçin Geçerli Olanlar vs. Araştırmalarda Kullanılanlar

| Özellik | Klinisyen Açısından Değeri | Araştırma / Makale (Hakem) Açısından Değeri | uSTAT Önceliği |
| :--- | :--- | :--- | :---: |
| **$P_{\text{overall}}$ Testi (Genel İlişki)** | Orta ("Bu belirteç sonuçla genel olarak ilişkili mi?") | **Zorunlu:** Hakemler hem genel etkiyi hem non-lineerliği ayrı $p$ değerleri olarak görmek ister. | **Faz 1 (Kritik)** |
| **Austin (2025) Mutlak Risk ($F(t_0)$)** | **Kritik:** Hekim ve hasta "1 veya 5 yılda başıma bir şey gelme riski % kaç?" sorusuna yanıt arar. | **Yüksek:** Modern klinik epidemiyoloji dergilerinde mutlak risk projeksiyonu tercih edilir. | **Faz 2 (Kritik)** |
| **Nadir / Minimum Risk Referansı** | **Çok Yüksek:** Biyobelirteç için fizyolojik olarak hedeflenen ideal değerdir (en düşük risk noktası). | **Yüksek:** U ve J tipi eğrilerde medyan yerine nadir referansı biyolojik mekanizmayı açıklar. | **Faz 3 (Yüksek)** |
| **Yoğunluk / Histogram Katmanı** | Yüksek: Verinin yoğun olduğu güvenli bölge ile veri kıtlığından sapan uçları ayırt eder. | **Çok Yüksek:** Uçlardaki bükülmelerin hasta azlığından kaynaklanmadığını kanıtlar. | **Faz 3 (Yüksek)** |
| **Knot Bütçesi ve AIC Kıyaslaması** | Düşük (Klinisyen model arkasındaki knot matematiğiyle ilgilenmez). | **Çok Yüksek:** "Neden 4 knot seçildi?" sorusunun metodolojik yanıtıdır. | **Faz 1 (Yüksek)** |
| **Kritik Eşik ($HR=1.0$ Geçişi)** | **Kritik:** Tedaviye veya müdahaleye nerede başlanacağını tartışmak için kullanılır; ancak $HR=1.0$ geçişi seçilen referans değere bağlıdır, tek başına klinik eşik kanıtı sayılmaz. | **Yüksek:** Sonuç ve tartışma bölümlerinde klinik çıkarım yapmak için kullanılır. | **Faz 3 (Yüksek)** |
| **Poisson / Sayım RCS Modeli** | Yüksek: Hastanede yatış süresi, acil başvuru sayısı, atak sayısı gibi olaylarda gereklidir. | **Yüksek:** Sayım verisiyle çalışan halk sağlığı ve klinik araştırmalarında talep edilir. | **Faz 4 (Orta)** |

---

## 4. uSTAT Mevcut Altyapı Durum Analizi (Gap Analysis)

### Güçlü Yönler:
1. **Harrell Basis Doğruluğu:** `backend/services/rcs_basis.py` Harrell RMS §2.4.5 (Restricted Cubic Splines) formülasyonunu tam olarak uygular.
2. **Çift Model Entegrasyonu:** Tek değişkenli `/api/models/rcs` (Cox, Logistic, Linear) ve çok değişkenli `/api/models/survival/cox_rcs` (1 veya 2 spline + tensor interaction + lineer kovaryatlar) mevcuttur.
3. **Crude vs Adjusted Eğrileri:** Düzeltilmiş model çalıştırıldığında arkaplanda ham (crude) model de hesaplanarak üst üste çizdirilebilmektedir.
4. **Log Ölçek & Rug Plot:** $HR/OR$ simetrisi için log-scale ve veri noktaları rug gösterimi bulunmaktadır.

### Eksiklikler ve Geliştirme Alanları:
1. **$P_{\text{overall}}$ Eksikliği:** Backend sadece non-lineerlik testini (`nonlinearity_p`, $k-2$ df) hesaplamaktadır. Tüm spline terimlerinin ortak testi olan `overall_p` ($k-1$ df) hesaplanmamaktadır.
2. **Knot Kısıtlılığı:** Sadece 3, 4, 5 knot desteklenmektedir. Harrell'ın 6 ve 7 knot tabloları eksiktir.
3. **Mutlak Risk Eksikliği (Austin 2025):** Cox-RCS yalnızca rölatif hazard oranı vermekte, kümülatif insidans $F(t_0)$ üretmemektedir.
4. **Referans Değer Seçeneği:** Sadece medyan veya manuel sayı girilebilmektedir; otomatik "Nadir / Minimum Risk" tespit mekanizması yoktur.
5. **Görsel Katman (Xu 2025):** Veri dağılımı sadece tek boyutlu çizgi izleri (rug) olarak gösterilmekte, ikincil eksende saydam KDE/histogram bulunmamaktadır.
6. **AIC Kıyaslama Tablosu:** Kullanıcı farklı knot sayılarını tek ekranda AIC ile kıyaslayamamaktadır.

---

## 5. Önceliklendirilmiş Geliştirme Yol Haritası (Implementation Roadmap)

### Faz 1: Akademik & İstatistiksel Bütünlük (Öncelik: Çok Yüksek)

> **Durum (1 Kasım 2026): tamamlandı.** `overall_p` (genel ilişki, $k-1$ df) ve
> `nonlinearity_p` iki RCS uç noktasında da raporlanıyor; etkileşim terimleri
> rms::anova gibi iki teste de dâhil. Doğrusal modelde test F, lojistik ve Cox'ta
> ki-kare. 6 ve 7 knot ile $n < 100$ dış knot kuralı eklendi. 3, 4, 5 knot için
> AIC karşılaştırma tablosu sonuç ekranında. Değerler R `rms` (lrm, ols, cph) ile
> `backend/tests/test_rcs.py` içinde doğrulandı.
* **Hedef:** Makale yazarlarının ve hakemlerin zorunlu kıldığı hipotez testlerini ve knot seçeneklerini tamamlamak.
* **Kapsam:**
  1. `backend/routers/models/cox.py`: Model parametrelerinden lineer ve kübik tüm terimleri kapsayan $P_{\text{overall}}$ ortak Wald testinin (`overall_wald`, `overall_df`, `overall_p`) hesaplanması.
  2. `backend/services/rcs_basis.py`: Harrell RMS Tablo 2.3'e uygun olarak 6 ve 7 düğüm persentillerinin eklenmesi.
  3. `frontend/src/components/RCSPanel.tsx`: Sonuç panelinde hem $P_{\text{overall}}$ hem de $P_{\text{non-linear}}$ değerlerinin birlikte gösterilmesi.
  4. 3, 4 ve 5 düğümlü modeller için tek tıkla AIC karşılaştırma özeti.

### Faz 2: Austin (2025) Klinik Mutlak Risk & Kümülatif İnsidans Modu (Öncelik: Yüksek)

> **Durum (1 Kasım 2026): tamamlandı.** Tek değişkenli Cox-RCS, seçilen $t_0$
> ufkunda (boş bırakılırsa medyan takip süresi) $F(t_0 \mid x)$ eğrisini diğer
> kovaryatlar ortalamada olacak şekilde döndürüyor. Güven bandı R
> `survfit.coxph` ile aynı yöntemle (Tsiatis varyansı, `conf.type = "log"`)
> hesaplanıyor; değerler `backend/tests/test_rcs_absolute_risk.py` içinde R ile
> doğrulandı. Sonuç ekranında "Hazard ratio | Absolute risk" geçişi, ufukta
> risk altındaki hasta sayısı ve az hasta kaldığında uyarı var. Çok değişkenli
> Cox-RCS (iki spline terimi) için mutlak risk henüz yok.
* **Hedef:** Klinisyenlerin doğrudan hasta başında kullanabileceği mutlak risk projeksiyonu sağlamak.
* **Kapsam:**
  1. `backend/routers/models/cox.py`: Breslow taban kümülatif tehlike fonksiyonu ($H_0(t)$) üzerinden kullanıcının seçeceği $t_0$ zaman ufkunda (örn. 1 yıllık, 3 yıllık, 5 yıllık veya veri medyan takip süresinde) kümülatif insidans eğrisi ($F(t_0 | X) = 1 - S_0(t_0)^{\exp(X\beta)}$) ve delta metoduyla %95 güven bandı hesaplanması.
  2. `frontend/src/components/RCSPanel.tsx`: Cox modunda **"Görünüm: Rölatif Hazard Oranı (HR) | Mutlak Risk (Olay Olasılığı %)"** geçiş düğmesi.

### Faz 3: Xu (2025) Yayın Seviyesinde Görselleştirme & Nadir Referansı (Öncelik: Yüksek)
* **Hedef:** Grafik estetiğini ve biyolojik yorumlanabilirliği en üst düzeye çıkarmak.
* **Kapsam:**
  1. **Nadir (Minimum Risk) Tespiti:** Risk fonksiyonunun minimum olduğu $X$ değerinin sayısal optimizasyonla bulunup otomatik referans ($HR/OR=1.0$) olarak atanabilmesi.
  2. **Eğri Altı Yoğunluk Histogramı / KDE:** Plotly grafiğinde ikincil Y ekseni üzerinde değişkenin dağılımını gösteren şeffaf yoğunluk katmanı.
  3. **Kritik Eşik İşaretçileri:** Güven aralığının 1.0 çizgisini aştığı istatistiksel anlamlılık kesişim noktalarının grafikte vurgulanması.

### Faz 4: Model Yelpazesi Genişletme (Öncelik: Orta)
* **Hedef:** Sayım ve oran tipi klinik sonuç değişkenlerini kapsamak.
* **Kapsam:**
  1. `model_type="poisson"` ve `"quasipoisson"` desteği (statsmodels GLM üzerinden).
  2. İnsidans Hız Oranı ($IRR$) eğrileri ve non-lineerlik testleri.

---

## 6. Doğrulama ve Kabul Kriterleri (QA & Parity)
Geliştirmeler tamamlandığında:
- R `rms::rcs` ve `rcssci` paketleriyle parite testleri (`qa/validation_cards/rcs.yaml`).
- Backend birim ve regresyon testleri (`backend/tests/test_rcs.py`).
- Frontend etkileşim ve render testleri (`frontend/src/components/RCSPanel.test.tsx`).
- Tip denetimi (`tsc -b`) ve linter geçişi.
