# uSTAT ve *Oxford Handbook of Medical Statistics* (Peacock & Peacock, 2. Baskı) Karşılaştırmalı İnceleme Raporu

**Referans Kaynak:** Janet L. Peacock & Philip J. Peacock, *Oxford Handbook of Medical Statistics*, 2nd Edition (2020), Oxford University Press (624 sayfa).  
**Tarih:** 2026-10-05  
**Yetki Seviyesi:** `edit` — Rapor dokümanı oluşturuldu; mevcut analiz kodları değiştirilmedi.  
**Kapsam:** Kitabın 15 ana bölümü ve uSTAT'ın `backend/routers/`, `backend/services/`, `backend/ustat_engine/` mimarisi.

---

## 1. Yönetici Özeti

*Oxford Handbook of Medical Statistics*, Birleşik Krallık Ulusal Sağlık Sistemi (NHS), Oxford Üniversitesi ve uluslararası klinik araştırmalarda (RCT) altın standart kabul edilen kılavuz eserdir. Kitap; temel hipotez testlerinin ötesinde **klinik epidemiyoloji (doğrudan/dolaylı standartlaştırma, SMR)**, **klinik çalışma metodolojisi (başlangıca göre değişim / baseline adjustment, özet ölçütler yaklaşımı)** ve **sağlık ekonomisi (maliyet verisi modelleme)** standartlarını eksiksiz belirler.

uSTAT'ın mevcut koduyla karşılaştırıldığında:
1. **Güçlü ve Doğrulanmış Alanlar:** uSTAT; Poisson regresyonda `exposure` (kişi-yılı ofseti), Gamma GLM (maliyet analizleri için log/identity link), GEE (Genelleştirilmiş Tahmin Denklemleri), Karışık Doğrusal Modeller (MixedLM), tanısal test Wilson/Simel GA'ları, Bland-Altman uyum sınırları için Student t tabanlı kesin güven aralıkları ve meta-analiz motoru gibi ileri düzey medikal istatistik yöntemlerini doğru şekilde uygulamaktadır.
2. **Kritik Metodolojik Eksiklikler:**
   - **Epidemiyolojik Doğrudan ve Dolaylı Standartlaştırma (SMR):** uSTAT'ta yaşa/cinsiyete göre standartlaştırılmış mortalite/morbidite oranı ve beklenen vaka hesabı **tamamen yoktur**.
   - **Seri/Boylamsal Verilerde Özet Ölçütler Yaklaşımı (Summary Measures / Hasta Başına AUC):** Tekrarlayan ölçümlerde trapezoit kuralıyla hasta başına AUC, pik değer, pike ulaşma süresi ve bireysel eğim (slope) çıkarma modülü yoktur.
   - **Klinik Çalışmalarda Başlangıç Düzeltmesi (Baseline Adjustment / ANCOVA Kılavuzu):** Ham fark skorları ($\Delta = Y_{\text{son}} - Y_{\text{ilk}}$) yerine başlangıcın ortak değişken (covariate) alındığı ANCOVA analizini zorunlu kılan/yönlendiren klinik araştırma akışı eksiktir.
   - **Tek Boyutlu Ki-Kare Uyum İyiliği Testi (Goodness-of-Fit):** uSTAT yalnızca iki boyutlu çapraz tablolarda ki-kare testi yapabilmektedir; gözlenen frekansları kuramsal oranlarla (Mendel, 1:1:1:1, Hardy-Weinberg) karşılaştıran tek örneklem uyum testi yoktur.
   - **McNemar Eşleştirilmiş Oran Farkı %95 Güven Aralığı:** McNemar testi yalnızca diskordant odds oranı ve p değeri dönmekte; klinik raporlamanın ana çıktısı olan eşleştirilmiş oranlar farkı ($p_1 - p_2$) ve güven aralığı hesaplanmamaktadır (`ci_low/high: None`).
   - **Ağırlıklı Kappa (Weighted Kappa) Güven Aralığı ve Anlamlılık Testi:** Kod içinde ağırlıklı kappa seçildiğinde standart hata, z değeri, p değeri ve güven aralığı hesaplanmayıp doğrudan `None` döndürülmektedir.
   - **Geometrik Ortalama ve Ters Dönüşüm (Back-Transformation):** Çarpık biyolojik belirteçler, viral yük ve antikor titreleri için zorunlu olan geometrik ortalama `descriptive.py` içinde yer almamaktadır.
   - **Küme Randomize Çalışma (Cluster Trial) Örneklem Büyüklüğü (Tasarım Etkisi / ICC):** Güç analizinde kümeleme düzeltmesi (VIF / Design Effect) eksiktir.

---

## 2. Bölüm Bazında Ayrıntılı Karşılaştırma Matrisi

| Bölüm & Başlık | Oxford Handbook Standartları | uSTAT Mevcut Durumu | İlgili Dosya & Satır | Durum |
| :--- | :--- | :--- | :--- | :--- |
| **Bölüm 2 & 11: RCT Tasarımı & Başlangıç Analizi** | Başlangıç ve sonlanım ölçümleri olan RKÇ'lerde 3 yöntem:<br>1. Sadece sonlanım t-testi<br>2. Değişim skoru ($Y_1 - Y_0$) t-testi<br>3. ANCOVA ($Y_1 \sim \text{Grup} + Y_0$) [Altın Standart; Vickers & Altman 2001] | `advanced_anova.py` içinde ANCOVA var; fakat RKÇ'ler için özel rehberli "Başlangıca Göre Değişim / ANCOVA" akışı veya ham fark skorlarına karşı uyarı mekanizması yok. | `backend/routers/advanced_anova.py` | ⚠️ Kısmi / Rehberlik Eksik |
| **Bölüm 2 & 11: Küme Randomize Çalışmalar** | Küme içi korelasyon katsayısı (ICC, $\rho$), Tasarım Etkisi ($\text{VIF} = 1 + (m-1)\rho$), küme düzeltmeli örneklem büyüklüğü hesabı. | Güç analizinde kümeleme (cluster) düzeltmesi yok. Bireysel testler için standart n hesaplanıyor. | `backend/ustat_engine/stats/power.py` | ❌ Eksik |
| **Bölüm 6: Veri Özetleme & Geometrik Ortalama** | Log-normal dağılan klinik verilerde Geometrik Ortalama ($\exp(\frac{1}{n}\sum \ln x_i)$), geometrik standart sapma ve asimetrik %95 GA. Harmonik ortalama. | Aritmetik ortalama ve harmonik ortalama var; **Geometrik ortalama yok**. | `backend/routers/stats/descriptive.py` (s. 187, 515) | ❌ Eksik |
| **Bölüm 8: İki Oran Farkı Güven Aralıkları** | Bağımsız iki grup için oran farkı ($p_1 - p_2$) ve %95 GA (Newcombe/Wilson). | `/two_proportions` içinde Newcombe GA mevcut. | `backend/routers/categorical.py` | ✅ Mevcut |
| **Bölüm 8: McNemar Eşleştirilmiş Oranlar Farkı** | Eşleştirilmiş 2x2 tablolarda oran farkı ($d = (b-c)/n$), standart hata ($\text{SE} = \frac{1}{n}\sqrt{b+c - (b-c)^2/n}$) ve %95 GA. | Yalnızca diskordant OR ($b/c$) ve p değeri dönüyor; oran farkı ve %95 GA yok (`ci_low: None, ci_high: None`). | `backend/routers/categorical.py` (s. 407-408) | ⚠️ Hatalı / Eksik Çıktı |
| **Bölüm 8 & 10: Çarpık Maliyet Verisi & GLM** | Sağlık ekonomisinde ortalama maliyet hesabı aritmetik ortalama gerektirir. Gamma GLM (log link) veya Duan smearing. | Gamma GLM (`log`, `identity`, `inverse` link) eksiksiz mevcut. Katsayılar residual df üzerinden t ile test ediliyor. | `backend/routers/models/glm.py` (s. 265-349) | ✅ Mevcut |
| **Bölüm 9: Tanısal Testler & Fagan Nomogramı** | Duyarlılık, Özgüllük, PPV, NPV (Wilson GA), Pozitif/Negatif Olabilirlik Oranları (Simel log GA), Test öncesi/sonrası olasılıklar. | `services/diagnostic_ci.py` içinde Wilson ve Simel log GA fonksiyonları mevcut. | `backend/services/diagnostic_ci.py` | ✅ Mevcut |
| **Bölüm 10: Ağırlıklı Kappa (Weighted Kappa)** | Sıralı kategorilerde lineer ve kuadratik ağırlıklı kappa; asimptotik standart hata (ASE) ve %95 GA. | Lineer ve kuadratik ağırlıklı kappa hesaplanıyor; ancak **SE, z, p ve %95 GA `None` olarak bırakılmış**. | `backend/routers/stats/correlation.py` (s. 767-770) | ⚠️ Eksik İnferans |
| **Bölüm 10: Bland-Altman Uyum Sınırları GA** | Ortalama fark (bias) için Student t GA; Üst ve alt uyum sınırları (LOA) için $\text{SE}_{\text{LoA}} = s \sqrt{\frac{1}{n} + \frac{1.96^2}{2(n-1)}}$ ile %95 GA. | Tam olarak bu formülle Student t tabanlı hesaplanıyor. Orantısal regresyon eğimi mevcut. | `backend/routers/agreement.py` (s. 56-80) | ✅ Mevcut |
| **Bölüm 10: Ki-Kare Uyum İyiliği Testi (1-way)** | Tek kategorik değişkenin kuramsal oranlara uygunluğu ($\chi^2 = \sum (O-E)^2 / E$, $df = k-1$). | Sadece iki değişkenli çapraz tablo testi (`scipy.stats.chi2_contingency`) var; tek boyutlu uyum testi yok. | `backend/routers/stats/inferential.py` (s. 146) | ❌ Eksik |
| **Bölüm 10: Doğrudan & Dolaylı Standartlaştırma (SMR)** | Farklı yaş piramitlerine sahip popülasyonlarda yaşa göre standartlaştırılmış ölüm/olay hızları (Direct) ve Standardized Mortality Ratio ($SMR = O/E$, Byar Poisson GA). | uSTAT'ta epidemiyolojik standartlaştırma ve SMR hesabı bulunmuyor. | `backend/routers/` (Yok) | ❌ Eksik |
| **Bölüm 10: Aktüeryal Yaşam Tablosu (Life Table)** | Zaman aralıklarına bölünmüş verilerde Cutler-Ederer yöntemiyle etkin risk altındaki kişi sayısı ($l' = l - w/2$) ve kümülatif sağkalım. | Yalnızca sürekli zamanlı Kaplan-Meier var; aralık tabanlı aktüeryal yaşam tablosu yok. | `backend/routers/survival.py` | ❌ Eksik |
| **Bölüm 11: Seri Veri Özet Ölçütler (AUC)** | Boylamsal takiplerde trapezoit kuralıyla hasta başına AUC, pik değer, tepeye ulaşma zamanı, lineer eğim (slope). | Yalnızca RM-ANOVA ve MixedLM var; klinik deneylerde tercih edilen Matthews özet ölçüt çıkarma aracı yok. | `backend/routers/repeated.py` | ❌ Eksik |
| **Bölüm 12: Genelleştirilmiş Tahmin Denklemleri (GEE)** | Tekrarlayan ve kümelenmiş yanıtlarda marjinal modelleme; exchangeable/autoregressive korelasyon yapıları. | GEE Gaussian, Binomial, Poisson aileleri ve AR(1)/Exchangeable/Independence yapılarıyla mevcut. | `backend/routers/models/glm.py` (s. 500-565) | ✅ Mevcut |
| **Bölüm 12: Poisson Regresyon & Takip Süresi (Offset)** | Olay sayılarını kişi-yılı ofsetiyle modelleyerek İnsidans Hız Oranı (IRR) hesaplama; aşırı yayılım kontrolü. | `exposure_col` parametresi ile kişi-yılı ofseti ve Pearson $\chi^2/df$ aşırı yayılım kontrolü mevcut. | `backend/routers/models/glm.py` (s. 100-185) | ✅ Mevcut |
| **Bölüm 13: Meta-Analiz** | Sabit ve rastgele etkiler modelleri, Paule-Mandel/REML $\tau^2$, Hartung-Knapp düzeltmesi, Egger ve Begg testleri. | `backend/ustat_engine/meta.py` içinde DL, PM, REML, Hartung-Knapp, huni grafiği ve regresyon testleri eksiksiz. | `backend/ustat_engine/meta.py` | ✅ Mevcut |

---

## 3. Derinlemesine Metodolojik Bulgular

### 3.1 Klinik Çalışmalarda Başlangıca Göre Değişim (Change from Baseline in RCTs)

Oxford Handbook (Bölüm 11, s. 460–464), bir tedavinin etkinliğini değerlendirirken en sık yapılan metodolojik hataya özel bir alt bölüm ayırmıştır:
- **Hata (Yöntem 2 - Ham Değişim Skoru):** Hastaların sonlanım skoru ile başlangıç skoru arasındaki farkı ($Y_{\text{takip}} - Y_{\text{başlangıç}}$) hesaplayıp iki grup arasında iki örneklem t-testi yapmak veya gruplar içinde eşleştirilmiş t-testi çalıştırmak. Bu yöntem "ortalamaya regresyon" (regression to the mean) yanılgısına ve başlangıç seviyesi dengesizliklerinin oluşturduğu karıştırıcı etkiye (confounding) karşı son derece kırılgandır.
- **Altın Standart (Yöntem 3 - Kovaryat Olarak Başlangıç Değerli Regresyon / ANCOVA):**  
  $$Y_{\text{takip}} = \beta_0 + \beta_1 \cdot \text{Grup} + \beta_2 \cdot Y_{\text{başlangıç}} + \epsilon$$
  Peacock & Peacock, Vickers & Altman (BMJ 2001) ve Matthews (2006) çalışmalarına atıfla ANCOVA'nın istatistiksel gücünün en yüksek olduğunu ve başlangıçtaki şans eseri farklılıkları tamamen düzelttiğini kanıtlamaktadır.
- **uSTAT Durumu:** uSTAT `backend/routers/advanced_anova.py` içinde genel ANCOVA desteğine sahiptir. Ancak arayüzde ve klinik analiz akışlarında araştırmacıyı ham fark skorlarından koruyan, "Tedavi Öncesi / Tedavi Sonrası Karşılaştırma (ANCOVA ile Düzeltilmiş)" adı altında rehberlik eden bir iş akışı bulunmamaktadır.

### 3.2 Boylamsal Verilerde Özet Ölçütler Yaklaşımı (Summary Measures Approach / AUC)

Klinik pratikte hastalar belirli günlerde (örneğin 0, 19, 49, 84, 168. günler) takip edildiğinde, tekrarlayan ölçümler ANOVA'sı (RM-ANOVA) kayıp vizitler ve eşit olmayan zaman aralıkları nedeniyle çökmektedir. Oxford Handbook (Bölüm 11, s. 444–451), Matthews vd. (BMJ 1990) metodolojisini temel alarak her hasta için tek bir özet ölçüt türetilmesini ve grupların bu ölçüt üzerinden karşılaştırılmasını önerir:
1. **Trapezoit Kuralı ile Alan (AUC):**  
   $$\text{AUC} = \sum_{i=1}^{n-1} \frac{1}{2} (t_{i+1} - t_i) (y_i + y_{i+1})$$
   Toplam takip süresine bölünerek standardize edilmiş ağırlıklı ortalama elde edilir.
2. **Pik Yanıt ($Y_{\max}$) ve Pike Ulaşma Zamanı ($T_{\max}$)**
3. **Bireysel Değişim Hızı (Lineer Eğim / Slope $\beta_i$)**
- **uSTAT Durumu:** uSTAT kullanıcıya veri manipülasyonu veya hesaplama araçlarında hasta düzeyinde trapezoit AUC veya eğim türetme seçeneği sunmamaktadır.

### 3.3 Epidemiyolojik Doğrudan ve Dolaylı Standartlaştırma (SMR)

Bölüm 10 (s. 428–431), farklı yaş veya risk profiline sahip hastane/bölge verilerini karşılaştırırken kaba ölüm hızlarının (crude death rate) yanıltıcı olduğunu gösterir:
- **Doğrudan Standartlaştırma (Direct Standardization):** İncelenen popülasyonun yaşa özel ölüm hızları standart bir popülasyona (örneğin DSÖ Dünya Standart Popülasyonu) uygulanarak yaşa göre düzeltilmiş hız hesaplanır.
- **Dolaylı Standartlaştırma (Indirect Standardization & SMR):** İncelenen gruptaki vaka sayısı az olduğunda, standart popülasyonun yaşa özel hızları incelenen gruba uygulanarak "Beklenen Ölüm Sayısı" ($E$) bulunur:
  $$\text{SMR} = \frac{O}{E} \times 100$$
  Byar yaklaşımı veya kesin Poisson dağılımı ile %95 güven aralığı hesaplanır:
  $$\text{Alt GA} = \frac{O}{E} \left(1 - \frac{1}{9O} - \frac{1.96}{3\sqrt{O}}\right)^3, \quad \text{Üst GA} = \frac{O+1}{E} \left(1 - \frac{1}{9(O+1)} + \frac{1.96}{3\sqrt{O+1}}\right)^3$$
- **uSTAT Durumu:** Bu analiz uSTAT'ta yer almamaktadır; epidemiyoloji ve kamu sağlığı araştırmacıları için önemli bir eksikliktir.

### 3.4 McNemar Testinde Eşleştirilmiş Oran Farkı Güven Aralığı Eksikliği

Bölüm 8 (s. 320–323), eşleştirilmiş binary verilerde (örneğin aynı biyopsi örneğinde A patoloğu vs B patoloğu veya tedavi öncesi vs sonrası pozitiflik) yalnızca p değeri ve odds oranının yetersiz olduğunu; temel etkinin **Oran Farkı ($p_1 - p_2$)** ve güven aralığı olduğunu belirtir:
$$d = \frac{b - c}{n}, \quad \text{SE}(d) = \frac{1}{n} \sqrt{b + c - \frac{(b-c)^2}{n}}$$
- **uSTAT İncelemesi (`backend/routers/categorical.py`, s. 404–420):**
  ```python
  es = {"name": "odds_ratio_discordant", "value": round(or_val, 4) if np.isfinite(or_val) else None,
        "ci_low": None, "ci_high": None, "magnitude": ""}
  ```
  Görüldüğü üzere diskordant odds oranı için güven aralığı hesaplanmadığı gibi, klinik olarak asıl yorumlanan eşleştirilmiş oran farkı ($d$) ve Newcombe/Wald güven aralığı çıktı nesnesinde tamamen eksiktir.

### 3.5 Ağırlıklı Kappa (Weighted Kappa) İnferans Boşluğu

Bölüm 10 (s. 412–413), sıralı derecelendirmelerde (örneğin evre 1, 2, 3, 4 tümör derecelemesi) uyumun Ağırlıklı Kappa ile ölçülmesini ve güven aralığı ile raporlanmasını şart koşar.
- **uSTAT İncelemesi (`backend/routers/stats/correlation.py`, s. 767–770):**
  ```python
  if req.weights:
      # Existing normal-theory SE/CI and H0 variance apply to nominal kappa
      # only. Do not relabel those quantities as weighted-kappa intervals.
      ci_low = ci_high = se = se_null = z_stat = p_value = None
  ```
  Nominal kappa formülünün ağırlıklı kappaya uymayacağı doğru şekilde tespit edilmiş, ancak ağırlıklı kappa için Fleiss-Cohen-Everitt standart hatası uygulanmak yerine tüm inferans değerleri `None` yapılarak bırakılmıştır. Kullanıcı ağırlıklı kappa çalıştırdığında p değeri ve güven aralığı alamamaktadır.

---

## 4. Üç Temel Kitabın Karşılaştırmalı Özeti

uSTAT projesi kapsamında incelenen üç eserin odak alanları ve uSTAT'a katkıları:

| Alan | Alpar (2020) | Petrie & Sabin (2020) | Peacock & Peacock (2020) | uSTAT Yol Haritası Önceliği |
| :--- | :--- | :--- | :--- | :--- |
| **Odak Alanı** | Klasik parametrik/nonparametrik testler, SPSS çıktı eşdeğerliği, geçerlik-güvenirlik (ölçek uyarlama). | Klinik tıp araştırmaları, 2x2 etki büyüklükleri (NNT, ARR), tanısal test kesin GA'ları, Bland-Altman. | RKÇ klinik protokolleri, başlangıç düzeltmesi (ANCOVA), boylamsal özet ölçütler (AUC), epidemiyolojik standartlaştırma (SMR). | 3 kitabın kesişimi tamamlandı; Oxford Handbook spesifik klinik araştırma ihtiyaçlarını tanımlıyor. |
| **Eksik Kalan İstatistiki Test** | Tek örneklem işaret/Wilcoxon, Scheffé posthoc, split-half güvenirlik. | Eşdeğerlik/non-inferiority güç analizi (TOST). | 1-way ki-kare uyum iyiliği, Aktüeryal yaşam tablosu (Cutler-Ederer), SMR / Standartlaştırma. | Yüksek |
| **Eksik Kalan Çıktı/GA** | Sıra ortalamaları (MWU), standartlaştırılmış regresyon katsayıları ($\beta$). | Çözüldü (Bölüm 5'te NNT, Breslow-Day, Simel LR, BA GA'ları eklendi). | McNemar eşleştirilmiş oran farkı GA'sı, Ağırlıklı Kappa GA ve p değeri, Geometrik ortalama. | Çok Yüksek (Hemen eklenebilir) |

---

## 5. Eyleme Dönüştürülebilir Geliştirme Önerileri

### 5.1 Öncelik 1: Çıktı ve İnferans Düzeltmeleri (Düşük Efor / Yüksek Etki)
1. **McNemar Eşleştirilmiş Oran Farkı GA'sı (`categorical.py`):**
   - Eşleştirilmiş oran farkı $d = (b-c)/n$ ve standart hatası eklenmeli.
   - Wald veya Newcombe eşleştirilmiş skor güven aralığı hesaplanarak API yanıtındaki `ci_low` ve `ci_high` doldurulmalı.
2. **Ağırlıklı Kappa İçin Standart Hata ve GA (`correlation.py`):**
   - `statsmodels` veya Cicchetti-Allison / Fleiss-Cohen-Everitt formülü kullanılarak ağırlıklı kappa standart hatası, z skoru, p değeri ve %95 güven aralığı hesaplanmalı (`None` döndürülmemeli).
3. **Tanımlayıcı İstatistiklere Geometrik Ortalama Eklenmesi (`descriptive.py`):**
   - Pozitif değerler için $\exp(\text{mean}(\ln(x)))$ hesaplanmalı; sıfır veya negatif değer varlığında bilgilendirici uyarı verilmelidir.

### 5.2 Öncelik 2: Yeni Klinik ve Biyoistatistiksel Yöntemler (Orta Efor)
1. **Tek Örneklem Ki-Kare Uyum İyiliği Testi (Goodness-of-Fit):**
   - Kullanıcının teorik oranlar girebileceği veya eşit dağılım varsayan tek değişkenli `scipy.stats.chisquare(f_obs, f_exp)` testi eklenmeli.
2. **Seri Verilerde Özet Ölçütler (AUC & Slope Çıkarıcı):**
   - Veri yönetimi / hesaplama sekmesine, hasta kimliği ve zaman sütunları seçildiğinde her hasta için trapezoit AUC, pik değer ve eğim türeten bir dönüştürücü eklenmeli.
3. **Klinik Çalışma Başlangıç Düzeltmesi (Baseline Adjustment / ANCOVA Sihirbazı):**
   - Tedavi öncesi ve sonrası ölçümler seçildiğinde, kullanıcıyı ham değişim skorları yerine ANCOVA'ya yönlendiren ve başlangıç dengesini görselleştiren bir analiz paneli eklenmeli.

### 5.3 Öncelik 3: Epidemiyolojik İleri Yöntemler (Geniş Kapsamlı)
1. **Doğrudan ve Dolaylı Standartlaştırma (SMR) Modülü:**
   - Standart popülasyon ağırlıkları kütüphanesi (DSÖ Dünya Standart Nüfusu, Avrupa Standart Nüfusu) ile yaşa göre düzeltilmiş hızlar ve SMR hesaplayıcı modülü.
2. **Küme Randomize Çalışma Güç Analizi:**
   - Tasarım etkisi (Design Effect = $1 + (m-1)\rho$) çarpanı `ustat_engine/stats/power.py` modülüne opsiyonel parametre olarak eklenmeli.
