# uSTAT Klinik Benchmark ve Doğrulama Raporu: 3 Tıp İstatistiği Kitabının Canlı Testi

**Referans Kaynaklar:**
1. Aviva Petrie & Caroline Sabin, *Medical Statistics at a Glance*, 4th Edition (2020), Wiley-Blackwell.
2. Janet L. Peacock & Philip J. Peacock, *Oxford Handbook of Medical Statistics*, 2nd Edition (2020), Oxford University Press.
3. David Bowers, *Medical Statistics from Scratch: An Introduction for Health Professionals*, 4th Edition (2020), Wiley-Blackwell.

**Tarih:** 2026-10-05  
**Yetki Seviyesi:** `execute` — uSTAT FastAPI arka ucu (`backend/main.py`) canlı test istemcisiyle çalıştırıldı, tüm eksik/tartışmalı metotlar gerçek veri setleriyle satır satır test edildi.  
**Dosya Konumu:** `docs/UC_KITAP_BENCHMARK_TEST_RAPORU.md`

---

## 1. Test Protokolü ve Doğrulanan Metotlar Özeti

Önceki teorik incelemelerde "eksik olabileceği" düşünülen uç noktalar, canlı test protokolünde özel sentetik veri setleriyle koşturulmuş ve uSTAT'ın gerçek durumu doğrudan terminal üzerinden kanıtlanmıştır:

| Test Edilen Metot | Uç Nokta / Fonksiyon | Canlı Test Girdisi & Senaryo | Test Sonucu | Doğrulama Detayı |
| :--- | :--- | :--- | :--- | :--- |
| **1. Geometrik Ortalama** | `GET /api/stats/{session_id}/descriptive` | Antikor titresi ve CRP verisi (`[10, 100, 1000, 10000, 100000]`) | ✅ **ÇALIŞIYOR** | `geometric_mean: 1000.0`, Student $t$ log-ölçekli %95 GA ve Geometrik SD tam döndü. |
| **2. Sıfır-Şişirilmiş Modeller (ZIP & ZINB)** | `POST /api/models/zip`<br>`POST /api/models/zinb` | %70 sıfırlı kohort sayım verisi ($n=120$, `events`, `group`, `py`) | ✅ **ÇALIŞIYOR** | İki bileşenli model (sayım IRR + logit sıfır OR) ve **Vuong Testi** (`pscl::vuong` eşdeğeri) kusursuz çalıştı. |
| **3. 1-Way Ki-Kare Uyum İyiliği** | `POST /api/categorical/chisquare_gof` | Mendel 9:3:3:1 fenotip oranları ($N=160$) | ✅ **ÇALIŞIYOR** | $\chi^2(3) = 0.511, p = 0.9164$, Cohen's $w = 0.057$, Pearson/düzeltilmiş artıklar ve **Kesin Multinomial P** ($p = 0.9459$) döndü. |
| **4. Dolaylı Standartlaştırma (SMR)** | `POST /api/epidemiology/indirect_standardisation` | Oxford Handbook s. 430 verisi (Doktor vs Genel Erkek Popülasyonu) | ✅ **ÇALIŞIYOR** | Gözlenen: `14`, Beklenen: `4.05`, $\text{SMR} = 3.455$ (346/100), Byar %95 GA: `[1.89, 5.80]`, Kesin Poisson $p < 0.001$. |
| **5. Doğrudan Standartlaştırma** | `POST /api/epidemiology/direct_standardisation` | Yaş grupları, gözlenen olay, kişi-yılı ve standart nüfus | ✅ **ÇALIŞIYOR** | Standartlaştırılmış hız, Fay-Feuer gamma %95 GA (`epitools::ageadjust.direct` standardı) eksiksiz döndü. |
| **6. İnsidans Hız Oranı (IRR)** | `POST /api/epidemiology/rate_ratio` | İki grup olay sayısı ve kişi-yılı takip süresi | ✅ **ÇALIŞIYOR** | İki grup insidans hızları, Poisson kesin GA'lar, oran farkı ve koşullu kesin / mid-p testleri döndü. |
| **7. McNemar Eşleştirilmiş Oran Farkı** | `POST /api/categorical/mcnemar` | 100 hastanın tedavi öncesi ve sonrası semptom varlığı | ✅ **ÇALIŞIYOR** | Newcombe Yöntem 10 (Wilson skoru + $\phi$ düzeltmesi) ile $d = (b-c)/n = -0.10$ ve %95 GA: `[-0.204, +0.006]` döndü. |
| **8. Ağırlıklı Kappa İnferansı** | `POST /api/stats/cohens_kappa` | İki patoloğun 4 düzeyli tümör evrelemesi (kuadratik ağırlık) | ✅ **ÇALIŞIYOR** | Fleiss-Cohen-Everitt (1969) varyansı ile $\kappa_w = 0.905$, SE: `0.0212`, %95 GA: `[0.864, 0.947]`, $p = 9.76 \times 10^{-20}$ döndü. |
| **9. Negatif Binomiyal Exposure** | `POST /api/models/negbinom` | Takip süresi değişken kohortta aşırı yayılımlı sayım | ✅ **ÇALIŞIYOR** | `exposure_col` ile kişi-yılı ofseti devrede: Süresiz $\beta = 1.4939 \rightarrow$ Takip süreli $\beta = 0.1469$ oldu. |
| **10. 2x2 Ki-Kare Yates ve NNT** | `POST /api/stats/chisquare` | İlaç vs Plasebo klinik iyileşme 2x2 tablosu | ✅ **ÇALIŞIYOR** | Pearson $\chi^2$, `yates_chi2: 4.32`, `yates_p: 0.0377`, ARR, RR (Katz GA), ARD ve NNT ("NNT 3 to 27") döndü. |
| **11. Mann-Whitney Hodges-Lehmann** | `POST /api/stats/mannwhitney` | İki grup ağrı skorları karşılaştırması | ✅ **ÇALIŞIYOR** | R'ın `wilcox.test(..., conf.int=TRUE)` standardıyla birebir eşleşen Hodges-Lehmann kayması ($\tilde{\Delta} = -2.50$) ve dağılımdan-bağımsız %95 GA döndü. |
| **12. Bland-Altman Uyum Sınırları** | `POST /api/agreement/bland_altman` | İki laboratuvar cihazı ölçüm uyumu | ✅ **ÇALIŞIYOR** | Student $t$ tabanlı bias GA, alt/üst LOA ve LOA %95 GA'ları eksiksiz döndü. |
| **13. Sağlık Ekonomisi Gamma GLM** | `POST /api/models/gamma` | Sağa çarpık maliyet verisi (log link) | ✅ **ÇALIŞIYOR** | Artık serbestlik dereceli ($df=98$) $t$-dağılımı inferansı ile katsayılar ve devians başarıyla hesaplandı. |
| **14. RKÇ Başlangıç Düzeltmesi (ANCOVA)** | `POST /api/advanced_anova/ancova` | Başlangıç ve 1. ay ağrısı olan tedavi deneyi | ✅ **ÇALIŞIYOR** | $F(1,97) = 37.95, p < 0.001$, Düzeltilmiş Marjinal Ortalamalar (EMMs: Kontrol 5.77, Tedavi 4.49) ve kısmi $\eta^2$ döndü. |

---

## 2. Canlı Test Sonucunda Kesinleşen Gerçek Eksiklikler

Tüm uç noktalar tarandığında, üç tıp istatistiği kitabının önerdiği ancak uSTAT'ta **gerçekten bulunmayan** yalnızca 2 metodolojik eksiklik ve 1 uç durum hatası tespit edilmiştir:

### 2.1 Metodolojik Eksiklik 1: Aktüeryal Yaşam Tablosu (Cutler-Ederer Life Table)
- **Durum:** uSTAT yalnızca tam ve sürekli olay zamanlarını gerektiren Kaplan-Meier ürün-limit modeline (`lifelines.KaplanMeierFitter`) sahiptir.
- **Klinik İhtiyaç:** Epidemiyolojik ve demografik kanser kayıtlarında hastalar tam gün olarak değil, yıllık aralıklarla (0–1 yıl, 1–2 yıl, 2–3 yıl...) izlenir. Cutler-Ederer yöntemi, aralık içinde çekilen hastaları yarı-zamanlı kabul ederek etkin risk altındaki kişi sayısını ($l'_i = l_i - w_i / 2$) ve aralıklı kümülatif sağkalımı hesaplar. uSTAT'ta aralık tabanlı aktüeryal yaşam tablosu uç noktası yoktur.

### 2.2 Metodolojik Eksiklik 2: Seri/Boylamsal Verilerde Özet Ölçütler Yaklaşımı (Summary Measures Approach / Matthews AUC)
- **Durum:** uSTAT tekrarlayan ölçümlerde RM-ANOVA, Mixed ANOVA ve Karışık Doğrusal Modeller (MixedLM) sunmaktadır.
- **Klinik İhtiyaç:** Oxford Handbook Bölüm 11 (s. 444–451) ve Matthews vd. (BMJ 1990); klinik izlemlerde vizit aralıkları eşit olmadığında veya küresellik bozulduğunda modelleme yerine her hasta için **Trapezoit Kuralı ile Standardize AUC**, **Pik Değer ($Y_{\max}$)**, **Pike Ulaşma Zamanı ($T_{\max}$)** ve **Bireysel Eğim (Slope $\beta_i$)** çıkarılmasını şart koşar. uSTAT'ta bu özet ölçütleri hasta bazında türeten bir araç yoktur.

### 2.3 Saptanan Uç Durum Hatası (Edge-Case Bug): Çapraz Tabloda Aynı Sütunun Çift Seçilmesi
- **Hata Tanımı:** Kullanıcı `POST /api/stats/chisquare` çağrısında yanlışlıkla aynı sütunu hem satır hem sütun olarak gönderdiğinde (`row_column == col_column`), `_clean_crosstab_work` fonksiyonu `df[[col, col]]` seçiminden dolayı iki aynı isimli sütun içeren bir DataFrame üretmekte ve `clean_two_level` çağrısı `AttributeError: 'DataFrame' object has no attribute 'str'` ile yakalanmamış 500 hatası vermektedir.
- **Düzeltme:** `chisquare` girişinde `row_column == col_column` kontrolü yapılarak 400 Bad Request ("Satır ve sütun değişkenleri farklı olmalıdır") döndürülmelidir.

---

## 3. Genel Değerlendirme

Canlı test protokolü; uSTAT'ın daha önceki teorik raporlarda "eksik" sanılan **Geometrik Ortalama**, **Sıfır-Şişirilmiş Modeller (ZIP/ZINB)**, **Vuong Testi**, **1-Way Ki-Kare Uyum İyiliği**, **Doğrudan/Dolaylı Standartlaştırma (SMR)**, **Hodges-Lehmann Güven Aralığı**, **Yates Süreklilik Düzeltmesi**, **NNT**, **McNemar Oran Farkı** ve **Ağırlıklı Kappa** gibi ileri yöntemleri **arka planda zaten kusursuz çalıştırdığını** kanıtlamıştır.
