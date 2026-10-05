# uSTAT ve *Medical Statistics from Scratch* (Bowers, 4. Baskı) Karşılaştırmalı İnceleme Raporu

**Referans Kaynak:** David Bowers, *Medical Statistics from Scratch: An Introduction for Health Professionals*, 4th Edition (2020), John Wiley & Sons Ltd.  
**Kapsam ve Bağlam:** *Medical Statistics at a Glance* (Petrie & Sabin) ve *Oxford Handbook of Medical Statistics* (Peacock & Peacock) raporlarıyla birlikte değerlendirilmiştir.  
**Tarih:** 2026-10-05  
**Durum:** Rapor kod ile doğrulandı, saptanan gerçek eksiklikler giderildi (Bölüm 4).

---

## 1. Doğrulama Sonucu (kod ile karşılaştırma)

İlk sürümdeki her iddia güncel kodla karşılaştırıldı.

### 1.1 Yanlış çıkan iddia

| İlk iddia | Gerçek durum | Kod |
| :--- | :--- | :--- |
| 2x2 ki-karede Yates düzeltmesi gösterilmiyor | 2x2 tabloda `yates_chi2` ve `yates_p` zaten dönüyor; `correction=False` yalnızca Pearson satırı için | `routers/stats/inferential.py` |

### 1.2 Doğru çıkan iddialar ve giderilmesi

| Bulgu | Önceki durum | Şimdiki durum |
| :--- | :--- | :--- |
| Negatif binomiyalde exposure/offset | `NegBinomRequest` içinde yok | Eklendi (`exposure_col`) |
| ZIP ve ZINB modelleri | Yok | Eklendi (`/api/models/zip`, `/zinb`) |
| Vuong testi | Yok | Eklendi (ham, AIC ve BIC düzeltmeli) |
| Hodges-Lehmann medyan farkı ve %95 GA | Yok | Eklendi (Mann-Whitney, eşleştirilmiş ve tek örneklem Wilcoxon) |
| McNemar eşleştirilmiş oran farkı ve GA | Yok, uyumsuz OR için `ci_low/high: None` | Eklendi (Newcombe yöntem 10, uyumsuz OR için kesin GA) |
| Ağırlıklı kappa SE, z, p, GA | Tamamı `None` | Eklendi (Fleiss-Cohen-Everitt varyansı) |
| Doğrudan/dolaylı standartlaştırma, SMR | Yok | Eklendi (`/api/epidemiology/*`) |
| Hız oranı ve hız farkı (kişi-yılı) | Yok | Eklendi (`/rate_ratio`) |

### 1.3 Bu turda yapılmayanlar

- Seri veri özet ölçütleri (trapezoit AUC, pik değer, eğim): yok.
- RKÇ başlangıca göre düzeltme (ANCOVA) sihirbazı: genel ANCOVA var (`routers/advanced_anova.py`), yönlendirici arayüz akışı yok.
- Yeni alanların ve uç noktaların arayüzde gösterimi (frontend): henüz yok, yalnızca API ve `result_text` düzeyinde.
- Vaka ölüm oranı (CFR) ve bebek ölüm oranı (IMR) için ayrı uç nokta yok; `/rate_ratio` ve doğrudan standartlaştırma bu hızları genel biçimde hesaplayabilir.

---

## 2. Bowers ve Önceki İki Kitabın Işığında Karşılaştırma

### 2.1 Sayma verisi regresyon hiyerarşisi [Bowers Bölüm 24]

Poisson, negatif binomiyal, ZIP, ZINB sıralaması ve modeller arası tercih testi artık uSTAT'ta mevcuttur.

- **Poisson ve Negatif Binomiyal:** ikisi de `exposure_col` ile kişi-yılı / takip süresini modele `exposure` olarak alır; IRR hız oranı olur. Negatif binomiyalde exposure hem ortak ML uyumuna (`alpha`) hem de katsayı tablosunu üreten GLM uyumuna aktarılır.
- **ZIP ve ZINB** (`backend/routers/models/count.py`): sayım kısmı (log IRR, IRR, GA) ve sıfır-şişirme kısmı (logit, OR, GA) ayrı tablolarda döner. `inflation_predictors` verilmezse sıfır-şişirme yalnızca sabit terimle kurulur. ZINB için `alpha` ve `theta` raporlanır. Gözlenen sıfır sayısı, modelin ve standart modelin beklediği sıfır sayısı ile birlikte verilir.
- **Vuong testi:** ZIP için Poisson, ZINB için negatif binomiyal ile gözlem düzeyinde log-olabilirlik farkları üzerinden hesaplanır; pscl'deki AIC ve BIC düzeltmeleri uygulanır, `vuong.preferred` AIC düzeltmeli istatistiğe göre `zero_inflated`, `standard` veya `neither` olur.
- **Uyum notu:** ZINB, veri Poisson'a yakınsa statsmodels'in varsayılan optimize edicisinde başarısız olabiliyor. Birden çok optimize edici denenir, alpha > 0 olan en iyi yakınsamış çözüm seçilir.

### 2.2 Nonparametrik konum farkı [Bowers Bölüm 15]

- Mann-Whitney: tüm ikili farkların (grup 1 eksi grup 2) medyanı, Hodges-Lehmann tahmin edicisi olarak verilir.
- Eşleştirilmiş ve tek örneklem Wilcoxon: farkların Walsh ortalamalarının medyanı.
- %95 GA R'ın `wilcox.test(conf.int = TRUE)` kuralını izler: ikisi de 50'den küçük ve bağ yoksa kesin sıra istatistikleri, aksi halde bağ ve süreklilik düzeltmeli normal yaklaşım. 13 rastgele veri kümesinde GA sınırları R ile 2e-4 içinde uyuştu.
- 4 milyondan fazla çift için sayısal alanlar açıklayıcı notla `null` döner, örnekleme yapılmaz.
- Mann-Whitney isteği `alpha` alanı içermediği için %95 kullanılır.

### 2.3 Yates düzeltmeli ki-kare [Bowers Bölüm 18]

Zaten mevcuttur (bkz. 1.1). İlk sürümdeki "eksik" iddiası hatalıydı.

### 2.4 Eşleştirilmiş oranlar farkı [Bowers Bölüm 15, Oxford Handbook Bölüm 8]

- `/mcnemar` yanıtına `paired_difference` eklendi: d = (b - c)/n, yani `col1` pozitif oranı eksi `col2` pozitif oranı. GA, Newcombe (1998) yöntem 10'dur; `note` alanı hangi sütunun hangisi olduğunu belirtir.
- Uyumsuz OR (`odds_ratio_discordant`) için `ci_low` ve `ci_high` artık b/(b+c) üzerinde kesin binom aralığının dönüşümüdür. c = 0 veya b + c = 0 ise `None` kalır.
- R'da `prop.test` (Wilson) ve `binom.test` ile kurulan karşılaştırma 1e-6 içinde uyuştu.

### 2.5 Ağırlıklı kappa [Bowers Bölüm 21, Oxford Handbook Bölüm 10]

- Doğrusal ve kuadratik ağırlıklı kappa için Fleiss-Cohen-Everitt (1969) asimptotik varyansı uygulandı; `se`, `ci_low`, `ci_high`, `se_null`, `z`, `p` ve ağırlıklı `po`, `pe` dolduruluyor.
- `psych::cohen.kappa` ve `DescTools::CohenKappa` ile SE ve GA 1e-6 içinde uyuşuyor. Kappa değeri hâlâ scikit-learn'ündür.
- Özdeşlik ağırlıklarında null SE ve z, mevcut ağırlıksız uç noktayla tam aynıdır.
- **Bilinen küçük fark:** ağırlıksız yolun `se` değeri Cohen'in basitleştirilmiş formülüdür (`sqrt(po(1-po)/(n(1-pe)^2))`), tam varyans formülü ise psych'in ağırlıksız değerini verir (deneme tablosunda 0.06839 ve 0.06872). Mevcut değerler değiştirilmediği için 2x2 tabloda `weights="linear"` ile `weights=none` yaklaşık %0.5 farklı SE verir. İsterseniz ağırlıksız SE tam formüle çekilebilir.

### 2.6 Epidemiyolojik standartlaştırma ve hızlar [Bowers Bölüm 7]

Uç noktalar `/api/epidemiology` altındadır ve oturumdaki veri çerçevesi yerine satır içi tabaka tablosu alır.

- **`/direct_standardisation`:** kaba hız (kesin Poisson GA), doğrudan yaşa standartlaştırılmış hız ve Fay-Feuer gamma GA'sı (`epitools::ageadjust.direct` ile aynı formül), tabaka tablosu (hız, GA, ağırlık, katkı yüzdesi).
- **`/indirect_standardisation`:** gözlenen O, beklenen E, SMR (ve x100 biçimi), Byar yaklaşık GA, kesin Poisson GA, H0: SMR = 1 için kesin iki yönlü Poisson p değeri. O = 0 ise alt sınır 0, O < 10 ise uyarı.
- **`/rate_ratio`:** iki grubun hızları (kesin GA), hız oranı (log GA), hız farkı (Wald GA), koşullu kesin binom p değeri, mid-p ve Wald p; ayrıca bir grupta olay sıfırsa da çalışan Clopper-Pearson tabanlı hız oranı GA'sı.
- Girdi doğrulaması 422 ve açık mesajla döner. `reference_rate` varsayılan olarak ham hızdır (`reference_multiplier = 1`); tabloda 100.000 başına hız kullanılıyorsa çarpan girilmelidir.
- `epitools` kurulu olmadığı için `ageadjust.direct` ile karşılaştırma testi atlanıyor; ilk kurulumda doğrulanmalıdır.

---

## 3. Üç Kitabın Sentez Matrisi (güncel)

| Metodolojik Konu | Petrie & Sabin | Peacock & Peacock | Bowers | uSTAT durumu |
| :--- | :--- | :--- | :--- | :--- |
| ZIP ve ZINB, Vuong testi | Değinilmedi | Kısa bahis | Ayrıntılı (Böl. 24) | ✅ Eklendi |
| Negatif binomiyalde exposure | Yok | Önerildi | Gerekli (Böl. 24) | ✅ Eklendi |
| Hodges-Lehmann farkı ve GA | Bahsedildi | Tabloda verildi | Vurgulandı (Böl. 15) | ✅ Eklendi |
| Yates düzeltmeli ki-kare (2x2) | Opsiyonel | Değinildi | SPSS standardı (Böl. 18) | ✅ Zaten vardı |
| McNemar oran farkı ve GA | Vurgulandı | Formül verildi | Ayrıntılı (Böl. 15) | ✅ Eklendi |
| Ağırlıklı kappa SE ve GA | Bahsedildi | Standart kabul | Gerekli (Böl. 21) | ✅ Eklendi |
| Standartlaştırma ve SMR | İşlendi | Ayrıntılı (Böl. 10) | Özel bölüm (Böl. 7) | ✅ Eklendi |
| Seri veri özet ölçütleri (AUC) | Yok | Özel bölüm (Böl. 11) | Yok | ❌ Yok |
| RKÇ başlangıç düzeltmesi (ANCOVA) | Bahsedildi | Özel bölüm (Böl. 11) | Bahsedildi | ⚠️ ANCOVA var, rehberlik yok |
| Tanısal testler (Wilson GA, Simel LR, ROC) | Tam bölüm | Tam bölüm | Tam bölüm (Böl. 27) | ✅ Çözüldü |
| Bland-Altman uyum sınırları ve GA | Tam bölüm | Tam bölüm | Tam bölüm (Böl. 21) | ✅ Çözüldü |
| Maliyet verisi Gamma GLM | Yok | Özel bölüm (Böl. 10) | Yok | ✅ Zaten vardı (`/gamma`) |

---

## 4. Uygulanan Düzeltmeler (2026-10-05)

### 4.1 Değişen dosyalar

| Dosya | Değişiklik |
| :--- | :--- |
| `backend/routers/models/glm.py` | `/negbinom` `exposure_col`, `rate_model`, `result_text` |
| `backend/routers/models/count.py` (yeni), `routers/models/__init__.py` | `/zip`, `/zinb`, Vuong testi |
| `backend/services/hodges_lehmann.py` (yeni) | İki örneklem, tek örneklem ve eşleştirilmiş HL tahmini ve GA |
| `backend/routers/stats/nonparametric.py`, `backend/routers/repeated.py` | Mann-Whitney ve Wilcoxon yanıtlarına `hodges_lehmann` |
| `backend/routers/categorical.py` | McNemar `paired_difference` ve uyumsuz OR GA'sı |
| `backend/services/weighted_kappa.py` (yeni), `backend/routers/stats/correlation.py` | Ağırlıklı kappa SE, GA, z, p |
| `backend/services/standardisation.py` (yeni), `backend/routers/epidemiology.py` (yeni), `backend/main.py` | Doğrudan/dolaylı standartlaştırma, SMR, hız oranı |

### 4.2 Testler

Yeni test dosyaları: `test_negbin_exposure.py`, `test_zero_inflated.py`, `test_hodges_lehmann.py`, `test_mcnemar_paired_ci.py`, `test_weighted_kappa_se.py`, `test_standardisation.py`. `test_stats_new_methods.py` içindeki bir assertion, ağırlıklı kappa için eski `None` davranışını kodluyordu; güncellendi. Tüm backend paketi: 2482 test geçti, 1 atlandı; tek başarısızlık `test_anova_scheffe_table`, yerelde `scikit_posthocs` paketi kurulu olmadığı için (`requirements.txt` içinde tanımlı, CI'da kurulur).

### 4.3 Kalan iş listesi

1. Frontend: `paired_difference`, `hodges_lehmann`, ağırlıklı kappa SE ve GA, `/zip` ve `/zinb` paneli ile Vuong sonucu, negatif binomiyal ve Poisson için takip süresi seçici, epidemiyoloji paneli (stratum tablosu girişi).
2. Seri veri özet ölçütleri ve başlangıç düzeltmeli ANCOVA sihirbazı.
3. Ağırlıksız kappa SE'sinin tam varyans formülüne çekilmesi (isteğe bağlı).
4. `epitools`, `pscl`, `exact2x2` kurulduğunda R pariteli doğrulama.
