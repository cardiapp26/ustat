# uSTAT ve *Medical Statistics at a Glance* (Petrie & Sabin, 4. Baskı) Karşılaştırmalı İnceleme Raporu

**Referans Kaynak:** Aviva Petrie & Caroline Sabin, *Medical Statistics at a Glance*, 4th Edition (2020), Wiley-Blackwell.  
**Tarih:** 2026-10-05  
**Durum:** Rapor kod ile doğrulandı, saptanan gerçek eksiklikler giderildi (Bölüm 5).

---

## 1. Doğrulama Sonucu (kod ile karşılaştırma)

İlk sürümdeki her iddia güncel kodla karşılaştırıldı.

### 1.1 Yanlış çıkan iddialar (eksiklik yok, rapordan çıkarıldı)

| İlk iddia | Gerçek durum | Kod |
| :--- | :--- | :--- |
| `/one_proportion` Wald aralığı kullanıyor | Wilson aralığı kullanıyor (`proportion_confint(method="wilson")`) | `routers/categorical.py` |
| `/two_proportions` fark GA'sı dönmüyor | `diff_prop` ve Newcombe %95 GA (`ci_diff_low/high`) zaten dönüyor | `routers/categorical.py` |
| Durbin-Watson yok | Mevcut | `services/assumptions.py`, `routers/diagnostics.py` |
| Cook uzaklığı / etkili gözlem taraması yok | Mevcut (Cook D, eşik sayımı) | `routers/diagnostics.py` |
| Güç motorunda kayıp (attrition) düzeltmesi yok | Mevcut (`N / (1 - L)`) | `ustat_engine/stats/power.py` |

### 1.2 Doğru çıkan iddialar

| Bulgu | Önceki durum | Şimdiki durum |
| :--- | :--- | :--- |
| NNT / NNH / ARR / RRR | Yok (chi-square 2x2'de yalnızca RR) | Eklendi (Bölüm 5.1) |
| Mantel-Haenszel homojenlik testi | Yok | Eklendi (Breslow-Day, Tarone düzeltmeli) |
| Poisson offset / exposure | Yok | Eklendi (`exposure_col`) |
| Poisson aşırı yayılım kontrolü | Yok | Eklendi (Pearson χ² / df) |
| Tanı testi duyarlılık, özgüllük, PPV, NPV, LR GA'ları | Yok | Eklendi (Wilson, Simel) |
| Bland-Altman uyum sınırı GA'ları | Yok | Eklendi |
| Bland-Altman ortalama fark GA'sında sabit 1.96 | Evet | Student t'ye çevrildi |
| Passing-Bablok yerine Theil-Sen | Evet | Gerçek Passing-Bablok (1983) |
| KM medyan sağkalım GA'sı | Yok | Eklendi |

### 1.3 Doğru ama bu turda yapılmayanlar

- Eşdeğerlik / non-inferiority örneklem büyüklüğü (TOST güç hesabı): `power.py` içinde yok.
- Tekrarlanabilirlik katsayısı (RC = 2.77 s_w): yok.
- Küme randomize çalışmalar için tasarım etkisi (Design Effect) hesaplayıcısı: yok.
- Standartlaştırılmış beta ve Mann-Whitney sıra ortalamaları (Alpar kitabı karşılaştırması): ayrıca kontrol edilmedi.
- Yeni alanların arayüzde gösterimi (frontend): henüz yok, yalnızca API ve `result_text` düzeyinde.

---

## 2. Kitap Bölümleriyle Karşılaştırma (düzeltilmiş)

### Bölüm 23, 24 ve 40: Kategorik veriler, iki oran, EBM

| Standart | uSTAT durumu |
| :--- | :--- |
| Tek oran GA (Wilson) | Var |
| İki oran farkı GA (Newcombe) | Var |
| ARR, RRR, RR GA'sı (Katz) | Eklendi (`risk_measures`) |
| NNT / NNH ve Altman (1998) aralığı | Eklendi; GA sıfırı kapsarsa iki parçalı aralık (`ci_spans_zero`) |

### Bölüm 31 ve 32: Hızlar, Poisson, GLM

- `exposure_col` ile kişi-yılı / takip süresi modele `exposure` olarak verilir; IRR hız oranı olur.
- Pearson dispersiyon (`dispersion`, `overdispersed` > 1.5) ve negatif binom / robust SE önerisi raporlanır.

### Bölüm 34: Tabakalı analiz

- CMH ve ortak OR vardı; Breslow-Day (Tarone düzeltmeli) `homogeneity_test` olarak eklendi. p < alpha ise ortak OR'nin yanıltıcı olabileceği uyarısı döner. Simetrik tabakalarda statsmodels NaN döndürdüğü için bu durumda alan `null` ve açıklama notu verilir.

### Bölüm 35: Varsayımlar

- Durbin-Watson, Cook uzaklığı, normallik, eşvaryanslılık, doğrusallık kontrolleri mevcuttur (ilk sürümdeki "eksik" iddiası hatalıydı).

### Bölüm 36: Örneklem büyüklüğü

- Güç motoru kayıp düzeltmesini destekler. Eşdeğerlik ve non-inferiority örneklem büyüklüğü eksik kalmıştır.

### Bölüm 38: Tanı araçları

- Duyarlılık, özgüllük, PPV, NPV, doğruluk için Wilson GA; LR+ ve LR- için Simel log-yöntemi GA'sı eklendi (optimal ve manuel kesim noktası için; ROC eğrisinin ~300 noktasına eklenmedi, performans nedeniyle).
- İsteğe bağlı `prevalence` ile Fagan yaklaşımı (`post_test`: ön olasılık, pozitif ve negatif sonuç sonrası olasılık) eklendi.

### Bölüm 39: Uyum

- Bland-Altman: ortalama fark GA'sı artık Student t ile; alt ve üst uyum sınırı GA'ları (Bland & Altman: SE = s·√(1/n + 1.96²/(2(n-1)))) eklendi. Uyum sınırlarının kendisi standart olarak ortalama ± 1.96 SD kalmıştır.
- Passing-Bablok: Theil-Sen yaklaşımı kaldırıldı; Passing & Bablok (1983) algoritması (kaydırılmış medyan, C_γ sınırları, Cusum doğrusallık testi) uygulandı. R `mcr` paketi kurulu olmadığı için R ile karşılaştırma yapılamadı; makaleden bağımsız yazılmış döngü uygulaması ve kesin doğru testleriyle doğrulandı.
- Tekrarlanabilirlik katsayısı eksik.

### Bölüm 41 ve 42: Kümelenmiş veri

- GEE ve LMM mevcut; Design Effect hesaplayıcısı eksik.

### Bölüm 44: Sağkalım

- KM medyan sağkalım için %95 GA (`median_survival_ci_low/high`, lifelines `median_survival_times`) eklendi (`cox.py` ve landmark servisi). Medyan ulaşılmadıysa `null`.

---

## 3. Alpar ve Petrie & Sabin Karşılaştırması (güncel)

| Alan | Alpar | Petrie & Sabin | uSTAT |
| :--- | :--- | :--- | :--- |
| NNT / NNH / ARR / RRR | İkincil | Temel EBM | Var |
| Poisson takip süresi | Teorik | Zorunlu | Var |
| Tanı testi GA'ları | Nokta formülleri | STARD standardı | Var |
| Bland-Altman LoA GA | Formül ve grafik | Altman-Bland 1986 | Var |
| Passing-Bablok | Bahsedilmez | Laboratuvar standardı | Var (gerçek algoritma) |
| KM medyan GA | Özet | Klinik standart | Var |
| CMH homojenlik | Kısmi | Zorunlu | Var |
| Eşdeğerlik örneklem büyüklüğü | Bahsedilmez | Var | Eksik |

---

## 4. Kalan İş Listesi

1. Frontend: `risk_measures`, `homogeneity_test`, ROC CI'ları, `post_test`, LoA CI'ları, Poisson `exposure_col` ve dispersiyon, KM medyan GA alanlarının panellerde gösterimi ve Poisson için takip süresi seçici.
2. TOST / non-inferiority örneklem büyüklüğü (`ustat_engine/stats/power.py`).
3. Tekrarlanabilirlik katsayısı ve Design Effect hesaplayıcısı.
4. Passing-Bablok için R `mcr` ile pariteli doğrulama (paket kurulunca).

---

## 5. Uygulanan Düzeltmeler (2026-10-05)

### 5.1 Değişen dosyalar

| Dosya | Değişiklik |
| :--- | :--- |
| `backend/services/risk_measures.py` (yeni) | ARD, RR (Katz), RRR, NNT/NNH (Altman) yardımcıları |
| `backend/routers/categorical.py` | `/two_proportions` `risk_measures`; `/mantel_haenszel` `homogeneity_test` |
| `backend/routers/stats/inferential.py` | `/chisquare` 2x2 `risk_measures`, isteğe bağlı `alpha` |
| `backend/routers/models/glm.py` | Poisson `exposure_col`, `dispersion`, `overdispersed` |
| `backend/routers/agreement.py` | Bland-Altman t-GA ve LoA GA; gerçek Passing-Bablok |
| `backend/services/diagnostic_ci.py` (yeni) | Wilson, Simel, Fagan |
| `backend/routers/stats/nonparametric.py` | ROC metrik GA'ları, `prevalence`, `post_test` |
| `backend/services/km_median_ci.py` (yeni) | KM medyan GA yardımcısı |
| `backend/routers/models/cox.py`, `backend/services/survival_advanced_service.py` | `median_survival_ci_low/high` |

### 5.2 Testler

Yeni test dosyaları: `test_ebm_risk_measures.py`, `test_diagnostic_ci.py`, `test_km_median_ci.py`, `test_agreement_corrections.py`, `test_poisson_exposure.py`. Tüm backend paketi (`test_stats_new_methods.py` hariç): 2327 test geçti. `test_stats_new_methods.py` bu işten bağımsız olarak `ModuleNotFoundError` veriyor (başka bir çalışmanın izlenmeyen dosyası).

### 5.3 Notlar

- statsmodels 0.14.4, `exposure` bir pandas Series iken NaN katsayı döndürüyor; uç nokta numpy dizisi geçirir ve katsayı sonlu değilse 400 döner.
- Bland-Altman uyum sınırları ortalama ± 1.96 SD olarak bırakıldı (standart tanım); yalnızca ortalama fark GA'sında t çarpanı kullanılır.
