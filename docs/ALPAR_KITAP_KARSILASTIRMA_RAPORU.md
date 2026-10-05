# uSTAT Biyoistatistik Analizleri Değerlendirme Raporu

**Referans kaynak:** Prof. Dr. Reha Alpar, *Spor, Sağlık ve Eğitim Bilimlerinden Örneklerle Uygulamalı İstatistik ve Geçerlik - Güvenirlik* (Detay Yayıncılık, 641 sayfa PDF).
**Tarih:** 2026-10-05
**Yetki seviyesi:** `review_only` — kod değiştirilmedi.

Bu incelemede kitabın 13 ana bölümünde ele alınan testler ve yöntemler (parametrik/nonparametrik testler, korelasyon, regresyon, geçerlik-güvenirlik ve faktör analizi) uSTAT kaynak koduyla karşılaştırıldı:
`backend/routers/stats/`, `backend/ustat_engine/`, `backend/routers/categorical.py`, `backend/routers/repeated.py`, `backend/routers/advanced_anova.py`, `backend/routers/models/linear.py`, `backend/services/assumptions.py`, `backend/routers/reliability.py`, `backend/routers/factor.py`.

> [!NOTE]
> **Doğrulama durumu.** Bulguların çoğu doğrudan kod okunarak doğrulandı (**kodla desteklenen**). Hiçbir bulgu, test çalıştırılarak yeniden üretilmedi. Aşağıda **(doğrulanmadı)** olarak işaretlenen maddelerde ilgili kod okunmadı; bunlar hipotez olarak değerlendirilmeli. Kitap sayfa numaraları OCR'ı bozuk İçindekiler sayfasından alındığı için yaklaşık değerlerdir.

---

## 1. Yönetici Özeti

uSTAT temel ve klinik testlerin çoğunu doğru şekilde yürütüyor. Ancak kitabın tez ve yayın raporlama standardına göre bazı çıktı bileşenleri, etki büyüklükleri ve yaygın testler eksik:

1. **Doğrusal regresyonda standartlaştırılmış β yok.** Çıktıda yalnızca ham B, SE, t, p, GA ve VIF var.
2. **Mann-Whitney U ve Kruskal-Wallis'te sıra ortalamaları ve sıra toplamları yok.** Yalnızca medyan ve IQR raporlanıyor.
3. **Fark güven aralıkları yok.** Bağımsız t-testinde ortalama farkının GA'sı, iki oran z-testinde oran farkının GA'sı verilmiyor.
4. **2×2 tablolarda Yates düzeltmeli ki-kare ve göreli risk (RR) yok.**
5. **ICC iki puanlayıcıya ve tek bir modele sabitlenmiş.** k ≥ 3 puanlayıcı ve diğer ICC modelleri desteklenmiyor.
6. **Eksik testler:** işaret testi, split-half (Spearman-Brown/Guttman), KR-20/KR-21, Scheffé, çift serili/tetrakorik korelasyon, Gamma, Somers' d, ağırlıklı Kappa ve marjinal homojenlik (Bowker/Stuart-Maxwell).

---

## 2. Bölüm Bazında Bulgular

### Bölüm 3, 4, 6, 7 — Tanımlayıcı İstatistikler, Standartlaştırma, Normallik

| Yöntem | Kitap (yakl.) | uSTAT | Kod | Bulgu |
|---|---|---|---|---|
| T skoru (T = 50 + 10z) | Böl. 6, s. 150–155 | ❌ | `backend/routers/compute.py` `TRANSFORMS` | `zscore` var, T skoru yok. |
| Varyasyon katsayısı, çeyrek sapma, harmonik ortalama | Böl. 4 | ⚠️ | `backend/routers/stats/descriptive.py` `descriptive()` | mean/std/median/IQR var; CV, (Q3−Q1)/2 ve harmonik ortalama yok. |
| Çarpıklık/basıklık SE'leri | Böl. 4 | ⚠️ | `descriptive.py` | `normality.py` SPSS G1/G2 ve SE kullanıyor; `descriptive` ise `scipy_stats.skew(s)` (bias=True, g1) kullanıyor ve SE vermiyor. Bu yüzden iki sekme farklı çarpıklık değeri gösterebilir. |
| Normallik dönüşümleri (arcsin√p, 1/X) | Böl. 7, s. 177–182 | ⚠️ | `compute.py` | ln, log10, sqrt, square, exp var; açı (arcsin) ve ters dönüşüm yok. |
| Detrended Q-Q, dal-yaprak grafiği | Böl. 3 / 7 | ❌ | `normality.py` | Standart Q-Q ve histogram var; bu ikisi yok. |

### Bölüm 9 — Hipotez Testleri

#### 9.4 Tek örneklem testleri
- Tek örneklem t-testi (`ustat_engine/stats/ttest.py`) ve tek oran testi (`categorical.py` `/one_proportion`) mevcut.
- **Eksik:** tek örneklem Wilcoxon işaretli sıralar testi (medyan = θ₀) ve tek örneklem işaret testi.
- **Ek risk (kodla desteklenen):** `/one_proportion` GA'yı Wald formülü ve sabit `1.96` ile hesaplayıp [0,1] aralığına kırpıyor. Küçük n veya uç oranlarda bu GA'nın kapsamı düşük kalır. Wilson veya Clopper-Pearson daha uygundur (`/binomial` zaten `binomtest().proportion_ci` kullanıyor). Ayrıca `r_code` alanı `prop.test(...)` gösteriyor; bu R fonksiyonu varsayılan olarak Yates düzeltmesi uygular ve uSTAT'ın düzeltmesiz z-testiyle aynı p değerini vermez.

#### 9.5 Bağımsız iki örneklem testleri
1. **Bağımsız t-testi** (`ustat_engine/stats/ttest.py` `run_ttest`): Levene ile Student/Welch seçimi ve Hedges' g mevcut.
   - Eksik: ortalama farkı (X̄₁ − X̄₂), farkın SE'si ve %95 GA'sı.
   - Eksik: kitapta verilen η² = t² / (t² + df).
2. **Mann-Whitney U** (`stats/nonparametric.py` `mannwhitney`): yalnızca U, medyan ve IQR raporlanıyor.
   - Eksik: grup sıra ortalamaları ve sıra toplamları (SPSS/tez tablolarında standart).
   - Eksik: bağ düzeltmeli asimptotik Z.
3. **İki oran z-testi** (`categorical.py` `/two_proportions`): oran farkı (p₁ − p₂) ve %95 GA'sı yok. `r_code` yine Yates'li `prop.test`.
4. **2×2 ki-kare** (`stats/inferential.py` `chisquare`): `chi2_contingency(ct, correction=False)` çağrılıyor.
   - Eksik: Yates düzeltmeli ki-kare.
   - Eksik: göreli risk (RR) ve GA'sı (yalnızca OR var).
   - Eksik: beklenen frekanslar tablosu. Hesaplanıyor (`_expected`) ama yanıtta dönmüyor.

#### 9.6 Bağımsız k grup
1. **Tek yönlü ANOVA** (`inferential.py` `anova`):
   - Eksik: SS/df/MS içeren varyans tablosu. Yanıtta yalnızca F, df ve p var.
   - Eksik: Scheffé ve Bonferroni post-hoc. Mevcut seçenekler Tukey, Games-Howell ve Dunnett.
2. **Kruskal-Wallis** (`nonparametric.py` `kruskal`): grup sıra ortalamaları yok, yalnızca medyan ve çeyrekler var. Post-hoc olarak Dunn (Holm/Bonferroni/FDR) mevcut.
3. **r×c ki-kare:** düzeltilmiş standartlaştırılmış artıklar (|z| > 1,96) yok. Anlamlı bir sonuçta hangi hücrenin farkı yarattığı raporlanmıyor.

#### 9.7 Bağımlı iki örneklem
- Eşleştirilmiş t-testi (`repeated.py` `paired_ttest`): tam (fark ortalaması, SD, d_z, GA).
- Wilcoxon işaretli sıralar (`repeated.py` `wilcoxon_signed_rank`): W⁺ etki büyüklüğü için iç hesapta kullanılıyor ama yanıtta W⁺, W⁻ ve Z ayrı ayrı verilmiyor.
- **İşaret testi yok.**
- McNemar (`categorical.py` `/mcnemar`): mevcut, tam/asimptotik seçimi yapıyor. 2×2 dışındaki tabloları hata vererek reddediyor. Bu yüzden **marjinal homojenlik (Bowker / Stuart-Maxwell) testi yok.**

#### 9.8 Bağımlı k grup
- Tekrarlı ölçümler ANOVA (`repeated.py` `rm_anova`): Mauchly, Greenhouse-Geisser ve Huynh-Feldt mevcut.
- Friedman (`repeated.py` `friedman`): Kendall W ve Holm düzeltmeli Wilcoxon post-hoc mevcut; koşul sıra ortalamaları yok.
- Cochran Q (`categorical.py` `/cochran_q`): anlamlı sonuçta ikili McNemar post-hoc yapılıp yapılmadığı **(doğrulanmadı)**.

### Bölüm 10 — Çok Etkenli Denemeler
- İki yönlü ANOVA (`advanced_anova.py`): Type II, EMM, Tukey-Kramer ve basit etkiler mevcut.
- Karma ANOVA (`repeated.py` `mixed_anova`): mevcut, hata katmanları doğru ayrılmış.
- ANCOVA: mevcut, eğim homojenliği kontrolü yapılıyor.
- Önemli bir eksik saptanmadı.

### Bölüm 11 — Korelasyon Katsayıları (`stats/correlation.py`)

| Katsayı | uSTAT | Not |
|---|---|---|
| Pearson r, kısmi korelasyon | ✅ | Fisher z GA ve t testi mevcut. |
| Spearman ρ, Kendall τ | ✅ | |
| Nokta çift serili (r_pb) | ⚠️ | Pearson'a eşdeğer ama ayrı bir seçenek olarak sunulmuyor. |
| Çift serili (r_b), tetrakorik | ❌ | |
| Kontenjans katsayısı C / C_adj | ❌ | Yalnızca Cramér V var. |
| Eta (η) korelasyon oranı | ❌ | |
| Goodman-Kruskal Gamma | ❌ | |
| Somers' d | ❌ | |

**Ek risk (kodla desteklenen):** `correlation_pair` içinde `result_text`, `method_used == 'pearson'` değilse "Spearman" yazıyor. Kendall seçildiğinde metin yanlış yöntem adını gösterir.

### Bölüm 12 — Doğrusal Regresyon (`models/linear.py`, `services/assumptions.py`)
1. **Standartlaştırılmış β yok:** β = B · s_X / s_Y.
2. **Regresyon ANOVA tablosu yok:** SS_reg, SS_res, MS değerleri dönmüyor; yalnızca F ve p var.
3. **Durbin-Watson yok:** varsayım kontrolleri yalnızca Shapiro/Jarque-Bera, Breusch-Pagan ve Rainbow.
4. **Tolerans (1/VIF) yok:** VIF mevcut.

### Bölüm 13 — Geçerlik, Güvenirlik, Faktör Analizi

#### Geçerlik
ROC (AUC, DeLong GA, Youden, duyarlılık/özgüllük/PPV/NPV), çoklu ROC DeLong karşılaştırması ve Bland-Altman mevcut. Bland-Altman ICC yanıtında da kodla doğrulandı; ayrı `agreement.py` dosyası okunmadı.

#### Sayısal güvenirlik (`reliability.py`)
- Mevcut: Cronbach α, McDonald ω, düzeltilmiş madde-toplam korelasyonu, madde silinirse α.
- ❌ Split-half (Spearman-Brown, Guttman)
- ❌ KR-20 / KR-21
- ❌ Standartlaştırılmış α
- ❌ Alt-üst %27 madde ayırt edicilik analizi
- ❌ Taban/tavan etkisi
- **Ek risk:** ω, `sklearn.FactorAnalysis` (ML tabanlı) ile hesaplanıyor; `r_code` alanı ise `psych::omega` gösteriyor. Bu iki yöntem farklı değer verebilir **(doğrulanmadı)**.

#### Puanlayıcılar arası güvenirlik
- ICC (`correlation.py` `/icc`): `k = 2` sabit. Yalnızca ICC(A,1) (iki yönlü rastgele, tek ölçüm, mutlak uyum) hesaplanıyor. Model, tür ve tek/ortalama seçimi yok; 3 ve üzeri puanlayıcı desteklenmiyor.
- Cohen κ ve Fleiss κ mevcut. **Ağırlıklı κ (doğrusal/kuadratik) yok.**

#### Faktör analizi (`factor.py`)
- Mevcut: KMO, Bartlett, özdeğer > 1, Varimax, Promax, açıklanan varyans.
- **Eksik:** korelasyon matrisinin determinantı (|R| > 0,00001 kuralı). Değer `calculate_bartlett` içinde hesaplanıyor ama yanıtta dönmüyor.
- **Etiket uyumsuzluğu (kodla desteklenen):** EFA etiketi "Principal Axis" diyor ve `r_code` `fm = "pa"` gösteriyor. Oysa uygulama `sklearn.FactorAnalysis` kullanıyor; bu ML benzeri bir kestirimdir, asal eksen faktörlemesi değildir. Yükler R çıktısıyla örtüşmeyebilir.

---

## 3. Önceliklendirilmiş Öneriler

### Öncelik 1 — Mevcut çıktıların zenginleştirilmesi
1. Doğrusal regresyona standartlaştırılmış β, tolerans, ANOVA tablosu ve Durbin-Watson eklenmesi.
2. Mann-Whitney, Kruskal-Wallis ve Friedman çıktılarına sıra ortalamaları ve sıra toplamları, Mann-Whitney'e Z eklenmesi.
3. t-testine ortalama farkı GA'sı, iki oran testine oran farkı GA'sı eklenmesi. Tek oran GA'sının Wilson'a çevrilmesi.
4. 2×2 ki-kareye Yates, RR ve beklenen frekansların; r×c tablolara düzeltilmiş artıkların eklenmesi.
5. ICC'nin k ≥ 3 puanlayıcıya ve ICC model seçeneklerine genişletilmesi.
6. Etiket ve metin uyumsuzluklarının giderilmesi: EFA "Principal Axis" etiketi, Kendall sonuç metni, `prop.test` için `r_code`.

### Öncelik 2 — Eksik yöntemlerin eklenmesi
1. İşaret testi (tek örneklem ve eşleştirilmiş), tek örneklem Wilcoxon.
2. Marjinal homojenlik (Bowker / Stuart-Maxwell).
3. Split-half, KR-20/21, standartlaştırılmış α.
4. Gamma, Somers' d, kontenjans katsayısı, ağırlıklı κ.
5. ANOVA için Scheffé ve Bonferroni post-hoc ile SS/MS tablosu.

> [!IMPORTANT]
> AGENTS.md kuralı gereği her yeni istatistik, hakemli bir kütüphane fonksiyonuna eşlenmelidir; formüller elle yazılmamalıdır. Örnekler: `statsmodels.stats.stattools.durbin_watson`, `statsmodels.stats.contingency_tables.SquareTable.homogeneity`, `pingouin.intraclass_corr` (yalnızca bağımlılık politikası izin veriyorsa), `scipy.stats.binomtest` ve `sklearn.metrics.cohen_kappa_score(weights=...)`. Yanıt yapısı değiştirildiğinde ilgili frontend paneli de güncellenmelidir.
