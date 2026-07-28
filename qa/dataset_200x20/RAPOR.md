# uSTAT, 200×20 veri seti ile istatistiksel test ve Table 1 doğrulama raporu

**Tarih:** 2026-07-28  
**Veri seti:** `dataset_200x20.csv`, 200 satır × 20 kolon  
**Üreteç:** `generate_dataset.py`, seed=42  
**Koşucu:** `run_tests.py <alan>`  
**Ham sonuçlar:** `results/*.json`

## 1. Veri seti

Survival değişkenleri mevcut 20 kolonun içindedir; ek kolon değildir.

| Tür | Sayı | Değişkenler |
|---|---:|---|
| Sürekli | 10 | age, bmi, sbp, cholesterol, glucose, score, biomarker, pre, post, time |
| Kategorik | 10 | event, sex, group, education, smoking, diabetes, hypertension, stage, center, response |

11 değişkende yaklaşık %3–9 eksiklik vardır:

- MCAR: bmi, score, post, biomarker, glucose, smoking, education, stage, response
- MAR: sbp, yüksek yaşta daha sık; cholesterol, yüksek bmi'da daha sık
- Eksiksiz survival çifti: time ve event

Bilinçli ground-truth ilişkileri:

- `score ~ group`: B − A = +5, anlamlı sonuç beklenir
- `biomarker ~ group`: gerçek fark 0, anlamsız sonuç beklenir
- `bmi ~ sex`: M − F = +3
- `glucose ~ diabetes`: +22
- `pre − post`: +3, eşleştirilmiş testlerde anlamlı sonuç beklenir
- `corr(age, sbp)` ve `corr(age, cholesterol)`: pozitif
- `group` survival HR: 1.8
- `sex × group`: bağımsızlık kontrolü, anlamsız sonuç beklenir

## 2. Yöntem

Her denetim üç bileşen kullanır:

1. uSTAT endpoint sonucu, FastAPI TestClient ve gerçek in-memory oturum üzerinden
2. Bağımsız SciPy, statsmodels veya scikit-learn hesabı
3. Ground-truth yönü ve anlamlılık beklentisi

Eksik gözlemler ilgili testin tam-vaka kuralıyla dışlanır. Denetim n, test istatistiği, p-değeri, etki büyüklüğü, güven aralığı, karar ve yanıt sözleşmesini uygun olduğu yerde karşılaştırır.

## 3. Sonuç

| Alan | Test edilenler | Sonuç |
|---|---|---:|
| inferential | t-test, tek örneklem t, χ², Fisher, ANOVA, TOST, non-inferiority, power | 22/22 PASS |
| nonparametric | Mann-Whitney, Kruskal-Wallis, Jonckheere-Terpstra, ROC | 7/7 PASS |
| repeated | paired t, Wilcoxon, Friedman, RM-ANOVA | 10/10 PASS |
| categorical | binomial, tek/iki oran, McNemar, Cochran Q, Mantel-Haenszel, Cochran-Armitage | 14/14 PASS |
| correlation | Pearson, Spearman, matris, ICC, Cohen's κ | 10/10 PASS |
| agreement | Bland-Altman, Deming, Passing-Bablok, Lin CCC | 9/9 PASS |
| reliability | Cronbach α | 3/3 PASS |
| table1 | gruplu/grupsuz Table 1, Missing satırları, pub_tables, XLSX/DOCX, weighted descriptive | 15/15 PASS |
| survival | KM + log-rank, Cox PH, Cox uni/multi, RMST, E-value, Fine-Gray, Landmark | 17/17 PASS |
| **Toplam** | | **107/107 PASS, %100** |

Karşılaştırılan sayısal sonuçlar bağımsız referanslarla eşleşti. Ground-truth ilişkileri doğru yön ve anlamlılıkta yakalandı. Biomarker null kontrolü ve `sex × group` bağımsızlık kontrolü doğru biçimde anlamsız çıktı.

## 4. Uygulanan düzeltmeler

### 4.1 ANOVA 500 hatası

- Kök neden: interpretation üretirken tanımsız `df_den_report` kullanımı.
- Düzeltme: klasik ANOVA için `df_within`, Welch ANOVA için ondalıklı Welch payda serbestlik derecesi raporlanıyor.
- Regresyon: eşit varyans ve Welch yolları endpoint seviyesinde test ediliyor.

### 4.2 Table 1 eksiklik görünürlüğü

- Her eksik sürekli veya kategorik değişken için `Missing n (%)` satırı üretildi.
- Overall ve grup-bazlı eksik n (%) değerleri ayrı raporlanıyor.
- Eksiklik, kategorik seviye sayılmıyor; p-değeri ve SMD tam-vaka hesabı değişmiyor.
- API yanıtı, ekran, kopyalama/CSV/XLSX verisi ve pub_tables XLSX/DOCX yolu eksiklik satırını koruyor.
- Kullanıcı `selected_stats` içinde `missing` seçerse yinelenen satır oluşmuyor.

### 4.3 Bağımsız t-test etki büyüklüğü

- Hesap doğrulandı: değer Cohen's d değil, küçük örneklem düzeltmeli Hedges' g'dir.
- `effect_sizes[0].name`, interpretation ve frontend yöntem açıklaması Hedges' g ile uyumlu hale getirildi.
- QA, J düzeltmesini bağımsız hesaplayıp API değeriyle karşılaştırıyor.

### 4.4 ROC auto-direction

- Backend sözleşmesindeki `direction_requested`, `direction_used` ve `direction_flipped` alanları regresyon testine alındı.
- Frontend yalnız gerçek auto-flip olduğunda bildirim gösteriyor.
- Kullanılan yön ve flip bilgisi dışa aktarıma eklendi.

### 4.5 Mantel-Haenszel ortak OR güven aralığı

- `StratifiedTable.oddsratio_pooled_confint(alpha)` ile alpha-duyarlı ortak OR güven aralığı eklendi.
- `ci_low`, `ci_high` ve `ci_level` effect-size sözleşmesinde, yorumda ve dışa aktarımda bulunuyor.
- Sıfır hücrede sonlu sonuç korunuyor; tam ayrışmada nonfinite değer JSON-safe `null` ve uyarı oluyor.
- 95% CI bağımsız statsmodels hesabıyla eşleşti.

### 4.6 İstek alanı alias'ları

- Mevcut alan adları canonical kaldı; güvenli uzun/kısa biçimler Pydantic `AliasChoices` ile kabul ediliyor.
- Canonical ve alias birlikte verilirse canonical değer kazanıyor.
- Kapsam: `group_col/group_column`, `row_col/row_column`, `col_col/col_column`, `col1/column1`, `method1/column1`, ROC score/outcome alanları, ICC/κ rater alanları, weighted descriptive alanları ve ilgili çoğul biçimler.
- Belirsiz evrensel alias üreticisi eklenmedi. OpenAPI ve mevcut istemciler bozulmadı.
- Frontend API fonksiyonlarının etkilenen istekleri TypeScript interface'leriyle tipli hale getirildi.

### 4.7 Veri seti boyutu ve metadata

- Yinelenen `treatment` kolonu kaldırıldı; `group` iki kollu maruziyeti zaten temsil ediyor.
- Veri seti gerçekten 200×20 oldu.
- `ground_truth.json` 10 sürekli ve 10 kategorik kolonla tutarlı.
- Survival `event`, görüntülenen yuvarlanmış zamandan değil latent olay ve sansür zamanından hesaplanıyor; event rate 0.805.
- Üreteç mutlak kullanıcı yolu yerine kendi dizinine yazıyor.

## 5. Yazılım doğrulaması

Son tam doğrulama:

```text
Backend pytest: 1231 passed, 9 skipped
Frontend TypeScript: no errors
Frontend ESLint: no issues
Frontend Vitest: 46 files, 386 tests passed
Frontend production build: passed
İstatistik denetimi: 90 passed, 0 failed
```

Backend koşusunda 50 uyarı vardır. Bunlar mevcut deprecation, convergence, perfect-separation ve sayısal precision uyarılarıdır; test başarısızlığı değildir. Frontend build, mevcut büyük chunk ve etkisiz dynamic-import uyarılarını üretir; build başarılıdır.

## 5b. Survival doğrulaması (yeni — 17/17 PASS)

- **KM + log-rank:** χ² ve p lifelines ile birebir (χ²=20.25, p=6.8e-06); medyan survival A=16.2 / B=7.1 doğru; HR=1.8 tasarımı anlamlı yakalandı.
- **Cox PH:** HR(grpB)=2.048, p=1.03e-05 lifelines ile birebir; gerçek HR=1.8 tahmin CI içinde; n/n_events/n_excluded raporlanıyor.
- **Cox uni_multi, Fine-Gray, Landmark:** 200, doğru yanıt şeması.
- **RMST (τ=10):** grup RMST değerleri lifelines referansıyla uyumlu (A≈8.40).
- **E-value (HR=1.8):** formül birebir (E=3.0), `measure_type` ile.
- Survival kolonları (`time`, `event`) eksiklik deseninden muaf tutuldu; tüm survival endpoint'leri tam n=200 ile çalıştı.

## 6. Kapsam sınırı

107 denetim bu raporda listelenen endpoint, Table 1 akışları ve ana survival endpoint'lerini (KM, Cox, RMST, E-value, Fine-Gray, Landmark) kapsar. Kalan ileri survival modülleri (cox_tv, cox_rcs, frailty, multistate, joint_model, external_validation, ml_survival_benchmark, recurrent_lwyy, dynamic_prediction) bu koşucuda henüz test edilmedi; sonuç uSTAT içindeki her istatistiksel yöntemin eksiksiz doğrulaması olarak yorumlanmamalıdır.

## 7. Yeniden üretim

```bash
cd /Users/yh/Documents/projects/wiz3
.venv/bin/python qa/dataset_200x20/generate_dataset.py
for area in inferential nonparametric repeated categorical \
  correlation agreement reliability table1 survival; do
  .venv/bin/python qa/dataset_200x20/run_tests.py "$area"
done
```

Beklenen çıktı: `qa/dataset_200x20/results/*.json` içinde toplam 107 PASS ve 0 FAIL.
