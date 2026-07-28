# uSTAT Tests + Table istatistiksel yöntem denetimi (200×20)

Tarih: 2026-07-28
Kapsam: deterministik 200 satır × 20 sütun sentetik veri; Tests alt panelleri
(tanımlayıcı, sürekli/kategorik/çıkarımsal, tekrarlı ölçüm, ileri ANOVA) ve
Table 1.
Karar: **ürün kodunda düzeltme yapılmadı.** Bu belge yalnızca bulgu raporudur.

## Kısa sonuç

Sürekli değişken testleri, kategorik/contingency testleri, tekrarlı ölçüm
omnibusları, ANCOVA/MANCOVA ve Table 1, bağımsız R referanslarıyla makine
duyarlığında uyuşuyor. Önceki 10×20 denetiminde sample-size guard'ların
engellediği **iki-way ANOVA n=200'de temiz çalışıyor** ve R ile birebir
uyuşuyor. `fix(table1): stop counting missing values as a category` düzeltmesi
n=200'de de sağlam: eksik kategorik değer kategori gibi sayılmıyor.

Ancak **iki ciddi yöntem hatası** klinik kararı değiştiriyor ve düzeltilmedi:

1. **Mixed ANOVA denek-içi bağımlılığı modellemiyor** (K1). n=200'de sapma büyük
   ve sistematik: denek-içi F'ler doğru değerinin ~%20'si, denekler-arası F
   ~2,6× şişiriliyor.
2. **Non-inferiority tek yönlü p-değeri yanlış kuyruğu kullanıyor** (K2).
   Endpoint doğru p yerine `1 − doğru p` döndürüyor; sürekli ve ikili (RR)
   yolların ikisinde de.

Bunlara ek olarak, RM ANOVA post-hoc'unda dejenerasyon koruması olmaması
(Y2), RM ANOVA sphericity düzeltmesinin hiç uygulanmaması (Y1) ve birkaç
ikili-girdi doğrulama/etiket açığı bulundu. Yeşil test paketi bu adversarial
yöntem hatalarını kapsamıyor.

## Veri seti

- Dosya: `qa/statistical_audit_200x20/dataset.csv`
- SHA-256: `db1281b2e372b27cfae8594bba8cfb6fbc543898f842ffa5405bec26ad698300`
- Boyut: 200 × 20
- Üretim: `generate_dataset.py`, sabit tohum `20260728`, rastgelelik yok,
  hasta verisi yok
- Gruplar: `arm` A/B (100'er), `group3` Low/Medium/High (~67'şer),
  `factor2` X/Y (her arm içinde 50'şer, tam rank)
- Sürekli: yaş, normal/çarpık biyobelirteçler, ANCOVA sonuçları, üç tekrarlı
  skor
- Kategorik: ikili olay, ikili/üç-seviyeli kategori, tabaka (S1/S2), sıralı doz
  (0/1/2)
- Eksikler: `biomarker_normal` 5, `biomarker_skew` 3, `followup_score` 5,
  `category_three` 4
- Tekrarlı ölçüm denetimi: üç skor sütununda tam olan 195 kişi uzun biçime
  çevrildi (585 × 4). Kaynak veri değişmedi.

## Çalıştırma ve doğrulama

```bash
# bağımsız R referansı
Rscript qa/statistical_audit_200x20/reference.R \
  qa/statistical_audit_200x20/dataset.csv
# denetim scriptleri (her biri kendi alanını kapsar)
.venv/bin/python qa/statistical_audit_200x20/audit_continuous.py
.venv/bin/python qa/statistical_audit_200x20/audit_categorical.py
.venv/bin/python qa/statistical_audit_200x20/audit_repeated.py
.venv/bin/python qa/statistical_audit_200x20/audit_table1.py
```

Ortam: Python 3.10, R 4.5.2, NumPy 2.2.1, pandas 2.2.3, SciPy 1.15.1,
statsmodels 0.14.4. Denetim HEAD'i `8fe62b0` (working-tree diff hash ayrı
kayıtlı). Dört alt-denetim TestClient ile ürünü süreç-içi çağırdı;
sayısal karşılaştırmalar yukarıdaki R scriptiyle yapıldı.

## Sayısal doğruluk özeti

Aşağıdaki testler R referansıyla tolerans içinde (p ≤ 1e-6, istatistik ≤ 1e-4,
yuvarlama tabanlı farklar hariç):

| Test | Endpoint | Doğrulama |
|---|---|---|
| Tek örneklem t | `/api/stats/ttest` | t, p tam uyum |
| Bağımsız t (oto Student/Welch) | `/api/stats/ttest` | Student ile tam uyum; Levene→Student doğru |
| Tek yönlü ANOVA | `/api/stats/anova` | klasik F, p tam uyum |
| Mann-Whitney | `/api/stats/mannwhitney` | p tam uyum (bağlarla normal yaklaşıklama) |
| Kruskal-Wallis | `/api/stats/kruskal` | p tam uyum |
| Ki-kare 2×2 (Yates) | `/api/stats/chisquare` | Yates χ², p tam uyum |
| Fisher exact | `/api/stats/fisher` | p tam uyum |
| Binomial exact | `/api/categorical/binomial` | k/n doğru |
| One/two-proportion z | `/api/categorical/*` | p yöntemle tutarlı |
| McNemar (istatistik/p) | `/api/categorical/mcnemar` | simetrik olduğu için doğru |
| Cochran Q | `/api/categorical/cochran_q` | p doğru |
| Cochran-Armitage | `/api/categorical/cochran_armitage` | z-trend doğru |
| Eşleştirilmiş t | `/api/repeated/paired_ttest` | t, p tam uyum |
| Friedman | `/api/repeated/friedman` | χ², p tam uyum |
| RM ANOVA (omnibus) | `/api/repeated/rm_anova` | F, p tam uyum |
| **İki yönlü ANOVA** | `/api/advanced_anova/two_way_anova` | arm/factor2/etkileşim F,p **tam uyum** (n=10'da reddediliyordu) |
| ANCOVA | `/api/advanced_anova/ancova` | F, p tam uyum |
| MANCOVA (Pillai) | `/api/advanced_anova/mancova` | Pillai, F, p tam uyum |

Sayısal olarak **uyuşmayan tek iki test**: mixed ANOVA (K1, yöntem hatası) ve
non-inferiority (K2, yanlış kuyruk). Bunların altı dışında tüm sayısal
kontroller geçti.

## Table ile Tests tutarlılığı

Aynı yöntemi kullandıklarında Table ve Tests bu veri setinde aynı sonucu
veriyor. Table, varsayılan normalite ön-testine göre otomatik yöntem seçiyor;
bu yüzden bazı değişkenlerde Table ile Tests farklı yöntem seçer ama görünen
p yine uyuşur:

| Değişken | Table yöntemi | Tests yöntemi | Sonuç |
|---|---|---|---|
| `biomarker_normal` | Student t (`<0.001`) | Student t (Levene) | Uyumlu |
| `ancova_outcome` | Student t (`<0.001`) | Student t (Levene) | Uyumlu |
| `biomarker_skew` | Mann-Whitney (`<0.001`) | Welch t | Beklenen fark: Table normaliteye bakar, Tests bakmaz |
| `category_binary` | Ki-kare (`<0.001`) | Fisher (sparse guard burada tetiklenmez) | Beklenen fark: hücreler yoğun (min beklenen 43,5) |
| `category_three` | Ki-kare asimptotik (0,382) | — | R MC p 0,386; fark yöntem farklılığı, hata değil |

`biomarker_skew` örneği önemli ve **bir hata değil**: Table çarpık değişken
için Mann-Whitney seçerken Tests paneli koşulsuz t-test sunuyor. İkisi de
`<0.001` gösteriyor. Bu, "veriye-bağlı test seçimi"nin bir örneği; küçük n'de
oynak olabilir ama burada tutarlı.

## Eksik veri yönetimi (son düzeltme doğrulandı)

`fix(table1): stop counting missing values as a category` (HEAD `8fe62b0`)
n=200'de sağlam:

- `category_three` (4 eksik): `overall_n` = **196** (200−4). Doğru.
- Seviye listesi: `['Alpha','Gamma','Beta']` — tam 3 seviye. **`missing`/`nan`
  kategori olarak yer almıyor.**
- Hücre toplamları 196'yi veriyor; uyarı doğru üretiliyor.
- `age` (eksiksiz): `overall_n` = 200. Doğru.

Düzeltme veriye-özel değil: `descriptive.py:988-991` önce
`df[[var, group]].dropna()` yapıp sonra `astype(str)` + `crosstab` çağırıyor,
dolayısıyla NaN hiçbir zaman `"nan"` dizgisine dönüşmüyor. Aynı `.dropna()`
deseni kategorik SMD yolunda da (`descriptive.py:1051-1054`) uygulanıyor.
Mevcut testler (`test_missing_not_a_category.py`,
`test_table1_consistency.py`) bu yolu kapsıyor.

## Kritik bulgular

### K1: Mixed ANOVA denek hata tabakalarını kullanmıyor

`backend/routers/repeated.py:393-398`. Yorum satırı "subject as random"
diyor ama formülde `subject_col` yok:

```python
formula = f"Q('{value_col}') ~ C(Q('{within_col}')) * C(Q('{between_col}'))"
model = smf.ols(formula, data=sub).fit()
aov = anova_lm(model, typ=2)
```

Düz OLS factorial ANOVA çalışıyor; her etkideki F tek havuzlanmış Residual
üzerinden hesaplanıyor. Doğru split-plot modeli iki hata tabakası ister:
denekler-arası (arm) → subject-within-arm hatası; denek-içi (zaman,
etkileşim) → subject×time hatası. R referansı
`aov(score ~ arm*timepoint + Error(id/timepoint))`.

n=200'deki sapma:

| Etki | Endpoint F | R (doğru tabaka) F | Oran | Endpoint p | R p |
|---|---:|---:|---:|---:|---:|
| Zaman (denek-içi) | 692,96 | 3397,42 | 0,20 | ~0 | 9,3e-246 |
| Kol (denekler-arası) | 132,72 | 51,20 | 2,59 | ~0 | 1,7e-11 |
| Zaman × kol | 40,10 | 196,60 | 0,20 | ~0 | 1,3e-59 |

Bu veride her etkideki sonuç p<0,05 düzeyinde aynı kalıyor (hepsi ezici
anlamlı). Ama F'ler 0,20–2,6× çarpık; kısmi η², gözlenen güç ve sınırda
anlamlı/zayıf etkiler yanlış sınıflanır. Zayıf gerçek etkide denek-içi
F'nin doğru değerinin %20'sine düşmesi gerçek bir etkiyi anlamsız gösterebilir;
denekler-arası F'nin 2,6× şişmesi yanlış pozitif üretebilir. Endpoint ayrıca
`ezANOVA(within=..., between=...)` kullanıldığını iddia eden R kodu yayımlıyor
(`repeated.py:465-468`); bu Python hesabıyla uyuşmuyor. Klinik kararı tersine
çevirebilecek bir hatadır.

### K2: Non-inferiority tek yönlü p-değeri yanlış kuyruk

`backend/routers/stats/inferential.py:561-562` (upper) ve `:566-567` (lower).
Upper-bound için doğru test istatistiği `z = (tahmin − marj)/se` ve
`p = Φ(z)` iken kod `z = (marj − tahmin)/se` (işaret ters) hesaplayıp
`norm.cdf` çağırıyor; bu `1 − Φ((tahmin−marj)/se)` = **doğru p'nin
tamamlayıcısı**. Lower dalı simetrik şekilde bozuk.

Sürekli (MD, upper, marj=20, B vs A), n=200:

- Tahmin = 11,72 (doğru); Welch %90 GA `[10,05; 13,39]` (doğru).
- Marj 20, GA üst sınırı 13,39 < 20 ⇒ **non-inferiority gösterilir.**
- Doğru tek yönlü t p'si = **1,47e-14** (R `ni_cont_upper_p_t`).
- Endpoint `p_noninferity` = **1,0** ≡ `1 − doğru p`. Yanlış.

İkili (RR, upper, marj=3, B vs A):

- RR = 2,48 (doğru).
- Doğru tek yönlü p = **0,1583** (R `ni_binary_upper_p`).
- Endpoint `p_noninferity` = **0,8417** ≡ `1 − doğru p`. Yanlış.

Önemli nüans: **`non_inferior` karar bool'u bu veride tesadüfen doğru**
(`hi_disp < margin` CI kuralından bağımsız hesaplanıyor, `inferential.py:560`).
Yani GA-tabanlı karar doğru çıkar ama raporlanan p-değeri, yorum metni
(`inferential.py:575-577`) ve export satırı yanlış. Başka bir veri setinde
CI kuralı da yanıltıcı olabilir; p güvenilir değil.

İkincil bir kusur: sürekli dal SE'yi t-tabanlı güven aralığından
`z_one` (`norm.ppf(1-α)`) ile geri-çözüyor (`inferential.py:548-549`). Bu z ve t
ölçeklerini karıştırır; işaret düzelse bile p yine z-tabanlı kalır, düzenleyici
beklenen t-tabanlı p'den farklıdır.

**Klinik etki (ağır):** Non-inferiority başarısız bir çalışma p≈0 raporlayıp
"NI gösterildi" yorumu çıkarabilir (yukarıdaki sürekli örnekte olduğu gibi);
başarılı bir çalışma p≈1 gösterebilir. Doğrudan hasta güvenliği / düzenleyici
anlamlılık hatasıdır.

## Yüksek riskli bulgular

### Y1: RM ANOVA sphericity düzeltmesi hiç uygulanmıyor

`backend/routers/repeated.py:292-302`. Kod `hasattr(res, 'epsilon')` kontrol
ediyor ama statsmodels `AnovaResults` nesnesinde `epsilon` (veya `mauchly_w`)
attribute yok (doğrudan doğrulandı). Dolayısıyla `eps` her zaman `None`,
`assumptions` her zaman `[]`. 3+ zaman noktasında sphericity ihlali yaygın;
düzeltilmemiş denek-içi F liberal (Tip I hata şişirilmiş) raporlanıyor,
uyarı/düzeltme yok.

### Y2: RM ANOVA post-hoc dejenerasyon koruması yok (n'ye göre farklı davranır)

`backend/routers/repeated.py:304-328`. Bağımsız `paired_ttest` `t=Inf`'i ±9999
ile sınırlandırıyor (`repeated.py:55-57`) ama RM ANOVA post-hoc döngüsü
`sp.ttest_rel` çağırıp sonluluk kontrolü yapmıyor. Sabit farklı kontrast
probe'unda:

- n=10: scipy `t=Inf` üretir, global JSON sonluluk koruması geçerli omnibus
  dahil **tüm yanıtı** HTTP 400 yapar.
- n=200: kayan nokta yuvarlaması `t=4,47e16` (sonlu ama çöp), `p=0` üretir;
  endpoint **HTTP 200** döner ve post-hoc tablosunda anlamsız istatistik
  `significant: true` ile gösterilir.

Aynı mantık kusuru n'ye göre zıt davranış üretiyor; n=200'deki daha kötü
(sessiz çöp istatistik). Post-hoc döngüsüne açık sıfır-varyans/sonluluk guard
gerekir.

### Y3: İkili girdi doğrulaması tutarsız

`backend/routers/categorical.py`. `binomial_test`, `one_proportion_ztest`,
`cochran_q_test` sütunun gerçekten ikili olduğunu doğrulamıyor. 3 seviyeli
sütun gönderilirse HTTP 200 dönüp en sık seviyeyi "success" seçebilir
(binomial/one-proportion) veya `mat.values.astype(float)` ile anlamsız Q üretir
(cochran_q). Buna karşın `cochran_armitage` (`categorical.py:668-670`) ve
`noninferiority` (`inferential.py:499-500`) doğru şekilde reddediyor. Panel
içi tutarsızlık; çok-seviyeli veri sessizce yanlış ikili gibi işlenir.

### Y4: McNemar hücre etiketleri crosstab yönüyle ters

`backend/routers/categorical.py:325-331, 357-360, 381-385`. `pd.crosstab(col1,
col2)` kurulup `a,b = table[0]; c,d = table[1]` alınıyor. 0/1 kodlamasında
`table[0,0]` = (ön=0, son=0) = **her ikisi negatif** ama yanıt bunu
`"a (both +)"` / `concordant_a` olarak etiketliyor. İstatistik ve p, b,c'de
simetrik olduğu için doğru; uyumsuz OR `b/c` sayısal olarak doğru ama
yön-etiketleri ve OR yönü yorumu ters. Export/raporlarda concordant/discordant
ve OR yönü yanıltıcı.

### Y5: Mantel-Haenszel sonsuz OR'de tüm yanıtı kaybedebilir

`/api/categorical/mantel_haenszel`. Bu veri setinde endpoint HTTP 400 dönüyor
(`stratum` yalnızca 2 seviye; arm A yalnız S1'de, arm B yalnız S2'de ⇒ her
tabaka tablosunda tam-sıfır satır ⇒ pooled OR `NaN`). Bu bir veri artefaktı,
endpoint hatası değil. Ama genel endişe geçerli: tek bir tabakada sıfır hücre
`Inf` OR ürettiğinde global `_sanitize`/sonluluk guard (`inferential.py:46-54`)
tüm yanıtı reddediyor; kısmi-sonuç + uyarı stratejisi daha bilgilendirici olurdu.

## Orta / düşük risk ve yöntem sınırlamaları

- **Welch omnibus yok (Orta):** Tek yönlü ANOVA her zaman klasik `f_oneway`
  kullanıyor (`inferential.py:280`); heteroskedastisitede yalnızca post-hoc
  Games-Howell'a geçiyor. Omnibus Welch F sunulmuyor. Bu veride Levene
  geçerli, fark etmiyor ama heterojen varyanslı denemelerde kısıtlama.
- **Etki büyüklüğü GA'ları z-yaklaşıklama (Düşük):** Cohen's d/Hedges g ve
  rank-biserial GA'ları `z=1,96` normal yaklaşıklaması (`stat_utils.py:319,
  378-381`). p≈1e-36'da Hedges g GA'sı sıfırı geçebiliyor (görsel olarak yanıltıcı).
  Tanımlayıcı `ci95` doğru t-kritik kullanıyor.
- **Rank-biserial işaret yönelimi (Kozmetik):** `stat_utils.py:374-388` adlandırılmış
  grubun düşük rank'lara sahip olması yönünde; aynı A-vs-B karşılaştırmasında
  t-test Hedges g negatif, rank-biserial pozitif çıkıyor. İçsel tutarlı ama
  sunum yönelimsiz.
- **Çok seviyeli kategorik SMD diyagonal kovaryans (Düşük):** Table 1 SMD'si
  `descriptive.py:1036-1038` çok terimli kovaryans yerine `diag(p(1-p))`
  kullanıyor (off-diagonal `−p_i p_j` yok). Mahalanobis iskelet doğru; bu
  veride 0,194 (kod) vs 0,198 (tam çokterimli) farkı eşiği değiştirmiyor.
- **Table 1 yalnız biçimlenmiş p (Bilgi):** 3 ondalıklı `p_value` döndürülüyor,
  ham p yok. Table 1 için standart ama kesin karşılaştırma/meta-analiz engeli.
- **Wilcoxon sıfır/bağ yönetimi belgesiz (Düşük):** `repeated.py:114-160` p
  için exact/normal seçimini ve bağ/zeıro kuralını bildirmiyor; R
  `wilcox.test` ile bağlarda farklı p üretebilir, nedeni yanıtta yok.
- **R ki-kare referansı Monte Carlo kullandı (not):** `reference.R` üç-seviyeli
  tablo için `simulate.p.value=TRUE` kullandı; endpoint asimptotik. Yoğun
  hücrelerde fark önemsiz (0,004), yöntem farklılığıdır, hata değildir.

## Sonuç

Temel istatistik motoru sayısal olarak sağlam: sürekli, kategorik, tekrarlı
omnibus, ANCOVA/MANCOVA ve iki yönlü ANOVA R ile makine duyarlığında uyuşuyor;
Table 1 Tests paneliyle tutarlı ve eksik veri düzeltmesi n=200'de sağlam.
Ürün, **mixed ANOVA (K1) ve non-inferiority (K2) düzeltilmeden** klinik üretim
için yeterli doğrulanmış sayılmaz. Y1–Y5 düzeltmeleri klinik güven ve
tutarlılık için önerilir. Bu rapor yalnızca bulguları belgeler; kod değişikliği
yapılmadı, kullanıcı sonradan yapacak.
