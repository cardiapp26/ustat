# uSTAT Tests + Table istatistik denetimi

Tarih: 2026-07-28  
Kapsam: sabit 10 satır × 20 sütun sentetik veri; Tests alt panelleri ve Table 1  
Karar: ürün kodunda düzeltme yapılmadı

## Kısa sonuç

Temel testlerin çoğu bağımsız R/SciPy hesaplarıyla sayısal olarak uyuşuyor. Table
ve Tests, aynı yöntemi kullandıklarında bu veri setinde aynı sonucu veriyor.
Ancak üç yöntem hatası klinik yorum değiştirebilir:

1. Mixed ANOVA, denek içi bağımlılığı modellemiyor.
2. Non-inferiority `upper` p-değeri yanlış kuyruğu kullanıyor.
3. Bazı geçerli repeated-measures sonuçları, post-hoc `Inf` ürettiğinde bütünüyle
   HTTP 400 oluyor.

Ek yüksek riskli doğrulama ve kullanıcı arayüzü açıkları da bulundu. Bu sürüm,
klinik üretim kullanımı için bu bulgular düzeltilmeden yeterli doğrulanmış
sayılmamalı.

## Veri seti

- Dosya: `dataset.csv`
- SHA-256:
  `60f14859d7a125bd4db78835a6f92db5672473263bef9e00e13870a8f30f543e`
- Boyut: 10 × 20
- Sürekli değişkenler: yaş, biyobelirteçler, sonuçlar, üç tekrarlı ölçüm
- Kategorik değişkenler: ikili, üç seviyeli, tabaka, sıralı doz
- Eksikler: `biomarker_normal` 1, `biomarker_skew` 1,
  `category_three` 1, `followup_score` 1
- Üretim: rastgelelik yok; hasta verisi yok

Tekrarlı ölçüm denetiminde, üç ölçüm sütununda tam olan 9 kişi uzun biçime
çevrildi: 27 × 4. Kaynak veri değişmedi.

## Çalıştırma ve doğrulama

Denetim komutu:

```bash
.venv/bin/python qa/statistical_audit_10x20/audit.py
```

Bağımsız referans:

```bash
Rscript qa/statistical_audit_10x20/reference.R \
  qa/statistical_audit_10x20/dataset.csv
```

Ortam: Python 3.10.7, R 4.5.2, NumPy 2.2.1, pandas 2.2.3, SciPy 1.15.1,
statsmodels 0.14.4. Denetim başlangıç HEAD'i:
`8fe62b04c1f7f7d3bf6abadb08881f283e9ba3df`.
Audit JSON, çalıştırılan 11 ürün kaynağının SHA-256 değerlerini ve dirty diff
hash'ini de kaydediyor.

Son test koşuları:

- Backend: `1103 passed, 2 skipped, 82 warnings`
- Frontend Vitest: `46 passed` dosya, `374 passed` test
- TypeScript: hata yok
- 32 endpoint çağrısı: beklenmeyen HTTP durum kodu yok
- R ile standart test karşılaştırmaları: Mixed ANOVA dışındaki t, ANOVA,
  nonparametric, contingency, ANCOVA/MANCOVA ve RM ANOVA değerleri tolerans
  içinde; non-inferiority p-değerleri ayrı yöntem kontrolünde başarısız

Yeşil test paketi, aşağıdaki adversarial yöntem hatalarını kapsamıyor.

## Table ile Tests tutarlılığı

| Değişken | Table yöntemi ve p | Tests yöntemi ve ham p | Sonuç |
|---|---:|---:|---|
| `biomarker_normal` | Student t, `<0.001` | Student t, 0.0004678898 | Uyumlu |
| `biomarker_skew` | Student t, `0.009` | Student t, 0.0091463382 | Uyumlu |
| `ancova_outcome` | Student t, `0.005` | Student t, 0.004673627 | Uyumlu |
| `category_binary` | Fisher, `0.524` | Fisher, 0.523809524 | Uyumlu |
| `category_binary` | Fisher, `0.524` | Pearson χ², 0.518605016 | Fark beklenen: yöntem farklı |
| `category_three` | Fisher-Freeman-Halton MC, `1.000` | Pearson χ², 0.893597 | Fark beklenen: yöntem farklı |

`category_binary × arm` tablosunda bütün beklenen hücre sayıları 5'in altında.
Tests χ² sayısı Yates düzeltmeli hesapla eşleşiyor ve endpoint Fisher uyarısı
veriyor. Table otomatik olarak Fisher seçiyor. Dolayısıyla önceki p farkı aynı
istatistiğin farklı hesaplanması değil; farklı test seçimi.

Eksik kategorik değer, kategori gibi sayılmadı. `category_three` için Table
`n=9` döndürdü ve 9/10 satır kullanıldığı uyarısını üretti. Güncel
`Table1Panel` bu uyarıyı gösteriyor.

## Kritik bulgular

### K1: Mixed ANOVA yanlış hata yapısı kullanıyor

`backend/routers/repeated.py:393-399`, açıklamada deneği random effect saysa da
formülde `subject_col` yok. Kod düz OLS factorial ANOVA çalıştırıyor. Aynı
kişinin tekrarlı satırları bağımsız sayılıyor.

| Etki | Endpoint F, p | R split-plot F, p |
|---|---:|---:|
| Zaman | 5.2672, 0.013998 | 298.4722, 3.3181e-12 |
| Kol | 8.0706, 0.009788 | 2.7222, 0.142947 |
| Etkileşim | 0.0137, 0.986377 | 0.7778, 0.478297 |

Kol etkisi endpointte anlamlı, doğru denek hata terimiyle anlamsız. Klinik
karar tersine dönüyor.

### K2: Non-inferiority upper-bound p-değeri ters

`backend/routers/stats/inferential.py:559-567`, marjdan uzaklığı pozitif `z`
olarak kurup `norm.cdf(z)` çağırıyor. Upper-bound hipotezi için bu yön p'nin
tamamlayıcısını veriyor.

- Sürekli: MD=9; %90 GA [4.67733, 13.32267]; marj=20.
  Karar `non_inferior=true`; endpoint p=0.999986; doğru tek yönlü
  Welch-t p=0.0007640831.
- İkili RR: RR=2; %90 GA [0.75581, 5.29231]; marj=3.
  Karar `false`; endpoint p=0.753442; doğru p=0.246558.

Sürekli dal ayrıca t-tabanlı güven aralığından standart hata çıkarırken
`z_one` kullanıyor. t ve z ölçekleri karışıyor.

## Yüksek riskli bulgular

### Y1: RM ANOVA sphericity düzeltmesi uygulanmıyor

Ana RM ANOVA bu dengeli fixture'da R ile eşleşti:
F(2,16)=307, p=1.73075e-13. Ancak kod yalnız `res.epsilon` var mı diye bakıyor.
statsmodels `AnovaRM` Greenhouse-Geisser düzeltmesini uygulamıyor. Üç veya daha
fazla zaman noktasında sphericity bozulursa raporlanan p güvenilir değil.

### Y2: Sabit post-hoc farkı tüm RM ANOVA yanıtını düşürüyor

Geçerli dengeli veride omnibus sonuç hesaplanıyor. Bir paired post-hoc farkı
sabit olunca SciPy `t=Inf` üretiyor. Global JSON sonluluk koruması bütün yanıtı
HTTP 400 yapıyor; geçerli omnibus sonuç kayboluyor.

### Y3: Ordinal sıra sessizce alfabetikleşebiliyor

Jonckheere-Terpstra backend, sayı olmayan seviyeleri varsayılan alfabetik
sıralıyor. `Low/Medium/High`, `High/Low/Medium` olur. Bu sıralama trend yönünü
ve p-değerini değiştirebilir. `HypothesisPanel` seviye sırası göndermiyor.
Ayrı categorical trend panelinde manuel sıra alanı bulunuyor.

### Y4: Binary testler binary veri şartını doğrulamıyor

- Binomial, one-proportion ve two-proportion 3+ seviyeli değişkeni kabul edip
  en sık seviyeyi “success” seçebiliyor.
- Cochran Q sürekli sütunları kabul edip negatif Q ve 1'den büyük “proportion”
  üretebiliyor.
- Categorical UI, bütün kategorik ve sayısal sütunları binary seçiciye koyuyor.

Sonuç matematiksel görünebilir ama test tanımı geçersiz.

### Y5: Mantel-Haenszel geçerli test sonucunu sonsuz OR yüzünden kaybedebiliyor

Bazı tabakalarda sıfır hücre olduğunda ortak OR `Inf`; χ² ve p sonlu olsa bile
global JSON koruması endpointi HTTP 400 yapıyor.

### Y6: Etki yönü bazı çıktılarda yanlış veya kayıp

- Wilcoxon iki yönlü SciPy istatistiğini signed rank-biserial dönüşümünde
  kullanıyor; bütün farklar pozitif ve bütün farklar negatif olduğunda ikisi
  de `r=-1` olabilir.
- McNemar p-değeri doğru; fakat `a/b/c/d` açıklaması crosstab yönüyle ters,
  odds-ratio yönü yanlış yorumlanabilir.

### Y7: UI geçersiz/stale seçim gönderebiliyor

- Two-way ANOVA isteği `factor1: groupCol` gönderiyor; fakat `two_way`,
  `needsGroup` listesinde yok. Kullanıcı factor1 seçicisini göremiyor.
- Test türü değişince `col`, `col2`, `groupCol` yeni seçeneklere göre
  sıfırlanmıyor. Görünmeyen eski numeric seçim categorical endpointine
  gönderilebilir.
- API cevapları büyük ölçüde `any/object`; backend-frontend sözleşme kayması
  derleme zamanında yakalanmıyor.

## Orta risk ve yöntem sınırlamaları

- Table varsayılan normalite kontrolü gruplar içinde değil, havuzlanmış genel
  Shapiro-Wilk. Kullanıcı `within_group` seçebiliyor; varsayılan yöntem grup
  testi varsayımını tam sınamıyor.
- Table, normalite ön-testine göre t/Mann-Whitney veya ANOVA/Kruskal seçiyor.
  Bu veri-bağımlı test seçimi küçük n'de oynak. Grup başına minimum n koruması
  yok.
- İkiden fazla grup için klasik one-way ANOVA var; heteroskedastisite halinde
  Welch omnibus fallback yok. Sadece post-hoc Games-Howell'a geçebiliyor.
- Çok seviyeli kategorik SMD, multinomial negatif kovaryansları kullanmıyor.
  Bu nedenle çok seviyeli değişkenlerde SMD yanlı hesaplanabilir.
- Table yalnız biçimlenmiş p döndürüyor (`0.009`, `<0.001`); ham p yok.
- Table değişken başına p üretir; çoklu test düzeltmesi yok.
- Fisher-Freeman-Halton Monte Carlo: seed 42, 5000 örnek. Yanıt MC standart
  hatasını ve tekrar sayısını vermiyor.
- ANOVA Levene ihlalinde omnibus yine klasik F; yalnız post-hoc yöntemi
  değişiyor.
- MANCOVA UI Box M, multivariate normality ve multicollinearity iddia ediyor;
  backend assumption çıktısı üretmiyor.
- Cramér's V açıklaması bias correction çağrışımı yapıyor; formül düz V.
- Odds ratio magnitude, OR<1 için reciprocal kullanmıyor.
- Gatekeeping ayarlı p yaklaşık 0.0005 grid ile hesaplanıyor; geçersiz gamma
  sessizce sınırlandırılıyor.
- One/two-proportion z testleri bu fixture'daki n=5 kolları uyarısız çalıştırıyor.
  Küçük örnekte normal yaklaşım uygun değil.

## Yöntem kaynakları

- SciPy `chi2_contingency`: Yates düzeltmesi ve beklenen frekans notları  
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html>
- statsmodels `AnovaRM`: dengeli veri şartı; sphericity düzeltmesi uygulanmıyor  
  <https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html>
- statsmodels `MixedLM`: bağımsız gruplar/random effect yapısı  
  <https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html>
- R normal dağılım API: `lower.tail` tanımı  
  <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Normal.html>
- SciPy Shapiro-Wilk  
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html>
- SciPy one-way ANOVA varsayımları  
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html>

## Sınırlar

- 10 satır, büyük örneklem yaklaşımlarını, güç analizini veya gerçek dünya
  heterojenliğini doğrulamaz.
- Two-way ANOVA endpointi minimum 12 satır şartı nedeniyle bu fixture'ı doğru
  biçimde reddetti; sayısal yöntem fixture ile doğrulanmadı.
- Monte Carlo sonuç, deterministik seed ile yeniden üretilebilir; exact RxC
  Tests endpointi yok.
- Çalışma ağacı denetim sırasında başka süreç tarafından değişti. Rapor,
  yukarıdaki HEAD ve denetim anındaki dirty worktree üzerinde üretildi.
  Ürün kodu geri alınmadı, stage edilmedi, commit edilmedi.

## Sonraki düzeltme sırası

1. Mixed ANOVA modelini denek hata yapısıyla değiştir; R doğrulama testini CI'a ekle.
2. Non-inferiority upper/lower hipotez yönlerini düzelt; CI-karar-p invariants ekle.
3. Binary/cardinality doğrulamalarını endpoint ve UI'a ekle.
4. RM sphericity ve post-hoc non-finite davranışını düzelt.
5. Jonckheere sırasını zorunlu ve görünür yap.
6. Table ham p, yöntem/varsayım metadata ve çoklu test seçeneği ekle.
