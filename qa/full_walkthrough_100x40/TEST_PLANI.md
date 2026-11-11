# uSTAT tam tur test planı (100 x 40)

Veri: `qa/full_walkthrough_100x40/dataset.csv` (100 satır, 40 sütun, sentetik).
Yeniden üretim: `backend/.venv/bin/python qa/full_walkthrough_100x40/generate_dataset.py` (sabit tohum, aynı dosyayı verir).

Her aşama bir sekmeyi kapsar. Sırayla ilerle; bir aşama bitmeden sonrakine geçme.
Her adımda üç şeye bak: (1) hata veya boş sonuç var mı, (2) n ve dışlanan satır sayısı beklenenle aynı mı, (3) yön ve büyüklük "Beklenen" sütunuyla uyumlu mu.
Sayılar tek bir sentetik örneklemden gelir; küçük sapma normal, yön değişimi veya n farkı hata işaretidir.

## Değişken sözlüğü

| Sütun | Tür | Not |
|---|---|---|
| pid | kimlik | 1–100, analize girmemeli |
| site | kategorik (5) | küme / rastgele etki |
| age, bmi | sürekli | bmi sonuçla U biçimli ilişkili (RCS için) |
| sex | ikili (F/M) | |
| arm | ikili (control/treat) | karıştırılmış atama: yaş, diyabet, evre, instrument |
| stage | sıralı (I/II/III) | |
| smoke | kategorik (3) | 6 eksik |
| diabetes, htn | 0/1 | |
| sbp_base, sbp_m3, sbp_m6 | sürekli, tekrarlı | sbp_m6: 5 eksik; treat kolunda daha çok düşer |
| ldl, hdl, crp | sürekli | eksik 6, 8, 15; crp sağa çarpık |
| egfr, hb | sürekli | |
| biomarker, biomarker2 | sürekli | event_binary için AUC yaklaşık 0.63; biomarker2: 3 eksik |
| dose | 0/10/20/40 | control = 0 |
| instrument | sürekli | arm ile r yaklaşık 0.40, sonucu yalnız arm üzerinden etkiler |
| mediator | sürekli | arm ile artar, qol'u artırır |
| qol | sürekli sonuç | 9 eksik, yaşlılarda daha sık (MAR) |
| cost | pozitif, çarpık | Gamma GLM |
| admissions | sayım | aşırı yayılım (ort. 1.24, varyans 6.24) |
| followup_years | maruziyet | sayım modellerinde offset |
| grade | sıralı (mild/moderate/severe) | |
| event_binary | 0/1 sonuç | oran 0.38 |
| rater1, rater2 | 0/1 | uyum 0.70 |
| time, status | sağkalım | ay, 66 olay, 36 ayda idari sansür |
| cmp_status | 0/1/2 | 66 olay, 9 yarışan, 25 sansür |
| ic_l, ic_r | aralık sansürü | ic_r boş = sağ sansür (34 satır) |
| item1–item4 | Likert 1–5 | tek faktör |

## Aşama 0: Yükleme ve veri sekmesi

- [ ] CSV yükle. Beklenen: 100 satır, 40 sütun, yükleme uyarısı yok.
- [ ] Tür tespiti: `pid` kimlik; `stage`, `grade`, `smoke`, `site`, `arm`, `sex` kategorik; `diabetes`, `htn`, `event_binary`, `status` ikili; `dose` sayısal.
- [ ] `stage` (I < II < III) ve `grade` (mild < moderate < severe) düzey sırasını ayarla. Sonraki aşamalar bu sıraya dayanır.
- [ ] `dose` için tür geçersiz kılma: kategorik yap, sonra sayısala geri al. Tabloda değişmeli.
- [ ] Veri sözlüğü panelinde etiket ekle, kaydet, sayfayı yenile: kalıcı mı.
- [ ] Sütun sıralama, filtre, arama.

## Aşama 1: Summary ve normallik

- [ ] Tanımlayıcılar: `age`, `bmi`, `crp`, `cost`. Beklenen: `crp` ve `cost` sağa çarpık, medyan < ortalama.
- [ ] Geometrik ortalama: `crp`.
- [ ] Normallik: `age` (normal), `crp` (ret), `admissions` (ret).
- [ ] Eksik sayıları sözlük tablosuyla aynı mı.
- [ ] Split file: `arm` ile böl, sonuçlar iki grup için ayrı gelmeli; bölmeyi kapat.

## Aşama 2: Table 1

- [ ] Gruplama `arm`; satırlar: `age`, `sex`, `bmi`, `stage`, `smoke`, `diabetes`, `htn`, `ldl`, `crp`, `egfr`.
- [ ] Beklenen: n = 46 control, 54 treat. `crp` için medyan [IQR] ve parametrik olmayan test seçilmeli.
- [ ] `smoke` eksikleri ayrı satır veya dipnot olarak görünmeli, referans düzeye karışmamalı.
- [ ] SMD sütunu aç. Dışa aktar (docx, xlsx): tablo hücreleri dolu mu.

## Aşama 3: Tests

Parametrik ve parametrik olmayan:
- [ ] Bağımsız t: `sbp_m6` ~ `arm`. Beklenen: treat daha düşük.
- [ ] Mann-Whitney: `crp` ~ `arm`. Hodges-Lehmann satırı görünmeli.
- [ ] Eşli t ve Wilcoxon: `sbp_base` ile `sbp_m6`. n = 95.
- [ ] Tek yönlü ANOVA ve Kruskal-Wallis: `qol` ~ `stage`; post-hoc açık. Evre III en düşük.
- [ ] Jonckheere-Terpstra: `qol` ~ `stage` (sıralı).
- [ ] İki yönlü ANOVA: `qol` ~ `arm` x `stage`.
- [ ] ANCOVA: `sbp_m6` ~ `arm` + `sbp_base`.
- [ ] MANCOVA: `sbp_m6`, `qol` ~ `arm` + `age`.

Kategorik:
- [ ] Ki-kare ve Fisher: `event_binary` x `arm`.
- [ ] Cochran-Armitage eğilim: `event_binary` x `dose`.
- [ ] Mantel-Haenszel: `event_binary` x `arm`, tabaka `stage`.
- [ ] McNemar: `rater1` ile `rater2`.
- [ ] Binom ve tek oran z: `event_binary`, p0 = 0.30.

Diğer:
- [ ] Tekrarlı ölçümler: `sbp_base`, `sbp_m3`, `sbp_m6`, grup `arm`. Zaman x grup etkileşimi beklenir.
- [ ] Çokluk düzeltmesi (gatekeeping): yukarıdaki üç p değerini gir.
- [ ] Epidemiyoloji: hız oranı, olay `admissions`, kişi-zaman `followup_years`, grup `arm`.
- [ ] Non-inferiority: `event_binary` ~ `arm`, risk farkı, marj 0.10.

## Aşama 4: Correlation ve güvenilirlik

- [ ] Pearson, Spearman, Kendall: `age`, `bmi`, `sbp_base`, `egfr`, `crp`. Beklenen: `age` ile `egfr` negatif.
- [ ] Kappa: `rater1` ile `rater2`. Beklenen: orta düzey (uyum 0.70).
- [ ] Cronbach alfa: `item1`–`item4`. Madde arası korelasyon 0.31–0.46.
- [ ] Faktör / PCA: `item1`–`item4`. Tek faktör beklenir.

## Aşama 5: ROC

- [ ] Tek eğri: `biomarker` → `event_binary`. AUC yaklaşık 0.63, Youden kesim noktası.
- [ ] Çoklu eğri ve DeLong: `biomarker` ile `biomarker2`. n = 97 (eksikler düşer); fark anlamsız beklenir.
- [ ] Birleşik model: `biomarker` + `age` + `stage`.
- [ ] Threshold paneli: `biomarker` için kesim noktası.

## Aşama 6: Models, regresyon

- [ ] Lineer: `qol` ~ `age` + `arm` + `stage` + `mediator`. n = 91. Evre III katsayısı negatif.
- [ ] Lineer stepwise: aynı aday kümesi + `bmi`, `sex`, `diabetes`. Ölçüt AIC.
- [ ] Lojistik: `event_binary` ~ `age` + `stage` + `diabetes` + `arm` + `bmi`. Beklenen: `arm` treat OR < 1, yaş OR > 1. Hosmer-Lemeshow ve Pseudo R² etiketi görünmeli.
- [ ] Firth lojistik ve OR tablosu (tek + çok değişkenli): aynı değişkenler.
- [ ] Lojistikte `smoke` ekle: n = 94 olmalı (6 eksik dışlanır, referansa sayılmaz).
- [ ] Poisson: `admissions` ~ `age` + `arm` + `diabetes`, offset `followup_years`. Aşırı yayılım uyarısı beklenir.
- [ ] Negatif binom: aynı model. Alpha > 0 tahmin edilmeli.
- [ ] Gamma GLM: `cost` ~ `age` + `stage` + `arm`.
- [ ] Ordinal: `grade` ~ `age` + `stage` + `arm`. İki eşik, artan sırada.
- [ ] Multinomial: `smoke` ~ `age` + `sex`.
- [ ] LMM ve GEE: `qol` ~ `arm` + `age`, küme `site`. `pid` küme önerisi olarak gelmemeli.
- [ ] Çoklu sonuç regresyonu: `qol`, `sbp_m6` ~ `arm` + `age`.
- [ ] Model karşılaştırma: lojistik modelin `bmi` ile ve `bmi` olmadan iki sürümü.
- [ ] E-değeri: lojistik `arm` OR'si için.

## Aşama 7: RCS

- [ ] Lojistik RCS: yordayıcı `bmi`, sonuç `event_binary`, kovaryat `age`, `sex`, `stage`. Beklenen: U biçimi, doğrusal olmama testi anlamlıya yakın.
- [ ] Kovaryat özeti: `age`, `sex_M`, `stage_II`, `stage_III` satırları doğru adlarla.
- [ ] Düğüm karşılaştırması: 3, 4, 5 düğüm için AIC satırları.
- [ ] Referans seçenekleri: medyan, nadir, elle değer. Eğri referansta 1'den geçmeli.
- [ ] Dağılım katmanı ve "CI referansı dışlıyor" bölgeleri.
- [ ] Lineer RCS: `qol` ~ `bmi`. Poisson RCS: `admissions` ~ `bmi`, offset `followup_years`.
- [ ] Sonuç türünü Poisson'dan lojistiğe çevir: eski offset isteğe girmemeli.
- [ ] Cox-RCS: `time`, `status`, spline `bmi`, kovaryat `age`, `arm`, `stage`. Mutlak risk, ufuk 24 ay.
- [ ] Kovaryata `smoke` ekle: n = 94.

## Aşama 8: Sağkalım

- [ ] Kaplan-Meier: `time`, `status`, grup `arm`. Beklenen: treat eğrisi üstte, log-rank df = 1. Risk altındaki sayı tablosu.
- [ ] KM grup `smoke`: 6 satır dışlanır, log-rank df = 2.
- [ ] Medyan takip (ters KM) ve medyan sağkalım CI.
- [ ] Cox: `age` + `arm` + `stage` + `diabetes`. Beklenen: `arm` HR yaklaşık 0.47, evre III HR > 1, C yaklaşık 0.68. PH testi: global df = terim sayısı.
- [ ] Cox MICE: yordayıcılara `crp` ekle, eksik yöntemi MICE. Havuzlanmış HR, CI ve `fmi` görünmeli.
- [ ] Cox zamanla değişen ve ufuk analizi (12, 24 ay).
- [ ] Yaşam tablosu: 6 aylık aralıklar.
- [ ] Gelişmiş: RMST (tau = 30) `arm` ile; landmark (6 ay), `stage` düzey başına satır; yarışan riskler `cmp_status` (CIF ve Fine-Gray); aralık sansürü `ic_l`, `ic_r`.
- [ ] Cox tanıları: Schoenfeld, martingale, dfbeta.

## Aşama 9: Doğrulama, DCA, ML

- [ ] İç doğrulama: lojistik model, 200 bootstrap. Optimism düzeltilmiş AUC görünen AUC'den küçük olmalı.
- [ ] Sağkalım modeli iç doğrulaması: C-indeks ve kalibrasyon eğimi.
- [ ] Eklenen değer: temel `age` + `stage`, eklenen `biomarker`. Delta AUC, NRI, IDI.
- [ ] DCA: `event_binary`, modeller `biomarker` ve `age` + `stage` + `biomarker`. Eşik 0.05–0.60.
- [ ] ML: Random Forest, sonuç `event_binary`; özellik önemi. Sağkalım ML: `time`, `status`.

## Aşama 10: PSM, IPTW, Causal+

- [ ] PSM: tedavi `arm`, kovaryat `age`, `diabetes`, `stage`, `bmi`. Eşleşme sonrası SMD < 0.1 hedef; `stage` için düzey bazlı SMD.
- [ ] IPTW: aynı kovaryatlar, sonuç `event_binary`. Ağırlık dağılımı ve dengesi.
- [ ] IPTW sonuç olarak `grade` seç: 422 hata mesajı beklenir (0/1 değil).
- [ ] IV (2SLS): sonuç `qol`, tedavi `arm`, enstrüman `instrument`. İlk aşama F > 10 beklenir.
- [ ] Aracılık: `arm` → `mediator` → `qol`. Dolaylı etki pozitif.
- [ ] Alt grup ve forest: `arm` etkisi `sex`, `diabetes`, `stage` içinde.

## Aşama 11: Missing

- [ ] Eksiklik örüntüsü: 8 sütun eksikli.
- [ ] Little MCAR testi.
- [ ] MICE: `ldl`, `hdl`, `crp`, `qol`. 5 atama, tanı grafikleri.
- [ ] Missing guard: eksikli değişkenle model çalıştırırken uyarı ve çalıştır düğmesi yerinde kalmalı.

## Aşama 12: Visual, Compute, çıktı

- [ ] Grafikler: histogram (`crp`), kutu (`qol` ~ `stage`), saçılım (`age`, `egfr`), çubuk (`stage` x `arm`). Tema ve dışa aktarım (png, svg).
- [ ] Compute: `log_crp = log(crp)`, `pp = sbp_base - 80`; yeni sütun tabloda görünmeli, eksikler eksik kalmalı.
- [ ] Skor bileşimi: `item1`–`item4` toplamı.
- [ ] Çıktı görüntüleyici: üç sonucu ekle, sırala, docx olarak dışa aktar.
- [ ] Kayıtlı analizler: bir analizi kaydet, yeniden yükle, aynı sonuç.
- [ ] Bayat sonuç uyarısı: bir sonucu açıkken Compute ile sütun ekle; uyarı çıkmalı.

## Bu veri setinin kapsamadığı alanlar

Ayrı biçimde veri ister; `qa/models_audit_30x50/` altındaki yardımcı dosyaları kullan:

- Tekrarlayan olaylar: `dataset_recurrent.csv`
- Çok durumlu model: `dataset_multistate.csv`
- Uzun biçim (joint model, LMM tekrarlı): `dataset_long.csv`
- Dış doğrulama kohortu: `dataset_external.csv`
- Meta-analiz, güç analizi, zaman serisi, birleştirme (Merge), Bayes: kendi girdileri var, panel içinden elle.

## Hata kaydı

| Aşama | Adım | Beklenen | Görülen | Ekran görüntüsü |
|---|---|---|---|---|
| | | | | |
