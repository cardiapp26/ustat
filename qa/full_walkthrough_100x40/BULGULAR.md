# Tam tur bulguları (2026-11-12)

Veri: `dataset.csv` (100 x 40). Yöntem: uygulama tarayıcıda açıldı, paneller arayüzden çalıştırıldı.
Model sonuçları statsmodels ve lifelines ile bağımsız uyumlarla karşılaştırıldı.

## Çalıştırılan ve sorunsuz geçen adımlar

| Aşama | Adım | Sonuç |
|---|---|---|
| 0 | CSV yükleme | 100 x 40, tür ve eksik sayıları doğru |
| 1 | Summary: `pid`, `crp` | crp: n = 85, çarpık, geometrik ortalama 1.67 |
| 2 | Table 1, grup `arm`, 40 değişken | n = 46 / 54; `smoke` eksikleri ayrı satırda |
| 3 | t, Mann-Whitney, ANOVA, Kruskal-Wallis, Jonckheere, ki-kare, Fisher | hepsi sonuç verdi |
| 4 | Korelasyon (6 çift) | `age` ile `sbp_base` rho = 0.334 |
| 5 | ROC: `biomarker` | eğri çizildi |
| 6 | Lojistik | katsayılar bağımsız uyumla aynı (arm OR 0.413) |
| 6 | Lojistik + `smoke` | n = 94, 6 satır dışlandı |
| 6 | Lineer `qol` | n = 91 |
| 6 | Poisson | aşırı yayılım uyarısı çıktı (4.05) |
| 7 | Lojistik RCS, `bmi`, 4 kovaryat | n = 94; kovaryat adları ve AIC tablosu doğru |
| 7 | Cox-RCS | n = 94, 63 olay |
| 8 | Kaplan-Meier `arm`, `smoke` | smoke: 94 hasta, 3 grup |
| 8 | Cox PH | HR'ler bağımsız uyumla aynı (arm 0.4696, C 0.6825) |
| 8 | RMST (tau 30), landmark (6 ay), Fine-Gray, aralık sansürü | hepsi sonuç verdi; landmark `stage` için düzey başına satır |
| 10 | PSM, IPTW (tedavi `diabetes`) | çalıştı |
| 3 | Eşli t, Wilcoxon (`sbp_base`, `sbp_m6`) | n = 95, fark 7.69 |
| 3 | McNemar, Cochran-Armitage, Mantel-Haenszel, binom, iki yönlü ANOVA | hepsi sonuç verdi |
| 3 | Non-inferiority (RD) | çalıştı; marj sorunu aşağıda |
| 4 | Cronbach (alfa 0.700), faktör analizi (tek faktör, KMO 0.736) | beklendiği gibi |
| 6 | Firth, OR tablosu, ordinal (Brant testi), multinomial (n = 94), ZIP | hepsi sonuç verdi |
| 7 | Poisson RCS (offset `followup_years`) | n = 94, aşırı yayılım uyarısı |
| 9 | İç doğrulama, 200 bootstrap | görünen AUC 0.754, düzeltilmiş 0.710, eğim 0.78 |
| 10 | Aracılık (`dose`, `mediator`, `qol`) | n = 91, ACME 0.0585 |
| 12 | Gamma/NB paneli, LMM, Compute (`ldl / hdl`) | Compute: 41 sütun, eksikler korundu |

## Düzeltilen hatalar

1. **Summary: metinde ham kaçış dizileri.** Normallik satırı `\u2014 report Mean \u00B1 SD` (kaçış dizisi düz metin olarak) yazıyordu. `DescriptivePanel.tsx`.
2. **Summary: basıklık değeri yok.** Şeritte yalnız "Kurtosis SE" vardı; "Kurtosis" eklendi.
3. **RCS: "-2 dropped".** Kategorik kovaryat dummy'lere açılınca sayı eksiye düşüyordu. Metin "4 covariates entered as 6 terms" biçimine çevrildi. `RCSPanel.tsx`.
4. **Poisson ve negatif binom: BIC yanlış ölçekte.** AIC = 391.41 yanında BIC = -179.09 gösteriyordu (statsmodels GLM sapma tabanlı BIC). Log-olabilirlik tabanlı hesaba geçildi. `backend/routers/models/glm.py`, test: `backend/tests/test_fix_glm_bic.py`.

5. **Tekrarlı ölçüm, güvenilirlik ve kategorik panellerde ham kaçış dizileri.** `\u2014`, `\uD83D\uDCAC`, `\u26A0`, `\u03B7\u00B2` düz metin olarak basılıyordu (10 satır, 3 dosya).
6. **OR tablosu: "2/100 variables entered multivariate".** Payda satır sayısıydı; aday yordayıcı sayısı (4) olmalı. `ModelsPanel.tsx`.
7. **Non-inferiority: ölçek değişince marj 1.20 kalıyordu.** RD için 1.20 anlamsız (her zaman "kanıtlandı" çıkar). Varsayılan artık ölçekle değişiyor (RR/OR 1.20, RD 0.10); kullanıcının yazdığı değer korunuyor.
8. **Causal+: metin ikili maruziyet tüm satırları düşürüyordu.** `arm` (control/treat) seçilince "Not enough complete observations" hatası. IV, aracılık, target trial ve DiD artık iki düzeyli metni 0/1 kodluyor ve kodlamayı `exposure_coding` alanında bildiriyor; 3+ düzeyde açık 422. Test: `test_fix_causal_text_exposure.py`.
9. **LMM: grup başına tek satırda yanlış sonuç.** Panel gruplama değişkenini `pid` seçiyor; geniş veride ICC = 0.5, `age` SE 0.0233 (p < 0.001; OLS'te p = 0.11). Artık 422 ve açıklama. Test: `test_fix_lmm_singleton_groups.py`.
10. **PSM ve IPTW: metin ikili tedavi desteklenmiyordu.** `arm` listede yoktu. Artık iki düzeyli sütunlar da listeleniyor; kullanıcı tedavi düzeyini açıkça seçiyor (`treated_level`), seçim yapılmadan çalıştırılırsa 422. Yanıtta `treatment_coding`. Test: `test_fix_psm_iptw_text_treatment.py`, `treatmentLevels.test.ts`.

## Açık bulgular (düzeltilmedi)

1. **Missing guard uç değeri eksik sayıyor.** `admissions` = 20 "sentinel" olarak işaretli; Data sekmesi "eksik değil" diyor, model 100 satırla uyuyor, ama guard "1 of 100 rows have missing values" gösteriyor.
2. **`pid` sayısal değişken gibi sunuluyor.** Table 1'de seçili; RCS, Cox, t-testi listelerinde varsayılan.
3. **Model keşfi dağınık.** Negatif binom, Gamma, LMM "Visual > Models & Diagnostics" altında; arama kutusu "models" sekmesine yönlendiriyor.
4. **Survival Advanced: yöntem değişince süre ve olay seçimi sıfırlanıyor.**
5. **Lineer regresyon 0/1 sonuçla uyarısız çalışıyor.**
6. **RCS düğüm ve nadir ondalıkları tutarsız:** `20.2, 25.8, 29.34, 35.4`, nadir `25.0617`.
7. **p gösterimi:** `p = <0.001` (Summary, t-testi, CMH, non-inferiority özetleri).
8. **Compute:** `log(crp)` için hata "Column 'log' not found"; fonksiyon desteklenmediğini söylemiyor.
9. **DCA hazır olasılık sütunu istiyor;** modelden doğrudan eğri üretilemiyor.

## Çalıştırılmayan veya sonucu okunamayan adımlar

Düzey sırası ve sözlük, ANCOVA, MANCOVA, RM ANOVA, Friedman (çoklu seçim kurulamadı), gatekeeping, epidemiyoloji, kappa sonucu, DeLong, çoklu sonuç regresyonu, HR tablosu, Cox MICE ve PH testi, zamanla değişen Cox, threshold (sonuç okunamadı), alt grup, ML, dış doğrulama, DCA, target trial, DiD, RDD, Missing sekmesi analizleri, grafikler, dışa aktarım, kayıtlı analizler.

## Not

Düzeltmeler testlerle doğrulandı (backend 2737, frontend 1131). Backend dev sunucusu yeniden başlatılmadı; BIC, Causal+, LMM ve PSM/IPTW düzeltmeleri arayüzde yeniden başlatınca görünür.
