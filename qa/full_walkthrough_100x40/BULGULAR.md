# Tam tur bulguları (2026-11-12, güncelleme 2026-11-12)

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
| 3 | ANCOVA (`sbp_m6` ~ `arm` + `sbp_base`), MANCOVA, Friedman (3 SBP ölçümü) | sonuç verdi; MANCOVA gösterimi düzeltildi (aşağıda) |
| 4 | Kappa (`rater1`, `rater2`) | κ = 0.394; elle doğrulandı (uyum 0.70, beklenen 0.5048) |
| 5 | Çoklu ROC ve DeLong | n = 97, ΔAUC +0.005, p = 0.945; ortak örneklemde elle doğrulandı |
| 6 | HR tablosu (tek + çok değişkenli) | düzeltilmiş arm HR 0.47, Cox ile aynı |
| 6 | Threshold (`qol` ~ `bmi`) | n = 91, kırılma 33.4, LRT p = 0.034, düz profil uyarısı |
| 6 | Alt grup (maruziyet `biomarker`; `sex`, `diabetes`, `stage`) | etkileşim p'leri ayrı sütunda |
| 6 | Progressive adjustment (`arm`) | düzeltmeden sonra: Crude OR 0.55, Model 2 OR 0.44, n = 100 |
| 8 | Cox PH, MICE ile (yordayıcıya `crp`) | 100 satır, 66 olay, havuzlama notu, PH testi (global χ² 3.60, df 5) |
| 9 | Lojistik olasılığını sütun olarak kaydet + DCA | uçtan uca çalıştı (n = 100) |
| 10 | IV (`arm`, enstrüman `instrument`) | n = 91, ilk aşama F = 16.5 |
| 10 | Target trial (`arm`), RDD (`qol`, `bmi`, kesim 30) | RD -0.142; RDD LATE +2.47 (anlamsız, beklenen) |
| 10 | PSM (`arm`, düzey `treat`) | 54 tedavi, 46 kontrol, 44 çift |
| 11 | Little MCAR + bağımlılık taraması | `qol` için yaşla ilişki yakalandı |
| 12 | Grafik (histogram `crp` ~ `arm`), bayat sonuç uyarısı | çalıştı |

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
11. **p yazımı.** `p = <0.001` yerine `p < 0.001` (ortak biçimleyici). Kategorik panelde `p` harfi düşmüştü; geri eklendi.
12. **RCS ondalıkları** tutarlı gösteriliyor (`20.18, 25.60, 29.04, 34.73`).
13. **Compute:** `log(crp)` için "Function 'log' is not supported. Use LOG ..." mesajı.
14. **Survival Advanced:** yöntem değişince süre ve olay seçimi korunuyor.
15. **Kimlik sütunu:** `pid` Table 1'de ve model seçicilerinde varsayılan değil (`frontend/src/lib/columnDefaults.ts`, `backend/services/column_identifiers.py`).
16. **Missing guard:** şüpheli değerler eksikten ayrı gösteriliyor ("admissions: 1 flagged value").
17. **Lineer regresyon 0/1 sonuçta** lojistik öneren uyarı.
18. **Model keşfi:** arama Negatif binom, Gamma ve LMM için doğru ekranı açıyor; Regression listesinde bağlantı var.
19. **DCA için olasılık sütunu:** lojistik sonucunda "Save predicted probabilities" düğmesi.
20. **MANCOVA sonucu `[object Object]` basıyordu.** Çok değişkenli testler tablo, etki büyüklüğü tek satır olarak gösteriliyor. `HypothesisPanel.tsx`, test eklendi.
21. **Cox paneli MICE ve PH testine ulaşamıyordu.** Survival Advanced Cox isteğe `imputation` göndermiyor, PH testini, dışlanan satırları ve havuzlama notunu göstermiyordu. Eksik veri seçimi ve `CoxModelDetails` bileşeni eklendi.
22. **MICE kullanılmışken "MICE kullanın" uyarısı.** `add_missing_data_diagnostics` artık havuzlanmış sonuçta bu öneriyi vermiyor. Test: `test_fix_missing_diagnostics_mice.py`.
23. **Progressive adjustment "No model converged".** Maruziyeti eksik satırlar tam vaka maskesinden düşmüyordu (boolean sütunda `notna()` hep doğru), her model NaN yüzünden sessizce düşüyordu; metin maruziyet (`arm`) tamamen NaN oluyordu. Maske düzeltildi, metin maruziyet düzey yoluna geçiyor ve not düşülüyor. `backend/routers/multimodel.py`, testler `test_multimodel.py` içinde.
24. **Missing Overview şüpheli değeri eksik sayıyordu** (`admissions` 1 eksik). Artık yalnız boşluklar sayılıyor, şüpheli değer "+1 flagged" olarak yanında; sütun temizlik için listede kalıyor.

25. **Alt grup: metin maruziyet desteklenmiyordu.** `arm` listede yoktu. Artık iki düzeyli metin sütunlar listeleniyor; kullanıcı "exposed" düzeyi seçiyor (`exposure_level`), sonuç "Odds ratio compares arm = treat with control" diyor. Sayısal 0/1 kodlamayla birebir aynı sonuç (test). `backend/routers/subgroup.py`, `SubgroupPanel.tsx`.
26. **Alt grup paneli güven aralığı olmayan tabakada çöküyordu** (`null.toFixed`, "This panel hit an error"). Biçimleyici ve grafik korundu; "CI not estimable" yazıyor.
27. **Alt grupta tam ayrışma OR 0.00, p = 1.000 olarak gösteriliyordu** (`diabetes = 1`, 10 hasta). Artık "not estimable: completely separated" notu.
28. **ROC özetinde eğri başına n yoktu.** Her eğrinin n'i gösteriliyor ve dışa aktarılıyor; n'ler farklıysa DeLong'un ortak örneklemi kullandığını söyleyen not çıkıyor.

29. **Missing bağımlılık taraması çoklu karşılaştırmayı düzeltmiyordu.** Değişken başına ~33 t-testi düzeltmesiz p < 0.05 ile yapılıyor, MCAR olan `crp`, `hdl`, `ldl` "MAR" çıkıyordu. Artık Holm düzeltmesi var; yalnız düzeltme öncesi anlamlı olanlar ayrı listede "possibly related ... exploratory" olarak gösteriliyor. Not: tasarımdaki zayıf gerçek sinyal (`qol` ~ yaş, ham p = 0.039, 9 eksik) da bu kademeye düşüyor; 9 eksikle ayırt etmek mümkün değil. Test: `test_fix_missing_dependence_holm.py`.
30. **Causal+ sonuçları maruziyet kodlamasını söylemiyordu.** IV, aracılık, target trial ve DiD cümlesi metin maruziyette "arm was coded 1 = 'treat', 0 = 'control'." ile bitiyor.
31. **İki düzeyli maruziyette gereksiz "P for trend" ve sıra uyarısı** kaldırıldı (Progressive adjustment); metin "P for trend: p = 0.093" biçiminde.
32. **RDD sonuçları 6 ondalıkla gösteriliyordu;** 4 anlamlı basamağa indirildi.
33. **LMM gruplama varsayılanı `pid` idi.** Kimlik sütunu yalnız tekrar ediyorsa (uzun biçim) öneriliyor; yoksa tekrar eden ilk sütun (`site`), o da yoksa yine kimlik (geniş veriyi uzun biçime çevirebilmek için).

34. **ModelsPanel'deki ölü KM ve Cox dalları silindi** (`ModelsPanel.tsx`, `CoefTables.tsx`, `ForestPlot.tsx`, `guidance.ts`). Eski kayıtlı `cox` / `km` model değeri "linear"a dönüyor; bilinmeyen model artık sessizce Cox'a düşmüyor.

## Açık bulgular (düzeltilmedi)

Turda bulunan açık bulgu kalmadı.

## Yanlış tespit

- **"PSM: ortak değişken seçilmeden çalıştırılınca mesaj yok"** yanlıştı. Düğme ortak değişken yokken zaten devre dışı ve altında "0 selected" yazıyor; düğme durumunu yanlış anda ölçmüştüm.

## Çalıştırılmayan veya bu veriyle yapılamayan adımlar

- **Bu veriyle yapılamaz (uzun biçim gerekir):** RM ANOVA, Mixed ANOVA, DiD. Tekrarlayan olay, çok durumlu model, joint model ve dış doğrulama için `qa/models_audit_30x50/` dosyaları kullanılmalı.
- **Çalıştırılmadı:** düzey sırası ve sözlük düzenleme, gatekeeping, epidemiyoloji, çoklu sonuç regresyonu, zamanla değişen Cox, yaşam tablosu, ML, dış doğrulama, NRI / IDI, MICE çalıştırma (Missing sekmesinden), dışa aktarım (dosya indirir), kayıtlı analizler.

## Not

Tüm düzeltmeler testlerle doğrulandı; backend yeniden başlatılıp arayüzde de denendi.
