# Açık bulgular için uygulama planı

Bu belge, işi devralacak ajan için yazıldı. Önceki oturumun bağlamı sende yok; gereken her şey burada.

## Bağlam

- Proje: uSTAT, klinik biyoistatistik uygulaması. Kök: `/Users/yh/Documents/projects/wiz3`.
- Backend: FastAPI, `backend/` (Python 3.10, venv `backend/.venv`). Frontend: React + TypeScript + Vite, `frontend/`.
- Analizler backend'de çalışır; frontend `/api/...` üzerinden çağırır (Vite proxy, port 8000).
- 100 x 40 sentetik veriyle arayüz turu yapıldı. Girdi dosyaları bu klasörde:
  - `dataset.csv`: test verisi. `generate_dataset.py` aynı dosyayı yeniden üretir.
  - `TEST_PLANI.md`: aşama aşama test adımları ve değişken sözlüğü.
  - `BULGULAR.md`: düzeltilen 10 hata, açık 9 bulgu, çalıştırılamayan adımlar.
- Son commit: `19d3695` (main, push edildi). Bu plan o commit'in üstüne kurulur.

## Çalışma kuralları

- Kullanıcıya Türkçe yanıt ver.
- Uzun tire (em dash) hiçbir yerde kullanılmaz: sohbet, kod yorumu, commit mesajı, belge. Teslimden önce eklediğin satırları tara. Kısa tire yalnız sayı aralıklarında.
- claude.ai artifact'i üretme; çıktılar yerel dosya.
- Commit ve push yalnız kullanıcı isteyince. Biçim: `<type>: <description>`.
- Her faz en çok 5 dosyaya dokunur. Faz bitince doğrulamayı koş, kullanıcıya raporla, onay bekle; sonraki faza kendiliğinden geçme.
- Bir dosyayı düzenlemeden önce yeniden oku. 500 satırdan uzun dosyaları parça parça oku.
- Yanıt alanları yük taşır: backend yanıtında bir alanın adını veya anlamını değiştirmeden önce frontend'de o alanı grep'le.
- `docs/ALPAR_UYGULAMA_DURUMU.md` ve `scratch_toc.txt` bu işe ait değil; commit'e katma.
- Hata düzeltirken önce hatayı gösteren testi yaz, sonra düzelt.

## Doğrulama komutları

Backend (tam paket yaklaşık 2 dakika):

```bash
cd backend && .venv/bin/python -m pytest -q -p no:cacheprovider --deselect tests/test_stats_new_methods.py::test_anova_scheffe_table
```

`test_anova_scheffe_table` ortamda `scikit_posthocs` olmadığı için hariç; bu işle ilgisi yok.

Frontend:

```bash
cd frontend && ./node_modules/.bin/tsc -b && ./node_modules/.bin/eslint . --quiet && ./node_modules/.bin/vitest run
```

- Tip denetimi `tsc -b`; `--noEmit` burada hiçbir şey denetlemez.
- `npx` kullanma, `./node_modules/.bin` altından çalıştır.
- Venv'de ruff yok; sistemdeki `ruff check <dosya>` kullanılır.
- Beklenen başlangıç durumu: backend 2737 geçer, 3 atlanır; vitest 1131 geçer.

## Arayüz testi nasıl yapılır

- Sunucular `.claude/launch.json` içindeki `backend` (8000) ve `frontend` (5173) yapılandırmalarıyla başlatılır.
- Backend `--reload` olmadan çalışır: backend kodunu değiştirince sunucuyu yeniden başlat. Oturum verisi bellekte tutulduğu için yeniden başlatınca veri yeniden yüklenmelidir.
- Veri yükleme: sayfadaki `#file-input` öğesine `DataTransfer` ile dosya ver ve `change` olayı tetikle. Dosya içeriğini sayfaya almak için CSV'yi geçici bir yerel HTTP sunucusundan (CORS başlığıyla) `fetch` etmek yeterli; Vite `/@fs/` yolu 403 döner.
- React kontrollü `select` ve `input` için değer, native setter ile atanıp `change` / `input` olayı tetiklenerek değiştirilir.
- Çoklu seçimli `select` öğelerinde (`multiple`) seçenekleri tek tek `selected` yapıp `change` tetikle.
- Tek bir tarayıcı betiği 45 saniyede zaman aşımına uğrar: uzun işleri sayfada arka planda başlat, sonuçları bir `window` değişkenine yaz, sonra oku.
- Sonuç metnini okuyamıyorsan ekran görüntüsü al; "çalıştı" demek için sonucu görmüş olmalısın.

## Faz sırası

Önerilen sıra: 6, 1, 2, 3, 4, 5. Faz 6 öne alınır, çünkü backend yeniden başlatılınca son düzeltmeler arayüzde doğrulanır ve çıkan yeni bulgular diğer fazlara eklenebilir.

Kullanıcıdan beklenen üç karar var (Faz 3, 4, 5). Her birinde bir öneri yazılı. Kullanıcı açıkça "önerilerle ilerle" demediyse o faza başlamadan sor.

---

## Faz 6: Çalıştırılamayan test adımları

Amaç: `TEST_PLANI.md` içindeki kalan adımları arayüzden çalıştırmak ve son backend düzeltmelerini arayüzde doğrulamak.

Adımlar:

1. Backend'i yeniden başlat, `dataset.csv` dosyasını yükle.
2. Önce son düzeltmeleri arayüzde doğrula:
   - Poisson: BIC pozitif ve AIC'den büyük olmalı (eskiden -179.09 idi).
   - Causal+ > IV: maruziyet `arm`, enstrüman `instrument`, sonuç `qol`. Sonuç gelmeli.
   - Visual > Linear Mixed Model: gruplama `pid` iken 422 ve açıklama; `site` iken uyum.
   - PSM ve IPTW: tedavi `arm`, düzey `treat` seçilip çalıştırılmalı.
3. Kalan adımlar:
   - Aşama 0: `stage` ve `grade` düzey sırası, veri sözlüğü etiketi.
   - Tests: ANCOVA, MANCOVA, RM ANOVA, Mixed ANOVA, Friedman (çoklu seçim gerekir), gatekeeping, epidemiyoloji (hız oranı).
   - Correlation: kappa sonucu (`rater1`, `rater2`).
   - ROC: çoklu eğri ve DeLong (`biomarker`, `biomarker2`), birleşik model.
   - Models: çoklu sonuç regresyonu, HR tablosu, threshold, alt grup, Progressive Adjustment, ML.
   - Survival Advanced: Cox ile MICE (yordayıcıya `crp` ekle), PH testi, zamanla değişen Cox, yaşam tablosu.
   - Validation: dış doğrulama, NRI / IDI.
   - Causal+: target trial, DiD, RDD.
   - Missing: örüntü, MCAR testi, MICE.
   - Visual: grafikler ve dışa aktarım.
   - Çıktı görüntüleyici, kayıtlı analizler, bayat sonuç uyarısı.
4. Her adım için: sonuç geldi mi, n ve dışlanan satır sayısı beklenenle aynı mı, yön `TEST_PLANI.md` ile uyumlu mu.
5. Kuşkulu sayısal sonuçları bağımsız doğrula: aynı modeli `backend/.venv/bin/python` içinde statsmodels veya lifelines ile kur ve karşılaştır.

Bu veriyle yapılamayanlar (ayrı veri biçimi ister): tekrarlayan olay, çok durumlu model, joint model, dış doğrulama kohortu. Bunlar için `qa/models_audit_30x50/` altındaki yardımcı CSV dosyaları var.

Çıktı: `BULGULAR.md` güncellenir (çalışan adımlar tabloya, yeni bulgular açık listeye). Bu fazda kod değiştirme; yalnız kesin ve küçük bir hata görürsen kullanıcıya sor.

Bitti ölçütü: listedeki her adım ya "çalıştı" (sonuç görüldü) ya da "çalıştırılamadı" (nedeniyle) olarak kayıtlı.

---

## Faz 1: Küçük ve net düzeltmeler

1. **p yazımı.** `p = <0.001` yerine `p < 0.001`.
   - Görüldüğü yerler: Summary normallik satırı, t-testi, eşli t, Wilcoxon, CMH, non-inferiority özet cümleleri.
   - Önce kaynağı bul: `grep -rn "p = {" backend/routers backend/services` ve frontend'de `p = ` ile `<0.001` üreten biçimleyiciler. Cümleler çoğunlukla backend'de `result_text` olarak üretiliyor.
   - Tek tek yamalamak yerine ortak bir biçimleyiciden geçir (`p < 0.001` ya da `p = 0.021`).
   - Dikkat: testler bu cümleleri karşılaştırıyor olabilir; etkilenen testleri güncelle.
2. **RCS ondalıkları.** Düğümler `20.2, 25.8, 29.34, 35.4`, nadir `25.0617` olarak görünüyor.
   - Backend: `backend/routers/models/cox.py` içindeki `_round_sig`. Frontend: `frontend/src/components/RCSPanel.tsx`.
   - Hedef: düğümler ve nadir aynı kuralla, aynı basamak sayısıyla gösterilsin. Küçük ölçekli yordayıcıda (0.01 civarı) düğümler birbirine yapışmamalı; bunu koruyan test var (`backend/tests/test_fix_cox_rcs_audit.py`).
3. **Compute hata mesajı.** Formüle `log(crp)` yazınca "Column 'log' not found" dönüyor.
   - `backend/routers/compute.py`. Bilinmeyen ad ardından parantez geliyorsa "fonksiyon desteklenmiyor; log dönüşümü için Transform sekmesi" de.
4. **Survival Advanced seçimleri.** Yöntem değişince (Kaplan-Meier'den Cox PH'ye) süre ve olay seçimi boşalıyor.
   - `frontend/src/components/SurvivalAdvancedPanel.tsx` ve `frontend/src/components/survival/`. Süre, olay ve grup seçimini yöntemler arasında ortak tut.

Bitti ölçütü: her madde için test, tam paketler yeşil, arayüzde birer kez görüldü.

---

## Faz 2: Kimlik sütunu

Sorun: `pid` sayısal değişken gibi sunuluyor. Table 1'de varsayılan seçili; RCS, Cox, t-testi, ROC listelerinde ilk seçenek olduğu için varsayılan değer oluyor.

- Mevcut kurallar: frontend `isIdLike` (`frontend/src/components/VisualModelPanel.tsx`, yaklaşık satır 341), backend `_is_id_like` (`backend/routers/models/linear.py`, yaklaşık satır 638). İkisi aynı sonucu vermeli.
- Frontend kuralını ortak bir dosyaya taşı (`frontend/src/lib/`), birim testi yaz.
- Kimlik sütununu listelerden silme; yalnız varsayılan seçimden çıkar:
  - Table 1: varsayılan işaretli gelmesin (`frontend/src/components/Table1Panel.tsx`).
  - Varsayılanı "ilk sütun" olan seçiciler: ilk kimlik olmayan sütunu seç.
- 5 dosya sınırını aşarsa fazı ikiye böl (önce ortak yardımcı ve Table 1, sonra diğer paneller).

Bitti ölçütü: `dataset.csv` yüklenince hiçbir panelde `pid` varsayılan sonuç, yordayıcı veya süre değil; LMM gruplama önerisi bozulmadı.

---

## Faz 3: Missing guard ile uç değer tutarsızlığı

Sorun: `admissions` sütununda 20 değeri "şüpheli" işaretli. Üç yer üç ayrı şey söylüyor:

- Data sekmesi başlığı: "PRESENT, not missing".
- Missing guard: "1 of 100 rows have missing values".
- Model: 100 satırla uyuyor.

Kaynak: `backend/services/impute.py` içindeki `missing_info`, `flag_sentinels` sonucunu eksik sayısına katıyor (`count` = gerçek eksik + şüpheli). Yanıtta `raw_count` ve `n_implausible` ayrı ayrı zaten var. Guard: `frontend/src/components/MissingGuard.tsx`.

**Karar gerekli.**

- Öneri: guard yalnız gerçek eksikleri saysın (`raw_count`), şüpheli değerleri ayrı satırda "1 şüpheli değer, analize dahil" diye göstersin. Modelin yaptığıyla tutarlı olur.
- Alternatif: şüpheli değerler gerçekten analiz dışı bırakılsın. Sonuçları değiştirir ve gerçek uç değerleri sessizce atar; önerilmez.

Başlamadan önce doğrula: `apply_imputation` şüpheli değerleri düşürüyor mu? Tur sırasında Poisson 100 satırla uydu, yani düşürmüyor görünüyor; yine de kodu oku ve hangi uç noktaların `missing_info` sayısına güvendiğini grep'le.

Bitti ölçütü: aynı sütun için başlık, guard ve modelin n değeri aynı hikâyeyi anlatıyor.

---

## Faz 4: Model kullanımı

1. **Lineer regresyon 0/1 sonuçta uyarısız çalışıyor.**
   - Engelleme; yanıta uyarı ekle ve lojistik regresyonu öner. Backend `backend/routers/models/linear.py`, gösterim `frontend/src/components/ModelsPanel.tsx`.
2. **Model keşfi dağınık.**
   - Negatif binom, Gamma ve LMM "Visual > Models & Diagnostics" altında. Arama kutusu kaydı (`frontend/src/App.tsx`, test kaydı listesi, yaklaşık satır 193–212) bunları "models" sekmesine yönlendiriyor.
   - **Karar gerekli.**
     - Öneri: arama kayıtlarını doğru sekmeye yönlendir ve Models > Regression listesinin altına "Negatif binom, Gamma, LMM: Visual > Models & Diagnostics" bağlantısı koy.
     - Alternatif: bu modelleri Models sekmesine taşı. Daha doğru ama büyük iş; bu planın dışında tutulmalı.

Bitti ölçütü: aramada "Negative Binomial" seçilince modelin gerçekten bulunduğu ekran açılıyor.

---

## Faz 5: DCA için olasılık sütunu

Sorun: DCA paneli hazır bir olasılık sütunu istiyor (`frontend/src/components/DecisionCurvePanel.tsx`); modelden doğrudan eğri üretilemiyor. `TEST_PLANI.md` Aşama 9'daki DCA adımı bu yüzden yapılamadı.

**Karar gerekli.**

- Öneri: lojistik sonuç ekranına "tahmin edilen olasılığı sütun olarak kaydet" düğmesi. Küçük iş; DCA ve dış doğrulama da bundan yararlanır. Önce benzer bir mekanizma var mı bak (sonuç ekranında "Export Model (CSV)" ve "Predicted ..." bölümleri var).
- Alternatif: DCA paneline yordayıcı seçip modeli içeride kurma. Daha rahat, daha büyük.

Dikkat: yeni sütun eklemek oturum verisini değiştirir; bayat sonuç uyarısı (`StaleGuard`) ve geri alma davranışı bozulmamalı.

Bitti ölçütü: lojistik model kur, olasılığı kaydet, DCA'yı o sütunla çalıştır; eğri çıkıyor.

---

## Her fazın sonunda

1. Backend ve frontend doğrulama komutlarını koş; sayıları raporla.
2. Eklenen satırlarda uzun tire taraması yap.
3. `BULGULAR.md` dosyasını güncelle: düzelen madde "Düzeltilen hatalar" listesine, kalan "Açık bulgular" listesinde.
4. Kullanıcıya kısa rapor: ne değişti, ne doğrulandı, ne doğrulanamadı. Commit için izin iste.

## Kapsam dışı

- `backend/routers/models/cox.py` 2100 satırın üstünde (proje sınırı 800). Bölünmesi ayrı bir iş.
- `multistate.py`, `frailty.py`, `interval_censored.py` içleri hiç incelenmedi.
- Hücre ve satır sağ tık menülerinde emojiler duruyor; sütun menüsü yeniden düzenlendi, onlar düzenlenmedi.
