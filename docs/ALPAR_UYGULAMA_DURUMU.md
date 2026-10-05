# Alpar karşılaştırma raporu: uygulama durumu

Kaynak: `ALPAR_KITAP_KARSILASTIRMA_RAPORU.md`. Tarih: 2026-10-05.
Çalışma yetkisi: kod değiştirme ve yerel doğrulama. Commit veya yayın yapılmadı.

## Tamamlanan ek kapsam

| Alan | Eklenen veya düzeltilen özellik |
|---|---|
| Tanımlayıcılar | CV/CV%, çeyrek sapma, pozitif veride harmonik ortalama; normallikle ortak G1/G2, SE ve z değerleri |
| Dönüşümler | T skoru, 1/x, arcsin√p; geçersiz alan kontrolü |
| Hipotez testleri | Tek örneklem Wilcoxon; tek örneklem ve eşleştirilmiş işaret testi |
| Kategorik testler | Wilson tek oran GA; düzeltmesiz score testi ve uyumlu R kodu; Bowker/Stuart-Maxwell |
| Ki-kare | Beklenen frekans tablosu, düzeltilmiş standartlaştırılmış artıklar, kontenjans katsayısı |
| ANOVA | SS/df/MS tablosu; Scheffé ve Bonferroni; Welch altında klasik SS etki büyüklüklerinin açık etiketi |
| Tekrarlı ölçümler | Wilcoxon W+, W− ve asimptotik Z; Friedman koşul sıra ortalamaları ve toplamları |
| Korelasyon | Nokta çift serili seçenek; Kendall sonuç metni düzeltmesi; Somers' d; doğrusal/kuadratik ağırlıklı Kappa |
| Regresyon | Tolerans ve ANOVA tablosu; HC3/MICE için tablo sınırlamalarının açıklaması |
| t-testi | Ortalama farkı SE'si; klasik iki grup η²; Python ve R yanıtlarının eşlenmesi |
| Faktör analizi | Korelasyon determinantı; gerçek sklearn ML yöntemine uygun etiket ve R `fm="ml"`; yeniden üretim farkı notu |
| Güvenilirlik | Sabit maddelerde tanımsız korelasyonların `null` dönmesi; omega yöntem notu; Kappa'da gözlenmeyen ara kategori mesafesinin korunması |
| ICC sınır durumu | Sıfır artık varyansında F=0 hatasının giderilmesi; düzenli güven aralığının hesaplanamadığının açıklanması |

Önceki kapsamda standartlaştırılmış regresyon β, sıra istatistikleri, fark GA'ları,
Yates/RR, çok gözlemcili ICC seçenekleri ve Durbin-Watson da eklendi.

## İncelemeye hazır, kural istisnası bekleyen katsayılar

Gamma türetimi, split-half Spearman-Brown/Guttman ve KR-21 için belgelenmiş
formüllerle çalışan kod hazırlandı. Gamma, R `DescTools`; alfa, R `psych`;
post-hoc sonuçları, R `DescTools::ScheffeTest` ve `pairwise.t.test` ile
karşılaştırıldı. KR-20, 0/1 maddelerde ham alfayla aynı katsayı olarak sunulur.

`AGENTS.md` yeni istatistiklerin doğrudan kütüphane işlevine eşlenmesini
gerektiriyor. Bu katsayılardaki sınırlı formül istisnası kullanıcıya soruldu;
yanıt gelmeden politika açısından tamamlandı sayılmaz.

## Kullanım sınırları

- Ağırlıklı Kappa için anlamlı kategori sırası gerekir. SE, GA ve p tahmin edilmiyor.
- Split-half seçilen madde sırasının tek/çift maddelerine dayanır. Tek sayıda
  maddede Spearman-Brown yaklaşımı yaklaşık olarak etiketlenir.
- KR katsayıları yalnızca 0/1 maddelerde sunulur; KR-21 eşit madde güçlüğü
  varsayımına dayanır.
- Welch ANOVA altında SS tablosu ve η²/ω² klasik ayrışmayı betimler.
- Omega ve EFA için R çıktıları ayrı kütüphanelerin karşılaştırmasıdır;
  birebir aynı sonuç iddiası yoktur.

## Bu kapsamda eklenmeyen seçenekler

Biserial/tetrakorik korelasyon, latent normal dağılım modeli ve doğrulanmış
uygulama seçimi gerektiriyor. Alt-üst %27 ayırt edicilik ve taban/tavan analizi
için ölçek puanlama yönü ve kuramsal sınırlar gerekir. Detrended Q-Q ve dal-yaprak
grafikleri, mevcut raporlama düzeltmeleri dışında kalan görsel seçeneklerdir.
Bu maddeler raporda eksik olarak kalır.

## Doğrulama

Son tam backend koşusunda 2271 test; tam frontend koşusunda 1041 test geçti.
TypeScript derlemesi temiz; ESLint'te yalnızca önceden mevcut `SubgroupPanel`
uyarısı var. Üretim derlemesi başarılı; Python wheel ve R bundle kaynak
kimlikleri yeniden üretildi. `git diff --check` temiz.
