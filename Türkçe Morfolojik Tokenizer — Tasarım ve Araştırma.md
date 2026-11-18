# Türkçe Morfolojik Tokenizer — Tasarım ve Araştırma

Oct 28, 2026 · @Yusuf

## Özet

Önerilen tokenizer, ekleri **arkifonemik** ID'lerle (`-lAr`, `-DAn`) saklar ve yüzey biçimini decode sırasında kökün **leksikal fonoloji özelliklerinden** deterministik olarak yeniden üretir; kayıpsızlık, her kelime için "üret ve doğrula" kontrolüyle yapı gereği garanti edilir.

Örnek: `evlerimizdekiler` → `ROOT:ev · SFX:lAr · SFX:(H)mHz · SFX:DA · SFX:ki · SFX:lAr`. Not: senin yazdığın `ev, ler, miz, de, ki, ler` bölmesinde `i` kayboluyor; yüzey `ev-ler-imiz-de-ki-ler`. 1. çoğul iyelik eki `-(H)mHz`'dir, ünsüzden sonra `imiz` olarak gerçekleşir. Arkifonemik ID bu `i`'yi otomatik geri getirir.

Üç ayrı isim alanı istediğin ayrımı sağlar:

- **ROOT** — sözlükte bilinen kök (fonolojik sınıfıyla birlikte tek ID).
- **LEX** — BPE parçaları; uzun/nadir kökler ve yabancı kelimeler (İngilizce, İspanyolca) burada bölünür. `Danimarka`'daki `dan` bir LEX parçasıdır.
- **SFX** — çekim/yapım ekleri. Ayrılma hâli `-DAn` bir SFX ID'sidir; `dan` LEX parçasıyla hiçbir ID'yi paylaşmaz.

Temel karar: uzun kökleri **uzunluğa göre değil frekansa göre** bölmek. Uzun ama sık bir kök (`çamaşırhane`) tek ID almalı; kısa ama nadir bir kök LEX parçalarına düşebilir.

## Verilen kaynaklar: ne yapıyorlar, nerede kalıyorlar

İki kaynak iki uç noktadır: TurkishTokenizer allomorfları birleştirir ama kayıplıdır; Morpheus kayıpsızdır ama allomorfları birleştirmez. Hedef tasarım ikisinin iyi yanını birleştirir.

|  | [TurkishTokenizer](https://arxiv.org/pdf/2508.14292) (Bayram vd. 2025) | [Morpheus](https://github.com/lonewolf-rd/TurkishMorpheus) (Şakar 2026) | Bu tasarım |
| --- | --- | --- | --- |
| Segmentasyon | Sözlük: en uzun kök eşleşmesi + iteratif ek eşleme | Nöral sınır dedektörü (Morfessor + Kalbur öğretmen) | Morfotaktik FSA üzerinde sentez-yoluyla-analiz |
| Allomorf paylaşımı | Var (`-lAr`, `-dAn/-tAn` tek ID; `kitap/kitab` tek ID) | Yok, yüzey string'i saklanır | Var, arkifonemik ID |
| Kayıpsızlık | Yok: ham satırlarda %0 roundtrip, `saatlerde→saatlarda` gibi kanonik yeniden yazım | %100 (byte fallback) | %100, üret-doğrula ile |
| Ek zinciri | Ayrı tokenlar | Sık birleştirir (`leri`, `deki`) | Her morfem ayrı |
| Sözlük | \~22K kök, \~230 ek, 10K BPE, 32K vocab | 50K segment vocab | \~30K kök + \~400 ek + \~16K LEX |
| Boşluk | Ayrı `<space>` tokenı | Kayıpsız | Kökte örtük; istisnalar için token |

TurkishTokenizer'ın kaybı tasarım kusurudur: ek ID'si yüzeyi bilmez, ama decoder'ın onu yeniden üretmek için ihtiyaç duyduğu istisna bilgisi (ör. `saat` kökünün ince ünlü uyumu) hiçbir yerde saklanmaz. Morpheus'un README'sindeki fenomen tablosu bunu doğruluyor: ünsüz yumuşaması ve ödünç kelime uyum istisnalarında TurkishTokenizer sınırı doğru koyuyor ama yüzey eşleşmesi %0'a düşüyor.

Morpheus'un kendi sınırı: ünsüz yumuşamasında %14, ünlü düşmesinde %17 doğruluk; ek zincirlerini birleştirme eğiliminde (`demir | cilik`). Yani yüzey koruyan nöral yaklaşım düzensiz durumları öğrenemiyor.

Çıkarım: kayıpsızlık ile allomorf birleştirme çelişmez. Gereken şey, düzensizliği **token'a değil kök sözlük kaydına** yazmak.

## Daha önce bu detayda bir Türkçe tokenizer var mı?

Bulabildiğim kadarıyla yok: tam morfotaktik + arkifonemik ek ID'si + leksikal özellikten kayıpsız yeniden üretim + LEX/SFX isim alanı ayrımını birleştiren açık bir Türkçe tokenizer yayımlanmamış. Parçaların her biri ayrı ayrı mevcut.

| Çalışma | Ne sağlıyor | Eksik olan |
| --- | --- | --- |
| Oflazer iki-seviyeli analizör (1994), [TRmorph](https://lexitron.nectec.or.th/public/LREC-2010_Malta/pdf/109_Paper.pdf) (Çöltekin), Zemberek, [TRMOR](https://journals.tubitak.gov.tr/elektrik/vol27/iss5/41), [MorTur](https://aclweb.org/anthology/2020.lrec-1.491.pdf) | FST tabanlı tam analiz + üretim; leksikon, morfotaktik ve iki-seviyeli fonoloji kuralları; TRmorph'ta segmentasyon ve bilinmeyen kelime tahmincisi | Tokenizer değiller: vocab, ID, BPE fallback, LM'e uygun dizi yok. Kelime başına çok analiz döndürüyorlar (`dolardır` için 4, 5 ve 17 analiz) |
| [TurkishTokenizer](https://arxiv.org/pdf/2508.14292) | Allomorf paylaşımlı ek ID'leri, `<uppercase>`, BPE fallback | Kayıplı; açgözlü kök eşleşmesi; düzensizlik leksikonda kodlanmamış |
| [Morpheus](https://arxiv.org/abs/2606.18717) | Kayıpsız, nöral, kök-aile embedding'i | Allomorf paylaşımı yok; yumuşama/düşme zayıf |
| [MorphBPE](https://arxiv.org/abs/2502.00894) (ACL Findings 2026) | BPE birleştirmelerini morfem sınırını aşmayacak şekilde kısıtlar; inference standart BPE | Allomorf normalizasyonu yok; Türkçe'de denenmemiş (İng./Rus./Macar./Arap.) |
| MorphPiece (Jabbar 2023) | Önce morfem, sonra BPE (İngilizce) | Türkçe fonolojisi yok |
| [HeceTokenizer](https://arxiv.org/pdf/2604.10665) | Hece tabanlı, retrieval odaklı | Morfem değil hece |
| [Vocab Diet](https://arxiv.org/pdf/2510.17001) | Yüzey varyantlarını taban + dönüşüm vektörüyle kompoze etme | Tokenizer değil, embedding yeniden şekillendirme; senin composer çalışmanla doğrudan ilgili |

En yakın öncül aslında FST analizörleri. Bu yüzden önerim: tokenizer'ı sıfırdan bir morfoloji motoru olarak değil, **iki-seviyeli morfoloji fikrinin tokenizer'a uyarlanması** olarak kurmak. Zemberek/TRmorph'un leksikonlarını (özellikle kök özellik bayraklarını) başlangıç verisi olarak kullanmak düşünülebilir; lisansları kontrol edilmeli (TRmorph GPL).

## Mimari: token isim alanları ve vocab bütçesi

Vocab, ID aralıklarıyla ayrılmış altı isim alanından oluşur; 48K'lık bir bütçe örneği aşağıda. Ayrım ID aralığında yaşar, böylece modele ayrıca bir **token-type embedding** (ROOT/LEX/SFX/PUNCT/SPECIAL) eklemek bedavadır.

| İsim alanı | Örnek | Boyut (taslak) | Not |
| --- | --- | --- | --- |
| SPECIAL | `<bos>` `<eos>` `<pad>` `<cap>` `<caps>` `<glue>` `<sp>` `<nl>` `<tab>` `<apos>` | \~64 | `<glue>`: kelimeden önce boşluk yok; `<sp>`: noktalamadan önce boşluk var |
| BYTE | `<0x00>`…`<0xFF>` | 256 | Son çare; `<unk>` hiç üretilmez |
| SFX (arkifonem) | `PL:lAr`, `ABL:DAn`, `P1PL:(H)mHz`, `REL:ki`, `PROG:Hyor` | \~350–450 | Çekim + yapım + klitik (`mI`, bağlaç `dA` ayrı ID) |
| SFX (işaretli allomorf) | `ABL!dan`, `DAT!a` | \~600 | Sadece üretici yanlış yüzey verirdiyse kullanılır (özel isim, kısaltma) |
| ROOT | `ev`, `kitap#N`, `kalp#N:heart`, `git#V` | \~30K | Her ID = sözcükbirim + fonoloji sınıfı; sesteş ama farklı davranan kökler ayrı ID |
| LEX (BPE) | `▁dani`, `marka`, `▁commun`, `ism` | \~16K | Kök kalıntıları + yabancı kelimeler; `▁` = kök başı. BPE **eklerinden arındırılmış** kökler + yabancı metin üzerinde eğitilir |
| PUNCT / DIGIT | `.` `,` `'` `0`…`9` | \~200 | Sayılar rakam rakam |

Kelime grameri (decoder bunu ayrıştırır):

```
word  := case? (ROOT | LEX_start LEX_cont*) (apos)? SFX*
case  := <cap> | <caps>
```

Kök bölümünün toplamının bir kök olduğu bu gramerden bilinir: ilk SFX tokenına ya da yeni bir kelime başına kadar ardışık LEX parçaları tek bir kök gövdesidir. `dan` LEX parçası ile `ABL:DAn` farklı ID aralıklarındadır; model ikisini karıştıramaz.

Neden ayrı LEX BPE'si: BPE tam kelimeler üzerinde eğitilirse `lar`, `dan`, `ları` gibi ek kalıplarını birleştirme olarak öğrenir ve kök parçalarına sızdırır. Eklerinden soyulmuş kök gövdeleri üzerinde eğitilen BPE bu sızıntıyı yapısal olarak engeller. MorphBPE'nin "birleştirme morfem sınırını aşamaz" kısıtıyla aynı fikrin daha sert hali.

## Encode pipeline

Encode bir düşme kademesidir: her kelime önce tam analizle denenir, başarısız olursa daha kaba yollara iner; her adımın çıktısı decode edilip orijinal kelimeyle karşılaştırılmadan kabul edilmez.

1. **Pretokenize (kayıpsız).** NFC dışında normalizasyon yok. Kelime / sayı / noktalama / boşluk / URL-kod bloklarına ayır. Boşluk düzeni `<glue>`, `<sp>`, `<nl>` ile kodlanır.
2. **Büyük harf.** Türkçe-farkında küçültme (`İ→i`, `I→ı`); `<cap>` / `<caps>`. Karışık büyük-küçük (`iPhone`) doğrudan LEX'e gider, harf bilgisi LEX parçalarında kalır.
3. **Sentez-yoluyla-analiz.** Kök trie'si, her kökün **alternatif gövdeleriyle** (`kitab`, `ağz`, `hakk`, `gid`) önceden genişletilir. Eşleşen her kök için morfotaktik FSA'da ilerlenir: her durumda bir sonraki arkifonem, o anki fonolojik bağlama göre (son ünlü, son ünsüz, ünlüyle bitme, kök bayrakları) yüzey biçimine çevrilir ve kalan string'in başıyla karşılaştırılır. Bu, iki-seviyeli bir FST'nin Python'da yazılabilir eşdeğeridir; sonra Rust'a derlenebilir.
4. **Belirsizlik giderme.** Çok analiz normaldir (`kalem` = kalem / kale+m, `yüzü`, `okul`). Morfem dizileri üzerinde EM ile eğitilmiş bir unigram (ileride bigram) model en olası analizi seçer, Unigram-LM tokenizer mantığı. Deterministik olmalı: aynı kelime her zaman aynı tokenlara gider.
5. **Bilinmeyen kök tahmincisi.** Tam analiz yoksa ek zinciri sağdan soyulur; kalan gövde ≥ 2 hece olmalı, son ek gövdenin son ünlüsüyle uyumlu olmalı, ve gövde korpusta ≥ k farklı ek zinciriyle görülmüş olmalı. Gövde LEX parçalarına bölünür: `komünistlerden` → `▁komün·ist` + `PL` + `ABL`.
6. **Yabancı kelime.** Kelime düzeyinde dil kimliği (karakter n-gram sınıflandırıcı) + tahminci reddi → tüm kelime LEX. `Avrupa → avrup+DAT gibi sahte ek soyma bu kapıyla engellenir; prototipteki kural: gövde ≥ 2 ünlü ve (≥ 2 ek ya da korpusta tanıklı gövde).`
7. **Doğrulama.** `decode(tokens) == kelime` değilse bir sonraki analiz, sonra işaretli allomorf, sonra LEX, en son BYTE. Böylece kayıpsızlık bir umut değil bir invariant olur.

Performans: analiz önbelleği (kelime tipi → token listesi) Zipf dağılımı sayesinde token akışının \~%95+'ini tek sözlük aramasına indirir; Morpheus'un `warm_cache` yaklaşımıyla aynı.

## Morfofonoloji kataloğu: düzenli kurallar ve düzensizlikler

Kural: düzenli olan şey üreticide **kural** olarak, düzensiz olan şey kök kaydında **bayrak** olarak ya da ek kaydında **istisna** olarak yaşar. Hiçbir düzensizlik token'a yazılmaz.

**Düzenli kurallar (üretici)**

| Kural | Arkifonem | Örnek |
| --- | --- | --- |
| 2'li ünlü uyumu | `A` → a/e | ev+lAr → evler, kitap+lAr → kitaplar |
| 4'lü ünlü uyumu | `H` → ı/i/u/ü | göz+Hm → gözüm |
| Ünsüz benzeşmesi (sertleşme) | `D` → d/t, `C` → c/ç, `G` → g/k | kitap+DA → kitapta, ağaç+CH → ağaççı |
| Kaynaştırma ünlüsü düşer | `(H)` ünlüden sonra düşer | araba+(H)m → arabam |
| Kaynaştırma ünsüzleri | `(y)` `(n)` `(s)` `(ş)` | araba+(y)A → arabaya, araba+(s)H → arabası, iki+(ş)Ar → ikişer |
| Zamir n'si | 3. tekil iyelikten sonra hâl eki `n` alır | ev+(s)H+DA → evinde |
| Kopula düşmesi | `(y)DH`, `(y)mHş`, `(y)sA`, `(y)ken` | güzel+(y)DH → güzeldi, hasta+(y)DH → hastaydı |

**Leksikal bayraklar (kök kaydı)**

| Bayrak | Örnek | Neden kural olamaz |
| --- | --- | --- |
| `soften` (p/ç/t/k → b/c/d/ğ, nk → ng) | kitap→kitabı, ağaç→ağacı, renk→rengi | top→topu, hukuk→hukuku, saat→saati: tek heceli ve ödünç köklerde tahmin edilemez |
| `soften` (fiil) | git→gidiyor, et→ediyor, tat→tadıyor | bit→bitiyor, at→atıyor |
| `drop_vowel` (orta hece düşmesi) | ağız→ağzı, burun→burnu, oğul→oğlu, şehir→şehri, resim→resmi | bağır (organ)→bağrı ama bağır (fiil)→bağırıyor: sesteşler ayrı ROOT ID |
| `geminate` | hak→hakkı, his→hissi, tıb→tıbbı, af→affı, ret→reddi | Arapça kökenli, liste gerektirir |
| `harmony=front` (uyum istisnası) | saat→saati, rol→rolü, alkol→alkolü, kalp→kalbi, hal→hali | Son ünlü kalın ama ekler ince |
| `aorist=Hr` | al→alır, bil→bilir, gel→gelir (+ \~10 tek heceli fiil) | Diğer tek heceliler `-Ar`: yap→yapar |
| `caus=…` | bat→batır, çık→çıkar, ol→oldur, öl→öldür | Ettirgen allomorfu fiile özgü |
| `y_poss` | su→suyu, ne→neyi | Beklenen `-sH` yerine `-yH` |

**Ek istisnaları ve supletif durumlar**

- Uyuma girmeyen ekler: `-Hyor`'un ikinci hecesi (`yor` sabit), `-ken`, `-leyin`, `-gil`, `-mtırak` kısmen. `-ki` çoğunlukla sabit ama `dünkü`, `öbürkü`.
- Ünlü daralması: `-Hyor` önünde a/e → ı/i/u/ü: ara→arıyor, başla→başlıyor; de/ye → diyor/yiyor, ayrıca diyen, yiyecek (bayrak).
- Edilgen allomorfu bağlamsal: `-Hl` / `-Hn` (l'den sonra) / `-n` (ünlüden sonra): yaz→yazıl, bil→bilin, oku→okun.
- Supletif zamirler: ben+DAT → bana, sen+DAT → sana; o/bu/şu + hâl → onu/bunu (n). Bunlar `(kök, ek dizisi önü) → yüzey` override tablosunda tutulur.
- Kalıplaşmış birleşikler kök sayılır: kahvaltı, cumartesi, ayakkabı (çift iyelik: ayakkabısı), bilgisayar.
- Şapka: `kâr`/`kar`, `hâlâ`/`hala` farklı kökler; şapka ayrıca ince `l`/`k` bilgisi taşır (`hal`→`hali`).

## Decode ve kayıpsızlık garantisi

Decode = ayrıştır + üret: token akışı kelime gramerine göre kelimelere bölünür, her kelimenin kök gövdesi seçilir, sonra arkifonemler soldan sağa bağlam durumu güncellenerek yüzeye çevrilir.

```
state = Context(last_vowel, last_cons, ends_vowel, root_flags)
surface = stem(root, next_suffix)        # kitap -> kitab, ağız -> ağz
for sfx in suffixes:
    piece = realize(sfx, state)          # DAn -> dan/den/tan/ten
    surface, state = join(surface, piece, state)
apply_case(surface, case_token)
```

Kök gövde seçimi bir sonraki ekin ilk sesine bağlıdır (ünlüyle başlayan ek → `kitab`, ünsüzle başlayan → `kitap`). Bu yüzden `stem()` sonraki eki görerek karar verir.

LEX köklerde bayrak yoktur: üretici kök bilgisini yüzeyden çıkarır (son ünlü, son ünsüz) ve yumuşama yapmaz. Bu varsayım yanlışsa encode sırasındaki doğrulama başarısız olur ve işaretli allomorf devreye girer; yani hata sessizce kayba değil, biraz daha uzun bir diziye dönüşür.

İnvariant, test olarak: korpustaki her satır için `decode(encode(x)) == x`. Morpheus'un 3 seviyeli roundtrip setini (30K çekimli kelime, 5K ham satır, 14 satırlık stres seti) aynen benimsemek mantıklı.

Üretim sırasında ek bir kazanım: decoder bir FSA olduğu için, transformer üretim yaparken morfotaktik olarak geçersiz bir SFX'i (ör. `PL` → `PL`) logit maskesiyle engelleyebilirsin. BPE'de bu mümkün değil.

## Zor kenar durumlar

Bu durumların ortak noktası: ek uyumu yazıma değil **okunuşa** göre belirlenir, ve okunuş string'den çıkmaz. Çözüm her yerde aynı: okunuş tablosu varsa üret, yoksa işaretli allomorfa düş.

| Durum | Örnek | Strateji |
| --- | --- | --- |
| Özel isim + kesme | `Danimarka'dan` | `<cap> ROOT:danimarka <apos> ABL:DAn`; Danimarka sözlükte yoksa `<cap> ▁dani marka <apos> ABL:DAn` |
| Yabancı özel isim | `Shakespeare'e`, `Bordeaux'ya` | Okunuş yazımdan çıkmaz → `DAT!e`, `DAT!ya` işaretli allomorf |
| Sayılar | `2023'te`, `6'dan`, `10'a` | Rakamlar ayrı token; üretici son basamakların Türkçe okunuşunu bilir (üç→te, altı→dan, on→a) |
| Kısaltmalar | `ABD'ye`, `TBMM'de`, `NATO'ya` | Harf adı okunuşu (be-de → ye) ya da okunur kısaltma; tablo + işaretli allomorf |
| Ayrı yazılan klitikler | `geldin mi`, `sen de` | `mI` ve bağlaç `dA` ayrı kelime ama önceki kelimeyle uyumlu; üretici kelime sınırını aşan durum taşır. Bağlaç `de`, bulunma `-DA`'dan farklı ID |
| Yabancı kök + Türkçe ek | `komünistlerden`, `deploy'ladık`, `meeting'e` | Tahminci kökü LEX'e, ekleri SFX'e verir; uyum kökün yazımından tahmin edilir, tutmazsa işaretli allomorf |
| Saf yabancı metin | `the meeting was postponed`, `el gobierno` | Dil kimliği → tamamı LEX |
| Kod, URL, e-posta | `https://…`, `x_1 = 3` | Pretokenizer tamamını LEX/BYTE'a gönderir, ek analizi yok |

İngilizce ve İspanyolca için ayrı bir morfoloji yapmıyorsun; ama LEX BPE'sini bu dillerden yeterli metinle eğitirsen bu kelimeler 1–3 parçada kalır. Öneri: LEX eğitim karışımı ≈ %60 Türkçe kök gövdeleri, %25 İngilizce, %10 İspanyolca, %5 kod.

## Transformer eğitimine etkiler

Dürüst beklenti: küçük ölçekte (≤1B) morfoloji hassas görevlerde ve örnek verimliliğinde kazanç, eşit hesapta BPC'de nötr ile hafif kayıp arası; büyük ölçekte avantaj muhtemelen erir. Kanıt karışık ve bu tasarım tam olarak hiç test edilmedi.

**Beklenen kazançlar**

- **Veri paylaşımı.** `ABL:DAn` tek embedding; `-dan/-den/-tan/-ten` dört ayrı token yerine tüm örneklerden öğrenir. Aynısı `kitap/kitab`, `ağız/ağz` için. Nadir kök + sık ek kombinasyonlarında en büyük etki.
- **Fonoloji yükü modelden alınır.** Allomorf seçimi deterministik olduğu için model kapasitesini uyum tahminine değil morfosöz dizimine harcar. Yanlış uyumlu üretim (`evlar`) yapısal olarak imkânsız hâle gelir.
- **Yeni köklere genelleme.** Ismayilzada vd. (2025) büyük LLM'lerin Türkçe ve Fince'de yeni kökle kompozisyonel genellemede sert düştüğünü gösteriyor. Morpheus'ta nadir kökleri tam doğru bölme oranı %70'e karşı byte-BPE'de %10.
- **Kısıtlı üretim.** FSA maskesiyle geçersiz ek dizisi üretilemez.
- **Daha küçük vocab.** \~48K'lık vocab, 64K–256K'lık çok dilli vocab'lara göre daha küçük embedding/softmax.
- **Yorumlanabilirlik.** Ek ID'leri probing ve mekanistik analiz için temiz birimler.

**Beklenen maliyetler**

- **Dizi uzunluğu.** Her morfem ayrı token olduğu için kelime başına \~1.9–2.2 token beklenir; byte-BPE \~1.56. Morpheus'ta bu %30 daha uzun dizi, eşit token bütçesinde %23 daha az metin ve %2.8 daha yüksek BPC (1.451 vs 1.411, 58M model, tek seed) anlamına geldi. Senin tasarımın ek zincirlerini birleştirmediği için fertility Morpheus'tan da yüksek olabilir.
- **Bağlam penceresi.** Aynı token sayısında karakter cinsinden daha kısa bağlam.
- **Üretim hızı.** Karakter başına daha fazla forward pass; Morpheus'ta batch 1'de byte-BPE'nin \~%54'ü hız.
- **Analiz hataları etiket gürültüsüdür.** Yanlış disambiguasyon (`kale+m` vs `kalem`) modele tutarsız sinyal verir; tokenizer'ın kalitesi doğrudan veri kalitesidir.
- **Kelime düzeyi anlam birden çok tokena yayılır.** Model `ev·lAr·(H)mHz·DA·ki·lAr` zincirinden kelime anlamını katmanlarda kompoze etmek zorunda; sığ modellerde bu zarar verebilir.

**Kanıt durumu**

- Lehte: [MorphBPE](https://aclanthology.org/2026.findings-acl.2068.pdf) 300M ve 1B modellerde daha düşük cross-entropy ve daha hızlı yakınsama raporluyor; Macarca dahil. Toraman vd. (2023) morfolojik tokenizer'ın Türkçe'de rekabetçi olduğunu buldu. TurkishTokenizer çalışması TR% ve Pure% ile TR-MMLU arasında korelasyon iddia ediyor.
- Aleyhte: [Rethinking Tokenization for Rich Morphology](https://arxiv.org/pdf/2508.08424) algoritma seçiminin (Unigram vs BPE) morfolojik hizalamadan daha belirleyici olduğunu buldu; Arnett ve Bergen (2025) hizalamanın downstream varyansın az kısmını açıkladığını raporluyor.
- Yorum: TR% / Pure% gibi metrikler hizalamayı ölçer, model kalitesini değil. Asıl test eşit hesap ve eşit metin altında BPC ve morfoloji hassas benchmark'lar.

**Fertility maliyetini geri almak için iki yol**

1. **Kayıpsız birleştirme katmanı.** Çok sık `kök+ek` ya da `ek+ek` kombinasyonları (`ları`, `ında`, `lerinde`) birleşik ID alır ama bileşen ID'lerini metadata olarak taşır. Morpheus da bunu planlıyor.
2. **Kelime düzeyi composer.** Senin A3t / tiered composer çalışman tam bu boşluğa oturuyor: tokenizer morfemleri verir, composer kelime vektörünü kurar, transformer kelime düzeyinde çalışır; dizi uzunluğu kelime sayısına iner. Bu tokenizer, şu an kayıplı olan TurkishTokenizer v1.0.4 substratının kayıpsız bir yerine geçeni olabilir.

## Değerlendirme protokolü

Karşılaştırma byte-level BPE, Unigram, TurkishTokenizer ve Morpheus'a karşı, aynı korpus ve aynı vocab bütçesinde yapılmalı; ana metrik BPC, per-token perplexity değil.

| Katman | Metrik | Kaynak / set |
| --- | --- | --- |
| Kayıpsızlık | `decode(encode(x)) == x` oranı | 30K çekimli kelime, 5K ham satır, stres seti |
| Morfolojik hizalama | MorphScore F1, sınır F1, kök tam doğruluk | UD\_Turkish-Kenet, SIGMORPHON 2022 |
| Düzensizlik kapsamı | Fenomen başına yüzey doğruluğu | Kendi test setin: yumuşama, düşme, ikizleşme, uyum istisnası, supletif, özel isim, sayı (her biri ≥ 50 örnek) |
| Verimlilik | Token/kelime, karakter/token, encode hızı | TR-MMLU metni |
| LM kalitesi | BPC; eşit hesap **ve** eşit metin; ≥ 3 seed | Param-eşit GPT, 50M → 350M ölçek |
| Downstream | TurBLiMP, TR-MMLU, yeni kök genelleme probları | — |

İki protokol şart: eşit adım (eşit hesap) morfolojik tokenizer'ı cezalandırır çünkü daha az metin görür; eşit metin onu kayırır çünkü daha çok hesap harcar. İkisini birlikte raporlamak tartışmayı kapatır. Ölçek eğrisi (en az iki boyut) avantajın büyüdükçe erimesini görmek için gerekli.

## Yol haritası ve başlangıç kodu

Prototip (`turkmorph`, \~600 satır Python) yazıldı ve 43 test kelimesi + 5 cümlede `decode(encode(x)) == x` invariantını geçiyor. Mimari belgedeki tasarımın birebir küçük ölçekli hali: 89 kök, 46 ek, 252 LEX parçası, 710 vocab.

Seçme çıktılar:

```
evlerimizdekiler  -> R:ev#N S:PL S:P1PL S:LOC S:REL S:PL
kalktığımızda     -> R:kalk#V S:PART_DHK S:P1PL S:LOC
gidiyorum         -> R:git#V S:PROG S:Z1SG
ağzından          -> R:ağız#N S:P3SG S:ABL
saatlerde         -> R:saat#N S:PL S:LOC
bana              -> R:ben#PRON S:DAT
Danimarka'dan     -> <cap> R:danimarka#PROPER <apos> S:ABL
komünistlerden    -> L:▁komünist S:PL S:ABL
2023'te           -> D:2 D:0 D:2 D:3 <apos> S:LOC
Bordeaux'ya       -> <cap> L:▁bordeaux <apos> <lit> L:y L:a
```

Bilinen sınırlar: leksikon el yazımı ve küçük; belirsizlik giderme el ayarlı maliyetlerle (EM unigram değil); `-Hl/-Hn` edilgen, ettirgen, birleşik fiiller (`-Abil` dışı) ve ünlü uyumu istisnası olan ekler (`-ki → -kü`) yok; analiz naif DFS, büyük leksikonda trie gerekir.

**Fazlar**

1. **Leksikon (1–2 hafta).** Zemberek/TRmorph kök listesini al; bayrakları korpustan çıkar (ör. `kitab+V` biçimi korpusta varsa `soften`). \~30K kök, frekansa göre ROOT vocab'ına gir.
2. **Morfotaktik (1–2 hafta).** Tam ek envanteri (\~350): edilgen, ettirgen, dönüşlü, işteş, birleşik fiiller, `-ki` uyumu, yapım ekleri. Fenomen test seti ile.
3. **Belirsizlik giderme (1 hafta).** Korpusta tüm analizleri çıkar, EM ile morfem unigram olasılıkları, Viterbi seçimi.
4. **LEX BPE (birkaç gün).** Eklerinden soyulmuş gövdeler + İng./İsp. metin üzerinde HF `tokenizers` ile \~16K.
5. **Performans.** Kelime-tipi önbelleği; analizörü Rust'a taşı (PyO3).
6. **Deney.** Değerlendirme protokolündeki 6 katman; composer ablasyonuna substrat olarak bağla.

**Hemen sonraki adım:** 200 kelimelik fenomen test setini (yumuşama, düşme, ikizleşme, uyum istisnası, supletif, özel isim, sayı) şimdi yaz; leksikon büyüdükçe regresyon kalkanı olur.

## Kaynaklar

- [Tokens with Meaning: A Hybrid Tokenization Approach (Bayram vd., arXiv 2508.14292)](https://arxiv.org/pdf/2508.14292)
- [TurkishTokenizer HF Space (alibayram)](https://huggingface.co/spaces/alibayram/turkish_tiktokenizer)
- [TurkishMorpheus (GitHub)](https://github.com/lonewolf-rd/TurkishMorpheus) · [Morpheus makalesi (arXiv 2606.18717)](https://arxiv.org/abs/2606.18717)
- [MorphBPE (arXiv 2502.00894)](https://arxiv.org/abs/2502.00894) · [MorphBPE, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.2068.pdf)
- [Rethinking Tokenization for Rich Morphology (arXiv 2508.08424)](https://arxiv.org/pdf/2508.08424)
- [Vocab Diet (arXiv 2510.17001)](https://arxiv.org/pdf/2510.17001)
- [HeceTokenizer (arXiv 2604.10665)](https://arxiv.org/pdf/2604.10665)
- [TRmorph (Çöltekin, LREC 2010)](https://lexitron.nectec.or.th/public/LREC-2010_Malta/pdf/109_Paper.pdf)
- [TRMOR (Kayabaş vd., 2019)](https://journals.tubitak.gov.tr/elektrik/vol27/iss5/41)
- [MorTur / DiaMor (LREC 2020)](https://aclweb.org/anthology/2020.lrec-1.491.pdf)
