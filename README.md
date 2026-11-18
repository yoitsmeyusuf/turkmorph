# turkmorph

Kayıpsız Türkçe morfolojik tokenizer. Ekler arkifonemik ID'lerle (`S:PL` = `-lAr`,
`S:ABL` = `-DAn`) saklanır; yüzey biçimi decode sırasında kökün leksikal bayraklarından
deterministik olarak üretilir. Her kelime "üret ve doğrula" ile kabul edildiği için
`decode(encode(x)) == x` bir invarianttır.

Tasarım ve gerekçeler: `Türkçe Morfolojik Tokenizer — Tasarım ve Araştırma.md`.

*Lossless Turkish morphological tokenizer: suffixes are stored as archiphonemic ids
and surface forms are regenerated on decode, so `decode(encode(x)) == x` always holds.
Rust runtime, Python API. Code Apache-2.0, tables CC BY-SA 4.0 (see NOTICE).*

## Kurulum ve kullanım

```bash
pip install turkmorph
```

```python
from turkmorph.fast import fast_tokenizer

tok = fast_tokenizer()                       # Rust çalışma zamanı, tablolar pakette
tok.encode("Öğretmenlerimizden NATO'ya zeytinyağları")
# ['<cap>', 'R:öğretmen#N', 'S:PL', 'S:P1PL', 'S:ABL', '<capsb>', 'R:nato#PROPER',
#  '<apos>', 'S:DAT', 'R:zeytinyağı#N', 'S:PL|P3PL']
ids = tok.encode_ids(metin)                  # model için tamsayı ID'ler (sözlük: 58.667)
assert tok.decode_ids(ids) == metin          # kayıpsız
tok.encode_batch(metinler, threads=6)        # çok iş parçacıklı, GIL serbest
```

Saf Python sürümü aynı tokenları üretir: `from turkmorph.presets import full_tokenizer`.
Paketler Linux (x86_64, aarch64), macOS (Intel, Apple Silicon) ve Windows (x64) için
Python ≥ 3.10 ile hazır gelir; diğer platformlarda pip kaynak koddan derler (Rust gerekir).

## Dil modeli karşılaştırması: Morph vs BPE

Aynı GPT (6 katman, 384 boyut, 33M parametre), aynı sözlük boyutu ve aynı
BellaTurca metni; BPE aynı metinle eğitilmiş byte-level BPE. Değerler karakter başına
bit (BPC, düşük daha iyi), eğitimde hiç kullanılmamış test metninde; aksi belirtilmedikçe
3 seed ortalaması (seed sapması 0,002–0,007).

| Karşılaştırma | Morph | BPE | Morph avantajı |
|---|---:|---:|---:|
| Eşit metin, 30M karakter | 2,015 | 2,412 | %16,4 |
| Eşit hesaplama (BPE'ye 1,77× metin, aynı token sayısı) | 2,015 | 2,395 | %15,8 |
| Eşit metin, BPE en iyi öğrenme oranı/batch ile (tek seed) | 2,015 | 2,363 | %14,7 |
| Uzun bağlam (~2.800 karakterlik parçalar) | 2,076 | 2,421 | %14,2 |
| Eşit metin, 100M karakter (tek seed) | 1,887 | 2,308 | %18,2 |

Bu ölçümler IMST'nin henüz çıkarılmadığı önceki tablolarla (sözlük 58.723) yapıldı;
IMST'siz tablolarda UD test kök doğruluğu %93,94 → %93,91 değişti.
Sınırlar: tek model boyutu, tek derlem, en fazla 100M karakter. Morph aynı metin için
yaklaşık 1,8 kat token üretir; çıkarım maliyeti buna göre yüksektir. Deney kodu `lm/`,
protokol ayrıntıları aşağıda.

```
bash scripts/build_all.sh            # UD + Zemberek indir ve her şeyi kur
python3 scripts/prepare_bellaturca.py   # BellaTurca örneği (data/raw/bellaturca/*.jsonl) -> sayımlar (~2 dk)
SKIP_LEXICON=0 bash scripts/rebuild.sh  # indirilmiş veriden yeniden kurulum, 6 çekirdek (~6 dk)
python3 demo.py --full               # tam tokenizer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
python3 scripts/errors.py            # UD test hata analizi -> data/errors.json
python3 scripts/eval_noise.py        # yazım hatası / ASCII ölçümü -> data/eval_noise.json
python3 scripts/train_context.py      # (B) bağlam modeli, GPU ~12 dk -> data/context.pt
python3 -m venv --system-site-packages .venv-tt && .venv-tt/bin/pip install turkish-tokenizer==1.0.4
.venv-tt/bin/python scripts/compare_tt.py   # TurkishTokenizer 1.0.4 ile karşılaştırma -> data/compare_tt.json
.venv-tt/bin/python server/app.py          # test sunucusu: http://localhost:8765 (bağlam modu anahtarlı)
```

Çalışma zamanı bağımlılığı yok (Python ≥ 3.10 standart kütüphanesi). `TURKMORPH_PROCS`
paralel işçi sayısını belirler (varsayılan: `rebuild.sh` 6, betikler tüm çekirdekler).

Kaynak koddan çalışırken Rust/PyO3 çalışma zamanı `bash scripts/build_fast.sh` ile
derlenir ve tablolar depodaki `data/` klasöründen okunur. Dağıtım paketi
`bash scripts/build_wheel.sh` ile `dist/` altına üretilir: tablolar `turkmorph/data/`
içine kopyalanıp wheel'e eklenir. Başka bir tablo kümesi için
`fast_tokenizer(data_dir=...)` veya `full_tokenizer(data_dir=...)` kullanılabilir.
Yayın: `v*` etiketi GitHub Actions ile tüm platformlar için wheel üretip PyPI'ye yükler
(`.github/workflows/release.yml`).

Lisans: kod Apache 2.0. Tablolar (`data/`, pakette `turkmorph/data/`) CC BY-SA 4.0:
Zemberek (Apache 2.0), UD Türkçe ağaç bankaları BOUN/Kenet/Penn/Tourism/FrameNet ve
BellaTurca (CC BY-SA 4.0) verisinden öğrenilir. Ticari kullanımı kısıtlayan IMST
(CC BY-NC-SA) hiçbir tabloya girmez, yalnızca yerel değerlendirmede test kümesi
olarak kullanılır. Ayrıntılar `NOTICE` dosyasında.

İlk GPT eğitim pilotu, hazır `data/lm/morph/` verisiyle GPU'nun erişilebilir olduğu
terminalden `bash scripts/start_lm.sh` komutuyla başlatılır. Betik CUDA ileri/geri
geçişini kontrol eder; 6 katman, 384 boyut, 512 token bağlam ve 1 milyon tokenlık
nominal bütçeyle çalışır. Bu pilot tek başına tokenizer karşılaştırması değildir.
Pilot varsayılan `--duty-cycle 0.7` ile eğitim ve doğrulama adımlarından sonra bekler;
ölçülen çalışma süresinin yaklaşık %43'ü kadar boşta süre ekler. Bu, donanımsal güç
veya sıcaklık sınırı değildir. Daha fazla bekleme için `--duty-cycle 0.5`, beklemeyi
kapatmak için `--duty-cycle 1` verilebilir. Eğitim günlüğü kalan eğitim süresini
tahmin eder; doğrulama süresi bu tahmine dahil değildir.
`data/lm/runs/` altında her çalıştırma için ayrı dizine `train.log`, `config.json`,
eğitim sonunda ağırlıklar ve optimizer durumunu içeren `model.pt`, doğrulama sonunda
`result.json` yazılır. Checkpoint'ten devam etme henüz uygulanmadı.
Farklı bütçe/seed için örnek: `bash scripts/start_lm.sh --max-tokens 31632567 --seed 1`.

BPE pilotu için `python3 lm/prepare_bpe.py`, mevcut morfolojik `.bin` dosyalarından
metni geri çözer, kaynak korpusla birebir doğrular ve aynı sözlük boyutunda byte-level
BPE verisini `data/lm/bpe/` altında oluşturur. Bu komut mevcut BPE verisini üzerine
yazmaz. Tokenizer, metin SHA-256 değerleri ve kayıpsızlık bilgisi de saklanır.
Ardından `TRAIN_DATA=bpe bash scripts/start_lm.sh` aynı GPT ayarları ve 1 milyonluk
nominal token bütçesiyle BPE pilotunu çalıştırır. Eşit token bütçesi, tokenizer'lar
farklı uzunlukta diziler ürettiği için eşit miktarda eğitim metni anlamına gelmez.

Üç seed ile 10 milyon tokenlık doğrulama deneyi:

```bash
python3 -u lm/experiment.py --output-dir data/lm/experiments/token-budget-10m-3seeds
```

Bu komut seed 0/1/2 için morph ve BPE GPT'lerini sırayla, aynı ayarlarla eğitir.
Her çalışma 2.441 adımda 9.998.336 token görür; varsayılan çalışma oranı %70,
çalışmalar arası bekleme 30 saniyedir. Veri/kod SHA-256 değerleri `plan.json` içinde,
çalışma başına günlük ve model ayrı alt dizinlerde tutulur. Altı çalışma tamamlanınca
`summary.md` ve `summary.json` seed bazında BPC farkını, ortalamayı ve örnek standart
sapmasını verir. Üç seed tek başına istatistiksel anlamlılık iddiası sağlamaz.
Aynı komut tekrar çalıştırılırsa ayarları doğrulanan tamamlanmış çalışmalar atlanır.
Yarım kalmış bir eğitimin optimizer adımından devam etme henüz desteklenmez.
GPU olmadan planı kontrol etmek için aynı komuta `--dry-run` eklenebilir.

Eşit metin karşılaştırması için hazırlanan ayrı protokol:

```bash
python3 lm/prepare_equal_text.py
python3 -u lm/equal_text_experiment.py --output-dir data/lm/experiments/equal-text-30m-3seeds
```

İlk komut mevcut BPE referansının ilk 30.000.000 eğitim karakterini ve 2.000.030
doğrulama karakterinin tamamını ortak parçalara böler. Hazır veri varsa bu komut
üzerine yazmayı reddeder; doğrudan ikinci komut kullanılabilir. Her parça iki
tokenizer ile de en fazla 512 tokena sığar ve kayıpsızlığı doğrulanır. Hazırlanan
30M kümede 29.654 eğitim parçası, Morph için 9.438.624, BPE için 5.341.980 token var.

İki GPT her seed'de aynı parçaları aynı karıştırılmış sırayla, yerine koymadan,
bir kez görür; 8 parçalık batch ile 3.707 optimizer adımı çalışır. Örnek sırası,
görülen karakterler ve puanlanan doğrulama tokenları sonuç dosyasında doğrulanır.
Her iki modelin sözlüğüne ortak başlangıç işlevli bir BOS tokenı eklenir; böylece
parçanın ilk tokenı da puanlanır. Sağdaki padding kayıp hesabından çıkarılır,
kayıp batch'in karakter sayısına bölünür. İki model de aynı karakter bağlamından
yararlanır; token sayıları ve hesaplama maliyetleri farklıdır. Bu yeni protokolün
BPC'si önceki sabit-token penceresi sonuçlarıyla doğrudan kıyaslanmamalıdır.
Seed 0/1/2, %70 çalışma oranı, 30 saniye çalışmalar arası bekleme ve çalışma başına
model/günlük kaydı korunur. Sonuçlar deney dizininde `summary.md` ve `summary.json`
olarak yazılır; tamamlanmış çalışmalar tekrar başlatılmaz.

```python
from turkmorph.presets import full_tokenizer
tok = full_tokenizer()
tok.encode("Öğretmenlerimizden NATO'ya zeytinyağları")
# ['<cap>', 'R:öğretmen#N', 'S:PL', 'S:P1PL', 'S:ABL', '<capsb>', 'R:nato#PROPER', '<apos>', 'S:DAT',
#  'R:zeytinyağı#N', 'S:PL']
tok.encode_grouped("Kitaplarında 2023'te")      # composer için kelime grupları
# [['<cap>', 'R:kitap#N', 'S:PL', 'S:P3SG', 'S:LOC'], ['D:2', 'D:0', 'D:2', 'D:3', '<apos>', 'S:LOC']]
```

## Sonuçlar (UD Turkish test: BOUN, IMST, Kenet, Penn, Tourism, FrameNet; 49K alfabetik token)

`scripts/evaluate.py`, çıktı `data/eval.json`.

| Yapılandırma | Kapsam | Kök doğruluğu (oracle) | Kök+türetme doğruluğu (oracle) |
| --- | --- | --- | --- |
| A prototip: el leksikonu, el maliyetleri | 25.5% | 22.2% (22.2) | 23.2% (23.3) |
| B Zemberek leksikonu, el maliyetleri | 98.2% | 85.5% (89.8) | 91.0% (95.6) |
| C gözetimsiz EM (UD) | 98.2% | 87.9% (89.8) | 93.5% (95.6) |
| D lemma kısıtlı EM (UD) | 98.2% | 88.3% (89.8) | 93.9% (95.6) |
| **E BellaTurca 118M token + UD lemma ×100** (varsayılan) | **98.7%** | **88.8%** (90.6) | **94.0%** (96.0) |

- *Kapsam*: en az bir bilinen-kök analizi olan token oranı. BellaTurca'da bu oran %96.2.
- *Kök+türetme*: UD lemmaları bazen yapım ekini içerir (`ilgili`), bazen içermez (`yaptırmadım`→`yap`).
- *Oracle*: doğru lemma adaylar arasında var mı. Kalan farkın çoğu lemma kuralından:
  UD `sadece` kelimesini lemma sayıyor, biz `sade+EQU` diye ayırıyoruz.
- E, UD testinde D ile aynı seviyede ama 118M tokenlık genel korpustan öğrendiği için varsayılan.
  UD ağırlığı ×20 iken kök doğruluğu 0.4 puan düşüktü.

Tokenizer düzeyi (7.042 ham test cümlesi):

| Metrik | Değer |
| --- | --- |
| Kayıpsızlık | **100%** |
| Token / kelime | 2.22 (düz BPE, eşit bütçe: 1.31) |
| Token dağılımı | ROOT 38.5%, SFX 35.5%, özel 14%, PUNCT 9%, LEX 1.9%, BYTE ~0% |
| Vocab | 58.5K: ROOT 39.6K, LEX 16.1K, SFX 81 + işaretli 116, düzeltme 2.3K, BYTE 256 |
| Analiz hızı | tek çekirdek 0.5 ms/kelime (v0.3: 8.4 ms), önbellekten ~1 µs |

Kelime başına token sayısı bilinçli bir tercih: her morfem ayrı token ve uzun dizi maliyetini model tarafı
karşılayabilir. Bunun için iki araç hazır: birleştirme katmanı (`turkmorph/merge.py`) ve
composer'a gruplu girdi (`encode_grouped`).

## TurkishTokenizer (alibayram, `turkish-tokenizer` 1.0.4) ile karşılaştırma

`scripts/compare_tt.py`; TurkishTokenizer ayrı bir sanal ortamda (`pip install turkish-tokenizer==1.0.4`).

| | TurkMorph | TurkishTokenizer 1.0.4 |
| --- | --- | --- |
| Kayıpsız, UD test cümleleri (tam / baştaki-sondaki boşluk hariç) | **100% / 100%** | 0% / 87.5% |
| Kayıpsız, BellaTurca held-out belgeleri | **100% / 100%** | 0% / 2.3% |
| Kök = UD lemması | **88.5%** | 74.0% |
| Token / kelime (UD test) | 2.20 | **1.95** |
| Karakter / token | 2.99 | **3.39** |
| Hız (ilk geçiş / önbellekli, kelime/sn) | 2.7K / 555K | **1.3M / 1.4M** (Rust) |
| ASCII yazımda temiz kelimenin kökü bulunuyor | **96.1%** | 18.4% |
| Tek harf hatasında temiz kelimenin kökü bulunuyor | **84.0%** | 22.8% |

TurkishTokenizer yüzeyi kanonik biçime yeniden yazıyor: `saatlerde` → "saatlarda", `sütü` → "südü",
`2023'te` → "2023'da", `çocukken` → "çocukkan", `şundan` → "şünden". Fenomen setinde uyum istisnası
(%42), sayı (%54), kısaltma (%50), küçültme (%0) kategorilerinde yüzey bozuluyor; TurkMorph'ta hepsi %100.
TurkishTokenizer kelime başına daha az token kullanıyor. Bu tablodaki hız karşılaştırması
TurkMorph'un Python sürümüne aittir; mevcut Rust sürümünü ölçmek için `scripts/bench_fast.py` kullanılabilir.

## Test sunucusu (`server/app.py`)

Yerel web arayüzü: metni yazınca her kelimenin tokenlarını (namespace renkleri, ID, ek açıklaması),
decode'un yüzeyi token token nasıl kurduğunu (`Kitap → Kitaplar → Kitaplarımız → Kitaplarımızı`) ve
kayıpsızlık rozetini gösterir; aynı metnin TurkishTokenizer çıktısı yanında, decode farkı işaretli.

## Hata analizi (`scripts/errors.py`, UD test, 49.8K kelime)

Her kelime gerçek tokenizer'dan geçiyor, kök ve ekler UD lemma/özellikleriyle karşılaştırılıyor.
Uyuşmazlıklar üç türe ayrılıyor: **bağlam** (aynı kökle altın özelliği veren başka bir aday var),
**kural farkı** (UD `biz`'e Plur, `değil`'e Neg diyor), **gerçek hata** (doğru analiz hiç yok).

| Kök düzeyi | Pay |
| --- | --- |
| Kök doğru | **93.94%** (bağlam modunda 94.37%) |
| UD donmuş biçimi lemma sayıyor (`sadece`=`sade+EQU`) | 1.97% |
| Doğru kök adaydı, bağlamsız seçim yanlış (`gelecek`, `yüzde`, `altına`) | 1.66% (bağlam modunda 1.23%) |
| Doğru kök hiç yok (çoğu UD tuhaflığı: `âdet`, `yanlarına`→"yanla", `_`) | 0.79% |
| Özel isim / yabancı / kısaltma → LEX (tasarım gereği) | 1.31% |
| Analiz edilemeyen Türkçe kelime | **0.27%** (başta 0.83%) |

| Ek düzeyi, gerçek hata | Hâl | Sayı | İyelik | Olumsuzluk | Çatı | Süreklilik |
| --- | --- | --- | --- | --- | --- | --- |
| Kelimelerin payı | 0.24% | 0.19% | 0.20% (başta 0.49%) | 0.05% (başta 0.51%) | 0.63% | 0.44% |

## Yazım hataları ve ASCII yazım (`turkmorph/normalize.py`, `scripts/eval_noise.py`)

Hatalı yazılmış kelime LEX parçalarına dağılmak yerine, kastedilen kelimenin analizi ve farkı
kayıpsız olarak tutan bir işaretle kodlanır:

```
cocuklarimiz -> <ascii> R:çocuk#N S:PL S:P1PL     decode yüzeyi üretip Türkçe harfleri ASCII'ye çevirir
tesbit       -> R:tespit#N E:sub:3:b              3. harf b (del / ins / sub / swap, tek token)
Türkiye' nin -> ... P:' <bare> S:GEN!nin          kökten kopmuş ek zinciri
geliyom      -> R:gel#V S:PROG_C S:Z1SG_C          konuşma dili ekleri ayrı ID (geliyo, yapıcam, nerde)
```

Korumalar: yalnızca bilinen analizi olmayan kelimeler (ASCII yazımda Türkçe biçim ≥50× daha
sıksa nadir bir analize de baskın gelir: `vardi` → `vardı`, `varit+P3SG` değil); düzenleme
mesafesi tam 1; 4 harfli kelimede ≥50×, 5+ harfte ≥20× daha sık hedef; yabancı kelime ya da
çoğunlukla büyük harfle yazılan isim değil; hedef bir özel isim değil. Sözlük BellaTurca'nın
641K kelime tipinden (`data/wordfreq.tsv`).

Held-out BellaTurca kelimelerine yapay hata eklenerek ölçüldü (hepsi kayıpsız):

| | ASCII yazım | Tek harf hatası |
| --- | --- | --- |
| Doğru kelimenin analizine eşlendi | **96.7%** | **80.9%** |
| Hatalı biçim zaten geçerli bir kelime (dokunulmadı) | 1.0% | 8.2% |
| Kurtarılamadı | 2.2% | 4.1% |
| Yanlış kelimeye eşlendi | 0.1% | 6.8% |
| Token/kelime (temiz → normalizasyonlu / normalizasyonsuz) | 2.54 → 3.49 / 2.87 | 2.19 → 3.14 / 3.46 |

Düzeltme adayları frekans × hata olasılığıyla sıralanıyor: ünlü düşürme (`geçş`→`geçiş`), komşu tuş,
`ı/i`-`s/ş` karışması, çift basma daha olası. 2 harf hataları (`değrlendrme`) tek harf eşleşmesi yoksa
≥100× daha sık hedefe gidiyor. Bitişik yazılan ilgeç/ek ayrılıyor (`ikisinide` → `ikisini <glue> de`),
`ne` kaynaşmaları `<fuse>` ile tutuluyor (`nolur`, `naber`, `napıyon`).

Hatasız ama bilinmeyen kelimelerin (isim, yabancı, nadir) %18'i normalize ediliyor. Elle
incelenen örneklerin ~%80'i aslında metindeki gerçek hataların doğru düzeltilmesi
(`cok`, `sehirde`, `yaptğının`). Gerçek yanlış alarmlar (`nolur`, `ikisinide`) tokenların ~%0.1'i.

## Belirsizlik: açık bırakma (A) ve bağlamla seçme (B)

**A, deterministik mod (varsayılan).** Aynı kelime her zaman aynı tokenları alır. Etiketi yalnızca
bağlamla belli olan yerde tokenizer tahmin etmez, birleşik token verir (`scripts/build_unions.py`,
BellaTurca'da görülen 41 birleşim):

```
kitabı      R:kitap#N S:P3SG|ACC
kitapları   R:kitap#N S:P3PL|PL+P3SG|PL+ACC
evin        R:ev#N S:P2SG|GEN
```

Decode ilk üyeyle üretir. Hepsi aynı yüzeyi verdiği için kayıpsızlık korunur. Ayrıca sık kelimeler için
`data/overrides.tsv`: UD train/dev çoğunluk lemmasından 314, nadir kök denetiminden 9 düzeltme
(`yerine` → yer+P3SG+DAT, `ayrıcalık` → ayrı+CA+lIk).

**B, bağlam modu (`turkmorph.context.ContextTokenizer`, torch gerekir).** Karakter CNN ve 2 katmanlı
çift yönlü LSTM cümleyi okur, analizörün adayları arasından seçer. UD train ile eğitildi, dev ile
en iyi tur seçildi, test ile ölçüldü. Kökü ve bölmeyi model seçer. Etikette emin değilse
(olasılık < 0.9) A'nın birleşik tokenı kalır. Küçük harfli kelimeye özel isim kökü hiç sunulmaz.

| UD test | A (deterministik) | **A+B (bağlam)** |
| --- | --- | --- |
| Belirsiz kelimelerde doğru seçim (24.4K kelime) | 88.9% | **93.5%** |
| Kök doğru (tüm kelimeler) | 93.94% | **94.37%** |
| Kökte bağlam hatası | 827 | **611** |
| Hâl: yanlış etiket / açık | 24 / 1.152 | 91 / 638 |
| Kayıpsızlık (UD test, BellaTurca) | 100% | 100% |
| Hız | 2.7K kelime/sn (ilk geçiş) | ~0.9K kelime/sn (GPU) |

```
Parayı alan kişi kitapları okudu.
  A:  R:para S:ACC  R:alan#N         R:kişi  R:kitap S:P3PL|PL+P3SG|PL+ACC  R:oku S:PAST
  B:  R:para S:ACC  R:al#V S:PART_AN R:kişi  R:kitap S:PL S:P3SG|ACC         R:oku S:PAST
```

Model vocab ID'lerine bağlı olduğundan yeniden kurulumdan sonra tekrar eğitilmesi gerekir:
`CONTEXT=1 bash scripts/rebuild.sh`.

## Bağlam modeli gerekli mi? (`scripts/ambiguity.py`)

UD test tokenlarında bağlamsız seçimin ne kadarı etkilediği:

| Durum | Pay |
| --- | --- |
| Belirsiz değil ya da tek analiz açık ara önde | 77.5% |
| Yalnızca etiket belirsizliği (sınırlar aynı, ek ID'si farklı: `kitab+ı`) | 11.1% |
| Segmentasyon belirsizliği (kök/sınır farklı) | 9.3% |
| Analiz yok | 2.1% |

ACC/P3SG belirsizliği tokenların %5.1'inde var ve bağlamsız seçim bunların %57'sinde doğru.
Bağlam modeli en fazla tokenların **%2.2'sinin** etiketini değiştirebilir. Karar: tokenizer bağlamsız ve
deterministik kalıyor (aynı kelime hep aynı tokenlar). Bu belirsizliği bağlamı zaten gören dil modeli çözer.

## Mimari

| Dosya | İçerik |
| --- | --- |
| `turkmorph/phonology.py` | Arkifonem gerçekleştirme, yumuşama, düşme, sayı okunuşu, kısaltma harf okunuşu |
| `turkmorph/morphology.py` | `Root`/`Lexicon` (alternatif gövde indeksi), ek envanteri + FSA, artımlı üretici (`gen_start`/`gen_step`), sentez-yoluyla-analiz, işaretli allomorflar, maliyet modelleri |
| `turkmorph/zemberek.py` | Zemberek → `Root`: bayraklar, budama, donmuş kalıplar, şapka varyantları, özel isim filtresi, bileşik P3SG iç gövdeleri |
| `turkmorph/em.py` | Paralel analiz toplama (ilerleme + ETA), EM |
| `turkmorph/langid.py` | Türkçe/yabancı karakter n-gram NB |
| `turkmorph/bpe.py` | LEX BPE (artımlı, heap) |
| `turkmorph/merge.py` | Kayıpsız birleştirme katmanı (Çözüm 1), `sfx` ve `full` modları |
| `turkmorph/normalize.py` | Yazım hatası / ASCII normalizasyonu: düzenleme işlemleri, frekans sözlüğü, korumalar |
| `turkmorph/context.py` | (B) bağlam modeli ve `ContextTokenizer` |
| `turkmorph/tokenizer.py` | Pretokenize, büyük harf, düşme kademesi, decode, önbellek, vocab, `encode_grouped` |
| `turkmorph/corpus.py`, `presets.py` | CoNLL-U okuyucu; `demo_tokenizer()`, `full_tokenizer()` |
| `scripts/` | `prepare_bellaturca.py`, `build_lexicon.py`, `train_em.py`, `build_lex.py`, `evaluate.py`, `errors.py`, `eval_noise.py`, `ambiguity.py`, `build_all.sh`, `rebuild.sh` |
| `lm/` | GPT/Composer, BPC; tamamlanan GPT pilotları ve üç seed karşılaştırma betiği |
| `tests/data/phenomena.tsv` | 273 kelimelik fenomen seti (32 kategori, demo preset'e göre) |

## v0.6 değişiklikleri

- **EM başlangıcı:** BellaTurca EM'i eşit dağılım yerine UD lemma-kısıtlı modelle başlıyor; yerel optimum
  düzeldi (`ise` artık `is+DAT` değil, kopula `i+sA`). İsimden sonra kişi ekine önsel ceza (`anneciğim` → `P1SG`).
  Görülmemiş ek çiftleri FSA durumuna göre geri çekiliyor.
- **Leksikon:**
  - edilgenle türetilmiş fiiller budanmıyor (`dokun`, `bayıl`)
  - fiillerde ünlü düşmesi yalnız edilgenden önce (`çevril-`, `çeviren`)
  - yumuşama korpustan çıkarılıyor: tam çekimli biçimlerle, 71 ekleme (`diyaloğu`), 167 çıkarma (`aktivisti`)
  - özel isimlerin uyum istisnası korpustan (`Kemal'e`)
- **Morfoloji:** edilgen ardından edilgen (`dokunulsa`); `-CA`/`-DA` + `-lIk`; iyelikten sonra kişi
  (`tanesiyim`); `-lAr-ken`; `-CAsInA`, `-mAcA`.
- **Normalizasyon:**
  - hata modeli: ünlü düşürme, komşu tuş, ı/i karışması
  - 2 harf hataları
  - ASCII Türkçede yabancı kelime engeli kaldırıldı (`calisma`, `bagli`)
  - bitişik ilgeç/ek (`ikisinide`, `yada`)
  - `ne` kaynaşmaları (`nolur`, `naber`)
  - ASCII kontrolü artık bitişik ayırmadan önce yapılıyor
- **Araçlar:** `scripts/compare_tt.py`, `server/` (test arayüzü).

## v0.5 değişiklikleri

- **Gerçek hata düzeltmeleri** (UD hata analizinden):
  - P3SG ile biten zamirler iç gövde alıyor: `kendini`, `birbirlerine`, `hepsine`, `birine`.
  - Yeni ekler: sıra/üleştirme sayıları `-(H)ncH`/`-(ş)Ar` (`ikinci`, `birer`); fiilden isim `-(y)Hş` (`oluşu`);
    `-DHktAn`; `-(y)HncAyA`; `-mAzlHk`; olumsuz ulaçlar `-mAdAn`/`-mAksHzHn`; `-mAktA`;
    küçültme `-CHk` (`kedicik`).
  - İlgeçler kopula alıyor (`gibiydi`); `i-` kopulası (`ise`, `idi`, `iken`).
  - `de-` daralması TDK'ya göre (`deyince`, `deyip`, ama `diyecek`); `günkü`, `yılki`;
    hâlden sonra kişi eki (`karşıyayız`); `Beyoğlu'nda` (`-oğlu` bileşik isimleri).
- **Konuşma dili:** `geliyo(m/n/z/du)`, `yapıcam`/`yapcam`/`yapıcak`, `nerde`/`burdan`/`orda`.
  Ayrı ve daha pahalı ek ID'leri; standart biçim her zaman öncelikli.
- **Normalizasyon:** `<ascii>`, tek token düzeltme (`E:sub:3:b`), kopmuş ekler (`<bare>`).
- **Ölçüm:** `scripts/errors.py` (kök ve ek düzeyi, bağlam/kural farkı/gerçek hata ayrımı),
  `scripts/eval_noise.py` (yapay hata).

## v0.4 değişiklikleri

- **Bileşik P3SG** (`zeytinyağı`, `işbirliği`, `madensuyu`): kökte iç gövde saklanıyor. Çoğul ya da
  iyelik gelince üretici iç gövdeye geçiyor: `zeytinyağ-lar-ı`, `zeytinyağ-ım`, `işbirlik-ler-i`, `âdemoğul-lar-ı`.
  926 bileşiğin 918'i iç gövde aldı; kalanlar sözlükteki hatalı kayıtlar.
- **BellaTurca** (Özenli Derlem'den 1 GB, 119M token): özel isim kanıtı (≥100 büyük harfli geçiş),
  EM (UD lemma ×100 ile) ve 16K LEX BPE. %1'lik held-out belge kümesi hiçbir eğitimde kullanılmıyor.
- **Analizör 17× hızlı** (tek çekirdek 8.4 → 0.5 ms/kelime). Hepsi kaba kuvvet referansla
  6.000 kelimede birebir aynı sonuç veriyor:
  - alternatif gövde indeksi (aday kök 24 → 3.4)
  - ilk ekten sonra 1 harflik budama toleransı
  - artımlı üretim (her düğümde zinciri baştan üretmek yerine tek adım)
  - eklerin olası ilk harflerine göre ön filtre
- **Birleştirme katmanı** (`MorphMerge`): sık ek dizileri (`sfx`) ya da kök+ek dizileri (`full`) tek ID alır.
  Decode açar, kayıpsızlık korunur.
- **`encode_grouped`**: kelime düzeyi composer için gruplar; birleştirildiğinde `encode` çıktısını verir.
- **Kurulum süresi:** 6 çekirdekle ~6 dk (önce 16 çekirdekle ~45 dk). Her aşama süre ve ETA yazıyor.

## Kalan sınırlar

- **LM deneyi:** 1 milyon tokenlık tek-seed GPT pilotunda morph 2.199, BPE 2.321 BPC.
  Bunlar ön sonuçlardır; üç-seed/10 milyon token karşılaştırması henüz tamamlanmadı.
- **Leksikon:** `-Ayaz/-Akal/-Adur` birleşik fiilleri yok. Kişi adlarında uyum istisnası bayrağı eksik
  Özel isimlerin uyum istisnası artık korpustan çıkarılıyor (`Kemal'e`, 316 isim); isim korpusta yoksa çıkmaz.
- **Normalizasyon:** değiştirme+düşürme türü 2 harf hataları (`geliyrom`) bulunmuyor; tek harf düzeltmesinin
  %6.8'i yanlış kelimeye gidiyor (çoğu iki geçerli adaydan bağlamsız seçim).
- **Belirsizlik:** `ayrıcalık` → nadir `ayrıç` kökü; `pahalı` → `pahal+P3SG`. EM'in seçtiği nadir kökler.
- **Performans:** Python için kayıtlı analiz süresi 0.5 ms/kelime. Rust/PyO3 çalışma zamanı
  mevcut; Python ile token, sözlük ve decode eşitliği `tests/test_fast.py` ile sınanıyor.
