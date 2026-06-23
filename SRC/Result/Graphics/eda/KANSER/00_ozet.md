# KANSER - Kesifci Veri Analizi (Before)

_Otomatik uretildi: eda_before.py_

## 1. Veri Kumesi Yapisi

- **Satir x Kolon**: 388 x 353
- **Bellek**: 1.22 MB
- **Tam kopya satir**: 0
- **Sabit (tek degerli) kolon sayisi**: 69
- **Dtype dagilimi**: float64=343, string=8, object=1, Int64=1

## 2. Eksik Deger Analizi

- **Eksik iceren kolon sayisi**: 351 / 353
- **Toplam eksik hucre**: 78309 (%57.17)
- **En cok eksik kolon**: CAT_6 (%98.97)

## 3. Sinif Dagilimi (Label)

- Sinif **0**: 120 ornek (%30.93)
- Sinif **1**: 268 ornek (%69.07)
- **Dengesizlik orani (cogunluk/azinlik)**: 2.23 : 1

## 4. Aykiri Deger Analizi (IQR 1.5x kurali)

- **Aykiri deger iceren kolon**: 127 / 343
- **En yuksek aykiri oranli kolon**: AL_319 (%21.21)

## 5. Dagilimlar

- **CAT_1** benzersiz deger: 26
- **CAT_2** benzersiz deger: 7
- **CAT_3** benzersiz deger: 5
- **CAT_4** benzersiz deger: 5
- **CAT_5** benzersiz deger: 5
- **CAT_6** benzersiz deger: 1
- **AA_1** benzersiz deger: 21
- **AA_2** benzersiz deger: 22

## 6. Ozellik - Hedef Bagimliligi

- **Hedefle en iliskili ozellik**: AL_3 (r=0.536)

## 7. Korelasyon Isi Haritasi

- **|r|>0.95 olan ozellik cifti**: 879 (bkz. 07_yuksek_korelasyon_ciftleri.csv)
