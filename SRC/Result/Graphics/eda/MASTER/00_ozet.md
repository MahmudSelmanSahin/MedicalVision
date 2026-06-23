# MASTER - Kesifci Veri Analizi (Before)

_Otomatik uretildi: eda_before.py_

## 1. Veri Kumesi Yapisi

- **Satir x Kolon**: 2931 x 353
- **Bellek**: 9.21 MB
- **Tam kopya satir**: 0
- **Sabit (tek degerli) kolon sayisi**: 57
- **Dtype dagilimi**: float64=343, string=8, object=1, Int64=1

## 2. Eksik Deger Analizi

- **Eksik iceren kolon sayisi**: 351 / 353
- **Toplam eksik hucre**: 568464 (%54.94)
- **En cok eksik kolon**: CAT_6 (%97.71)

## 3. Sinif Dagilimi (Label)

- Sinif **0**: 782 ornek (%26.68)
- Sinif **1**: 2149 ornek (%73.32)
- **Dengesizlik orani (cogunluk/azinlik)**: 2.75 : 1

## 4. Aykiri Deger Analizi (IQR 1.5x kurali)

- **Aykiri deger iceren kolon**: 232 / 343
- **En yuksek aykiri oranli kolon**: EK_2 (%33.09)

## 5. Dagilimlar

- **CAT_1** benzersiz deger: 30
- **CAT_2** benzersiz deger: 7
- **CAT_3** benzersiz deger: 5
- **CAT_4** benzersiz deger: 5
- **CAT_5** benzersiz deger: 5
- **CAT_6** benzersiz deger: 3
- **AA_1** benzersiz deger: 24
- **AA_2** benzersiz deger: 25

## 6. Ozellik - Hedef Bagimliligi

- **Hedefle en iliskili ozellik**: AL_83 (r=-0.212)

## 7. Korelasyon Isi Haritasi

- **|r|>0.95 olan ozellik cifti**: 22 (bkz. 07_yuksek_korelasyon_ciftleri.csv)
