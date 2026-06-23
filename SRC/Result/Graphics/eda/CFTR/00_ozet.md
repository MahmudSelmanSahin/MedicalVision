# CFTR - Kesifci Veri Analizi (Before)

_Otomatik uretildi: eda_before.py_

## 1. Veri Kumesi Yapisi

- **Satir x Kolon**: 111 x 353
- **Bellek**: 0.35 MB
- **Tam kopya satir**: 0
- **Sabit (tek degerli) kolon sayisi**: 70
- **Dtype dagilimi**: float64=343, string=8, object=1, Int64=1

## 2. Eksik Deger Analizi

- **Eksik iceren kolon sayisi**: 338 / 353
- **Toplam eksik hucre**: 11864 (%30.28)
- **En cok eksik kolon**: CAT_6 (%98.2)

## 3. Sinif Dagilimi (Label)

- Sinif **0**: 21 ornek (%18.92)
- Sinif **1**: 90 ornek (%81.08)
- **Dengesizlik orani (cogunluk/azinlik)**: 4.29 : 1

## 4. Aykiri Deger Analizi (IQR 1.5x kurali)

- **Aykiri deger iceren kolon**: 113 / 343
- **En yuksek aykiri oranli kolon**: AL_23 (%21.82)

## 5. Dagilimlar

- **CAT_1** benzersiz deger: 23
- **CAT_2** benzersiz deger: 7
- **CAT_3** benzersiz deger: 5
- **CAT_4** benzersiz deger: 5
- **CAT_5** benzersiz deger: 5
- **CAT_6** benzersiz deger: 1
- **AA_1** benzersiz deger: 18
- **AA_2** benzersiz deger: 20

## 6. Ozellik - Hedef Bagimliligi

- **Hedefle en iliskili ozellik**: AL_27 (r=-0.615)

## 7. Korelasyon Isi Haritasi

- **|r|>0.95 olan ozellik cifti**: 0 (bkz. 07_yuksek_korelasyon_ciftleri.csv)
