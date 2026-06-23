# PAH - Kesifci Veri Analizi (Before)

_Otomatik uretildi: eda_before.py_

## 1. Veri Kumesi Yapisi

- **Satir x Kolon**: 372 x 353
- **Bellek**: 1.15 MB
- **Tam kopya satir**: 0
- **Sabit (tek degerli) kolon sayisi**: 91
- **Dtype dagilimi**: float64=344, string=7, object=1, Int64=1

## 2. Eksik Deger Analizi

- **Eksik iceren kolon sayisi**: 351 / 353
- **Toplam eksik hucre**: 71301 (%54.30)
- **En cok eksik kolon**: CAT_6 (%100.0)

## 3. Sinif Dagilimi (Label)

- Sinif **0**: 62 ornek (%16.67)
- Sinif **1**: 310 ornek (%83.33)
- **Dengesizlik orani (cogunluk/azinlik)**: 5.00 : 1

## 4. Aykiri Deger Analizi (IQR 1.5x kurali)

- **Aykiri deger iceren kolon**: 158 / 343
- **En yuksek aykiri oranli kolon**: AL_271 (%23.08)

## 5. Dagilimlar

- **CAT_1** benzersiz deger: 24
- **CAT_2** benzersiz deger: 7
- **CAT_3** benzersiz deger: 4
- **CAT_4** benzersiz deger: 4
- **CAT_5** benzersiz deger: 4
- **AA_1** benzersiz deger: 20
- **AA_2** benzersiz deger: 20

## 6. Ozellik - Hedef Bagimliligi

- **Hedefle en iliskili ozellik**: AL_29 (r=-0.518)

## 7. Korelasyon Isi Haritasi

- **|r|>0.95 olan ozellik cifti**: 129 (bkz. 07_yuksek_korelasyon_ciftleri.csv)
