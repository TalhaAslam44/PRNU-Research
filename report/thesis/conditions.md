# Thesis - final HVPF per condition

## unseen cameras (5-fold)

| cond                        |   clips |   correct |   hvpf_accepts |   pce_accepts |
|:----------------------------|--------:|----------:|---------------:|--------------:|
| AI: CogVideo                |     300 |     100   |            0   |           0   |
| AI: CogVideo+inj0.5         |     300 |     100   |            0   |          87   |
| AI: CogVideo+inj1           |     300 |     100   |            0   |          97.7 |
| AI: CogVideo+wa             |     300 |      99.7 |            0.3 |           0   |
| AI: CogVideo+yt             |     300 |      99.7 |            0.3 |           0   |
| AI: Kling                   |     269 |      99.3 |            0.4 |           0   |
| AI: Kling+inj0.5            |     269 |      98.9 |            0   |          85.5 |
| AI: Kling+inj1              |     269 |      98.1 |            0   |          98.5 |
| AI: Kling+wa                |     269 |      97.8 |            1.1 |           0   |
| AI: Kling+yt                |     269 |      97.8 |            0.7 |           0   |
| AI: Mora                    |     300 |      99.3 |            0.7 |           0   |
| AI: Mora+inj0.5             |     300 |      99.7 |            0   |          81   |
| AI: Mora+inj1               |     300 |      99.7 |            0   |          99   |
| AI: Mora+wa                 |     300 |     100   |            0   |           0   |
| AI: Mora+yt                 |     300 |      98   |            1   |           0   |
| AI: OpenSora                |     272 |      99.6 |            0.4 |           0   |
| AI: OpenSora+inj0.5         |     272 |     100   |            0   |          83.8 |
| AI: OpenSora+inj1           |     272 |     100   |            0   |          99.3 |
| AI: OpenSora+wa             |     272 |      99.6 |            0   |           0   |
| AI: OpenSora+yt             |     272 |      99.6 |            0.4 |           0   |
| AI: Pika                    |     300 |      99   |            0   |           0   |
| AI: Pika+inj0.5             |     300 |      99.7 |            0   |          66.3 |
| AI: Pika+inj1               |     300 |      99.7 |            0   |          96   |
| AI: Pika+wa                 |     300 |     100   |            0   |           0   |
| AI: Pika+yt                 |     300 |      99   |            0.3 |           0   |
| AI: SVD                     |     300 |      98.7 |            0.7 |           0   |
| AI: SVD+inj0.5              |     300 |      99.7 |            0   |          54.7 |
| AI: SVD+inj1                |     300 |      99.7 |            0   |          92.7 |
| AI: SVD+wa                  |     300 |     100   |            0   |           0   |
| AI: SVD+yt                  |     300 |      98   |            1.7 |           0   |
| AI: Sora                    |      51 |      82.4 |            9.8 |           0   |
| AI: Sora+inj0.5             |      51 |      82.4 |            2   |          76.5 |
| AI: Sora+inj1               |      51 |      90.2 |            0   |          96.1 |
| AI: Sora+wa                 |      51 |      51   |           29.4 |           0   |
| AI: Sora+yt                 |      51 |      74.5 |           21.6 |           0   |
| AI: Text2Video-Zero         |     300 |      99.3 |            0   |           0   |
| AI: Text2Video-Zero+inj0.5  |     300 |      99.7 |            0   |          47.7 |
| AI: Text2Video-Zero+inj1    |     300 |     100   |            0   |          88.3 |
| AI: Text2Video-Zero+wa      |     300 |     100   |            0   |           0   |
| AI: Text2Video-Zero+yt      |     300 |      99.3 |            0   |           0   |
| attack: denoise_a1          |     928 |      83   |           16.3 |          22.8 |
| attack: denoise_a2          |     928 |      91.9 |            7.5 |          19.5 |
| attack: denoise_a2+wa       |     928 |      58.6 |           40.7 |           1.1 |
| attack: denoise_a2+yt       |     928 |      56   |           43.4 |           6.9 |
| attack: denoise_a4          |     928 |      96.6 |            2.9 |          14.8 |
| attack: injection_a0.25     |     928 |      88.4 |           11.3 |          23.3 |
| attack: injection_a0.5      |     928 |      94.3 |            4.8 |          76.2 |
| attack: injection_a0.5+wa   |     928 |      74.4 |           25.3 |           1   |
| attack: injection_a0.5+yt   |     928 |      57   |           42.8 |          26.5 |
| attack: injection_a1        |     928 |      97.6 |            0.3 |          98.5 |
| attack: removal_a0.25       |     928 |      71.4 |           27.6 |          13   |
| attack: removal_a0.5        |     928 |      94   |            5.2 |           2.9 |
| attack: removal_a0.5+wa     |     928 |      53.3 |           45   |           0.5 |
| attack: removal_a0.5+yt     |     928 |      57.4 |           41.9 |           2.7 |
| attack: removal_a1          |     928 |      99   |            0.2 |           0.2 |
| attack: removal_opt         |     928 |      84.7 |           14   |           0.3 |
| attack: removal_opt+wa      |     928 |      50   |           49   |           0.2 |
| attack: removal_opt+yt      |     928 |      46.7 |           52.4 |           1   |
| genuine: native             |    4805 |      96.8 |           96.8 |          41   |
| genuine: real whatsapp copy |     423 |      79.2 |           79.2 |          10.2 |
| genuine: real youtube copy  |     411 |      89.3 |           89.3 |          14.4 |
| genuine: reencode           |     928 |      58   |           58   |          29.3 |
| genuine: reencode+wa        |     928 |      55.7 |           55.7 |           2.7 |
| genuine: reencode+yt        |     928 |      76.1 |           76.1 |          12.6 |
| genuine: reencode_crf18     |     928 |      71.3 |           71.3 |          32.5 |
| genuine: reencode_crf18+wa  |     928 |      54.1 |           54.1 |           2.8 |
| genuine: reencode_crf18+yt  |     928 |      79.4 |           79.4 |          12.6 |
| genuine: reencode_crf23     |     928 |      73.5 |           73.5 |          23.3 |
| genuine: reencode_crf28     |     928 |      90.5 |           90.5 |          12.9 |
| genuine: reencode_crf28+wa  |     928 |      52   |           52   |           1.5 |
| genuine: reencode_crf28+yt  |     928 |      81.2 |           81.2 |           8.1 |

**Genuine clips by source and stabilisation**

|                          |   clips |   accepted % |   PCE accepts % |
|:-------------------------|--------:|-------------:|----------------:|
| ('ACID', False)          |    4380 |         97.1 |            39.8 |
| ('ACID-attack', False)   |    5280 |         67.2 |            11.5 |
| ('VISION', False)        |     765 |         84.4 |            39.9 |
| ('VISION', True)         |     494 |         92.5 |             4.3 |
| ('VISION-attack', False) |    2410 |         67.7 |            26.6 |
| ('VISION-attack', True)  |    1590 |         78.1 |             2.3 |

## unseen datasets

| cond                    |   clips |   correct |   hvpf_accepts |   pce_accepts |
|:------------------------|--------:|----------:|---------------:|--------------:|
| AI: cogvideox           |     150 |      98.7 |            0   |           0   |
| AI: cogvideox+inj0.5    |     150 |      97.3 |            0   |          82.7 |
| AI: cogvideox+inj1      |     150 |      98   |            0   |          95.3 |
| AI: cogvideox+wa        |     150 |      50.7 |           20.7 |           0   |
| AI: cogvideox+yt        |     150 |      94   |            0.7 |           0   |
| AI: easyanimate         |     150 |      99.3 |            0   |           0   |
| AI: easyanimate+inj0.5  |     150 |     100   |            0   |          79.3 |
| AI: easyanimate+inj1    |     150 |     100   |            0   |          96.7 |
| AI: easyanimate+wa      |     150 |      75.3 |           10   |           0   |
| AI: easyanimate+yt      |     150 |      98.7 |            0.7 |           0   |
| AI: hunyuanvideo        |     150 |      99.3 |            0   |           0   |
| AI: hunyuanvideo+inj0.5 |     150 |      98.7 |            0   |          70   |
| AI: hunyuanvideo+inj1   |     150 |      98.7 |            0   |          97.3 |
| AI: hunyuanvideo+wa     |     150 |      76.7 |            8   |           0   |
| AI: hunyuanvideo+yt     |     150 |      98   |            0.7 |           0   |
| AI: ltxvideo            |     150 |      97.3 |            0.7 |           0   |
| AI: ltxvideo+inj0.5     |     150 |      98   |            0.7 |          70   |
| AI: ltxvideo+inj1       |     150 |      96.7 |            0   |          94   |
| AI: ltxvideo+wa         |     150 |      32.7 |           33.3 |           0   |
| AI: ltxvideo+yt         |     150 |      89.3 |            4.7 |           0   |
| genuine: native         |    1716 |      87.2 |           87.2 |          16.1 |

