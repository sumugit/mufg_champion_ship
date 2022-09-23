- EX001 : deberta-v3-large_5fold
    - SEED : 2022
    - CV : 0.80623
    - LB : 0.81702
- EX002 : deberta-v3-large_5fold
    - SEED : 1998
    - CV : 0.80501
    - LB : 0.82094
- EX003 : deberta-large_5fold
    - LGBM
        - SEED : 2014
        - TUN : {'num_leaves': 26, 'max_depth': 8, 'reg_lambda': 0.03565525423207128}
        - CV : 0.75808
        - LB : 
    - BERT
        - SEED : 2014
        - CV : 0.80529
- EX004 : deberta_large_5fold
    - SEED : 2018
    - CV : 0.80887
    - LB : 
- EX005 : roberta-large_5fold
    - SEED : 42
    - CV : 0.80100
    - LB : 0.81177
- EX006 : roberta-large_5fold (AWP)
    - SEED : 42
    - CV : 0.79292
    - LB : 0.81143
- EX007 : deberta-v3-large-5fold (AWP)
    - SEED : 2022
    - CV : 0.80696
    - LB : 0.81932
- EX008 : deberta-large-5fold (AWP)
    - SEED : 2014
    - CV : 0.80586
    - LB : 
- EX009 : deberta-large-5fold (FGM)
    - SEED : 2018
    - CV : 0.80887
    - LB : 
- EX0010 : deberta-v3-large-5fold (FGM)
    - BERT
        - SEED : 1998
        - CV : 0.8094
        - LB : 
    -LGBM
        - SEED : 1998
        - CV : 0.83940
        - LB : 
- Ensemble : 1,2,3,4,5,lgbm
    - CV : 0.82213
    - LB : 0.82881
- Ensemble : 7,2,3,4,5,lgbm
    - CV : 0.82317
- Ensemble : 7,2,3,4,6,lgbm
    - CV : 0.82354
- Ensemble : 6,7,8,9,10,lgbm10
    - CV : 0.8336
    - LB : 0.8298
- Stacking1 : 12345
    - {'num_leaves': 26, 'max_depth': 20, 'reg_lambda': 0.0970996171664033}
    - CV: 0.828777
    - LB : 0.78233
- Stacking2 : 12345 (not in lgbm)
    - SEED : 2022
    - CV : 0.82700
    - LB : 0.82938
- Stacking3 : 72346
    - SEED : 2022
    - CV : 0.82441
    - LB : 
- Stacking4 : 72345*
    - SEED : 2022
    - CV : 0.82684
    - LB : 0.83176
- Stacking5
    - SEED : 1998
    - CV : 0.82695
    - LB : 0.82897
- Stacking6
    - SEED : 1998
    - CV : 0.85842
    - LB : 0.68005
