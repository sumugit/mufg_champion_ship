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
    - LB : 
- Ensemble : 1,2,3,4,5,lgbm
    - CV : 0.82213
    - LB : 0.82821
