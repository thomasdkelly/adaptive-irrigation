# adaptive-irrigation

Code used for the article

...article citation...


## Instructions

1. clone repo

2. install reqs

```
virtualenv env
source env/bin/create
pip install -r requirements.txt
```

3. run scripts

    - `fixed_strategy.py` : optimizing fixed strategy that is applied in all years

    - `potential_strategy.py` : optimizing strategy for each year individually

    - `growth_stage_adaptation.py` : strategies re-optimized at the start of each growth stage

    - `seven_day_adaptation.py` : strategies re-optimized every seven days with or without perfect 7-day forecast

