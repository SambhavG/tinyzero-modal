modal deploy tinyzero_modal.py && ./run.sh


### regenerate data
python countdown_dataset.py
modal volume rm -r countdown-data train.parquet
modal volume rm -r countdown-data test.parquet
modal volume put countdown-data ./data/train.parquet /
modal volume put countdown-data ./data/test.parquet /