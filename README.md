# CausalAI Assistant (CAIA) 
*Note* : This repository is a work in progress and will be updated with additional annotations or files.

**CAIA** is a tool designed to generate causality-based answers to natural language queries. Given a dataset (CSV file), its description, and an accompanying question, CAIA automatically selects an appropriate causal inference method and identifies the relevant variables. It then implements the corresponding analysis and interprets the numerical results in the context of the original query.

## Installation 

```python
pip install -r requirements.txt
pip install .
```

## Dataset Information 

All datasets used to evaluate CAIA and the baseline models are available in the data/ directory. Specifically:

* `all_data`: Folder containing all CSV files from the QRData and real-world study collections.
* `synthetic_data`: Folder containing all CSV files corresponding to synthetic datasets.
* `qr_info.csv`: Metadata for QRData files. For each file, this includes the filename, description, causal query, reference causal effect, intended inference method, and additional remarks.
* `real_info.csv`: Metadata for the real-world datasets.
* `synthetic_info.csv`: Metadata for the synthetic datasets.

## Run 
To run the program, one can run
```python
python main/run_agent.py -f ${CSV_PATH} -d ${DATA_FOLDER} -t ${DATA_CATEGORY} -o ${OUTPUT_FOLDER} -l ${LLM_NAME}
```
**Args**
* `CSV_PATH`: A csv file containing the queries, the dataset description, name of the data file
* `DATA_FOLDER`: Path to the folder containing the data
* `DATA_CATEGORY`: The name of the dataset collection (real, qrdata, synthetic, or other name)
* `OUTPUT_FOLDER`: Folder where the output is saved
* `LLM_NAME`: Name of the LLM
  
A specific example, 
```python
python main/run_agent.py -f data/test.csv -d data/all_data -t test -o output -l gpt-4o-mini 
```

 
