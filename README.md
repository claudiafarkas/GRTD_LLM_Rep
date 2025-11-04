# GRTD_LLM_Rep
## EECS 6412 - Replication Project for the Paper: Generating Realistic Tabular Data with Large Language Models
### Paper found here: https://arxiv.org/pdf/2410.21717 



## Instructions to Run the Project: 

### Files:
- pred_llm_mini_adult.ipynb: Run synthetic labeling on the Adult dataset.
- pred_llm_mini_bank.ipynb: Run synthetic labeling on the Bank dataset.
- requirements.txt: Holds all package and library requirments 
- data/: Folder where input datasets and generated outputs are stored.
- README.md: This file.
- .gitignore: Ignores unnecessary files like .DS_Store.


### Installation:
1. Clone the Repo:
   - ```https://github.com/username/GRTD_LLM_Rep.git```
   -  ```cd GRTD_LLM_Rep.git```
  
2. Set up the Env with Requirments.txt
   - ```pip install -r requirements.txt ```

3. Running (you can run the notebooks directly in Google Colab):
	1. Open either:
    - pred_llm_mini_adult.ipynb
    - pred_llm_mini_bank.ipynb
	2. Follow the cell-by-cell instructions to:
	  - Load the dataset
    - Generate synthetic rows
    - Query LLM for labels
    - Run TSTR evaluation and metrics




### Notes
- Make sure to enable GPU in Colab (Runtime > Change runtime type > GPU) for faster training and inference, if not availoable the code will choose CPU (will take longer)
- All outputs (CSVs) are saved to the data/ directory.
   
