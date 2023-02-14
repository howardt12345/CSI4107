# CSI 4107 Group 1: Assignment 1

https://www.site.uottawa.ca/~diana/csi4107/A1_2023.htm

## Group Members

Howard Hao En Tseng, 300108234

Leah Young, 300118869

Shang Lin Heish, 300121996

### Contribution

| Name | Contribution |
| --- | --- |
| Howard Hao En Tseng | Initializing assignment project, "Getting Started" section, **Preprocessing**, initialized indexing, Topic querying with descriptions, README file |
| Leah Young | **Indexing** using the Pyterrier system |
| Shang Lin Heish | **Retrieval and Ranking**, processing the 50 queries and writing the results to Results.txt, fine-tuning results |

## About the Program
This program uses PyTerrier to index the documents and perform retrieval and ranking. The program also uses the NLTK library to preprocess the documents and queries.
Preprocessing is done by iterating through the files in the provided `AP_collection`, finding the documents in each file, and then tokenizing the documents and queries. The documents and queries are then converted to lowercase, and stopwords from the provided stopwords file are removed. The documents are also stemmed using the PorterStemmer algorithm. The queries are also lemmatized using the WordNetLemmatizer algorithm.
The documents are then indexed using PyTerrier. The queries are then processed and ranked using PyTerrier. The results are then written to a file called `Results.txt`.

# Getting Started

This section is the complete guide to setting up the project and running the program. Follow the steps below to get started.

Python 3.8 is **required** to install the packages used in this project. If you do not have Python 3.8 installed on your machine, you can download it from [here](https://www.python.org/downloads/release/python-380/).
Java 11 or greater is also required, and the JAVA_HOME environment variable must be set to the Java installation directory. Instructions on how to do so can be found [here](https://docs.oracle.com/cd/E19182-01/820-7851/inst_cli_jdk_javahome_t/).

### Recommended: Setting up a virtual environment

This is an optional, but recommended, step to set up a virtual environment for this project. This is especially important if your default Python version is not 3.8.
Before running the following commands, make sure you have python 3.8 installed on your machine. If not,  you can download it from [here](https://www.python.org/downloads/release/python-380/).

To set up the virtual environment, run the following commands:

```powershell
pip install virtualenv
python -m virtualenv .venv --python=python3.8
.venv/Scripts/Activate.ps1
```

### Installing packages

To install the packages used in this project, run the following command:

```bash
pip install -r requirements.txt
```

### Running the program

To run the program, run the following command:

```bash
python main.py
```

# Explanation of Algorithms




# Testing and Results

## Testing methodology



## Results

BM25:
| Run | MAP score |
| --- | --- |
| Titles Only | 0.3183 |
| Titles and Descriptions | 0.3214 |

TF-IDF:
| Run | MAP score |
| --- | --- |
| Titles Only | 0.3206 |
| Titles and Descriptions | 0.3238 |

### Discussion



## Optimizations

- Fine tuned the search parameters for retrieval to improve the MAP score
  - Using the same preprocessing algorthm as the one used in preprocessing improved map scores by [amount for both BM25 and TF-IDF]