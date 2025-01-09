# UNIVERSITA' DEGLI STUDI DI NAPOLI FEDERICO II
# NEURAL NETWORK AND DEEP LEARNING
## Authors

    Giuseppe Cicchella N97000452
    Raffaele D'Anna N97000455

---

# Confronto tra RProp Standard e le sue varianti per la classificazione di immagini MNIST
Implementazione di reti neurali artificiali addestrate sul dataset MNIST usando l'algoritmo di ottimizzazione Rprop (e le sue varianti). Include una libreria per la simulazione della propagazione in avanti e della retropropagazione con diverse funzioni di attivazione. Valutazione delle prestazioni attraverso accuratezza sui set di addestramento, validazione e test e tempo medio di addestramento.
All'interno della seguente repository, oltre ai file sorgenti, è presente un quaderno Jupyter che offre un ambiente interattivo per l’addestramento e l’analisi di reti neurali applicate alla classificazione delle cifre del dataset MNIST, permettendo agli utenti di esplorare diverse configurazioni di reti e algoritmi di ottimizzazione, con particolare focus sulle varianti dell’algoritmo Rprop (STANDARD, RPROP_PLUS, IRPROP).

## Prerequisiti

Prima di iniziare, è essenziale assicurarsi di avere i seguenti strumenti e dipendenze installati sul tuo sistema:

1. **Python** <img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" alt="Python Logo" width="60"/>  
   Verifica la tua versione:
   ```bash
   python --version

2. **Jupyter Notebook** <img src="https://jupyter.org/assets/homepage/main-logo.svg" alt="Jupyter Notebook" width="70"/>
   ```bash
   pip install notebook

## Istruzioni per l'uso
1. **Clona la repository**
   ```bash
   git clone https://github.com/rafdan28/ANN-Mnist-RProp.git
2. **Esecuzione del quaderno Jupyter**
   ```bash
   jupyter notebook

## Contenuti della repository
- __src/__: Contiene il quaderno Jupyter per testare la libreria.
- __src/nndlpy__: Contiene i file sorgenti della libreria.