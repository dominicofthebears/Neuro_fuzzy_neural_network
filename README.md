# Neuro_fuzzy_neural_network
Repository per il progetto di tesi "Sintesi di un algoritmo per la scopera di un modello neuro-fuzzy per la classificazione di cyber-minacce", A.A. 2021/2022

I dataset utilizzati sono disponibili ai seguenti link:
https://www.unb.ca/cic/datasets/maldroid-2020.html
https://www.unb.ca/cic/datasets/ids-2017.html

## Struttura del repository


        ├── anfis                            directory contenente il core del progetto, ovvero i file relativi alla rete neuro-fuzzy        	            
        │    ├ anfis_Mandami.py                 codice della rete neurale vera e propria         		
        │    ├ experimental.py                  parte di codice per l'apprendimento della rete neurale              
        │    ├ hyp_opt.py 	                    punto d'ingresso del programma, contenente la parte di ottimizzazione degli iperparametri
        │    ├ load_model.py                    contiene i metodi per la valutazione delle metriche di riferimento           		       
        │    ├ membership.py                    contiene i metodi per le funzioni di membership (gaussiane, triangolari ecc.)  
        │    ├ train_anfis.py                   contiene i metodi per l'apprendimento dei modelli e la loro ottimizzazione      
        ├── cn2                             directory contenente il codice ed i risultati relativi al cn2 rule learner
        │    ├ cn2_results                    directory ove sono salvati i risultati prodotti da cn2
        │    ├ cn2_rule_induction.py            contiene i metodi per l'istanziazione, il training e la valutazione di un rule learner e classifier
        ├── dataset_scripts                 directory contenente gli script per il preprocessing dei dataset              
        │    ├ CICIDS_script.py                 script per il preprocessing del dataset "CICIDS"            
        │    ├ maldroid_script.py               script per il preprocessing del dataset "maldroid"         
        ├── datasets                        directory contenente i dataset utilizzati 
        │    ├ CICIDS                         directory contenente il dataset "CICIDS"
        │    │  ├ 0_test_CICIDS2017Multiclass_R.csv  test set di CICIDS originale    
        │    │  ├ CICIDS_test_binary.csv        test set per la versione con target binario preprocessata di CICIDS 
        │    │  ├ CICIDS_test_multiclass.csv    test set per la versione con target multiclasse preprocessata di CICIDS  
        │    │  ├ CICIDS_train_binary.csv       training set per la versione con target binario preprocessata di CICIDS
        │    │  ├ CICIDS_train_multiclass.csv   training set per la versione con target multiclasse preprocessata di CICIDS
        │    │  └ train_CICIDS2017Multiclass_R.csv  training set di CICIDS originale 
        │    ├ maldroid                       directory contenente il dataset "maldroid"
        │    │  ├ feature_filtered_maldroid.csv  prima versione preprocessata di maldroid
        │    │  ├ maldroid_test_multiclass.csv   test set per la versione con target multiclasse preprocessata di maldroid  
        │    │  ├ maldroid_train_binary.csv     training set per la versione con target binario preprocessata di maldroid
        │    │  ├ maldroid_test_binary.csv      test set per la versione con target multiclasse preprocessata di maldroid
        │    │  └ maldroid_train_multiclass     training set per la versione con target multiclasse preprocessata di maldroid
        ├── feature_preprocessing            directory contenente i file utilizzati in fase di preprocessing
        │    ├ feature_ranking.py               contiene i metodi per la valutazione della Mutual Information delle features                               
        │    ├ feature_type_extraction.py       contiene metodi utilizzati inizialmente per determinare il tipo di ciascuna feature (continuo o discreto)           
        │    ├ preprocessing.py                 contiene i metodi di discretizzazione ed un frequency encoder usato in fase di preprocessing  
        ├── feature_preprocessing            directory contenente gli istrogrammi prodotti in alcuni esprimenti
        │    ├ maldroid                         directory contenente gli istogrammi prodotti in fase di discretizzazione delle features di maldroid per verificarne           │    │                                  la distribuzione di esempi
        ├── log_files                        directory contenente i file di log prodotti dagli esperimenti, sia per maldroid che per CICIDS
        ├── models                           directory contenente i file .h5, ovvero i modelli salvati al termine di ogni esperimento
        ├── plots                            directory contenente i grafici dell'errore dei modelli calcolato durante gli esperimenti
        │    ├ CICIDS                        directory contenente i grafici per il dataset CICIDS
        │    │  ├ 2  test set di CICIDS originale       directory contenente i grafici per il dataset CICIDS per esperimenti con 2 features
        │    │  └ 3  training set di CICIDS originale   directory contenente i grafici per il dataset CICIDS per esperimenti con 3 features
        │    ├ maldroid                      directory contenente i grafici per il dataset maldroid
        │    │  ├ 2  test set di CICIDS originale       directory contenente i grafici per il dataset maldroid per esperimenti con 2 features
        │    │  └ 3  training set di CICIDS originale   directory contenente i grafici per il dataset maldroid per esperimenti con 3 features
        ├── results                          directory contenente i file .csv nei quali sono salvati i risultati degli esperimenti
        ├── requirements.txt                 file contenente tutte le librerie necessarie per la corretta esecuzione del codice   		    
        └── README.md                        file di readme, esplicativo del contenuto delle directory e delle modalità di esecuzione del codice
        
## Modelli
I modelli vengono salvati nella directory "models", etichettati con il nome del dataset utilizzato seguito dal numero di features dell'esperimento e dall'approccio utilizzato (gaussiano o triangolare)

## Valutazione
Per ciascun esperimento, oltre al modello, vengono prodotti un file .csv salvato nella directory "results" con il nome del dataset seguito dal numero di features
e dall'approccio utilizzato ed un file di plot nella directory "plots", nella specifica directory del dataset e del numero di features dell'esperimento. Il nome
del file sarà "nomeDataset_numeroFeatures_learningRate"

## Installazione

    pip install -r requirements.txt
**Versione di python utilizzata: python 3.9**


## Guida all'utilizzo
Per avviare l'apprendimento di un modello, è necessario eseguire il file "hyp_opt.py", da ide o da linea di comando.<br> 
Il codice può essere eseguito in due differenti modi

  * Senza parametri: se si decide di eseguire il codice senza alcun parametro, l'esperimento di default che verrà condotto farà uso di maldroid come dataset, un numero di features pari a 2, approccio gaussiano e quindi senza alcun algoritmo di discretizzazione
  * Con parametri: se si decide di voler eseguire una determinata configurazione per anfis, bisogna inserire i seguenti parametri nell'ordine in cui sono elencati. Si precisa che i parametri sono Case sensitive, vanno quindi inseriti nella stessa forma indicata da questa guida
    1. Dataset: si può scegliere tra "maldroid" e "CICIDS"
    2. Target: si può scegliere tra "binary" per il target binario e "multiclass" per un target multiclasse
    3. Approccio: si può scegliere tra "gaussian" per l'approccio gaussiano e "triangular" per l'approccio triangolare
    4. Numero di features: si può scegliere qualsiasi valore intero superiore a 0 per il numero di features da tenere in considerazione
    5. Discretizzazione: se si è scelto l'approccio triangolare, è possibile selezionare i tre differenti algoritmi di discretizzazione inserendo "width" per l'equal width discretization, "frequency" per l'equal frequency discretization e "supervised" per la discretizzazione con algoritmo chimerge. Se invece si è selezionato l'approccio gaussiano, bisognerà necessariamente inserire il parametro "none" 

In questo modo verrà avviato l'apprendimento del modello di ANFIS, il quale richiederà 20 trial, ciascuno composto da 150 epoche a meno che non si verifichi la condizione di early stopping.

## File principali
Si propone una panoramica dei file principali e delle relative funzioni presenti nella directory "anfis", ovvero il core del programma stesso. <br>
Per maggiori dettagli, riferirsi ai commenti presenti nel codice

#### anfis_Mandami.py
Questo file rappresenta la rete neurale ANFIS vera e propria, ed il modo in cui è costituita a livello di layer singolo, nonchè il modo in cui essi vengono collegati e concatenati per ottenere la rete finale, realizzato tramite la classe "AnfisNet". Ciascun layer ed il suo funzionamento è descritto dai commenti presenti nel codice

#### experimental.py
In questo file troviamo il metodo "train_anfis_cat" utilizzato per addestrare la rete anfis. Tale metodo specifica il processo di apprendimento e validazione dei modelli, nonchè la funzione di errore di riferimento per la selezione del migliore di questi ultimi. Esso, infine, si occupa anche della stampa dei valori di loss e delle statistiche di accuracy per ciascuna epoca

#### hyp_opt.py
Questo file rappresenta il punto d'ingresso del codice, contiene la creazione e l'impostazione di hyperopt, e la funzione "fit_and_score" richiamata dalla stessa. Tramite la modifica di questo file si può andare a determinare lo spazio di ricerca ed in generale il flusso di esecuzione del codice.

#### load_model.py
Nel suddetto file troviamo il metodo "metrics", ovvero il metodo utilizzato per la valutazione delle metriche del miglior modello prodotto da ogni trial

#### membership.py
In questo file sono presenti le classi rappresentative delle varie funzioni di membership, quali quella triangolare, quella gaussiana, trapezoidale ecc.
Sono inoltre presenti i metodi "make_anfis" e "make_anfis_T", i quali si occupano di richiamare i metodi per la creazione delle funzioni di membership rispettivamente gaussiane e triangolari, per poi istanziare un oggetto di tipo "AnfisNet", a cui è delegata la creazione del modello

#### train_anfis.py
Infine, in questo file troviamo un metodo per effettuare il one hot encoding delle features in input, ed i metodi "opt" e "train", i quali si occupano di richiamare questo encoding, inserire i dati di input in due differenti tensori e darli in input alla rete. La differenza tra questi metodi è che il secondo viene richiamato al termine delle 20 prove per salvare il modello
        
