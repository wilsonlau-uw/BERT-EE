# BERT-EE
This repo contains the event extraction framework in the paper "Event-based clinical findings extraction from radiology reports with pre-trained language model". 
The framework directly processes annotated data saved in [BRAT's standoff format](https://brat.nlplab.org/standoff.html). 
The paths of training/ validation/ prediction data can be specified in config.ini file.  Alternatively, they can also be provided as command-line parameters.

Please refer to config.ini for each parameter setting.  There are separate sections to configure NER, RE and Model.  
Command-line parameters override the settings in config.ini.

The framework fine-tunes a single pre-trained BERT model for extracting named entities (sequence tagging) and relations,
 using multi-task learning (MTL), by minimizing the cross-entropy loss against the task's target labels.

Please cite our work.

```
@article{Wilson-Lau-2021-event-extraction,
    title = "Event-based clinical findings extraction from radiology reports with pre-trained language model",
    author = "Wilson Lau, Kevin Lybarger, Martin L. Gunn, Meliha Yetisgen",    
    url = "https://link.springer.com/article/10.1007/s10278-022-00717-5"
    }

```

### Acknowledgements
The framework is built upon code released in the following repo:

- https://github.com/chakki-works/seqeval 

