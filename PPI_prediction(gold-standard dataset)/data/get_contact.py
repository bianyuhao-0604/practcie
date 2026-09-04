import requests
import json
import pandas as pd
from multiprocessing import Process


def run_pdb_query(prot_id1, prot_id2):
    # looks for PDB structures with the specified uniprot IDs.
    # The number of distinct protein entities in the entry is limited to 2.
    myquery = {
        "query": {
            "type": "group",
            "nodes": [
                {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                                "operator": "in",
                                "negation": False,
                                "value": [
                                    prot_id1
                                ]
                            }
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                                "operator": "exact_match",
                                "value": "UniProt",
                                "negation": False
                            }
                        }
                    ],
                    "label": "nested-attribute"
                },
                {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                                "operator": "in",
                                "negation": False,
                                "value": [
                                    prot_id2
                                ]
                            }
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                                "operator": "exact_match",
                                "value": "UniProt",
                                "negation": False
                            }
                        }
                    ],
                    "label": "nested-attribute"
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "negation": False,
                        "value": 2
                    }
                }
            ],
            "logical_operator": "and",
            "label": "text"
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 25
            },
            "results_content_type": [
                "experimental"
            ],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ],
            "scoring_strategy": "combined"
        }
    }
    myquery = json.dumps(myquery)
    data = requests.get(f"https://search.rcsb.org/rcsbsearch/v2/query?json={myquery}")
    if data.status_code == 200:
        results = data.json()['result_set']
        if len(results) > 0:
            return [results[i]['identifier'] for i in range(len(results))]
    return None


def get_pdb_info(file, id):
    interactions = pd.read_csv(file, header=None, sep=' ')
    interactions.columns = ['prot1', 'prot2']
    # sample 1000 random rows
    #interactions = interactions.sample(1000)
    ppis_with_structures = {}
    count = 0
    for index, row in interactions.iterrows():
        if count % 100 == 0:
            print(f'row {count} of {len(interactions)}')
        prot1 = row['prot1']
        prot2 = row['prot2']
        result = run_pdb_query(prot1, prot2)
        if result is not None:
            ppis_with_structures[f'{prot1}_{prot2}'] = {'pdb_matches': result}
        count += 1
    df = pd.DataFrame.from_dict(ppis_with_structures, orient='index')
    # split index into prot1 and prot2
    df.index = df.index.str.split('_', expand=True)
    df.index.names = ['prot1', 'prot2']
    df.to_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/ppis_with_structures_{id}.csv')
    
    
if __name__ == "__main__":
    # Create process objects
    p1 = Process(target=get_pdb_info, args=('/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/Intra0_pos_rr.txt', 'Intra0_pos_rr'))
    p2 = Process(target=get_pdb_info, args=('/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/Intra1_pos_rr.txt', 'Intra1_pos_rr'))
    p3 = Process(target=get_pdb_info, args=('/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/Intra2_pos_rr.txt', 'Intra2_pos_rr'))

    # Start the processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for all processes to finish
    p1.join()
    p2.join()
    p3.join()
