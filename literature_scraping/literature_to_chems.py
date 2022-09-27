from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from elsapy.elsdoc import AbsDoc
import json
import pandas as pd
import os
from tqdm import tqdm
from p_tqdm import p_map
from chemdataextractor.doc import Document, Paragraph
import requests
import xmltodict
import cirpy


def getScopSearchResults(searchTerm):
    print('searching for ' + searchTerm + ' papers')
    doc_srch = ElsSearch('\"' + searchTerm + '\"', 'scopus')
    doc_srch.execute(client, get_all=True)
    print('doc_srch has', len(doc_srch.results), 'results.')
    with open(searchTerm.replace(' ', '_') + '_results.json', 'w') as file:
        json.dump(doc_srch.results, file)
    # return doc_srch.results


# def loadExistingScopResults():
#     with open('dump.json', 'r') as file:
#         scop_srch_results = json.load(file)
#     return scop_srch_results


def compileAbstracts(searchTerm):
    with open(searchTerm.replace(' ', '_') + '_results.json', 'r') as file:
        scop_srch_results = json.load(file)
    if os.path.isfile('all_scop_abstracts.csv'):
        df_all = pd.read_csv('all_scop_abstracts.csv')
    else:
        df_all = pd.DataFrame(columns=['doi', 'scp_id', 'title', 'abstract'])
    data = []

    def get_article_data(result):
        try:
            doi = result['prism:doi']
        except KeyError:
            # print(result)
            doi = None
        scpid = result['dc:identifier'].split(':')[1]
        title = result['dc:title']
        scp_doc = AbsDoc(scp_id=scpid)
        scp_doc.read(client)
        # try:
        abstract = scp_doc.data['item']['bibrecord']['head']['abstracts']
        # except KeyError:
        # return
        # except TypeError:
        # return
        return doi, scpid, title, abstract
    data = p_map(get_article_data, scop_srch_results[:10])
    # for result in tqdm(scop_srch_results):
    #     try:
    #         doi = result['prism:doi']
    #     except KeyError:
    #         # print(result)
    #         doi = None
    #     scpid = result['dc:identifier'].split(':')[1]
    #     title = result['dc:title']
    #     scp_doc = AbsDoc(scp_id=scpid)
    #     scp_doc.read(client)
    #     try:
    #         abstract = scp_doc.data['item']['bibrecord']['head']['abstracts']
    #     except KeyError:
    #         continue
    #     data.append([doi, scpid, title, abstract])
    print(data)
    # df = pd.DataFrame(data, columns=['doi', 'scp_id', 'title', 'abstract'])
    # df_all = df_all.merge(df, on='doi', how='outer')
    # df_all.to_csv('all_scop_abstracts.csv', index=False)


def get_all_chems():
    df = pd.read_csv('all_scop_abstracts.csv')

    for doc in tqdm(df['abstract']):
        # print("    " + str(index) + " / " + str(len(desc_data)), end='\r')
        if doc['abstract'] is None:
            continue
        doct = Document(doc['abstract'])
        p = doct.elements[0]
        abbrevslist = p.abbreviation_definitions
        abbrevs = []
        for abbrev in abbrevslist:
            if (abbrev[0] not in abbrevs):
                abbrevs.append(abbrev[0][0])
        for cem in doct.cems:
            if cem.text not in chems and cem.text not in abbrevs:
                chems.append(cem.text)
            # else:
            #     print('chem already exists')
        with open('all_scop_sf_chems.txt', 'w') as file:
            json.dump(chems, file)

    print("\n")

    with open('all_scop_sf_chems.txt', 'r') as file:
        chems = json.load(file)
    print(len(chems))
    unique_chems = []
    for chem in chems:
        if chem is not None and chem not in unique_chems:
            unique_chems.append(chem)
    print(len(unique_chems))

    with open('all_scop_unique_sf_chems.txt', 'w') as file:
        json.dump(unique_chems, file)

    return unique_chems


def main():
    # searchTerms = ['triplet triplet annihilation', 'triplet triplet fusion',
    #                'singlet fission', 'photon upconversion', 'photon downconversion',
    #                'luminescent downshifting', 'spectral conversion']
    searchTerms = ['triplet triplet annihilation']
    for searchTerm in searchTerms:
        # getScopSearchResults(searchTerm)
        compileAbstracts(searchTerm)
    # get_all_chems()
    # scop_srch_results = loadExistingScopResults()
    # compileAbstracts(scop_srch_results)


if __name__ == "__main__":
    client = ElsClient('1826bc4e710224ea3c32d0d415ac390e', num_res=50)
    main()
