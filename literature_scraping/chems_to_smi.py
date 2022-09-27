from chemdataextractor.doc import Document, Paragraph
from tqdm import tqdm
import json
import os
import requests
import xmltodict
import pubchempy as pcp
import cirpy


def getAllChems():
    with open('all_scop_sf_desc.txt', 'r') as file:
        desc_data = json.load(file)

    if os.path.isfile('all_scop_sf_chems.txt'):
        with open('all_scop_sf_chems.txt', 'r') as file:
            chems = json.load(file)
    else:
        chems = []
        with open('all_scop_sf_chems.txt', 'w') as file:
            json.dump(chems, file)

    for index, doc in enumerate(tqdm(desc_data)):
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


def loadAllChems():
    with open('all_scop_unique_sf_chems.txt', 'r') as file:
        chems = json.load(file)
    print(len(chems))
    return chems


def ChemToSmiles():
    with open('all_scop_unique_sf_chems.txt', 'r') as file:
        allChems = json.load(file)

    molnames = []
    smistr = []
    errCount = 0
    # totTime = 0
    # avgTime = 0
    skipped = 0

    if os.path.isfile('all_scop_smiles_pc_sf.txt'):
        with open('all_scop_smiles_pc_sf.txt', 'r') as file:
            smi_list = json.load(file)
    else:
        smi_list = []
        with open('all_scop_smiles_pc_sf.txt', 'w') as file:
            json.dump(smi_list, file)

    if os.path.isfile('all_scop_unique_smiles_pc_sf.txt'):
        with open('all_scop_unique_smiles_pc_sf.txt', 'r') as file:
            smi_list_unique = json.load(file)
    else:
        smi_list_unique = []
        with open('all_scop_unique_smiles_pc_sf.txt', 'w') as file:
            json.dump(smi_list_unique, file)

    for data in smi_list:
        molnames.append(data['name'])
        if data['smiles'] is None:
            errCount = errCount + 1

    for data in smi_list_unique:
        smistr.append(data['smiles'])

    for index, mol in enumerate(tqdm(allChems)):
        # print("    " + str(index) + " / " + str(len(allChems)) + " errs: " + str(errCount) + " skipped: " + str(
        #     skipped) + " avg time: " + str(avgTime), end='\r')

        #     if(mol!='9,10-diphenylanthracene'):
        #         continue

        if (mol in molnames):
            skipped = skipped + 1
            continue

        # start = time.time()

        # cid1 = []
        cid = None
        sid = None

        cid1 = pcp.get_compounds(mol, 'name')
        if (len(cid1) > 0):
            cid = cid1[0].cid
        else:
            results = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pccompound&term=\"" + mol + "\"&sort=relevance")
            resultsjson = xmltodict.parse(results.text)
            try:
                cidresults = resultsjson['eSearchResult']['IdList']['Id']
            except:
                cidresults = []
            if (type(cidresults) == str):
                cid = int(cidresults)
            else:
                if (len(cidresults) > 0):
                    cid = int(cidresults[0])
                else:
                    results = requests.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pccompound&term=" + mol + "&sort=relevance")
                    try:
                        resultsjson = xmltodict.parse(results.text)
                    except:
                        resultsjson = {}
                    try:
                        cidresults = resultsjson['eSearchResult']['IdList']['Id']
                    except:
                        cidresults = []
                    if (type(cidresults) == str):
                        cid = int(cidresults)
                    else:
                        if (len(cidresults) > 0):
                            cid = int(cidresults[0])
                        else:
                            results = requests.get(
                                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcsubstance&term=\"" + mol + "\"&sort=relevance")
                            try:
                                resultsjson = xmltodict.parse(results.text)
                            except:
                                resultsjson = []
                            try:
                                sidresults = resultsjson['eSearchResult']['IdList']['Id']
                            except:
                                sidresults = []
                            if (type(sidresults) == str):
                                sid = int(sidresults)
                            else:
                                if (len(sidresults) > 0):
                                    sid = int(sidresults[0])
        iupac_name = None
        inchi_key = None
        smiles = None
        dctype = None
        csid = None

        if (cid is not None):
            try:
                comp = pcp.Compound.from_cid(cid)
                iupac_name = comp.iupac_name
                inchi_key = comp.inchikey
                smiles = comp.canonical_smiles
                dctype = 'compound'
                csid = cid
            except:
                pass
        elif (sid is not None):
            try:
                comp = pcp.Substance.from_sid(sid)
                cid = comp.standardized_cid
                comp = pcp.Compound.from_cid(cid)
                iupac_name = comp.iupac_name
                inchi_key = comp.inchikey
                smiles = comp.canonical_smiles
                dctype = 'substance'
                csid = cid
            except:
                pass
        elif (cid is None and sid is None):
            errCount = errCount + 1
            smiles = cirpy.resolve(mol, 'smiles')
            if smiles is not None:
                iupac_name = cirpy.resolve(smiles, 'iupac_name')
                try:
                    inchi_key = cirpy.resolve(smiles, 'stdinchikey')
                except urllib.request.HTTPError:
                    inchi_key = None
                # smiles = smi
                dctype = 'cirpy'
            else:
                url = 'https://opsin.ch.cam.ac.uk/opsin/' + mol
                try:
                    r = requests.get(url).json()
                except json.JSONDecodeError:
                    r = {}
                try:
                    smiles = r['smiles']
                except KeyError:
                    pass
                try:
                    inchi_key = r['stdinchikey']
                except KeyError:
                    pass
                if smiles is not None:
                    dctype = 'opsin'

        temp = {}
        temp['name'] = mol
        temp['iupac_name'] = iupac_name
        temp['inchi_key'] = inchi_key
        temp['smiles'] = smiles
        temp['dctype'] = dctype
        temp['csid'] = csid
        smi_list.append(temp)

        if smiles is not None and smiles not in smistr:
            smi_list_unique.append(temp)
            smistr.append(smiles)

        with open("all_scop_smiles_pc_sf.txt", 'w') as file:
            json.dump(smi_list, file)
        with open("all_scop_unique_smiles_pc_sf.txt", 'w') as file:
            json.dump(smi_list_unique, file)

        # elapsed = time.time() - start
        # totTime = totTime + elapsed
        # avgTime = totTime / (index + 1)

    print(len(smi_list))
    print(len(smi_list_unique))


# chems = getAllChems()
# chems = loadAllChems()
ChemToSmiles()
