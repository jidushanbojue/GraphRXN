from rdkit import Chem
import os
import pandas as pd

def canonicalize_smi(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def make_reaction_smiles_Denmark(file, sheet_name):
    """
    file: reaction file
    """
    rxn_list = []
    df = pd.read_excel(file, sheet_name=sheet_name)
    for idx, line in df.iterrows():
        catalyst = canonicalize_smi(line['Catalyst'])
        imine = canonicalize_smi(line['Imine'])
        Thiol = canonicalize_smi(line['Thiol'])
        rxn = '*'.join([catalyst, imine, Thiol])
        rxn_list.append(rxn)
    return rxn_list, df['Output']

def make_reaction_smiles_Cernak(file, sheet_name):
    """
    file: reaction file
    """
    rxn_list = []
    df = pd.read_excel(file, sheet_name=sheet_name)
    for idx, line in df.iterrows():
        Electrophile = canonicalize_smi(line['Electrophile'])
        Nucleophile = canonicalize_smi(line['Nucleophile'])
        Catalyst = canonicalize_smi(line['Catalyst'])
        Base = canonicalize_smi((line['Base']))
        rxn = '*'.join([Electrophile, Nucleophile, Catalyst, Base])
        rxn_list.append(rxn)
    return rxn_list, df['Output']

    # print('Done')


if __name__ == '__main__':


    ### Denmark
    base_dir = '/data/baiqing/PycharmProjects/CMPNN-master-concat_1/data/'
    reaction_file = os.path.join(base_dir, 'Denmark_input_data.xlsx')


    sheet_name_list = ['FullCV_01',
                       'FullCV_02',
                       'FullCV_03',
                       'FullCV_04',
                       'FullCV_05',
                       'FullCV_06',
                       'FullCV_07',
                       'FullCV_08',
                       'FullCV_09',
                       'FullCV_10',
                       ]

    for sheet_name in sheet_name_list:
        rxn_list, output = make_reaction_smiles_Denmark(reaction_file, sheet_name=sheet_name)
        new_df = pd.DataFrame()
        new_df['rxn'] = rxn_list
        new_df['Output'] = output
        train_df = new_df[:601]
        test_df = new_df[601:]

        train_df.to_csv(os.path.join(base_dir, 'Denmark', sheet_name+'_train.csv'), index=False)
        test_df.to_csv(os.path.join(base_dir, 'Denmark', sheet_name+'_test.csv'), index=False)


    print('Done')

    ### Cernak
    # base_dir = '/data/baiqing/PycharmProjects/CMPNN-master-concat_1/data/'
    # reaction_file = os.path.join(base_dir, 'Cernak_and_Dreher_input_data.xlsx')
    #
    #
    # sheet_name_list = ['Blatt1',
    #                    'Blatt2',
    #                    'Blatt3',
    #                    'Blatt4',
    #                    'Blatt5',
    #                    'Blatt6',
    #                    'Blatt7',
    #                    'Blatt8',
    #                    'Blatt9',
    #                    'Blatt10',
    #                    ]
    #
    # for sheet_name in sheet_name_list:
    #     rxn_list, output = make_reaction_smiles_Cernak(reaction_file, sheet_name=sheet_name)
    #     new_df = pd.DataFrame()
    #     new_df['rxn'] = rxn_list
    #     new_df['Output'] = output
    #     train_df = new_df[:1076]
    #     test_df = new_df[1076:]
    #
    #     train_df.to_csv(os.path.join(base_dir, 'Cernak', sheet_name+'_train.csv'), index=False)
    #     test_df.to_csv(os.path.join(base_dir, 'Cernak', sheet_name+'_test.csv'), index=False)
    #
    #
    # print('Done')
