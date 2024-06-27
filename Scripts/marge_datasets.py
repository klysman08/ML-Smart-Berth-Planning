import pandas as pd
import matplotlib.pyplot as plt

#carregar arquivos csv 
navios_sines_imos = pd.read_csv('Datasets/dataset_navios_with_imos.csv')
imos_navios_caracteristicas = pd.read_csv('Datasets/df_imos_caracteristicas.csv')
porto_sines_caracteristicas = pd.read_csv('Datasets/porto_caracteristicas_terminais.csv', encoding='latin1')


#função para juntar os datasets com base no IMOs entre navios_sines_imos e imos_navios_caracteristicas
def merge_datasets():
    navios_sines_imos['Imo'] = navios_sines_imos['Imo'].astype(str)
    imos_navios_caracteristicas['IMO number'] = imos_navios_caracteristicas['IMO number'].astype(str)
    df = pd.merge(navios_sines_imos, imos_navios_caracteristicas, left_on='Imo', right_on='IMO number')
    return df

def novo_dataframe(df):
    df_modelagem = df[['Berth Name', 'Terminal Name', 'Time At Berth', 'Time At Port' , 'Vessel Type - Generic', 'Commercial Market','Voyage Distance Travelled', 'Voyage Speed Average', 'Year of build', 'Voyage Origin Port', 'Flag', 'Gross tonnage', 'Deadweight', 'Length', 'Breadth' ]]

    #remover linhas  com valores nulos
    df_modelagem = df_modelagem.dropna()

    #salvar novo dataset em csv
    df_modelagem.to_csv('Datasets/dataset_modelagem.csv', index=False)

if __name__ == '__main__':
    df = merge_datasets()
    novo_dataframe(df)
    print('Dataset criado com sucesso!')

